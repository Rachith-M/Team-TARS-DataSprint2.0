"""Screen a frozen local GGUF encoder on a labeled grouped fold."""

import argparse
import base64
import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from time import monotonic, sleep
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import LABELS, script
from sprint.data import SEED, dataset_key, folds, load_training
from sprint.encoder import variant_groups
from sprint.score_push import metrics
from sprint.train import EVAL_VIEWS, augmentation_rows, fit_fold, take


EMBEDDING_SLOTS = 32


class LocalOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def stratified_sample(rows, labels, limit, rng):
    by_class = [rows[labels[rows] == label] for label in LABELS]
    return np.sort(np.concatenate([rng.choice(group, min(len(group), limit), replace=False)
                                   for group in by_class]))


def extract(args, inputs, context_size=4096):
    with socket.socket() as address:
        address.bind(('127.0.0.1', 0))
        port = address.getsockname()[1]
    env = os.environ | {'GGML_BACKEND_PATH': str(args.backend.resolve()),
                        'PATH': str(args.backend.resolve().parent) + os.pathsep + os.environ['PATH']}
    command = [str(args.server.resolve()), '--model', str(args.model.resolve()), '--host', '127.0.0.1',
               '--port', str(port), '--no-webui', '--no-slots', '--embedding', '--pooling', 'last',
               '--device', 'Vulkan0', '--gpu-layers', 'all', '--parallel', str(EMBEDDING_SLOTS),
               '--ctx-size', str(context_size),
               '--batch-size', '2048', '--ubatch-size', '512']
    opener = build_opener(ProxyHandler({}), LocalOnly())

    def request(path, body=None, timeout=2):
        payload = None if body is None else json.dumps(body).encode('utf-8')
        query = Request(f'http://127.0.0.1:{port}{path}', data=payload,
                        headers={'Content-Type': 'application/json'})
        with opener.open(query, timeout=timeout) as response:
            return json.load(response)

    started = monotonic()
    with tempfile.TemporaryFile('w+b') as log:
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log, creationflags=flags)
        try:
            until = started + 90
            while True:
                try:
                    request('/health')
                    break
                except OSError:
                    if process.poll() is not None or monotonic() >= until:
                        raise
                    sleep(.25)
            vectors = []
            for offset in range(0, len(inputs), 64):
                batch = inputs[offset:offset + 64]
                result = request('/v1/embeddings', {'input': batch, 'encoding_format': 'base64'}, 120)
                rows = sorted(result['data'], key=lambda item: item['index'])
                if [row['index'] for row in rows] != list(range(len(batch))):
                    raise ValueError('Unexpected embedding indices from llama-server')
                vectors.extend(np.frombuffer(base64.b64decode(row['embedding'], validate=True), dtype='<f4')
                               for row in rows)
                if offset % 1024 == 0:
                    print(f'Encoded {len(vectors):,}/{len(inputs):,} in {monotonic() - started:.0f}s', flush=True)
        except Exception:
            log.seek(0)
            print(log.read().decode(errors='replace')[-6000:], flush=True)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    embeddings = np.stack(vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('Teacher returned an invalid embedding')
    return embeddings / norms, monotonic() - started


def run(args):
    texts, labels = load_training(args.train)
    fold_index = args.fold - 1
    outer_fit, outer_check = folds(texts, labels)[fold_index]
    rng = np.random.default_rng(SEED + fold_index)
    fit = stratified_sample(outer_fit, labels, args.train_per_class, rng)
    check = stratified_sample(outer_check, labels, args.validation_per_class, rng)
    settings = json.loads((args.reference / 'metrics.json').read_text(encoding='utf-8'))
    config = settings['config']
    if config['oos'] or settings['tta_weight'] != .35 or settings['cluster_weight']:
        raise ValueError('Expected the archived E5-small configuration')

    scripts = np.asarray([script(text) for text in texts])
    with np.load(args.reference / 'scores.npz') as archive:
        if not np.array_equal(archive['target'], labels) or not np.array_equal(
                np.flatnonzero(archive['fold_id'] == fold_index), outer_check):
            raise ValueError('The saved labels or fold do not match the training CSV')
        archived_scores = {view: archive[view][check] for view in EVAL_VIEWS}
    groups = variant_groups(texts, [])
    selected = {view: check for view in EVAL_VIEWS}
    selected['raw'] = np.union1d(fit, check)
    selected['aug'] = augmentation_rows(groups, fit, config['augment'], SEED + fold_index, check=check)
    unique_texts = list(dict.fromkeys(text for view, rows in selected.items() for text in take(groups[view], rows)))
    print(f'Fold {args.fold}: {len(fit):,} fitting rows, {len(check):,} evaluation rows, '
          f'{len(unique_texts):,} unique inputs', flush=True)
    embeddings, seconds = extract(args, unique_texts)
    positions = {text: i for i, text in enumerate(unique_texts)}
    vectors = {}
    for view, rows in selected.items():
        vectors[view] = np.empty((len(texts), embeddings.shape[1]), dtype=np.float32)
        vectors[view][rows] = embeddings[[positions[text] for text in take(groups[view], rows)]]

    def evaluate(scores):
        return metrics(scores, labels[check], scripts[check])

    def fit_and_score(embedding):
        with threadpool_limits(limits=args.threads):
            model = fit_fold(config, groups, labels, embedding, fit, SEED + fold_index, check=check)
            scores = {view: model.score(take(groups[view], check),
                                        None if embedding is None else embedding[view][check]) for view in EVAL_VIEWS}
        return evaluate(scores)

    probe = fit_and_score(vectors)
    lexical = fit_and_score(None)
    result = {'model_file': args.model.name, 'train_dataset_key': dataset_key(texts), 'fold': args.fold,
              'fitting_rows': len(fit), 'fitting_rows_per_label_cap': args.train_per_class,
              'evaluation_rows': len(check), 'evaluation_rows_per_label_cap': args.validation_per_class,
              'embedding_dimensions': embeddings.shape[1], 'encoded_queries': len(unique_texts),
              'embedding_seconds': seconds, 'pooling': 'last', 'backend': 'Vulkan0',
              'teacher_plus_lexical': probe, 'matched_lexical': lexical,
              'archived_e5_with_full_fold_training': evaluate(archived_scores),
              'gain_over_matched_lexical': {name: probe[name] - lexical[name] for name in probe}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('server', 'backend', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--train', type=Path, default=Path('train.csv'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--train-per-class', type=int, default=48)
    parser.add_argument('--validation-per-class', type=int, default=24)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.train_per_class, args.validation_per_class, args.threads) < 1:
        parser.error('Sample caps and thread count must be positive')
    if args.output.exists():
        parser.error(f'Output already exists at {args.output}')
    run(args)


if __name__ == '__main__':
    main()
