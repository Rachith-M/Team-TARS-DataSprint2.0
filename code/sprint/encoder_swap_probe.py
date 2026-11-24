"""Screen frozen replacement encoders with Bhasha's labeled, grouped training recipe."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import PARAMETER_CAP
from sprint.augment import tta_variant
from sprint.bhasha_audit import BASE_VIEWS, REQUEST_VIEWS, dictionary_inputs, metric_delta, route_metrics, score_parts
from sprint.data import SEED, dataset_key
from sprint.encoder import MAX_LEN, embed_bhasha, session, tokenized
from sprint.romansetu_distill import perturbation_interval
from sprint.score_push import fit_candidate, prepare, samples
from sprint.train import take


MODELS = {
    'horizon': ('Horizon-Labs/multilingual-embedding-small', '033028575b040790aef3064b2a045245ee54d238'),
    'hindsbert': ('l3cube-pune/hindi-sentence-similarity-sbert', '1ac1c75727339338dfd67de7346eb2fda0798267'),
}


def horizon_vectors(args, texts):
    from tokenizers import Tokenizer

    checkpoint = args.cache / args.model
    runtime = session(checkpoint / 'onnx' / 'model.onnx', args.threads)
    names = [item.name for item in runtime.get_inputs()]
    tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
    tokenizer.enable_truncation(MAX_LEN)
    pad = next(tokenizer.token_to_id(word) for word in ('<pad>', '[PAD]') if tokenizer.token_to_id(word) is not None)
    tokenizer.enable_padding(pad_id=pad)
    encoded = np.empty((len(texts), 1024), dtype=np.float32)
    started = perf_counter()
    for offset in range(0, len(texts), args.batch_size):
        batch = tokenized(texts[offset:offset + args.batch_size], tokenizer, names, prefixed=False)
        values = runtime.run(None, batch)[0]
        if values.shape != (len(batch['input_ids']), encoded.shape[1]):
            raise ValueError('Expected the published Horizon sentence embeddings')
        values = values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)
        encoded[offset:offset + len(values)] = values
        if offset and offset % (args.batch_size * 80) == 0:
            print(f'Encoded {offset:,}/{len(texts):,} in {perf_counter() - started:.0f}s', flush=True)
    return encoded, 140_493_696 + 384 * 1024


def vectors_for(args, views):
    texts = sorted(dict.fromkeys(text for current in views.values() for text in current), key=len)
    fingerprint = dataset_key(texts)
    revision = MODELS[args.model][1]
    path = args.cache / f'{args.model}-{revision[:8]}-{fingerprint[:16]}.npz'
    if not path.exists():
        print(f'Encoding {len(texts):,} distinct views with {args.model}', flush=True)
        if args.model == 'horizon':
            vectors, parameters = horizon_vectors(args, texts)
        else:
            import torch

            torch.set_num_threads(args.threads)
            values, parameters = embed_bhasha({'all': texts}, args.cache / args.model, 'cpu', args.batch_size)
            vectors = values['all']
        if len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError('Invalid replacement embeddings')
        temporary = path.with_suffix('.partial.npz')
        np.savez(temporary, vectors=vectors, parameters=parameters, fingerprint=fingerprint)
        temporary.replace(path)
    with np.load(path) as cached:
        matrix = cached['vectors']
        parameters = int(cached['parameters'])
        if cached['fingerprint'].item() != fingerprint or len(matrix) != len(texts):
            raise ValueError('The replacement cache does not match the requested views')
    index = {text: row for row, text in enumerate(texts)}
    return {name: matrix[[index[text] for text in rows]] for name, rows in views.items()}, parameters


def run(args):
    from intent_model import Views

    args.phrases = args.counterfactuals = True
    args.rewrites = None
    groups, labels, folds, _, _, scripts, config, _, variants, _ = prepare(args, include_reference_vectors=False)
    index = args.fold - 1
    fit, check = folds[index]
    seed = SEED + index
    groups['asr_tta'] = list(map(tta_variant, groups['asr']))
    source = {name: groups[name] for name in (*BASE_VIEWS, 'aug')}
    source.update({name: texts for name, texts, *_ in variants})
    needed = {name: set(check) for name in BASE_VIEWS}
    parts = samples(config, groups, labels, fit, check, variants, seed)
    for name, _, rows, _, _ in parts:
        needed.setdefault(name, set()).update(rows)
    rows = {name: np.array(sorted(positions)) for name, positions in needed.items()}
    views = {name: take(source[name], positions) for name, positions in rows.items()}
    dictionary, changed, native_text = dictionary_inputs(groups, fit, check)
    views.update({f'dictionary_{view}': texts for view, texts in native_text.items()})
    trimmed, parameters = vectors_for(args, views)
    matrix = {}
    for name, positions in rows.items():
        matrix[name] = np.full((len(source[name]), trimmed[name].shape[1]), np.nan, dtype=np.float32)
        matrix[name][positions] = trimmed[name]
    with threadpool_limits(limits=args.threads):
        model, counts = fit_candidate(config, groups, labels, matrix, variants, fit, check, seed)
        native_vectors = {view: trimmed[f'dictionary_{view}'] for view in REQUEST_VIEWS}
        candidate = score_parts(model, groups, matrix, check, changed, native_text, native_vectors)[1]
    total = (parameters + model.head.coef_.size + model.head.intercept_.size + model.features.parameter_count()
             + 2 * len(model.classes) + 8 + sum(map(len, dictionary)))
    with np.load(args.audit) as history:
        if not np.array_equal(history['labels'], labels) or not np.array_equal(np.sort(check), np.flatnonzero(history['fold_id'] == index)):
            raise ValueError('The archived Bhasha control is for a different grouped split')
        baseline = {view: history[f'always_dictionary_{view}'][check] for view in REQUEST_VIEWS}
    base = route_metrics(baseline, labels[check], scripts[check])
    result = route_metrics(candidate, labels[check], scripts[check])
    delta = metric_delta(result, base)
    keys = np.asarray(Views.from_texts(take(groups['raw'], check)).phonetic)
    interval = perturbation_interval(labels[check], baseline['perturbed'].argmax(1), candidate['perturbed'].argmax(1), keys)
    report = {'model': MODELS[args.model][0], 'revision': MODELS[args.model][1], 'screen_fold': args.fold,
              'train_counts': counts, 'baseline': base, 'candidate': result, 'delta': delta,
              'grouped_interval_95pct': interval, 'encoder_parameters': parameters,
              'estimated_pipeline_parameters': total, 'within_parameter_cap': total <= PARAMETER_CAP,
              'follow_up_warranted': (delta['perturbed'] >= .003 and delta['roman_asr'] >= -.002
                                      and total <= PARAMETER_CAP)}
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'AnveshanNLP'
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--cache', type=Path, default=root / 'encoder-replacement-screen')
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--audit', type=Path, default=Path('sprint_results/score_push/bhasha_full_audit_run/out_of_fold_scores.npz'))
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.threads, args.batch_size) < 1:
        parser.error('Use a new output path and positive batch size and thread count')
    run(args)


if __name__ == '__main__':
    main()
