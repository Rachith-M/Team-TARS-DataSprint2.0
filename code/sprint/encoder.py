import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import numpy as np

from sprint.augment import augment_batch, tta_variant
from sprint.data import SEED, dataset_key, load_data


CHECKPOINTS = {
    'e5-small': ('intfloat/multilingual-e5-small', '614241f622f53c4eeff9890bdc4f31cfecc418b3'),
    'e5-base': ('intfloat/multilingual-e5-base', 'd128750597153bb5987e10b1c3493a34e5a4502a'),
}
EXTENSIONS = {'encoder.onnx': 'model_qint8_avx512_vnni.onnx', 'tokenizer.json': 'tokenizer.json'}
MAX_LEN = 64


def embed_bhasha(texts_by_view, checkpoint, device, batch_size):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, add_pooling_layer=False,
                                      attn_implementation='sdpa').to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters < 290_000_000
    vocabulary = dict.fromkeys(text for texts in texts_by_view.values() for text in texts)
    ordered = sorted(vocabulary, key=len)
    dtype = torch.bfloat16 if device == 'xpu' else torch.float32
    started = perf_counter()
    with torch.inference_mode(), torch.autocast(device, dtype=dtype, enabled=device == 'xpu'):
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset:offset + batch_size]
            tokens = tokenizer(batch, max_length=MAX_LEN, truncation=True, padding=True, return_tensors='pt').to(device)
            hidden = model(**tokens).last_hidden_state.float()
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError('Encoder returned a non-finite vector')
            vocabulary.update(zip(batch, vectors))
            if offset and offset % (batch_size * 128) == 0:
                print(f'Encoded {offset}/{len(ordered)} in {perf_counter() - started:.0f}s', flush=True)
    return {view: np.stack([vocabulary[text] for text in texts]) for view, texts in texts_by_view.items()}, parameters


def fetch(cache, name):
    repo, revision = CHECKPOINTS[name]
    directory = Path(cache) / name
    directory.mkdir(parents=True, exist_ok=True)
    revision_file = directory / 'revision.txt'
    if not revision_file.exists() or revision_file.read_text(encoding='utf-8-sig').strip() != revision:
        for local, remote in EXTENSIONS.items():
            destination = directory / local
            staging = destination.with_suffix('.download')
            urlretrieve(f'https://huggingface.co/{repo}/resolve/{revision}/onnx/{remote}?download=true', staging)
            staging.replace(destination)
        revision_file.write_text(revision + '\n', encoding='utf-8')
    if any(not (directory / local).is_file() for local in EXTENSIONS):
        raise FileNotFoundError(f'Incomplete encoder cache at {directory}')
    return directory


def session(path, threads=2):
    import onnxruntime as ort
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = threads
    settings.inter_op_num_threads = 1
    return ort.InferenceSession(path if isinstance(path, bytes) else str(path), sess_options=settings,
                                providers=['CPUExecutionProvider'])


def tokenized(texts, tokenizer, names, max_length=MAX_LEN, prefixed=True):
    rows = tokenizer.encode_batch([f'query: {text}' if prefixed else text for text in texts])
    columns = {'input_ids': 'ids', 'attention_mask': 'attention_mask', 'token_type_ids': 'type_ids'}
    return {name: np.asarray([getattr(row, columns[name])[:max_length] for row in rows], dtype=np.int64)
            for name in names}


class Encoder:
    def __init__(self, directory, threads=2):
        from tokenizers import Tokenizer
        directory = Path(directory)
        self.runtime = session(directory / 'encoder.onnx', threads)
        self.names = [item.name for item in self.runtime.get_inputs()]
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=MAX_LEN)
        pad = self.tokenizer.token_to_id('<pad>')
        self.tokenizer.enable_padding(pad_id=pad if pad is not None else self.tokenizer.token_to_id('[PAD]'))

    def encode_batch(self, texts):
        inputs = tokenized(texts, self.tokenizer, self.names)
        hidden = self.runtime.run(None, inputs)[0]
        mask = inputs['attention_mask'][:, :, None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)

    def encode(self, texts, batch_size=32, name=''):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        started = perf_counter()
        matrix = None
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            vectors = self.encode_batch([texts[index] for index in batch])
            if matrix is None:
                matrix = np.empty((len(texts), vectors.shape[1]), dtype=np.float32)
            matrix[batch] = vectors
            if name and offset and offset % (batch_size * 128) == 0:
                print(f'{name}: {offset}/{len(texts)} encoded in {perf_counter()-started:.0f}s', flush=True)
        return matrix


def variant_groups(texts, test):
    augmented = augment_batch(texts, SEED + 1)
    natural = augment_batch(texts, SEED + 2, 'natural')
    asr = augment_batch(texts, SEED + 3, 'asr')
    return {'raw': texts, 'aug': augmented, 'natural': natural, 'asr': asr,
            'tta': list(map(tta_variant, texts)), 'natural_tta': list(map(tta_variant, natural)),
            'test': test, 'test_tta': list(map(tta_variant, test))}


def cache_embeddings(groups, directory, model_name, batch_size=32, threads=2, selected=None):
    directory = Path(directory)
    runtime = Encoder(fetch(directory, model_name), threads)
    paths = {}
    for name, texts in groups.items():
        if selected is not None and name not in selected:
            continue
        path = directory / model_name / f'{name}-{dataset_key(texts)[:16]}.npy'
        paths[name] = path
        if path.exists():
            continue
        existing = {}
        if name.endswith('tta'):
            origin = {'tta': 'raw', 'natural_tta': 'natural', 'test_tta': 'test'}[name]
            if origin in paths:
                existing = dict(zip(groups[origin], np.load(paths[origin], mmap_mode='r')))
        unique = list(dict.fromkeys(text for text in texts if text not in existing))
        if unique:
            existing.update(zip(unique, runtime.encode(unique, batch_size, name)))
        matrix = np.asarray([existing[text] for text in texts], dtype=np.float32)
        temporary = path.with_name(path.name + '.tmp.npy')
        np.save(temporary, matrix)
        os.replace(temporary, path)
        print(f'{model_name} {name} ready, {len(unique):,} new queries', flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--model', choices=CHECKPOINTS, default='e5-small')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--views', nargs='+')
    args = parser.parse_args()
    texts, _, test, _ = load_data(args.data)
    start = perf_counter()
    paths = cache_embeddings(variant_groups(texts, test), args.cache, args.model,
                             args.batch_size, args.threads, args.views)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    print(f'Encoder cache completed in {perf_counter() - start:.0f}s')


if __name__ == '__main__':
    main()
