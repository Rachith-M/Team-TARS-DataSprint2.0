"""Cache last-layer token states for a grouped, training-only Bhasha pilot."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from intent_model import Views
from sprint.augment import tta_variant
from sprint.bhasha_audit import CHECKPOINT_REVISION, REQUEST_VIEWS, dictionary_inputs, dictionary_view, inner_partition
from sprint.data import SEED, dataset_key, load_training
from sprint.encoder import MAX_LEN, session, tokenized, variant_groups
from sprint.score_data import splits
from sprint.score_native import reverse_dictionary
from sprint.train import take


def pilot_inputs(data, split):
    texts, labels = load_training(data / 'train.csv')
    folds, template, frames, scripts = splits(texts, labels)
    if split == 'template':
        (fit, check), keys, seed = template, frames, SEED + 9
    else:
        (fit, check), keys, seed = folds[int(split) - 1], np.asarray(Views.from_texts(texts).phonetic), SEED + int(split) - 1
    learn, dev = inner_partition(fit, labels, scripts, keys, seed)
    groups = variant_groups(texts, [])
    groups['asr_tta'] = list(map(tta_variant, groups['asr']))
    views = {}
    for name, train, evaluate, roles in (('dev', learn, dev, REQUEST_VIEWS), ('outer', fit, check, REQUEST_VIEWS)):
        _, changes, native = dictionary_inputs(groups, train, evaluate)
        for view in roles:
            first, normalized = roles[view]
            second = take(groups[normalized], evaluate)
            for index, text in zip(changes[view], native[view]):
                second[index] = text
            views[name, view] = take(groups[first], evaluate), second
    real = take(texts, learn)
    dictionary = reverse_dictionary(real)
    views['learn', 'overall'] = real, [dictionary_view(text, tta_variant(text), dictionary) for text in real]
    return groups, labels, scripts, keys, seed, fit, check, learn, dev, views


class TokenBank:
    def __init__(self, args, views):
        from tokenizers import Tokenizer

        self.texts = list(dict.fromkeys(text for pair in views.values() for part in pair for text in part))
        key = dataset_key(self.texts)
        self.index = {text: i for i, text in enumerate(self.texts)}
        cache = args.cache / f'last-layer-{CHECKPOINT_REVISION[:8]}-{key[:16]}'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            encoder = session(args.encoder, args.threads)
            names = [item.name for item in encoder.get_inputs()]
            tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
            tokenizer.enable_truncation(MAX_LEN)
            tokenizer.no_padding()
            lengths = [len(row.ids) for row in tokenizer.encode_batch(self.texts)]
            pad = next(tokenizer.token_to_id(token) for token in ('<pad>', '[PAD]')
                       if tokenizer.token_to_id(token) is not None)
            tokenizer.enable_padding(pad_id=pad)
            offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
            order = np.argsort(lengths, kind='stable')
            started = perf_counter()
            print(f'Caching {len(order):,} distinct views, {offsets[-1]:,} unpadded tokens', flush=True)
            with TemporaryDirectory(dir=cache.parent, prefix='token-states-') as temporary:
                target = Path(temporary) / 'bank'
                target.mkdir()
                storage = None
                try:
                    for start in range(0, len(order), args.batch_size):
                        rows = order[start:start + args.batch_size]
                        batch = tokenized(take(self.texts, rows), tokenizer, names, prefixed=False)
                        hidden = encoder.run(None, batch)[0]
                        if storage is None:
                            storage = np.lib.format.open_memmap(target / 'states.npy', mode='w+', dtype=np.float16,
                                                                shape=(int(offsets[-1]), hidden.shape[2]))
                        for i, row in enumerate(rows):
                            length = lengths[row]
                            storage[offsets[row]:offsets[row + 1]] = hidden[i, :length]
                        if start and start % (args.batch_size * 50) == 0:
                            print(f'{start:,}/{len(order):,} token views in {perf_counter() - started:.0f}s', flush=True)
                    storage.flush()
                finally:
                    del storage
                np.save(target / 'offsets.npy', offsets)
                (target / 'info.json').write_text(json.dumps({'key': key, 'texts': len(order)}), encoding='utf-8')
                target.rename(cache)
            print(f'Token bank saved to {cache} in {perf_counter() - started:.0f}s', flush=True)
        if json.loads((cache / 'info.json').read_text(encoding='utf-8')) != {'key': key, 'texts': len(self.texts)}:
            raise ValueError('The last-layer cache does not match the pilot texts')
        self.offsets = np.load(cache / 'offsets.npy')
        self.states = np.load(cache / 'states.npy', mmap_mode='r')
        if self.offsets[-1] != len(self.states) or len(self.offsets) != len(self.texts) + 1:
            raise ValueError('Invalid token cache offsets')

    def indices(self, views):
        return np.array([[self.index[text] for text in part] for part in views], dtype=np.int32).T

    def batch(self, pairs):
        indices = pairs.ravel()
        length = self.offsets[indices + 1] - self.offsets[indices]
        states = np.zeros((len(indices), length.max(), self.states.shape[1]), dtype=np.float32)
        for i, row in enumerate(indices):
            states[i, :length[i]] = self.states[self.offsets[row]:self.offsets[row + 1]]
        return states, np.arange(states.shape[1])[None, :] < length[:, None]


def parser():
    root = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'AnveshanNLP'
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument('--data', type=Path, default=Path.cwd())
    options.add_argument('--checkpoint', type=Path, default=root / 'bhasha-embed-v0-64b330ff')
    options.add_argument('--encoder', type=Path, default=root / 'bhasha-full-audit' / f'bhasha-{CHECKPOINT_REVISION[:8]}-encoder.fp32.onnx')
    options.add_argument('--cache', type=Path, default=root / 'bhasha-full-audit')
    options.add_argument('--split', choices=('1', '2', '3', '4', '5', 'template'), default='1')
    options.add_argument('--threads', type=int, default=8)
    options.add_argument('--batch-size', type=int, default=64)
    return options


if __name__ == '__main__':
    args = parser().parse_args()
    TokenBank(args, pilot_inputs(args.data, args.split)[-1])
