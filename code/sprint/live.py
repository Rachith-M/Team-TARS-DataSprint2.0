import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK
from sprint.data import SEED, folds, load_data
from sprint.encoder import cache_embeddings, variant_groups
from sprint.postprocess import average_tta
from sprint.train import fit_fold, score_view


def novelty_score(logits, measure, scale):
    known = logits[:, :FALLBACK]
    active = known > -5.99
    scaled = np.where(active, known * scale, -np.inf)
    if measure == 'softmax':
        return softmax(scaled, axis=1).max(axis=1)
    return (logsumexp(scaled, axis=1) - np.log(active.sum(axis=1))) / scale


def route(logits, gate=None):
    predictions = logits.argmax(axis=1)
    if gate:
        novelty = novelty_score(logits, gate['measure'], gate['scale']) < gate['threshold']
        predictions = np.where(novelty, FALLBACK, predictions)
    return predictions


def rates(labels, predictions, novel):
    known = (labels != FALLBACK) & ~novel
    fallback = (labels == FALLBACK) & ~novel
    return {'known_reject_rate': float(np.mean(predictions[known] == FALLBACK)),
            'novel_recall': float(np.mean(predictions[novel] == FALLBACK)),
            'class55_recall': float(np.mean(predictions[fallback] == FALLBACK)),
            'novel_rows': int(novel.sum()), 'known_rows': int(known.sum())}


def probe(config, groups, labels, embeddings, split, removed, style='raw'):
    fit, check = split
    eligible = fit[~np.isin(labels[fit], removed)]
    model = fit_fold(config, groups, labels, embeddings, eligible, SEED + len(removed), check=check)
    raw = score_view(model, groups, embeddings, style, check)
    variant = score_view(model, groups, embeddings, 'tta' if style == 'raw' else 'natural_tta', check)
    return raw, variant, labels[check], np.isin(labels[check], removed)


def calibrate(fit, max_reject):
    scores, labels, novel = (np.concatenate(pair) for pair in zip(*fit))
    prediction = route(scores)
    supervised = rates(labels, prediction, novel)
    best = None
    for measure in ('energy', 'softmax'):
        for scale in (2., 4., 7., 12.):
            value = novelty_score(scores, measure, scale)
            for cutoff in np.r_[-np.inf, np.quantile(value, np.linspace(.01, .48, 96))]:
                gate = {'measure': measure, 'scale': scale, 'threshold': float(cutoff)}
                result = rates(labels, np.where(value < cutoff, FALLBACK, prediction), novel)
                if result['known_reject_rate'] > max(max_reject, supervised['known_reject_rate']):
                    continue
                utility = result['novel_recall'] - .3 * result['known_reject_rate']
                if best is None or utility > best[0]:
                    best = utility, gate, result
    if not np.isfinite(best[1]['threshold']):
        return None, best[2]
    return best[1], best[2]


def summarize(raw_results, weight, withheld, max_reject):
    results = [(average_tta(raw, variant, weight), labels, novel)
               for raw, variant, labels, novel in raw_results]
    gate, tune = calibrate(results[:2], max_reject)
    scores, targets, novel = (np.concatenate(pair) for pair in zip(*results[2:]))
    return {'gate': gate, 'calibration': tune, 'test_without_gate': rates(targets, route(scores), novel),
            'test_with_gate': rates(targets, route(scores, gate), novel),
            'calibration_withheld_classes': withheld[:2].tolist(), 'test_withheld_classes': withheld[2:].tolist()}


def run(directory, data, cache, threads, max_reject):
    settings = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    texts, labels, test, _ = load_data(data)
    groups = variant_groups(texts, test)
    config = {**settings['config'], 'oos': 0}
    paths = cache_embeddings(groups, cache, config['encoder'], threads=threads) if config['encoder'] else None
    embeddings = {name: np.load(path, mmap_mode='r') for name, path in paths.items()} if paths else None
    withheld = np.random.default_rng(SEED + 80).permutation(np.arange(FALLBACK))[:24].reshape(4, 6)
    with threadpool_limits(limits=threads):
        results = []
        for i, (split, removed) in enumerate(zip(folds(texts, labels), withheld)):
            results.append(probe(config, groups, labels, embeddings, split, removed,
                                 style='raw' if i % 2 == 0 else 'natural'))
            print(f'Live probe {i+1}/4', flush=True)
        report = summarize(results, settings['tta_weight'], withheld, max_reject)
    np.savez_compressed(directory / 'live_probe_logits.npz', withheld=withheld, **{
        f'{number}_{name}': array for number, result in enumerate(results)
        for name, array in zip(('raw', 'tta', 'target', 'novel'), result)
    })
    (directory / 'live.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--known-reject-budget', type=float, default=.15)
    args = parser.parse_args()
    run(args.directory, args.data, args.cache, args.threads, args.known_reject_budget)


if __name__ == '__main__':
    main()
