import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import LABELS, Views
from sprint.data import SEED
from sprint.score_push import compare, prepare
from sprint.train import EVAL_VIEWS, fit_fold, score_view


def unit(vectors, center):
    vectors = vectors - center
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


def retrieve(vectors, raw, augmented, center, classes, batch_size=128):
    result = np.empty((len(vectors), len(LABELS)), dtype=np.float32)
    for offset in range(0, len(vectors), batch_size):
        queries = unit(vectors[offset:offset + batch_size], center)
        similarity = np.maximum(queries @ raw.T, queries @ augmented.T)
        for label, neighbors in enumerate(classes):
            local = similarity[:, neighbors]
            top = min(3, local.shape[1])
            result[offset:offset + len(queries), label] = np.partition(local, -top, axis=1)[:, -top:].mean(axis=1)
    return result


def rare_roman(groups, scripts, fit, check):
    roman = Views.from_texts([groups['raw'][i] for i in fit]).roman
    vocabulary = {word for words in roman for word in words.split()}
    roman = Views.from_texts([groups['raw'][i] for i in check]).roman
    return (scripts[check] == 'roman') & np.array([
        sum(word not in vocabulary for word in words.split()) * 5 >= max(1, len(words.split()))
        for words in roman
    ])


def run(args):
    started = perf_counter()
    groups, labels, original, challenge, _, scripts, config, vectors, _, saved = prepare(args)
    pooled = {view: np.empty_like(saved[view]) for view in EVAL_VIEWS}
    difficult = np.zeros(len(labels), dtype=bool)
    results = {'folds': []}
    for index, (fit, check) in enumerate([*original, challenge]):
        if index < len(original):
            baseline = {view: saved[view][check] for view in EVAL_VIEWS}
        else:
            model = fit_fold(config, groups, labels, vectors, fit, SEED + 9, check=check)
            baseline = {view: score_view(model, groups, vectors, view, check) for view in EVAL_VIEWS}
        center = vectors['raw'][fit].mean(axis=0)
        raw = unit(vectors['raw'][fit], center)
        augmented = unit(vectors['aug'][fit], center)
        classes = [np.flatnonzero(labels[fit] == label) for label in LABELS]
        with threadpool_limits(limits=args.threads, user_api='blas'):
            hybrid = {view: baseline[view] + retrieve(vectors[view][check], raw, augmented, center, classes)
                      for view in EVAL_VIEWS}
        subset = rare_roman(groups, scripts, fit, check)
        entry = compare(hybrid, baseline, check, labels, scripts, len(fit))
        entry['low_coverage_roman'] = compare({v: hybrid[v][subset] for v in EVAL_VIEWS},
                                              {v: baseline[v][subset] for v in EVAL_VIEWS},
                                              check[subset], labels, scripts, len(fit))
        if index < len(original):
            difficult[check] = subset
            for view in EVAL_VIEWS:
                pooled[view][check] = hybrid[view]
            results['folds'].append(entry)
        else:
            results['challenge'] = entry
        print(f"{index + 1}: perturbed {entry['delta']['perturbed']:+.5f}, Roman ASR "
              f"{entry['delta']['roman_asr']:+.5f}, low-coverage Roman "
              f"{entry['low_coverage_roman']['delta']['overall']:+.5f}", flush=True)
    results['pooled'] = compare(pooled, saved, np.arange(len(labels)), labels, scripts, None)
    results['low_coverage_roman'] = compare({v: pooled[v][difficult] for v in EVAL_VIEWS},
                                           {v: saved[v][difficult] for v in EVAL_VIEWS},
                                           np.flatnonzero(difficult), labels, scripts, None)
    results['elapsed_seconds'] = perf_counter() - started
    args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/score_push/retrieval.json'))
    parser.add_argument('--threads', type=int, default=4)
    parser.set_defaults(rewrites=None, phrases=False, counterfactuals=False)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
