import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from intent_model import LABELS
from sprint.data import SEED
from sprint.score_push import compare, prepare, samples
from sprint.score_retrieval import rare_roman
from sprint.train import EVAL_VIEWS, Router, take


def fit_pairs(matrix, labels, weights, scores, config, seed):
    pairs = set()
    alternatives = scores.copy()
    alternatives[np.arange(len(labels)), labels] = -np.inf
    rivals = alternatives.argmax(axis=1)
    for label in LABELS:
        candidates = Counter(rivals[labels == label]).most_common(4)
        pairs.update(tuple(sorted((label, rival))) for rival, _ in candidates)
    fitted = {}
    for low, high in sorted(pairs):
        positions = np.flatnonzero((labels == low) | (labels == high))
        target = labels[positions]
        counts = np.array([(target == low).sum(), (target == high).sum()])
        weight = (counts.mean() / counts) ** config['balance']
        svm = LinearSVC(C=config['c'], tol=.004, dual=True, random_state=seed,
                        class_weight=dict(zip((low, high), weight)), max_iter=2500)
        svm.fit(matrix[positions], target, sample_weight=weights[positions])
        fitted[low, high] = svm
    return fitted


def specialize(matrix, scores, pairs):
    output = scores.copy()
    order = np.argsort(scores, axis=1)
    best, rival = order[:, -1], order[:, -2]
    options = np.sort(np.column_stack((best, rival)), axis=1)
    rows = np.arange(len(scores))
    close = scores[rows, best] - scores[rows, rival] <= .45
    for (low, high), svm in pairs.items():
        select = np.flatnonzero(close & (options[:, 0] == low) & (options[:, 1] == high))
        if not len(select):
            continue
        low_score, high_score = scores[select, low], scores[select, high]
        margin = .5 * (high_score - low_score) + .5 * svm.decision_function(matrix[select])
        best_score = np.maximum(low_score, high_score)
        output[select, low] = best_score - np.maximum(margin, 0)
        output[select, high] = best_score + np.minimum(margin, 0)
    return output


def run(args):
    groups, labels, original, challenge, _, scripts, config, vectors, _, saved = prepare(args)
    results = {}
    for name, (fit, check), seed in (
            ('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)):
        parts = samples(config, groups, labels, fit, check, [], seed)
        texts = [item for _, text, ids, _, _ in parts for item in take(text, ids)]
        target = np.concatenate([target for _, _, _, target, _ in parts])
        weights = np.concatenate([np.full(len(ids), weight) for _, _, ids, _, weight in parts])
        embedding = np.concatenate([vectors[view][ids] for view, _, ids, _, _ in parts])
        router = Router(config)
        with threadpool_limits(limits=args.threads):
            matrix = router.matrix(texts, embedding, fit=True)
            router.fit_matrix(matrix, target, weights)
            train_scores = matrix @ router.head.coef_.T + router.head.intercept_
            pairs = fit_pairs(matrix, target, weights, train_scores, config, seed)
            baseline, candidate = {}, {}
            for view in EVAL_VIEWS:
                query = router.matrix(take(groups[view], check), vectors[view][check])
                baseline[view] = query @ router.head.coef_.T + router.head.intercept_
                if name == 'fold1':
                    np.testing.assert_allclose(baseline[view], saved[view][check], atol=.004, rtol=0)
                candidate[view] = specialize(query, baseline[view], pairs)
        entry = compare(candidate, baseline, check, labels, scripts, len(fit))
        subset = rare_roman(groups, scripts, fit, check)
        entry['low_coverage_roman'] = compare({v: candidate[v][subset] for v in EVAL_VIEWS},
                                              {v: baseline[v][subset] for v in EVAL_VIEWS},
                                              check[subset], labels, scripts, len(fit))
        entry['pair_count'] = len(pairs)
        results[name] = entry
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: perturbed {entry['delta']['perturbed']:+.5f}, Roman ASR "
              f"{entry['delta']['roman_asr']:+.5f}, low-coverage Roman "
              f"{entry['low_coverage_roman']['delta']['overall']:+.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/score_push/specialists.json'))
    parser.add_argument('--threads', type=int, default=4)
    parser.set_defaults(rewrites=None, phrases=False, counterfactuals=False)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
