import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from sprint.augment import cluster_key
from sprint.data import load_data, write_submission


def logit_scale(logits, labels):
    def loss(scale):
        scaled = logits * scale
        return np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels])

    return float(minimize_scalar(loss, bounds=(.10, 35), method='bounded').x)


def average_tta(raw, variants, weight):
    return (1 - weight) * raw + weight * variants


def cluster_logits(texts, logits, weight=1.0, fold_id=None):
    by_key = defaultdict(list)
    for index, text in enumerate(texts):
        key = cluster_key(text)
        if key:
            by_key[(0 if fold_id is None else fold_id[index], key)].append(index)
    merged = logits.copy()
    for rows in by_key.values():
        if len(rows) > 1:
            merged[rows] = (1 - weight) * logits[rows] + weight * logits[rows].mean(axis=0)
    return merged


def pseudo_candidates(raw, tta, oof, labels, scale, cap=36, minimum=.90):
    probabilities = softmax(oof * scale, axis=1)
    oof_target = probabilities.argmax(axis=1)
    oof_confidence = probabilities.max(axis=1)
    probabilities = softmax(average_tta(raw, tta, .5) * scale, axis=1)
    target = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    stability = (raw.argmax(axis=1) == target) & (tta.argmax(axis=1) == target)
    chosen = []
    for label in range(55):
        for threshold in (.80, .85, .90, .93, .96, .98):
            reference = (oof_target == label) & (oof_confidence >= threshold)
            if reference.sum() < 12 or np.mean(labels[reference] == label) < minimum:
                continue
            indices = np.flatnonzero((target == label) & (confidence >= threshold) & stability)
            chosen.extend(indices[np.argsort(-confidence[indices])[:cap]])
            break
    selected = np.array(sorted(chosen), dtype=int)
    return selected, target[selected], confidence[selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', type=Path)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--tta-weight', type=float, default=.5)
    parser.add_argument('--cluster-weight', type=float, default=0)
    args = parser.parse_args()
    _, _, test, ids = load_data(args.data)
    arrays = np.load(args.scores)
    scores = average_tta(arrays['test'], arrays['test_tta'], args.tta_weight)
    if args.cluster_weight:
        scores = cluster_logits(test, scores, args.cluster_weight)
    write_submission(args.output, ids, scores.argmax(axis=1))


if __name__ == '__main__':
    main()
