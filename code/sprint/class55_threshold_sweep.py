import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax

from intent_model import FALLBACK, Views, script
from sprint.class55_recall import VIEWS, counts, margin, view_rows
from sprint.data import load_training
from sprint.depth_ladder import read_plan
from sprint.depth_ladder_report import result, write_json


PROBABILITIES = (.60, .70, .80, .90)
LOGIT_PENALTIES = (.30, .50, 1., 2.)


def evaluate(labels, logits):
    baseline = logits.argmax(1)
    best_supported = logits[:, :FALLBACK].argmax(1)
    is_fallback = baseline == FALLBACK
    valid = labels != FALLBACK
    probability = softmax(logits.astype(np.float64), axis=1)[:, FALLBACK]
    fallback_margin = margin(logits)
    rules = {'A2_argmax': is_fallback}
    rules.update({f'probability_{cut:.2f}': is_fallback & (probability >= cut) for cut in PROBABILITIES})
    rules.update({f'logit_penalty_{cut:.2f}': is_fallback & (fallback_margin > cut) for cut in LOGIT_PENALTIES})
    measured = {}
    for name, retain in rules.items():
        pred = np.where(retain, FALLBACK, best_supported)
        routed = is_fallback & ~retain
        measured[name] = {**counts(labels, pred),
                          'supported_micro_recall': float((pred[valid] == labels[valid]).mean()),
                          'vetoed_false_fallbacks_restored_to_true_intent': int((routed & valid & (pred == labels)).sum()),
                          'vetoed_false_fallbacks_still_misrouted': int((routed & valid & (pred != labels)).sum()),
                          'vetoed_true_fallbacks': int((routed & ~valid).sum())}
    return measured


def run(train, plan, scores):
    raw, labels = load_training(train)
    keys = np.asarray(Views.from_texts(raw).phonetic)
    scripts = np.asarray([script(text) for text in raw])
    fold_id, _ = read_plan(plan, raw, labels, keys)
    pool = {name: [] for name in VIEWS}
    folds = []
    for fold in (1, 2):
        rows = np.flatnonzero(fold_id == fold)
        logits = result(scores / f'fold{fold}-A2' / 'logits.npz', labels, scripts, rows)[0]
        output = {'fold': fold, 'views': {}}
        for name in VIEWS:
            mask = np.isin(rows, view_rows(name, rows, scripts))
            source = 'asr' if name == 'roman_asr' else name
            values = labels[rows[mask]], logits[source][mask]
            output['views'][name] = evaluate(*values)
            pool[name].append(values)
        folds.append(output)
    pooled = {name: evaluate(np.concatenate([part[0] for part in variants]),
                             np.concatenate([part[1] for part in variants])) for name, variants in pool.items()}
    return {'source': 'fixed diagnostic sweep on the previously evaluated pilot folds',
            'softmax_temperature': 1., 'temperature_fitted': False, 'threshold_selected': False,
            'probability_cutoffs': PROBABILITIES, 'logit_penalties': LOGIT_PENALTIES,
            'folds': folds, 'pooled': pooled}


def main():
    parser = argparse.ArgumentParser()
    for name in ('train', 'plan', 'scores', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, run(args.train, args.plan, args.scores))
    print(f'CLASS55_THRESHOLD_READY {args.output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
