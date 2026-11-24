"""Calibrate live fallback from the organizer's labeled Bhasha out-of-fold rows."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from intent_model import FALLBACK
from sprint.data import load_training
from sprint.defense_artifacts import POLICY, SUBMITTED_SHA256
from sprint.gate_experiment import specificity_threshold
from sprint.gate_runtime import actions
from sprint.postprocess import logit_scale
from sprint.score_push import fingerprint


AUDIT = Path('sprint_results/score_push/bhasha_full_audit_run/out_of_fold_scores.npz')


def evaluate(logits, labels, mask, scale, rule):
    values = logits[mask].astype(np.float64) * scale
    top = values.argmax(1)
    risk = 1 - softmax(values, axis=1).max(axis=1)
    routed = actions(risk, top, rule, True)
    supported = labels[mask] != FALLBACK
    correct = supported & (top == labels[mask])
    accepted = routed == 'accept'
    return {'supported': int(supported.sum()), 'supported_correct_accept': int((correct & accepted).sum()),
            'supported_wrong_accept': int((supported & ~correct & accepted).sum()),
            'supported_fallback': int((supported & ~accepted).sum()),
            'correct_supported_fallback': int((correct & ~accepted).sum()),
            'correct_supported': int(correct.sum()), 'class55': int((~supported).sum()),
            'class55_fallback': int((~supported & ~accepted).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=AUDIT)
    parser.add_argument('--train', type=Path, default=Path('train.csv'))
    parser.add_argument('--output', type=Path, default=POLICY)
    args = parser.parse_args()
    _, labels = load_training(args.train)
    with np.load(args.audit) as saved:
        if not np.array_equal(saved['labels'], labels):
            raise ValueError('Archived scores and training labels differ')
        fit = saved['fold_id'] < 3
        logits = {view: saved[f'always_dictionary_{view}'] for view in ('overall', 'perturbed', 'asr')}
    clean = logits['overall']
    scale = logit_scale(clean[fit], labels[fit])
    top = clean.argmax(1)
    correct_fit = fit & (top == labels) & (labels != FALLBACK)
    risk = 1 - softmax(clean[correct_fit].astype(np.float64) * scale, axis=1).max(1)
    direct = np.zeros(len(risk), dtype=bool)
    rule = {'threshold': specificity_threshold(risk, direct, .03),
            'accept_below': specificity_threshold(risk, direct, .07)}
    result = {'source': 'Organizer-labeled training rows, always-dictionary Bhasha grouped OOF logits',
              'audit_sha256': fingerprint(args.audit), 'submission_sha256': SUBMITTED_SHA256,
              'model_logit_scale': scale, 'rule': rule,
              'calibration': evaluate(clean, labels, fit, scale, rule),
              'holdout': {view: evaluate(matrix, labels, ~fit, scale, rule) for view, matrix in logits.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result['holdout'], indent=2))


if __name__ == '__main__':
    main()
