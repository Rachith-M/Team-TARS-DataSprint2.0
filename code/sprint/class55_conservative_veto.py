import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, Views, script
from sprint.class55_asymmetric_gate import negative_weights, route, summarize, supported_gap
from sprint.class55_lexical_gate import FALLBACK_CEILING, SUPPORTED_GAP, confidence_veto
from sprint.class55_recall import MACRO_TOLERANCE, VIEWS, counts, encode, train_probe, variants, view_rows
from sprint.data import load_training
from sprint.depth_ladder import read_plan
from sprint.depth_ladder_report import result, write_json


CLEAN_FALLBACK_VETO_BUDGET = .02


def clean_veto_cut(scores, old_cut):
    if not len(scores):
        raise ValueError('Clean calibration needs Class 55 rows')
    rank = int(np.floor(CLEAN_FALLBACK_VETO_BUDGET * len(scores)))
    return min(old_cut, float(np.partition(scores, rank)[rank]))


def fold_results(fold, rows, labels, scripts, early, all_views, results):
    fit = np.flatnonzero((rows != fold) & ~early[fold - 1])
    cal = np.flatnonzero(early[fold - 1])
    check = np.flatnonzero(rows == fold)
    features, model, old_cut, add_cut = train_probe(all_views, labels, scripts, fit, cal, negative_weights(labels[fit]))
    positives = cal[labels[cal] == FALLBACK]
    scores = model.decision_function(encode(features, all_views['overall'], positives))
    new_cut = clean_veto_cut(scores, old_cut)
    logits = result(results / f'fold{fold}-A2' / 'logits.npz', labels, scripts, check)[0]
    record = {'fold': fold, 'lexical_add_at': add_cut, 'previous_lexical_veto_below': old_cut,
              'stricter_lexical_veto_below': new_cut, 'clean_calibration_class55_rows': len(positives),
              'clean_calibration_class55_below_previous_cut': int((scores < old_cut).sum()),
              'clean_calibration_class55_below_stricter_cut': int((scores < new_cut).sum()), 'views': {}}
    arrays = {}
    for name, texts in all_views.items():
        eval_rows = view_rows(name, check, scripts)
        z = logits['asr' if name == 'roman_asr' else name][np.isin(check, eval_rows)]
        y = labels[eval_rows]
        lexical = model.decision_function(encode(features, texts, eval_rows))
        probability = softmax(z.astype(np.float64), axis=1)
        margin = supported_gap(probability)
        add = margin < SUPPORTED_GAP
        remove = confidence_veto(z)
        base = z.argmax(1)
        former = route(z, lexical, old_cut, add_cut, add, remove)
        candidate = route(z, lexical, new_cut, add_cut, add, remove)
        outputs = {'A2': base, 'weighted_two_sided': former, 'tighter_veto': candidate}
        restored = (former != FALLBACK) & (candidate == FALLBACK)
        lost_true = (y == FALLBACK) & (base == FALLBACK) & (former != FALLBACK)
        entry = {'metrics': {key: counts(y, predictions) for key, predictions in outputs.items()},
                 'restored_genuine_fallbacks': int((restored & (y == FALLBACK)).sum()),
                 'restored_false_fallbacks': int((restored & (y != FALLBACK)).sum())}
        if name == 'overall':
            entry['previously_vetoed_genuine_fallbacks'] = {
                'lexical_decision_scores': summarize(lexical[lost_true]),
                'raw_supported_probability_gaps': summarize(margin[lost_true]),
                'raw_fallback_probabilities': summarize(probability[lost_true, FALLBACK]),
                'retained_by_stricter_veto': int((lost_true & (candidate == FALLBACK)).sum())}
        record['views'][name] = entry
        arrays[name] = y, outputs
    return record, arrays


def main():
    parser = argparse.ArgumentParser()
    for name in ('train', 'plan', 'results', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw, labels = load_training(args.train)
    views = Views.from_texts(raw)
    scripts = np.asarray([script(item) for item in raw])
    ids, early = read_plan(args.plan, raw, labels, np.asarray(views.phonetic))
    all_views = variants(raw)
    with threadpool_limits(limits=2):
        one, two = (fold_results(fold, ids, labels, scripts, early, all_views, args.results) for fold in (1, 2))
    records = [one[0], two[0]]
    pooled = {}
    for name in VIEWS:
        (y1, p1), (y2, p2) = one[1][name], two[1][name]
        pooled[name] = {key: counts(np.concatenate((y1, y2)), np.concatenate((p1[key], p2[key]))) for key in p1}
    checks = {'macro_preserved_on_each_fold_and_view': all(view['metrics']['tighter_veto']['macro_f1'] >=
                                                              view['metrics']['A2']['macro_f1'] - MACRO_TOLERANCE
                                                              for record in records for view in record['views'].values()),
              'clean_recall_preserved': pooled['overall']['tighter_veto']['recall'] >= pooled['overall']['A2']['recall'],
              'perturbed_and_noisy_roman_recall_improved': all(pooled[name]['tighter_veto']['recall'] > pooled[name]['A2']['recall']
                                                               for name in ('perturbed', 'roman_asr')),
              'precision_preserved_on_all_views': all(view['tighter_veto']['precision'] >= view['A2']['precision']
                                                         for view in pooled.values())}
    write_json(args.output, {'diagnostic_only': True, 'temperature_fitted': False, 'raw_probability_gap_below_to_add': SUPPORTED_GAP,
                             'raw_probability_gap_at_least_to_veto': SUPPORTED_GAP, 'raw_fallback_probability_below_to_veto': FALLBACK_CEILING,
                             'max_clean_calibration_fallback_veto_rate': CLEAN_FALLBACK_VETO_BUDGET, 'folds': records,
                             'pooled': pooled, 'checks': checks, 'passed': all(checks.values())})
    print(f'CLASS55_VETO_AUDIT_READY {args.output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
