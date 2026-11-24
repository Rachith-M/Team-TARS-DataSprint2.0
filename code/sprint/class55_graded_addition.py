import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, Views, script
from sprint.class55_asymmetric_gate import ALARM, negative_weights, route, summarize, supported_gap
from sprint.class55_conservative_veto import clean_veto_cut
from sprint.class55_lexical_gate import SUPPORTED_GAP, confidence_veto
from sprint.class55_recall import MACRO_TOLERANCE, VIEWS, counts, encode, train_probe, variants, view_rows
from sprint.data import load_training
from sprint.depth_ladder import read_plan
from sprint.depth_ladder_report import result, write_json


HIGH_GAP = .30
HIGH_PRECISION = .95


def strong_cut(scores, positive, low_cut):
    thresholds = np.unique(scores[scores >= low_cut])
    for cutoff in thresholds:
        include = scores >= cutoff
        correct = int(positive[include].sum())
        number = int(include.sum())
        if correct / number >= HIGH_PRECISION:
            return float(cutoff), {'classified_as_fallback': number, 'correct': correct, 'precision': correct / number}
    return None, {'classified_as_fallback': 0, 'correct': 0, 'precision': None}


def evaluate_fold(fold, ids, early, labels, scripts, texts, results):
    fit = np.flatnonzero((ids != fold) & ~early[fold - 1])
    cal = np.flatnonzero(early[fold - 1])
    check = np.flatnonzero(ids == fold)
    features, model, old_cut, low_add = train_probe(texts, labels, scripts, fit, cal, negative_weights(labels[fit]))
    cal_scores = model.decision_function(encode(features, texts['overall'], cal))
    positive = labels[cal] == FALLBACK
    veto_cut = clean_veto_cut(cal_scores[positive], old_cut)
    high_add, calibration = strong_cut(cal_scores, positive, low_add)
    a0 = result(results / f'fold{fold}-A0' / 'logits.npz', labels, scripts, check)[0]
    a2 = result(results / f'fold{fold}-A2' / 'logits.npz', labels, scripts, check)[0]
    record = {'fold': fold, 'low_add': low_add, 'high_add': high_add, 'veto_cut': veto_cut,
              'strong_lexical_calibration': calibration, 'views': {}}
    arrays = {}
    for name, content in texts.items():
        rows = view_rows(name, check, scripts)
        mask = np.isin(check, rows)
        key = 'asr' if name == 'roman_asr' else name
        logits = a2[key][mask]
        base = logits.argmax(1)
        frozen = a0[key][mask].argmax(1)
        lexical = model.decision_function(encode(features, content, rows))
        gaps = supported_gap(softmax(logits.astype(np.float64), axis=1))
        uncertain = gaps < SUPPORTED_GAP
        moderate = (gaps >= SUPPORTED_GAP) & (gaps < HIGH_GAP)
        strong = lexical >= (np.inf if high_add is None else high_add)
        veto = confidence_veto(logits)
        old_gate = route(logits, lexical, veto_cut, low_add, uncertain, veto)
        new_gate = route(logits, lexical, veto_cut, low_add, uncertain | (moderate & strong), veto)
        y = labels[rows]
        recovered = (new_gate == FALLBACK) & (old_gate != FALLBACK)
        outputs = {'A0': frozen, 'A2': base, 'conservative_veto': old_gate, 'graded_addition': new_gate}
        item = {'metrics': {arm: counts(y, pred) for arm, pred in outputs.items()},
                'tier2_true_additions': int((recovered & (y == FALLBACK)).sum()),
                'tier2_false_additions': int((recovered & (y != FALLBACK)).sum())}
        if name == 'overall':
            missed = (y == FALLBACK) & (base != FALLBACK)
            captured_by_frozen = missed & (frozen == FALLBACK)
            item['a2_false_negative_summary'] = {
                'all_missed': int(missed.sum()), 'caught_by_a0': int(captured_by_frozen.sum()),
                'raw_probability_gaps_among_missed': summarize(gaps[missed]),
                'lexical_scores_among_missed': summarize(lexical[missed]),
                'missed_in_moderate_gap_band': int((missed & moderate).sum()),
                'missed_in_moderate_band_and_above_high_cut': int((missed & moderate & strong).sum()),
                'missed_caught_by_a0_in_moderate_gap_band': int((captured_by_frozen & moderate).sum()),
                'new_false_alarm_fallbacks': int((recovered & (y == ALARM)).sum())}
        record['views'][name] = item
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
    forms = Views.from_texts(raw)
    scripts = np.asarray([script(text) for text in raw])
    ids, early = read_plan(args.plan, raw, labels, np.asarray(forms.phonetic))
    texts = variants(raw)
    with threadpool_limits(limits=2):
        first, second = (evaluate_fold(fold, ids, early, labels, scripts, texts, args.results) for fold in (1, 2))
    folds = [first[0], second[0]]
    pooled = {}
    for name in VIEWS:
        (y1, p1), (y2, p2) = first[1][name], second[1][name]
        pooled[name] = {arm: counts(np.concatenate((y1, y2)), np.concatenate((p1[arm], p2[arm]))) for arm in p1}
    clean = [row['views']['overall']['metrics'] for row in folds]
    mean_f1_delta = float(np.mean([view['graded_addition']['f1'] - view['A0']['f1'] for view in clean]))
    checks = {'class55_f1_vs_a0_within_0_010': mean_f1_delta >= -.010,
              'clean_fold1_macro_preserved': clean[0]['graded_addition']['macro_f1'] >= clean[0]['A2']['macro_f1'],
              'clean_recall_preserved': pooled['overall']['graded_addition']['recall'] >= pooled['overall']['A2']['recall'],
              'perturbed_and_noisy_roman_recall_preserved': all(pooled[name]['graded_addition']['recall'] >=
                                                                pooled[name]['conservative_veto']['recall'] for name in VIEWS[1:]),
              'all_folds_and_views_macro_within_0_002': all(view['metrics']['graded_addition']['macro_f1'] >=
                    view['metrics']['A2']['macro_f1'] - MACRO_TOLERANCE for row in folds for view in row['views'].values())}
    write_json(args.output, {'diagnostic_only': True, 'temperature_fitted': False, 'tier2_gap_max_exclusive': HIGH_GAP,
              'tier2_min_clean_calibration_precision': HIGH_PRECISION, 'folds': folds, 'pooled': pooled,
              'mean_clean_class55_f1_delta_vs_a0': mean_f1_delta, 'checks': checks, 'passed': all(checks.values())})
    print(f'CLASS55_GRADED_ADDITION_READY {args.output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
