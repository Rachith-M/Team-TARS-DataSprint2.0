import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, Views, script
from sprint.class55_asymmetric_gate import negative_weights, route, summarize, supported_gap
from sprint.class55_conservative_veto import clean_veto_cut
from sprint.class55_graded_addition import strong_cut
from sprint.class55_lexical_gate import SUPPORTED_GAP, confidence_veto
from sprint.class55_recall import MACRO_TOLERANCE, VIEWS, counts, encode, train_probe, variants, view_rows
from sprint.data import load_training
from sprint.depth_ladder import read_plan
from sprint.depth_ladder_report import result, write_json


LEAD_MAX = .10
SVM_CONFIRM = 0.
PROBABILITY_FLOOR = .45
ANCHORED_LEAD_MAX = .25
ANCHORED_SUPPORTED_GAP = .15
VETO_RULES = ('narrow_lead', 'confidence_floor', 'both_vetoes', 'anchored_veto')


def distributions(labels, predictions, scores):
    flagged = predictions == FALLBACK
    quantiles = [.1, .25, .5, .75, .9]
    return {key: {'rows': int(mask.sum()), **{name: np.quantile(values[mask], quantiles).tolist()
            for name, values in scores.items()}} for key, mask in (
                ('true_fallbacks', flagged & (labels == FALLBACK)),
                ('false_fallbacks', flagged & (labels != FALLBACK)))}


def decisions(logits, lexical, low, add, strong, usual_veto):
    a2 = logits.argmax(1)
    supported_intent = logits[:, :FALLBACK].argmax(1)
    probabilities = softmax(logits.astype(np.float64), axis=1)
    p55 = probabilities[:, FALLBACK]
    lead = p55 - probabilities[:, :FALLBACK].max(1)
    gap = supported_gap(probabilities)
    base = route(logits, lexical, low, add, gap < SUPPORTED_GAP, confidence_veto(logits))
    current_fallback = (a2 == FALLBACK) & (base == FALLBACK)
    narrow = current_fallback & (lead < LEAD_MAX) & (lexical < SVM_CONFIRM)
    floor = current_fallback & (p55 < PROBABILITY_FLOOR) & (lexical < (np.inf if strong is None else strong))
    anchored = current_fallback & (lead < ANCHORED_LEAD_MAX) & (lexical < usual_veto) & (gap >= ANCHORED_SUPPORTED_GAP)
    outputs = {'A2': a2, 'conservative_veto': base,
               'narrow_lead': np.where(narrow, supported_intent, base),
               'confidence_floor': np.where(floor, supported_intent, base),
               'both_vetoes': np.where(narrow | floor, supported_intent, base),
               'anchored_veto': np.where(anchored, supported_intent, base)}
    return outputs, {'p55': p55, 'p55_lead_over_best_supported': lead, 'weighted_svm': lexical}


def veto_breakdown(y, prior, pred):
    vetoed = (prior == FALLBACK) & (pred != FALLBACK)
    valid = y != FALLBACK
    return {'false_fallbacks_removed': int((vetoed & valid).sum()),
            'false_fallbacks_restored_to_gold': int((vetoed & valid & (pred == y)).sum()),
            'true_fallbacks_removed': int((vetoed & ~valid).sum())}


def fold_results(fold, labels, scripts, ids, early, texts, results):
    train = np.flatnonzero((ids != fold) & ~early[fold - 1])
    check = np.flatnonzero(ids == fold)
    cal = np.flatnonzero(early[fold - 1])
    features, probe, usual_veto, add = train_probe(texts, labels, scripts, train, cal, negative_weights(labels[train]))
    lexical_cal = probe.decision_function(encode(features, texts['overall'], cal))
    positive = labels[cal] == FALLBACK
    low = clean_veto_cut(lexical_cal[positive], usual_veto)
    strong, cal_stats = strong_cut(lexical_cal, positive, add)
    logits = {arm: result(results / f'fold{fold}-{arm}' / 'logits.npz', labels, scripts, check)[0]
              for arm in ('A0', 'A2')}
    record = {'fold': fold, 'strong_lexical_cut': strong, 'strong_lexical_calibration': cal_stats,
              'usual_veto': usual_veto, 'views': {}}
    arrays = {}
    for name, values in texts.items():
        rows = view_rows(name, check, scripts)
        mask = np.isin(check, rows)
        key = 'asr' if name == 'roman_asr' else name
        lexical = probe.decision_function(encode(features, values, rows))
        decisions_by_arm, scores = decisions(logits['A2'][key][mask], lexical, low, add, strong, usual_veto)
        decisions_by_arm['A0'] = logits['A0'][key][mask].argmax(1)
        y = labels[rows]
        prior = decisions_by_arm['conservative_veto']
        lost = (y == FALLBACK) & (prior == FALLBACK) & (decisions_by_arm['narrow_lead'] != FALLBACK)
        record['views'][name] = {
            'metrics': {arm: counts(y, pred) for arm, pred in decisions_by_arm.items()},
            'veto_breakdown': {arm: veto_breakdown(y, prior, decisions_by_arm[arm]) for arm in VETO_RULES},
            'old_narrow_lead_true_vetoes': {'rows': int(lost.sum()),
                'weighted_svm_scores': summarize(lexical[lost]),
                'at_or_above_usual_veto': int((lost & (lexical >= usual_veto)).sum()),
                'preserved_by_anchored_veto': int((lost & (decisions_by_arm['anchored_veto'] == FALLBACK)).sum())}}
        arrays[name] = y, decisions_by_arm, scores
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
    scripts = np.asarray([script(item) for item in raw])
    ids, early = read_plan(args.plan, raw, labels, np.asarray(forms.phonetic))
    texts = variants(raw)
    with threadpool_limits(limits=2):
        one, two = (fold_results(fold, labels, scripts, ids, early, texts, args.results) for fold in (1, 2))
    folds = [one[0], two[0]]
    pooled = {}
    diagnostic = {}
    for view in VIEWS:
        (y1, p1, s1), (y2, p2, s2) = one[1][view], two[1][view]
        y = np.concatenate((y1, y2))
        preds = {arm: np.concatenate((p1[arm], p2[arm])) for arm in p1}
        pooled[view] = {arm: counts(y, pred) for arm, pred in preds.items()}
        if view == 'overall':
            diagnostic = distributions(y, preds['conservative_veto'], {key: np.concatenate((s1[key], s2[key])) for key in s1})
    clean = [fold['views']['overall']['metrics'] for fold in folds]
    checks = {}
    for arm in VETO_RULES:
        delta = float(np.mean([row[arm]['f1'] - row['A0']['f1'] for row in clean]))
        checks[arm] = {'mean_clean_class55_f1_delta_vs_a0': delta,
                       'mean_class55_f1_within_0_010_of_a0': delta >= -.01,
                       'clean_fold1_macro_at_least_a2': clean[0][arm]['macro_f1'] >= clean[0]['A2']['macro_f1'],
                       'clean_recall_at_least_0_80': pooled['overall'][arm]['recall'] >= .80,
                       'noisy_recall_above_a2': all(pooled[name][arm]['recall'] > pooled[name]['A2']['recall'] for name in VIEWS[1:]),
                       'all_folds_and_views_macro_within_0_002': all(view['metrics'][arm]['macro_f1'] >=
                        view['metrics']['A2']['macro_f1'] - MACRO_TOLERANCE for fold in folds for view in fold['views'].values())}
        checks[arm]['passed'] = all(value for value in checks[arm].values() if isinstance(value, bool))
    write_json(args.output, {'diagnostic_only': True, 'a2_cutoffs_fitted': False, 'temperature_fitted': False,
                             'fixed_p55_lead_max': LEAD_MAX, 'fixed_svm_confirmation': SVM_CONFIRM,
                             'anchored_veto_lead_below': ANCHORED_LEAD_MAX,
                             'anchored_veto_supported_gap_at_least': ANCHORED_SUPPORTED_GAP,
                             'fixed_p55_floor': PROBABILITY_FLOOR, 'quantiles': [.1, .25, .5, .75, .9],
                             'current_clean_fallback_distributions': diagnostic, 'folds': folds, 'pooled': pooled, 'checks': checks})
    print(f'CLASS55_WEAK_FALLBACK_READY {args.output.name} {args.output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
