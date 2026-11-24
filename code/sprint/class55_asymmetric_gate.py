import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, Views, script
from sprint.class55_lexical_gate import CLEAN_GATE_BUDGET, SUPPORTED_GAP, confidence_veto
from sprint.class55_recall import MACRO_TOLERANCE, VIEWS, counts, encode, limit, train_probe, variants, view_rows
from sprint.data import load_training
from sprint.depth_ladder import read_plan
from sprint.depth_ladder_report import result, write_json


ALARM = 49


def supported_gap(probability):
    ordered = np.partition(probability[:, :FALLBACK], -2, axis=1)[:, -2:]
    return ordered[:, 1] - ordered[:, 0]


def trigrams(line):
    for token in line.split():
        if token.isalpha():
            padded = f'^{token}$'
            yield from (padded[i:i + 3] for i in range(len(padded) - 2))


def distortion(lines, vocabulary):
    scores = []
    for line in lines:
        grams = list(trigrams(line))
        scores.append(sum(g not in vocabulary for g in grams) / max(1, len(grams)))
    return np.asarray(scores)


def negative_weights(labels):
    freq = np.bincount(labels, minlength=FALLBACK + 1)
    negatives = labels != FALLBACK
    weights = np.ones(len(labels))
    weights[negatives] = negatives.sum() / (np.count_nonzero(freq[:FALLBACK]) * freq[labels[negatives]])
    return weights


def route(logits, score, low, high, add, remove):
    base = logits.argmax(1)
    supported = logits[:, :FALLBACK].argmax(1)
    return np.where((score >= high) & add, FALLBACK,
                    np.where((base == FALLBACK) & (score < low) & remove, supported, base))


def summarize(values):
    return {'rows': len(values), 'min_median_max': np.quantile(values, [0., .5, 1.]).tolist() if len(values) else []}


def fold_run(context, fold, balanced):
    labels, roman_forms, scripts, fold_id, early, texts, results = context
    check = np.flatnonzero(fold_id == fold)
    calibration = np.flatnonzero(early[fold - 1])
    training = np.flatnonzero((fold_id != fold) & ~early[fold - 1])
    weights = negative_weights(labels[training]) if balanced else None
    features, model, low, high = train_probe(texts, labels, scripts, training, calibration, weights)
    a2 = result(results / f'fold{fold}-A2' / 'logits.npz', labels, scripts, check)[0]
    fit_vocabulary = Counter(gram for i in training for gram in trigrams(roman_forms[i]))
    familiar = {gram for gram, frequency in fit_vocabulary.items() if frequency >= 3}
    clean_calibration = distortion([roman_forms[i] for i in calibration], familiar)
    noise_cut = limit(clean_calibration, CLEAN_GATE_BUDGET)
    saved = {}
    record = {'fold': fold, 'negative_intent_balanced': balanced, 'probe_veto_below': low,
              'probe_add_at': high, 'character_noise_cut': noise_cut, 'views': {}}
    if weights is not None:
        record['weight_summary'] = summarize(weights[labels[training] != FALLBACK])
    for name, text in texts.items():
        rows = view_rows(name, check, scripts)
        mask = np.isin(check, rows)
        logits = a2['asr' if name == 'roman_asr' else name][mask]
        prediction = logits.argmax(1)
        probability = softmax(logits.astype(np.float64), axis=1)
        gap = supported_gap(probability)
        uncertain = gap < SUPPORTED_GAP
        confident_veto = confidence_veto(logits)
        score = model.decision_function(encode(features, text, rows))
        noisy = Views.from_texts([text[i] for i in rows]).roman
        noisy = distortion(noisy, familiar) >= noise_cut
        outputs = {'A2': prediction, 'lexical': route(logits, score, low, high, True, True),
                   'addition_gate': route(logits, score, low, high, uncertain, True),
                   'asymmetric_margin': route(logits, score, low, high, uncertain, confident_veto),
                   'character_noise_addition_gate': route(logits, score, low, high, noisy, True)}
        y = labels[rows]
        entry = {'metrics': {key: counts(y, pred) for key, pred in outputs.items()},
                 'character_gate_share': float(noisy.mean())}
        if fold == 1 and name == 'overall':
            alarms = (y == ALARM) & (prediction == ALARM) & (outputs['lexical'] == FALLBACK)
            entry['alarm_false_additions'] = {'raw_supported_probability_margin': summarize(gap[alarms]),
                                               'lexical_score': summarize(score[alarms]),
                                               'raw_A2_alarm_probability': summarize(probability[alarms, ALARM]),
                                               'blocked_on_these_rows': {key: int((pred[alarms] == ALARM).sum())
                                                                         for key, pred in outputs.items()}}
        saved[name] = y, outputs
        record['views'][name] = entry
    return record, saved


def evaluate(context, balanced):
    first, second = (fold_run(context, fold, balanced) for fold in (1, 2))
    folds = [first[0], second[0]]
    pooled = {}
    for name in VIEWS:
        (y1, p1), (y2, p2) = first[1][name], second[1][name]
        y = np.concatenate((y1, y2))
        pooled[name] = {key: counts(y, np.concatenate((p1[key], p2[key]))) for key in p1}
    checks = {}
    for candidate in (arm for arm in pooled['overall'] if arm != 'A2'):
        checks[candidate] = {
            'all_folds_macro_within_0_002': all(view['metrics'][candidate]['macro_f1'] >=
                                                view['metrics']['A2']['macro_f1'] - MACRO_TOLERANCE
                                                for fold in folds for view in fold['views'].values()),
            'clean_recall_preserved': pooled['overall'][candidate]['recall'] >= pooled['overall']['A2']['recall'],
            'noisy_recall_improved': all(pooled[name][candidate]['recall'] > pooled[name]['A2']['recall']
                                        for name in ('perturbed', 'roman_asr')),
            'precision_preserved_all_views': all(pooled[name][candidate]['precision'] >= pooled[name]['A2']['precision']
                                                 for name in VIEWS),
        }
    return {'folds': folds, 'pooled': pooled, 'checks': checks,
            'passes_all_checks': [key for key, values in checks.items() if all(values.values())]}


def main():
    parser = argparse.ArgumentParser()
    for name in ('train', 'plan', 'results', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw, labels = load_training(args.train)
    views = Views.from_texts(raw)
    scripts = np.asarray([script(text) for text in raw])
    fold_id, early = read_plan(args.plan, raw, labels, np.asarray(views.phonetic))
    context = labels, views.roman, scripts, fold_id, early, variants(raw), args.results
    with threadpool_limits(limits=2):
        uniform = evaluate(context, False)
        print('Uniform lexical weighting evaluated', flush=True)
        balanced = None if uniform['passes_all_checks'] else evaluate(context, True)
    write_json(args.output, {'diagnostic_only': True, 'temperature_fitted': False,
                             'addition_supported_probability_margin_below': SUPPORTED_GAP,
                             'character_addition_calibration_budget': CLEAN_GATE_BUDGET,
                             'uniform_negative_weights': uniform, 'inverse_intent_negative_weights': balanced})
    print(f'CLASS55_ASYMMETRIC_AUDIT_READY {args.output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
