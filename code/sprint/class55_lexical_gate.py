import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, LABELS, Views, script
from sprint.class55_recall import MACRO_TOLERANCE, VIEWS, counts, encode, limit, train_probe, variants, view_rows
from sprint.data import SCRIPTS, load_training
from sprint.depth_ladder import read_plan
from sprint.depth_ladder_report import result, write_json


CLEAN_GATE_BUDGET = .10
FALLBACK_CEILING = .60
SUPPORTED_GAP = .10


def novelty(view, vocabulary):
    scores = []
    for line in view.roman:
        words = [word for word in line.split() if word.isalpha()]
        scores.append(sum(word not in vocabulary for word in words) / (len(words) + 2))
    return np.asarray(scores)


def novelty_gate(train_view, calibration_view, scripts):
    vocabulary = {word for line in train_view.roman for word in line.split() if word.isalpha()}
    scores = novelty(calibration_view, vocabulary)
    cuts = {tag: limit(scores[mask], CLEAN_GATE_BUDGET) for tag in SCRIPTS if (mask := scripts == tag).any()}
    return vocabulary, cuts, limit(scores, CLEAN_GATE_BUDGET)


def confidence_veto(logits):
    probability = softmax(logits.astype(np.float64), axis=1)
    best_two = np.partition(probability[:, :FALLBACK], -2, axis=1)[:, -2:]
    gap = best_two[:, 1] - best_two[:, 0]
    return (probability[:, FALLBACK] < FALLBACK_CEILING) & (gap >= SUPPORTED_GAP)


def class_changes(labels, initial, corrected):
    before = precision_recall_fscore_support(labels, initial, labels=LABELS, zero_division=0)
    after = precision_recall_fscore_support(labels, corrected, labels=LABELS, zero_division=0)
    changes = [{'class': int(i), 'support': int(before[3][i]), 'precision': [before[0][i], after[0][i]],
                'recall': [before[1][i], after[1][i]], 'f1': [before[2][i], after[2][i]],
                'f1_delta': after[2][i] - before[2][i]} for i in LABELS if after[2][i] != before[2][i]]
    groups = {
        'valid_requests_restored_to_true_intent': (labels != FALLBACK) & (initial != labels) & (corrected == labels),
        'valid_requests_newly_misrouted': (labels != FALLBACK) & (initial == labels) & (corrected != labels),
        'genuine_fallbacks_recovered': (labels == FALLBACK) & (initial != FALLBACK) & (corrected == FALLBACK),
        'genuine_fallbacks_lost': (labels == FALLBACK) & (initial == FALLBACK) & (corrected != FALLBACK),
        'changed_but_still_wrong': (initial != corrected) & (initial != labels) & (corrected != labels),
    }
    return {'per_intent': sorted(changes, key=lambda item: item['f1_delta']),
            'changed_rows': int((initial != corrected).sum()),
            'transitions': {name: {'count': int(mask.sum()), 'actual_labels': dict(Counter(map(int, labels[mask]))),
                                   'new_predictions': dict(Counter(map(int, corrected[mask])))}
                            for name, mask in groups.items()}}


def run(train, plan, results):
    raw, labels = load_training(train)
    views = Views.from_texts(raw)
    scripts = np.asarray([script(text) for text in raw])
    fold_id, early = read_plan(plan, raw, labels, np.asarray(views.phonetic))
    texts = variants(raw)
    folds = []
    pooled = {name: [] for name in VIEWS}
    with threadpool_limits(limits=2):
        for fold in (1, 2):
            check = np.flatnonzero(fold_id == fold)
            calibration = np.flatnonzero(early[fold - 1])
            training = np.flatnonzero((fold_id != fold) & ~early[fold - 1])
            features, probe, low, high = train_probe(texts, labels, scripts, training, calibration)
            vocabulary, cuts, default = novelty_gate(views.take(training), views.take(calibration), scripts[calibration])
            a2 = result(results / f'fold{fold}-A2' / 'logits.npz', labels, scripts, check)[0]
            record = {'fold': fold, 'novelty_cutoffs': cuts, 'views': {}}
            for name, text in texts.items():
                rows = view_rows(name, check, scripts)
                items = [text[i] for i in rows]
                mask = np.isin(check, rows)
                logits = a2['asr' if name == 'roman_asr' else name][mask]
                baseline = logits.argmax(1)
                supported = logits[:, :FALLBACK].argmax(1)
                score = probe.decision_function(encode(features, text, rows))
                promotion = score >= high
                veto = (baseline == FALLBACK) & (score < low)
                lexical = np.where(promotion, FALLBACK, np.where(veto, supported, baseline))
                cutoff = np.asarray([cuts.get(script(item), default) for item in items])
                eligible = novelty(Views.from_texts(items), vocabulary) >= cutoff
                confident = confidence_veto(logits)
                preds = {'A2': baseline, 'lexical': lexical, 'novelty_gated_lexical': np.where(eligible, lexical, baseline),
                         'lexical_margin_veto': np.where(promotion, FALLBACK,
                                                        np.where(veto & confident, supported, baseline)),
                         'margin_veto': np.where((baseline == FALLBACK) & confident, supported, baseline)}
                y = labels[rows]
                entry = {'metrics': {key: counts(y, p) for key, p in preds.items()},
                         'novelty_gate_eligible_rows': int(eligible.sum()), 'rows': len(rows)}
                if name == 'overall':
                    entry['lexical_attribution'] = class_changes(y, baseline, lexical)
                record['views'][name] = entry
                pooled[name].append((y, preds))
            folds.append(record)
            print(f'Clean attribution and gates evaluated for fold {fold}', flush=True)
    summary = {name: {key: counts(np.concatenate([part[0] for part in parts]),
                                  np.concatenate([part[1][key] for part in parts])) for key in parts[0][1]}
               for name, parts in pooled.items()}
    checks = {}
    for candidate in ('lexical', 'novelty_gated_lexical', 'lexical_margin_veto', 'margin_veto'):
        checks[candidate] = {
            'clean_fold1_macro_within_tolerance': folds[0]['views']['overall']['metrics'][candidate]['macro_f1'] >=
            folds[0]['views']['overall']['metrics']['A2']['macro_f1'] - MACRO_TOLERANCE,
            'perturbed_and_noisy_roman_recall_improved': all(summary[name][candidate]['recall'] > summary[name]['A2']['recall']
                                                             for name in ('perturbed', 'roman_asr')),
            'precision_preserved_all_views': all(values[candidate]['precision'] >= values['A2']['precision']
                                                 for values in summary.values()),
        }
    return {'diagnostic_only': True, 'temperature_fitted': False, 'rule': {'raw_softmax_class55_ceiling': FALLBACK_CEILING,
            'raw_softmax_supported_gap': SUPPORTED_GAP, 'inner_clean_novelty_budget': CLEAN_GATE_BUDGET},
            'folds': folds, 'pooled': summary, 'checks': checks}


def main():
    parser = argparse.ArgumentParser()
    for name in ('train', 'plan', 'results', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, run(args.train, args.plan, args.results))
    print(f'CLASS55_GATED_AUDIT_READY {args.output.stat().st_size}', flush=True)


if __name__ == '__main__':
    main()
