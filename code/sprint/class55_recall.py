import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, Views, script
from sprint.data import SCRIPTS, SEED, load_training, macro
from sprint.depth_ladder import read_plan, validation_views
from sprint.depth_ladder_report import result, write_json


VIEWS = ('overall', 'perturbed', 'roman_asr')
LOGIT_RAISE = .3
PROBE_FPR = .002
PROBE_VETO_POSITIVE_BUDGET = .10
MACRO_TOLERANCE = .002


def counts(y, pred):
    gold, flagged = y == FALLBACK, pred == FALLBACK
    tp = int((gold & flagged).sum())
    fp = int((~gold & flagged).sum())
    fn = int((gold & ~flagged).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(1, tp + fp),
            'recall': tp / max(1, tp + fn), 'f1': 2 * tp / max(1, 2 * tp + fp + fn),
            'macro_f1': macro(y, pred)}


def histogram(values):
    return {str(label): count for label, count in Counter(map(int, values)).most_common()}


def attribution(y, old, new):
    truth = y == FALLBACK
    before, after = old == FALLBACK, new == FALLBACK
    return {'false_rejects_added_by_actual_intent': histogram(y[~truth & ~before & after]),
            'false_rejects_removed_by_actual_intent': histogram(y[~truth & before & ~after]),
            'missed_fallbacks_added_by_predicted_intent': histogram(new[truth & before & ~after]),
            'missed_fallbacks_recovered_by_original_prediction': histogram(old[truth & ~before & after])}


def margin(logits):
    return logits[:, FALLBACK] - logits[:, :FALLBACK].max(1)


def diagnostics(y, scripts, control, a2):
    old, new = control.argmax(1), a2.argmax(1)
    pr = {arm: float(average_precision_score(y == FALLBACK, margin(z)))
          for arm, z in [('A0', control), ('A2', a2)]}
    shifts = {arm: {name: np.round(np.quantile(margin(z)[mask], [.1, .5, .9]), 5).tolist()
                   for name, mask in [('true_class55', y == FALLBACK), ('supported', y != FALLBACK)]}
              for arm, z in [('A0', control), ('A2', a2)]}
    return {'A0': counts(y, old), 'A2': counts(y, new), 'average_precision': pr, 'margin_quantiles': shifts,
            'mistake_changes': attribution(y, old, new),
            'per_script': {name: {'A0': counts(y[mask], old[mask]), 'A2': counts(y[mask], new[mask])}
                           for name in SCRIPTS if (mask := scripts == name).any()}}


def limit(scores, rate):
    n = min(int(np.floor(rate * len(scores))), len(scores) - 1)
    cut = np.partition(scores, len(scores) - n - 1)[len(scores) - n - 1]
    return float(np.nextafter(np.float64(cut), np.inf))


def variants(raw):
    data = validation_views(raw)
    return {'overall': data['overall'], 'perturbed': data['perturbed'], 'roman_asr': data['asr']}


def view_rows(name, rows, scripts):
    return rows[scripts[rows] == 'roman'] if name == 'roman_asr' else rows


def encode(features, texts, rows):
    return features.encode(Views.from_texts([texts[index] for index in rows]))


def train_probe(texts, labels, scripts, training, calibration, sample_weight=None):
    from intent_model import Features

    features = Features()
    fit = features.encode(Views.from_texts([texts['overall'][index] for index in training]), fit=True)
    model = LinearSVC(C=2., dual=True, random_state=SEED, tol=.002)
    model.fit(fit, labels[training] == FALLBACK, sample_weight=sample_weight)
    negative, positive = [], []
    for name, view in texts.items():
        rows = view_rows(name, calibration, scripts)
        scores = model.decision_function(encode(features, view, rows))
        fallback = labels[rows] == FALLBACK
        negative.append(limit(scores[~fallback], PROBE_FPR))
        positive.append(float(np.sort(scores[fallback])[int(np.floor(PROBE_VETO_POSITIVE_BUDGET * fallback.sum()))]))
    strong, weak = max(negative), min(positive)
    return features, model, min(strong, weak), strong


def compare(y, outputs, by_fold):
    metrics = {key: counts(y, pred) for key, pred in outputs.items()}
    baseline = metrics['A2']
    checks = {}
    for name in ('raised_threshold', 'lexical_probe'):
        selected = metrics[name]
        checks[name] = {'recall_up': selected['recall'] > baseline['recall'],
                        'precision_preserved': selected['precision'] >= baseline['precision'],
                        'macro_preserved_in_both_folds': all(fold[name]['macro_f1'] >= fold['A2']['macro_f1'] - MACRO_TOLERANCE
                                                            for fold in by_fold)}
    return {'metrics': metrics, 'checks': checks}


def run(train, plan, results):
    raw, labels = load_training(train)
    keys = np.asarray(Views.from_texts(raw).phonetic)
    scripts = np.asarray([script(text) for text in raw])
    fold_id, early = read_plan(plan, raw, labels, keys)
    texts = variants(raw)
    families = defaultdict(set)
    for key, label in zip(keys, labels):
        families[key].add(int(label))
    fallback = labels == FALLBACK
    support = {'train_rows': len(labels), 'train_class55': int(fallback.sum()),
               'class55_by_script': dict(Counter(scripts[fallback].tolist())),
               'class55_family_count': len(set(keys[fallback])),
               'class55_rows_in_families_with_supported_labels': sum(len(families[key]) > 1 for key in keys[fallback])}
    folds = []
    pooled = {name: [] for name in VIEWS}
    with threadpool_limits(limits=2):
        for fold in (1, 2):
            check = np.flatnonzero(fold_id == fold)
            calibration = np.flatnonzero(early[fold - 1])
            training = np.flatnonzero((fold_id != fold) & ~early[fold - 1])
            old = result(results / f'fold{fold}-A0' / 'logits.npz', labels, scripts, check)[0]
            a2 = result(results / f'fold{fold}-A2' / 'logits.npz', labels, scripts, check)[0]
            features, model, low, high = train_probe(texts, labels, scripts, training, calibration)
            entry = {'fold': fold, 'probe_fit_rows': len(training), 'probe_calibration_rows': len(calibration),
                     'probe_features': features.parameter_count(), 'probe_veto_below': low, 'probe_promote_at': high,
                     'views': {}}
            for name, view in texts.items():
                rows = view_rows(name, check, scripts)
                select = np.isin(check, rows)
                source = 'asr' if name == 'roman_asr' else name
                y, old_z, new_z = labels[rows], old[source][select], a2[source][select]
                before, baseline = old_z.argmax(1), new_z.argmax(1)
                supported = new_z[:, :FALLBACK].argmax(1)
                lexical = model.decision_function(encode(features, view, rows))
                two_sided = np.where(lexical >= high, FALLBACK,
                                     np.where((baseline == FALLBACK) & (lexical < low), supported, baseline))
                raised = np.where(margin(new_z) >= LOGIT_RAISE, FALLBACK, supported)
                outputs = {'A0': before, 'A2': baseline, 'raised_threshold': raised, 'lexical_probe': two_sided}
                pooled[name].append((y, old_z, new_z, scripts[rows], outputs))
                entry['views'][name] = {'audit': diagnostics(y, scripts[rows], old_z, new_z),
                                        'experiment': {key: counts(y, p) for key, p in outputs.items()}}
            folds.append(entry)
            print(f'Class 55 analysis completed for fold {fold}', flush=True)
    summary = {}
    for name, parts in pooled.items():
        y, a0, a2, tags = (np.concatenate([part[col] for part in parts]) for col in range(4))
        outputs = {key: np.concatenate([part[4][key] for part in parts]) for key in parts[0][4]}
        summary[name] = {'audit': diagnostics(y, tags, a0, a2), 'experiment': compare(
            y, outputs, [row['views'][name]['experiment'] for row in folds])}
    threshold_checks = {key: {name: view['experiment']['checks'][key] for name, view in summary.items()}
                        for key in ('raised_threshold', 'lexical_probe')}
    return {'source': 'retrospective paired audit on the two completed pilot folds',
            'protocol': {'raised_class55_margin': LOGIT_RAISE, 'inner_max_probe_fpr': PROBE_FPR,
                         'inner_max_probe_veto_positive_rate': PROBE_VETO_POSITIVE_BUDGET,
                         'max_per_fold_macro_loss': MACRO_TOLERANCE, 'lexical_svc_c': 2.},
            'dataset': support, 'folds': folds, 'pooled': summary,
            'all_views_passed': {key: all(all(view.values()) for view in checks.values())
                                 for key, checks in threshold_checks.items()}}


def main():
    parser = argparse.ArgumentParser()
    for name in ('train', 'plan', 'results', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = run(args.train, args.plan, args.results)
    write_json(args.output, data)
    print(json.dumps({'dataset': data['dataset'], 'all_views_passed': data['all_views_passed'],
                      'pooled': {key: val['experiment']['metrics'] for key, val in data['pooled'].items()}}), flush=True)


if __name__ == '__main__':
    main()
