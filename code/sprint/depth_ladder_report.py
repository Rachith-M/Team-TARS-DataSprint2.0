import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from intent_model import LABELS
from sprint.data import macro
from sprint.romansetu_distill import perturbation_interval


ARMS = ('A0', 'A1', 'A2')
PAIRS = (12, 47, 36, 9, 27, 44)
VIEWS = ('overall', 'perturbed', 'asr')


def write_json(path, data):
    temporary = path.with_suffix(f'{path.suffix}.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def measures(labels, scripts, logits):
    output = {name: logits[name].argmax(1) for name in VIEWS}
    roman = scripts == 'roman'
    precision, recall, f1, support = precision_recall_fscore_support(labels, output['perturbed'], labels=LABELS, zero_division=0)
    oos = float(f1[55])
    clean_oos = float(precision_recall_fscore_support(labels, output['overall'], labels=[55], zero_division=0)[2][0])
    return {'overall': macro(labels, output['overall']), 'perturbed': macro(labels, output['perturbed']),
            'roman': macro(labels[roman], output['overall'][roman]),
            'roman_asr': macro(labels[roman], output['asr'][roman]), 'class55_f1': clean_oos,
            'class55_perturbed_f1': oos,
            'class55_roman_asr_f1': float(precision_recall_fscore_support(labels[roman], output['asr'][roman],
                                                                        labels=[55], zero_division=0)[2][0]),
            'per_class': {str(i): {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'support': int(support[i])} for i in (*PAIRS, 55)}}


def paired(control, candidate):
    delta = {key: val - control[key] for key, val in candidate.items() if key != 'per_class'}
    delta['per_class'] = {key: {column: val - control['per_class'][key][column]
                                    for column, val in col.items() if column != 'support'}
                          for key, col in candidate['per_class'].items()}
    return delta


def result(path, labels, scripts, expected):
    with np.load(path) as archive:
        if not np.array_equal(archive['rows'], expected):
            raise ValueError(f'Unexpected evaluation rows in {path.name}')
        logits = {key: archive[key] for key in VIEWS}
    if any(x.shape != (len(expected), 56) or not np.isfinite(x).all() for x in logits.values()):
        raise ValueError(f'Invalid logits in {path.name}')
    return logits, measures(labels[expected], scripts[expected], logits)


def tables(output, rows):
    metric = [key for key in rows[0]['control'] if key != 'per_class']
    main = output / 'paired_folds.csv'
    per_class = output / 'paired_intents.csv'
    with main.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['split', 'arm', 'metric', 'A0', 'candidate', 'delta'])
        writer.writerows((row['split'], row['arm'], key, row['control'][key],
                          row['candidate'][key], row['delta'][key]) for row in rows for key in metric)
    with per_class.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['split', 'arm', 'class', 'support', 'metric', 'A0', 'candidate', 'delta'])
        writer.writerows((row['split'], row['arm'], label, row['candidate']['per_class'][label]['support'], key,
                          row['control']['per_class'][label][key], row['candidate']['per_class'][label][key], val)
                         for row in rows for label, entry in row['delta']['per_class'].items() for key, val in entry.items())


def pilot(output, labels, scripts, fold_id):
    rows = []
    failures = {arm: [] for arm in ARMS}
    for number in (1, 2):
        expected = np.flatnonzero(fold_id == number)
        jobs = {arm: output / f'fold{number}-{arm}' for arm in ARMS}
        if any(not ((job / 'complete.json').is_file() or (job / 'failed.json').is_file()) for job in jobs.values()):
            return {'status': 'incomplete', 'winner': None}
        for arm, job in jobs.items():
            if (job / 'failed.json').is_file():
                failures[arm].append(number)
        if number in failures['A0']:
            continue
        values = {arm: result(job / 'logits.npz', labels, scripts, expected)[1]
                  for arm, job in jobs.items() if number not in failures[arm]}
        rows.extend({'split': f'fold{number}', 'arm': arm, 'control': values['A0'],
                     'candidate': values[arm], 'delta': paired(values['A0'], values[arm])}
                    for arm in ('A1', 'A2') if arm in values)
    summary = {}
    for arm in ('A1', 'A2'):
        delta = [row['delta'] for row in rows if row['arm'] == arm]
        if len(delta) != 2:
            summary[arm] = {'mean_gain': None, 'checks': {'two_valid_pairs': False}, 'hard_stop': True}
            continue
        mean = {key: float(np.mean([row[key] for row in delta])) for key in ('perturbed', 'roman_asr', 'class55_f1')}
        tests = {'mean_perturbed_ge_0_008': mean['perturbed'] >= .008,
                 'each_fold_nonnegative': all(x['perturbed'] >= 0 for x in delta),
                 'mean_noisy_roman_ge_minus_0_005': mean['roman_asr'] >= -.005,
                 'mean_class55_ge_minus_0_010': mean['class55_f1'] >= -.010}
        summary[arm] = {'mean_gain': mean, 'checks': tests,
                        'hard_stop': mean['perturbed'] < .003 or any(x['perturbed'] < -.010 for x in delta)}
    summary['A2']['checks']['within_0_003_of_A1'] = (
        summary['A1']['mean_gain'] is not None and summary['A2']['mean_gain'] is not None and
        summary['A2']['mean_gain']['perturbed'] >= summary['A1']['mean_gain']['perturbed'] - .003)
    eligible = [arm for arm in ('A1', 'A2') if all(summary[arm]['checks'].values())]
    winner = max(eligible, key=lambda arm: summary[arm]['mean_gain']['perturbed'], default=None)
    if rows:
        tables(output, rows)
    return {'status': 'go' if winner else 'stop', 'winner': winner, 'failed_folds': failures,
            'arms': summary, 'folds': rows}


def grouped(output, labels, scripts, fold_id, keys, winner):
    rows, controls, candidates, indices = [], [], [], []
    for number in range(1, 6):
        expected = np.flatnonzero(fold_id == number)
        control, a = result(output / f'fold{number}-A0' / 'logits.npz', labels, scripts, expected)
        candidate, b = result(output / f'fold{number}-{winner}' / 'logits.npz', labels, scripts, expected)
        rows.append({'split': f'fold{number}', 'arm': winner, 'control': a, 'candidate': b, 'delta': paired(a, b)})
        controls.append(control)
        candidates.append(candidate)
        indices.append(expected)
    rows_for_score = np.concatenate(indices)
    base = {view: np.concatenate([r[view] for r in controls]) for view in VIEWS}
    new = {view: np.concatenate([r[view] for r in candidates]) for view in VIEWS}
    a = measures(labels[rows_for_score], scripts[rows_for_score], base)
    b = measures(labels[rows_for_score], scripts[rows_for_score], new)
    delta = paired(a, b)
    ci = perturbation_interval(labels[rows_for_score], base['perturbed'].argmax(1), new['perturbed'].argmax(1), keys[rows_for_score])
    gains = [row['delta']['perturbed'] for row in rows]
    tests = {'pooled_gain_ge_0_010': delta['perturbed'] >= .010, 'four_fold_wins': sum(x > 0 for x in gains) >= 4,
             'worst_fold_ge_minus_0_005': min(gains) >= -.005, 'paired_ci_lower_gt_0': ci[0] > 0,
             'no_noisy_roman_loss': delta['roman_asr'] >= 0, 'class55_within_0_010': delta['class55_f1'] >= -.010,
             'pair_classes_within_0_020': all(delta['per_class'][str(i)]['f1'] >= -.020 for i in PAIRS)}
    return {'passed': all(tests.values()), 'checks': tests, 'interval_95': ci, 'folds': rows,
            'pooled': {'control': a, 'candidate': b, 'delta': delta}}


def template(output, labels, scripts, expected, winner):
    a = result(output / 'template-A0' / 'logits.npz', labels, scripts, expected)[1]
    b = result(output / f'template-{winner}' / 'logits.npz', labels, scripts, expected)[1]
    delta = paired(a, b)
    checks = {'gain_ge_0_008': delta['perturbed'] >= .008, 'no_noisy_roman_loss': delta['roman_asr'] >= 0,
              'class55_within_0_010': delta['class55_f1'] >= -.010,
              'pair_classes_within_0_020': all(delta['per_class'][str(i)]['f1'] >= -.020 for i in PAIRS)}
    return {'split': 'template', 'arm': winner, 'passed': all(checks.values()), 'checks': checks,
            'task2_floor': b['perturbed'] >= .866, 'final_floor': b['perturbed'] >= .885,
            'control': a, 'candidate': b, 'delta': delta}
