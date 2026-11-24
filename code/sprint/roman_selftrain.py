import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from intent_model import LABELS, clean, script
from sprint.data import SEED, dataset_key, folds, load_training, macro
from sprint.encoder import variant_groups
from sprint.postprocess import logit_scale
from sprint.train import Router, take


CUTS = (.8, .9, .95)
PRIMARY = .9
CLASSES = (12, 36, 27, 5, 52)
LIMIT = 3 * 60 * 60


def embeddings(cache, groups):
    return {name: np.load(cache / 'e5-small' / f'{name}-{dataset_key(groups[name])[:16]}.npy', mmap_mode='r')
            for name in ('raw', 'natural')}


def source_matrix(groups, vectors, rows, check):
    forbidden = {clean(groups['raw'][i]) for i in check}
    synthetic = [i for i in rows if clean(groups['natural'][i]) not in forbidden]
    texts = take(groups['raw'], rows) + take(groups['natural'], synthetic)
    encoder = np.concatenate((vectors['raw'][rows], vectors['natural'][synthetic]))
    router = Router({'lexical': 'cross-script', 'embedding_weight': 1.4})
    matrix = router.matrix(texts, encoder, fit=True)
    return router, matrix, np.r_[rows, synthetic], np.r_[np.ones(len(rows)), np.full(len(synthetic), .7)]


def classifier(weights, seed, epochs):
    return SGDClassifier(loss='log_loss', alpha=.000035, learning_rate='constant', eta0=.02,
                         class_weight=weights, max_iter=epochs, tol=None, random_state=seed)


def class_weights(labels):
    counts = np.bincount(labels, minlength=56)
    if np.any(counts == 0):
        raise ValueError('A source fold is missing one or more intent classes')
    return {i: float((counts.mean() / count) ** .38) for i, count in enumerate(counts)}


def logits(model, matrix):
    return matrix @ np.asfortranarray(model.coef_, dtype=np.float32).T + model.intercept_.astype(np.float32)


def aligned_matrix(router, groups, vectors, name, indices):
    return router.matrix(take(groups[name], indices), vectors[name][indices])


def check_time(start):
    if perf_counter() - start >= LIMIT:
        raise TimeoutError


def train_fold(groups, vectors, labels, dev, fit, check, number, start):
    seed = SEED + number
    source = fit[dev[fit]]
    inner, calibration = folds(take(groups['raw'], source), labels[source], n_splits=4)[0]
    calibration = source[calibration]
    cal_router, cal_x, cal_ids, cal_weight = source_matrix(groups, vectors, source[inner], np.r_[calibration, check])
    cal_model = classifier(class_weights(labels[cal_ids]), seed, 26).fit(cal_x, labels[cal_ids], sample_weight=cal_weight)
    z = logits(cal_model, aligned_matrix(cal_router, groups, vectors, 'natural', calibration))
    scale = logit_scale(z, labels[calibration])
    del cal_router, cal_x, cal_model
    check_time(start)

    router, train, ids, weight = source_matrix(groups, vectors, source, check)
    weights = class_weights(labels[ids])
    base = classifier(weights, seed, 26).fit(train, labels[ids], sample_weight=weight)
    pool = fit[~dev[fit]]
    pool_x = aligned_matrix(router, groups, vectors, 'raw', pool)
    check_x = aligned_matrix(router, groups, vectors, 'raw', check)
    pool_prob = softmax(scale * logits(base, pool_x), axis=1)
    confidence, pseudo = pool_prob.max(axis=1), pool_prob.argmax(axis=1)
    base_prob = softmax(scale * logits(base, check_x), axis=1)
    base_pred = base_prob.argmax(axis=1)
    true = labels[check]
    f1_base = f1_score(true, base_pred, labels=CLASSES, average=None, zero_division=0)
    initial = macro(true, base_pred)
    median = float(np.median(base_prob.max(axis=1)))

    def continue_head(features, target, sample_weight):
        head = classifier(weights, seed + 100, 8)
        head.fit(features, target, sample_weight=sample_weight, coef_init=base.coef_.copy(),
                 intercept_init=base.intercept_.copy())
        return softmax(scale * logits(head, check_x), axis=1)

    control = continue_head(train, labels[ids], weight)
    control_f1 = macro(true, control.argmax(axis=1))
    comparisons = []
    for cut in CUTS:
        check_time(start)
        selected = confidence >= cut
        if selected.any():
            matrix = sparse.vstack((train, pool_x[selected]), format='csr')
            prob = continue_head(matrix, np.r_[labels[ids], pseudo[selected]], np.r_[weight, confidence[selected]])
        else:
            prob = base_prob
        pred = prob.argmax(axis=1)
        f1 = macro(true, pred)
        comparison = {'cut': cut, 'pseudo_rows': int(selected.sum()), 'pool_rows': len(pool),
                      'coverage': float(selected.mean()), 'roman_macro_f1': f1,
                      'paired_gain': f1-initial, 'gain_vs_labeled_continuation': f1-control_f1,
                      'median_max_softmax': float(np.median(prob.max(axis=1))),
                      'class_f1_delta': dict(zip(map(str, CLASSES),
                          (f1_score(true, pred, labels=CLASSES, average=None, zero_division=0)-f1_base).tolist()))}
        comparison['median_confidence_shift'] = comparison['median_max_softmax'] - median
        comparisons.append((comparison, selected))

    for comparison, selected in comparisons:
        count = int(np.sum(pseudo[selected] == labels[pool[selected]]))
        comparison['pseudo_correct'] = count
        comparison['pseudo_accuracy'] = count / comparison['pseudo_rows'] if comparison['pseudo_rows'] else None
    return {'fold': number + 1, 'roman_holdout_rows': len(check), 'labeled_rows_with_augmentations': len(ids),
            'source_calibration_scale': scale, 'baseline_roman_macro_f1': initial,
            'labeled_continuation_macro_f1': control_f1, 'baseline_median_max_softmax': median,
            'baseline_class_f1': dict(zip(map(str, CLASSES), f1_base.tolist())), 'cuts': [row for row,_ in comparisons]}


def write_table(output, result):
    columns = ('fold', 'roman_holdout_rows', 'cut', 'baseline_f1', 'labeled_continuation_f1', 'self_training_f1',
               'paired_gain', 'gain_vs_labeled_continuation', 'pseudo_rows', 'pool_rows', 'pseudo_accuracy',
               'coverage', 'baseline_median_max_softmax', 'self_training_median_max_softmax', 'median_confidence_shift',
               *(f'class_{label}_f1_change' for label in CLASSES))
    with (output / 'fold_results.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for fold in result['folds']:
            for cut in fold['cuts']:
                writer.writerow((fold['fold'], fold['roman_holdout_rows'], cut['cut'], fold['baseline_roman_macro_f1'],
                                 fold['labeled_continuation_macro_f1'], cut['roman_macro_f1'], cut['paired_gain'],
                                 cut['gain_vs_labeled_continuation'], cut['pseudo_rows'], cut['pool_rows'],
                                 cut['pseudo_accuracy'], cut['coverage'], fold['baseline_median_max_softmax'],
                                 cut['median_max_softmax'], cut['median_confidence_shift'],
                                 *(cut['class_f1_delta'][str(label)] for label in CLASSES)))


def run(data, cache, output):
    start = perf_counter()
    texts, labels = load_training(data / 'train.csv')
    groups = variant_groups(texts, [])
    vector = embeddings(cache, groups)
    scripts = np.array(list(map(script, texts)))
    parts = folds(texts, labels)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'started.json'
    with lock.open('x', encoding='utf-8') as file:
        json.dump({'seed': SEED, 'cuts': CUTS, 'primary': PRIMARY, 'dataset_sha256': dataset_key(texts)}, file)

    result = {'device': 'CPU', 'primary_cut': PRIMARY, 'folds': []}
    ended = 'complete'
    try:
        for number, (fit, check) in enumerate(parts):
            check_time(start)
            pure = scripts != 'mixed'
            fit = fit[pure[fit]]
            check = check[scripts[check] == 'roman']
            row = train_fold(groups, vector, labels, scripts == 'devanagari', fit, check, number, start)
            result['folds'].append(row)
            print(json.dumps(row), flush=True)
            if len(result['folds']) == 2:
                pair = [fold['cuts'][1]['paired_gain'] for fold in result['folds']]
                if np.mean(pair) < .003 or min(pair) < -.010:
                    ended = 'early_stop'
                    break
    except TimeoutError:
        ended = 'time_cap'
    rows = [fold['cuts'][1] for fold in result['folds']]
    gains = [row['paired_gain'] for row in rows]
    denominator = sum(row['pseudo_rows'] for row in rows)
    result.update({'status': ended, 'elapsed_seconds': perf_counter()-start,
                   'primary_mean_paired_gain': float(np.mean(gains)) if gains else None,
                   'primary_weighted_pseudo_accuracy': sum(row['pseudo_correct'] for row in rows)/denominator
                   if denominator else None,
                   'criteria': {'completed_five_folds': len(rows) == 5,
                                'mean_gain_ge_0_010': bool(gains and np.mean(gains) >= .010),
                                'positive_ge_four': sum(gain > 0 for gain in gains) >= 4,
                                'no_fold_worse_than_minus_0_005': bool(gains and min(gains) >= -.005),
                                'pseudo_accuracy_ge_0_95': bool(denominator and
                                    sum(row['pseudo_correct'] for row in rows)/denominator >= .95)}})
    (output / 'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    write_table(output, result)
    print(json.dumps({key: val for key,val in result.items() if key != 'folds'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/roman_selftrain_run'))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data, args.cache, args.output)


if __name__ == '__main__':
    main()
