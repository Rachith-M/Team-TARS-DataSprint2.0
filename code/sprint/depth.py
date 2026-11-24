import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import script
from sprint.data import SEED, dataset_key, folds, load_training
from sprint.diagnostics import write_rows
from sprint.encoder import variant_groups
from sprint.muril import VIEWS, baseline, evaluate, json_file
from sprint.train import fit_fold, score_view, take


DEPTHS = ((2500, .004), (5000, .001), (10000, .00025))
MIN_TWO = .003
BUDGET = 3 * 3600
METRICS = ('overall', 'roman', 'perturbed', 'class55_f1')


def prepare(args):
    texts, labels = load_training(args.data / 'train.csv')
    groups = variant_groups(texts, [])
    roman = np.asarray([script(text) == 'roman' for text in texts])
    parts = folds(texts, labels)
    scores = baseline(args.reference, labels, roman, parts)
    config = json.loads((args.reference / 'metrics.json').read_text(encoding='utf-8'))['config']
    vectors = {name: np.load(args.cache / 'e5-small' / f'{name}-{dataset_key(groups[name])[:16]}.npy', mmap_mode='r')
               for name in (*VIEWS, 'aug')}
    return groups, labels, roman, parts, scores, config, vectors


def inner_split(groups, labels, fit):
    train, valid = folds(take(groups['raw'], fit), labels[fit], n_splits=4)[0]
    return fit[train], fit[valid]


def choose_depth(groups, labels, roman, vectors, config, fit, check, inner, weight, seed):
    train, valid = inner
    first = None
    best = -1.
    scores = []
    for max_iter, tol in DEPTHS:
        params = {**config, 'max_iter': max_iter, 'tol': tol}
        router = fit_fold(params, groups, labels, vectors, train, seed, check=np.r_[valid, check])
        logits = {name: score_view(router, groups, vectors, name, valid) for name in VIEWS}
        metric = evaluate(logits, labels[valid], roman[valid], weight)['perturbed']
        scores.append({'max_iter': max_iter, 'tolerance': tol, 'iterations': int(router.head.n_iter_),
                       'inner_perturbed': metric})
        if metric > best + .0001:
            first, best = params, metric
    return first, scores


def report(args, name, base, records, pooled=None, status='running', **extra):
    gains = [row['deltas']['perturbed'] for row in records]
    checks = {'five_folds': len(gains) == 5,
              'mean_gain_ge_0_010': float(np.mean(gains)) >= .010 if len(gains) == 5 else None,
              'at_least_four_wins': sum(gain > 0 for gain in gains) >= 4 if len(gains) == 5 else None,
              'no_fold_below_minus_0_005': min(gains) >= -.005 if gains else None,
              'pooled_roman_ge_0_865': pooled['roman'] >= .865 if pooled else None,
              'pooled_class55_ge_0_85': pooled['class55_f1'] >= .85 if pooled else None,
              'int8_p95_le_30_ms': None, 'parameters_le_300m': None}
    data = {'variant': name, 'baseline': base, 'folds': records, 'pooled': pooled, 'status': status,
            'mean_paired_perturbed_gain': float(np.mean(gains)) if gains else None,
            'checks': checks, **extra}
    json_file(args.output / f'{name}.json', data)
    write_rows(args.output / f'{name}_folds.csv', ('fold', *(f'{prefix}_{metric}' for prefix in ('selected', name, 'delta')
                                                         for metric in METRICS)),
               ((row['fold'], *(source[metric] for source in (base['folds'][row['fold']-1], row['metrics'],
                                                              row['deltas']) for metric in METRICS)) for row in records))
    return data


def fold_result(number, logits, check, labels, roman, base, **extra):
    scores = evaluate(logits, labels[check], roman[check], base['tta_weight'])
    deltas = {name: scores[name]-base['folds'][number][name] for name in METRICS}
    return {'fold': number+1, 'metrics': scores, 'deltas': deltas, **extra}


def run_d1(args, inputs):
    start = perf_counter()
    groups, labels, roman, parts, base, config, vectors = inputs
    records, status = [], 'complete'
    pooled = {name: np.zeros((len(labels), 56), dtype=np.float32) for name in VIEWS}
    for number, (fit, check) in enumerate(parts):
        seed = SEED + number
        selected, inner = choose_depth(groups, labels, roman, vectors, config, fit, check,
                                       inner_split(groups, labels, fit), base['tta_weight'], seed)
        router = fit_fold(selected, groups, labels, vectors, fit, seed, check=check)
        logits = {name: score_view(router, groups, vectors, name, check) for name in VIEWS}
        for name, values in logits.items():
            pooled[name][check] = values
        row = fold_result(number, logits, check, labels, roman, base, selected=selected, inner=inner,
                          outer_iterations=int(router.head.n_iter_))
        records.append(row)
        print(json.dumps(row), flush=True)
        if len(records) == 2 and np.mean([x['deltas']['perturbed'] for x in records]) < MIN_TWO:
            status = 'dropped_after_two'
            break
        report(args, 'D1', base, records, status='running', elapsed_seconds=perf_counter()-start)
    metrics = evaluate(pooled, labels, roman, base['tta_weight']) if len(records) == 5 else None
    result = report(args, 'D1', base, records, metrics, status, elapsed_seconds=perf_counter()-start)
    if metrics and all(result['checks'][key] for key in result['checks'] if key not in ('int8_p95_le_30_ms', 'parameters_le_300m')):
        from sprint.depth_export import export_d1
        export_d1(args, inputs, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('D1', 'partial'))
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--model', type=Path, default=Path('sprint_results/final'))
    parser.add_argument('--checkpoint', type=Path, default=Path('.sprint_cache/e5-small-torch'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/depth_paired'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('auto','xpu','cuda'), default='auto')
    parser.add_argument('--previous-gpu-seconds',type=float,default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f'{args.stage}_started.json').open('x', encoding='utf-8') as file:
        json.dump({'seed': SEED, 'freeze_depth_checkpoints': DEPTHS, 'early_drop': MIN_TWO,
                   'top_blocks': {'D2': 4, 'D3': 8}, 'encoder_lr': 1e-5, 'layer_decay': .9,
                   'gpu_budget_seconds': BUDGET, 'previous_gpu_seconds':args.previous_gpu_seconds,
                   'checkpoint': str(args.checkpoint)}, file)
    with threadpool_limits(limits=args.threads):
        inputs = prepare(args)
        if args.stage == 'D1':
            run_d1(args, inputs)
        else:
            from sprint.depth_partial import run_partial
            run_partial(args, inputs)


if __name__ == '__main__':
    main()
