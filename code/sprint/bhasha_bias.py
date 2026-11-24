"""Tune Macro-F1 decision biases on inner Bhasha holdouts and evaluate on outer folds."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import LABELS, Views
from sprint.augment import tta_variant
from sprint.bhasha_audit import (BASE_VIEWS, REQUEST_VIEWS, cached_vectors, dictionary_inputs,
                                 inner_partition, metric_delta, route_metrics, score_parts)
from sprint.data import SEED
from sprint.export import quantized
from sprint.postprocess import logit_scale
from sprint.romansetu_distill import perturbation_interval
from sprint.score_push import fit_candidate, prepare


ARMS = ('base', 'class_prior', 'per_class', 'class_prior_guarded', 'per_class_guarded')
VIEW_WEIGHTS = {'overall': .2, 'perturbed': .6, 'asr': .2}
PRIOR_GRID = (0., .15, -.15, .30, -.30, .45, -.45)
BIAS_GRID = (0., -.125, .125, -.25, .25, -.375, .375, -.50, .50)
L2 = .002


def int8_head(model):
    """Apply the export's separate lexical and scaled semantic quantizers in memory."""
    weights = model.head.coef_.T
    boundary = model.features.parameter_count()
    embedding_weight = model.config['embedding_weight']
    lexical, lexical_scale = quantized(weights[:boundary])
    semantic, semantic_scale = quantized(weights[boundary:] * embedding_weight)
    model.head.coef_ = np.asfortranarray(np.vstack((lexical * lexical_scale,
                                                   semantic * (semantic_scale / embedding_weight))).T)
    return model


def macro_for_predictions(labels, predicted, actual_counts):
    predicted_counts = np.bincount(predicted, minlength=len(LABELS))
    hits = labels == predicted
    true_counts = np.bincount(labels[hits], minlength=len(LABELS))
    total = actual_counts + predicted_counts
    return float(np.divide(2 * true_counts, total, out=np.zeros(len(LABELS)), where=total > 0).mean())


def objective(logits, labels, actual_counts, bias):
    return sum(VIEW_WEIGHTS[view] * macro_for_predictions(labels, (values + bias).argmax(1), actual_counts)
               for view, values in logits.items()) - L2 * float(np.mean(bias ** 2))


def coordinate_biases(logits, labels, actual_counts):
    bias = np.zeros(len(LABELS), dtype=np.float32)
    best = objective(logits, labels, actual_counts, bias)
    for _ in range(2):
        for class_id in LABELS:
            selected = bias[class_id]
            competitors = {}
            for view, scores in logits.items():
                adjusted = scores + bias
                adjusted[:, class_id] = -np.inf
                alternative = adjusted.argmax(1)
                competitors[view] = alternative, adjusted[np.arange(len(labels)), alternative] - scores[:, class_id]
            for value in BIAS_GRID:
                bias[class_id] = value
                score = 0.
                for view, (alternative, gap) in competitors.items():
                    wins = (value > gap) | ((value == gap) & (class_id < alternative))
                    predicted = np.where(wins, class_id, alternative)
                    score += VIEW_WEIGHTS[view] * macro_for_predictions(labels, predicted, actual_counts)
                score -= L2 * float(np.mean(bias ** 2))
                if score > best + 1e-5:
                    best, selected = score, value
            bias[class_id] = selected
    return bias


def calibrate(logits, labels, scripts, fitting_counts):
    scale = logit_scale(logits['overall'], labels)
    scaled = {view: (values * scale).astype(np.float32) for view, values in logits.items()}
    actual_counts = np.bincount(labels, minlength=len(LABELS))
    prior = -np.log(fitting_counts / np.median(fitting_counts)).astype(np.float32)
    best_tau = max(PRIOR_GRID, key=lambda tau: objective(scaled, labels, actual_counts, prior * tau))
    candidates = {'class_prior': prior * best_tau, 'per_class': coordinate_biases(scaled, labels, actual_counts)}
    baseline = route_metrics(scaled, labels, scripts)
    settings = {}
    for name, bias in candidates.items():
        delta = metric_delta(route_metrics({view: values + bias for view, values in scaled.items()}, labels, scripts), baseline)
        enabled = (delta['perturbed'] > 0 and delta['overall'] >= -.003
                   and delta['roman_asr'] >= -.001 and delta['class55_f1'] >= -.02)
        settings[name] = {'calibration_delta': delta, 'guard_passed': enabled, 'logit_bias': (bias / scale).tolist()}
    return scale, best_tau, settings


def run(args):
    groups, labels, folds, challenge, frames, scripts, config, _, variants, _ = prepare(
        args, include_reference_vectors=False)
    groups['asr_tta'] = list(map(tta_variant, groups['asr']))
    views = {view: groups[view] for view in (*BASE_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    vectors, _ = cached_vectors(args, views, 'base')
    keys = np.asarray(Views.from_texts(groups['raw']).phonetic)
    plan = [(f'fold{i + 1}', fit, check, SEED + i, keys) for i, (fit, check) in enumerate(folds)]
    plan.append(('template_holdout', *challenge, SEED + 9, frames))
    native, partitions = {}, {}
    for name, fit, check, seed, group_keys in plan:
        inner_fit, calibration = inner_partition(fit, labels, scripts, group_keys, seed)
        partitions[name] = inner_fit, calibration
        for role, train, val in (('inner', inner_fit, calibration), ('outer', fit, check)):
            _, changed, texts_by_view = dictionary_inputs(groups, train, val)
            native[name, role] = changed, texts_by_view
    converted = {f'{name}_{role}_{view}': texts for (name, role), (_, by_view) in native.items()
                 for view, texts in by_view.items()}
    native_vectors, _ = cached_vectors(args, converted, 'dictionary')

    def score(name, role, fit, blocked, rows, seed):
        changed, texts = native[name, role]
        embeddings = {view: native_vectors[f'{name}_{role}_{view}'] for view in REQUEST_VIEWS}
        with threadpool_limits(limits=args.threads):
            fitted, _ = fit_candidate(config, groups, labels, vectors, variants, fit, blocked, seed)
            return score_parts(int8_head(fitted), groups, vectors, rows, changed, texts, embeddings)[1]

    oof = {arm: {view: np.empty((len(labels), len(LABELS)), dtype=np.float32) for view in REQUEST_VIEWS} for arm in ARMS}
    report = {'method': 'Inner holdout calibration and untouched grouped outer evaluation using simulated export int8 heads',
              'view_weights': VIEW_WEIGHTS, 'prior_grid': PRIOR_GRID, 'coordinate_grid': BIAS_GRID,
              'bias_penalty': L2, 'folds': [], 'status': 'running'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, fit, check, seed, _ in plan:
        print(f'{name}: fitting the calibration model', flush=True)
        inner_fit, inner_check = partitions[name]
        inner_logits = score(name, 'inner', inner_fit, np.r_[check, inner_check], inner_check, seed)
        counts = np.bincount(labels[inner_fit], minlength=len(LABELS))
        scale, prior_tau, settings = calibrate(inner_logits, labels[inner_check], scripts[inner_check], counts)
        print(f'{name}: scoring the outer fold with frozen biases', flush=True)
        base = score(name, 'outer', fit, check, check, seed)
        arms = {'base': base}
        for arm, policy in settings.items():
            bias = np.asarray(policy['logit_bias'], dtype=np.float32)
            arms[arm] = {view: values + bias for view, values in base.items()}
            arms[f'{arm}_guarded'] = arms[arm] if policy['guard_passed'] else base
        measured = {arm: route_metrics(logits, labels[check], scripts[check]) for arm, logits in arms.items()}
        deltas = {arm: metric_delta(measured[arm], measured['base']) for arm in ARMS[1:]}
        result = {'split': name, 'calibration_rows': len(inner_check), 'evaluation_rows': len(check),
                  'calibration_scale': scale, 'prior_tau': prior_tau, 'decisions': settings,
                  'metrics': measured, 'delta': deltas}
        if name == 'template_holdout':
            result['perturbed_paired_group_interval_95pct'] = {
                arm: perturbation_interval(labels[check], base['perturbed'].argmax(1),
                                           arms[arm]['perturbed'].argmax(1), frames[check]) for arm in ARMS[1:]}
            report[name] = result
        else:
            for arm, logits in arms.items():
                for view, values in logits.items():
                    oof[arm][view][check] = values
            report['folds'].append(result)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(name, {arm: round(delta['perturbed'], 6) for arm, delta in deltas.items()}, flush=True)
    report['five_fold_mean'] = {arm: {metric: float(np.mean([r['metrics'][arm][metric] for r in report['folds']]))
                                     for metric in measured[arm]} for arm in ARMS}
    report['five_fold_delta'] = {arm: metric_delta(report['five_fold_mean'][arm], report['five_fold_mean']['base'])
                                 for arm in ARMS[1:]}
    report['paired_group_interval_95pct'] = {arm: perturbation_interval(labels, oof['base']['perturbed'].argmax(1),
                                                                       oof[arm]['perturbed'].argmax(1), keys)
                                             for arm in ARMS[1:]}
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: report[name] for name in ('five_fold_mean', 'five_fold_delta', 'paired_group_interval_95pct')},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    for name in ('checkpoint', 'cache', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.threads, args.batch_size) < 1:
        parser.error('Provide a new output path and positive thread and batch counts')
    args.phrases = args.counterfactuals = True
    args.rewrites = None
    run(args)


if __name__ == '__main__':
    main()
