"""Audit full-recipe Bhasha on grouped training folds and test a two-view dictionary route."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, PARAMETER_CAP, Views, script
from sprint.augment import tta_variant
from sprint.data import SEED, dataset_key, macro
from sprint.encoder import embed_bhasha
from sprint.postprocess import average_tta
from sprint.score_native import native_view, reverse_dictionary
from sprint.score_push import WEIGHT, fit_candidate, metrics, prepare
from sprint.score_retrieval import rare_roman
from sprint.train import EVAL_VIEWS, take


CHECKPOINT_REVISION = '64b330ff6703ffebdfc55c412a6a8aaf1b2728c8'
BASE_VIEWS = (*EVAL_VIEWS, 'asr_tta')
REQUEST_VIEWS = {'overall': ('raw', 'tta'), 'perturbed': ('natural', 'natural_tta'), 'asr': ('asr', 'asr_tta')}
ARMS = ('base', 'always_dictionary', 'selective')
MARGINS = (.025, .05, .10, .20, .35, .50, .75)


def dictionary_view(text, normalized, dictionary):
    if script(text) in ('roman', 'mixed'):
        return native_view(normalized, *dictionary)
    return normalized


def top_gap(scores):
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def select_views(gaps, confirmed, threshold):
    return confirmed & (gaps <= threshold) if threshold is not None else np.zeros_like(confirmed)


def cached_vectors(args, views, name):
    texts = sorted(dict.fromkeys(text for rows in views.values() for text in rows), key=len)
    fingerprint = dataset_key(texts)
    path = args.cache / f'bhasha-{CHECKPOINT_REVISION[:8]}-{name}-{fingerprint[:16]}.npz'
    if not path.exists():
        import torch

        torch.set_num_threads(args.threads)
        print(f'Encoding {len(texts):,} distinct {name} inputs in FP32', flush=True)
        started = perf_counter()
        vectors, parameters = embed_bhasha({'all': texts}, args.checkpoint, 'cpu', args.batch_size)
        temporary = path.with_suffix('.partial.npz')
        np.savez(temporary, vectors=vectors['all'], parameters=parameters, fingerprint=fingerprint)
        temporary.replace(path)
        print(f'{name} embeddings cached in {perf_counter() - started:.0f}s', flush=True)
    with np.load(path) as archive:
        matrix = archive['vectors']
        parameters = int(archive['parameters'])
        if archive['fingerprint'].item() != fingerprint or len(matrix) != len(texts):
            raise ValueError(f'Cached Bhasha inputs do not match {path}')
    positions = {text: index for index, text in enumerate(texts)}
    return {view: matrix[[positions[text] for text in rows]] for view, rows in views.items()}, parameters


def inner_partition(fit, labels, scripts, group_keys, seed):
    strata = [f'{labels[i]}_{scripts[i] == "roman"}' for i in fit]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 405)
    train, calibration = next(splitter.split(fit, strata, group_keys[fit]))
    return fit[train], fit[calibration]


def dictionary_inputs(groups, fit, check):
    dictionary = reverse_dictionary(take(groups['raw'], fit))
    changed, variants = {}, {}
    for view, (source, normalized) in REQUEST_VIEWS.items():
        original = take(groups[normalized], check)
        native = [dictionary_view(text, normal, dictionary)
                  for text, normal in zip(take(groups[source], check), original)]
        changed[view] = np.flatnonzero([a != b for a, b in zip(original, native)])
        variants[view] = take(native, changed[view])
    return dictionary, changed, variants


def combine_views(raw, changed, translated_scores):
    base, dictionary, gaps, eligible = {}, {}, {}, {}
    for view, (source, second) in REQUEST_VIEWS.items():
        base[view] = average_tta(raw[source], raw[second], WEIGHT)
        gaps[view] = top_gap(raw[source])
        dictionary[view] = base[view].copy()
        eligible[view] = np.zeros(len(raw[source]), dtype=bool)
        rows = changed[view]
        if len(rows):
            dictionary[view][rows] = average_tta(raw[source][rows], translated_scores[view], WEIGHT)
            eligible[view][rows] = True
    return base, dictionary, gaps, eligible, raw


def score_parts(model, groups, vectors, check, changed, native_text, native_vectors):
    raw = {view: model.score(take(groups[view], check), vectors[view][check]) for view in BASE_VIEWS}
    translated = {view: model.score(native_text[view], native_vectors[view]) for view in REQUEST_VIEWS if len(changed[view])}
    return combine_views(raw, changed, translated)


def routed_outputs(parts, threshold):
    base, dictionary, gaps, eligible, _ = parts
    masks = {view: select_views(gaps[view], eligible[view], threshold) for view in base}
    logits = {view: np.where(masks[view][:, None], dictionary[view], base[view]) for view in base}
    return logits, masks


def route_metrics(logits, labels, scripts):
    pred = {view: matrix.argmax(1) for view, matrix in logits.items()}
    roman = scripts == 'roman'
    return {**{name: macro(labels, output) for name, output in pred.items()},
            'roman_asr': macro(labels[roman], pred['asr'][roman]),
            'class55_f1': float(f1_score(labels == FALLBACK, pred['overall'] == FALLBACK, zero_division=0))}


def metric_delta(candidate, control):
    return {name: value - control[name] for name, value in candidate.items()}


def choose_margin(parts, labels, scripts):
    baseline = route_metrics(parts[0], labels, scripts)
    chosen = None
    best = 0.
    trials = []
    for threshold in MARGINS:
        scores, masks = routed_outputs(parts, threshold)
        delta = metric_delta(route_metrics(scores, labels, scripts), baseline)
        acceptable = (delta['perturbed'] > 0 and delta['overall'] >= -.003
                      and delta['roman_asr'] >= -.001 and delta['class55_f1'] >= -.02)
        objective = delta['perturbed'] + .25 * delta['roman_asr']
        trials.append({'margin': threshold, 'delta': delta, 'perturbed_routes': int(masks['perturbed'].sum()),
                       'acceptable': acceptable})
        if acceptable and objective > best:
            chosen, best = threshold, objective
    return {'chosen_margin': chosen, 'baseline': baseline, 'trials': trials}


def score_cpu(runtime, texts, normalized, dictionary, threshold):
    if len(texts) == 1 and (threshold is None or threshold == np.inf):
        text = texts[0]
        second = dictionary_view(text, normalized[0], dictionary) if threshold is not None else normalized[0]
        if second == text:
            return runtime.raw(texts)
        logits = runtime.raw([text, second])
        return average_tta(logits[:1], logits[1:], WEIGHT)
    raw = runtime.raw(texts)
    selected = select_views(top_gap(raw), np.ones(len(texts), dtype=bool), threshold)
    second = [dictionary_view(text, variant, dictionary) if active else variant
              for text, variant, active in zip(texts, normalized, selected)]
    changed = np.flatnonzero([a != b for a, b in zip(texts, second)])
    if len(changed):
        raw[changed] = average_tta(raw[changed], runtime.raw(take(second, changed)), WEIGHT)
    return raw


def cpu_benchmark(args, model, groups, labels, scripts, check, native, threshold, expected):
    from sprint.muril_export import encoder_onnx
    from sprint.runtime import Runtime
    from sprint.score_bhasha import bundle, runtime_scores

    dictionary, changed, native_text = native
    dictionary_entries = sum(map(len, dictionary))
    encoder = args.cache / f'bhasha-{CHECKPOINT_REVISION[:8]}-encoder.fp32.onnx'
    if not encoder.exists():
        intermediate = encoder.with_suffix('.partial.onnx')
        encoder_onnx(args.checkpoint, intermediate)
        intermediate.replace(encoder)
    rng = np.random.default_rng(SEED + 7)
    chosen = rng.choice(len(check), size=200, replace=False)
    batches = [('overall', chosen), ('perturbed', chosen[:150]), ('asr', chosen[:50])]
    thresholds = {'base': None, 'always_dictionary': np.inf, 'selective': threshold}
    with TemporaryDirectory(dir=args.cache, prefix='paired-router-') as temp:
        directory = Path(temp) / 'model'
        parameters = bundle(directory, encoder, model, args.checkpoint, 5.)
        if parameters + dictionary_entries > PARAMETER_CAP:
            raise ValueError('The deployment exceeds the cap when dictionary entries are counted')
        runtime = Runtime(directory, args.cpu_threads)
        views = {view: take(groups[view], check) for view in BASE_VIEWS}
        unique = list(dict.fromkeys(text for rows in (*views.values(), *native_text.values()) for text in rows))
        print(f'Template holdout: checking {len(unique):,} distinct inputs on CPU ONNX', flush=True)
        combined = dict(zip(unique, runtime_scores(runtime, unique)))
        raw = {view: np.stack([combined[text] for text in texts]) for view, texts in views.items()}
        translated = {view: np.stack([combined[text] for text in texts]) for view, texts in native_text.items() if texts}
        parts = combine_views(raw, changed, translated)
        selective, _ = routed_outputs(parts, threshold)
        onnx_scores = dict(zip(ARMS, (*parts[:2], selective)))
        onnx_metrics = {arm: route_metrics(scores, labels[check], scripts[check]) for arm, scores in onnx_scores.items()}
        parity = {view: {arm: float(np.mean(onnx_scores[arm][view].argmax(1) == expected[arm][view].argmax(1)))
                         for arm in ARMS} for view in REQUEST_VIEWS}
        queries = [text for view, positions in batches for text in take(views[REQUEST_VIEWS[view][0]], positions)]
        durations = {arm: [] for arm in ARMS}
        for text in queries[:20]:
            for cutoff in thresholds.values():
                score_cpu(runtime, [text], [tta_variant(text)], dictionary, cutoff)
        for number, i in enumerate(rng.permutation(len(queries))):
            text = queries[i]
            for offset in range(len(ARMS)):
                arm = ARMS[(number + offset) % len(ARMS)]
                start = perf_counter()
                score_cpu(runtime, [text], [tta_variant(text)], dictionary, thresholds[arm])
                durations[arm].append(1000 * (perf_counter() - start))
        del runtime
    latency = {arm: {'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95)),
                     'p99_ms': float(np.percentile(values, 99)), 'max_ms': max(values),
                     'queries_at_or_over_50ms': sum(value >= 50. for value in values)}
               for arm, values in durations.items()}
    return {'parameters': parameters, 'dictionary_entries': dictionary_entries,
            'cpu_threads': args.cpu_threads, 'queries': len(queries),
            'nonadaptive_views_batched': True,
            'argmax_agreement_with_fp32': parity, 'latency': latency, 'template_metrics': onnx_metrics,
            'delta': {arm: metric_delta(onnx_metrics[arm], onnx_metrics['base']) for arm in ARMS[1:]}}


def audit_tables(output, scores, labels, scripts):
    from sprint.diagnostics import class_rows, confusion_cost

    with (output / 'per_class.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('arm', 'view', 'script', 'class_id', 'support', 'predicted_count',
                         'true_positives', 'precision', 'recall', 'f1'))
        for arm, by_view in scores.items():
            for view, logits in by_view.items():
                writer.writerows((arm, view, *row) for row in class_rows(labels, logits.argmax(1), scripts))
    with (output / 'confusion_pairs.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('arm', 'view', 'script', 'rank', 'true_class', 'predicted_class', 'errors', 'macro_f1_cost'))
        for arm, by_view in scores.items():
            for view, logits in by_view.items():
                for name, mask in (('overall', np.ones(len(labels), dtype=bool)), ('roman', scripts == 'roman')):
                    pairs = confusion_cost(labels[mask], logits[mask].argmax(1))
                    writer.writerows((arm, view, name, rank, *row) for rank, row in enumerate(pairs, 1))


def run(args, groups, labels, folds, challenge, frames, scripts, config, variants, vectors, encoder_parameters):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    progress = output / 'results.json'
    keys = np.asarray(Views.from_texts(groups['raw']).phonetic)
    plan = [(f'fold{i+1}', fit, check, SEED + i, keys) for i, (fit, check) in enumerate(folds)]
    plan.append(('template_holdout', *challenge, SEED + 9, frames))
    sources, prepared_native = {}, {}
    for name, fit, check, seed, group_keys in plan:
        inner_fit, calibration = inner_partition(fit, labels, scripts, group_keys, seed)
        sources[name] = (inner_fit, calibration)
        for role, before, after in (('inner', inner_fit, calibration), ('outer', fit, check)):
            prepared_native[name, role] = dictionary_inputs(groups, before, after)
    native_views = {f'{name}_{role}_{view}': texts for (name, role), (_, _, texts_by_view) in prepared_native.items()
                    for view, texts in texts_by_view.items()}
    native_vectors, _ = cached_vectors(args, native_views, 'dictionary')
    all_scores = {arm: {view: np.empty((len(labels), 56), dtype=np.float32) for view in REQUEST_VIEWS} for arm in ARMS}
    fold_id = np.full(len(labels), -1, dtype=np.int8)
    report = {'encoder_revision': CHECKPOINT_REVISION, 'encoder_parameters': encoder_parameters,
              'asr_metrics_use_two_views': True, 'margins': MARGINS, 'folds': [], 'status': 'running'}

    def model_outputs(name, role, model, rows):
        _, changed, native_text = prepared_native[name, role]
        embeddings = {view: native_vectors[f'{name}_{role}_{view}'] for view in REQUEST_VIEWS}
        return score_parts(model, groups, vectors, rows, changed, native_text, embeddings)

    for index, (name, fit, check, seed, _) in enumerate(plan):
        started = perf_counter()
        inner_fit, calibration = sources[name]
        print(f'{name}: fitting the inner calibration head', flush=True)
        with threadpool_limits(limits=args.threads):
            inner, _ = fit_candidate(config, groups, labels, vectors, variants, inner_fit,
                                     np.concatenate((check, calibration)), seed)
            inner_scores = model_outputs(name, 'inner', inner, calibration)
            calibration_result = choose_margin(inner_scores, labels[calibration], scripts[calibration])
            print(f"{name}: margin {calibration_result['chosen_margin']}; fitting the outer head", flush=True)
            del inner
            model, counts = fit_candidate(config, groups, labels, vectors, variants, fit, check, seed)
            parts = model_outputs(name, 'outer', model, check)
        selective, masks = routed_outputs(parts, calibration_result['chosen_margin'])
        scores = dict(zip(ARMS, (*parts[:2], selective)))
        measured = {arm: route_metrics(logits, labels[check], scripts[check]) for arm, logits in scores.items()}
        low = rare_roman(groups, scripts, fit, check)
        entry = {'split': name, 'evaluation_rows': len(check), 'training_rows': counts,
                 'parameters': encoder_parameters + model.features.parameter_count() + model.head.coef_.size + model.head.intercept_.size,
                 'calibration': calibration_result, 'metrics': measured,
                 'legacy_base_single_view_asr': metrics(parts[-1], labels[check], scripts[check]),
                 'delta': {arm: metric_delta(measured[arm], measured['base']) for arm in ARMS[1:]},
                 'eligible': {view: int(mask.sum()) for view, mask in parts[3].items()},
                 'selectively_routed': {view: int(mask.sum()) for view, mask in masks.items()},
                 'low_coverage_roman': {'rows': int(low.sum()), **{
                     arm: route_metrics({k: v[low] for k, v in logits.items()}, labels[check][low], scripts[check][low])
                     for arm, logits in scores.items()}}, 'seconds': perf_counter() - started}
        if entry['parameters'] > PARAMETER_CAP:
            raise ValueError(f'{name} exceeded the parameter cap')
        if name != 'template_holdout':
            fold_id[check] = index
            for arm, logits in scores.items():
                for view, matrix in logits.items():
                    all_scores[arm][view][check] = matrix
            report['folds'].append(entry)
            if len(report['folds']) == len(folds):
                if (fold_id < 0).any():
                    raise ValueError('Missing out-of-fold predictions')
                np.savez_compressed(output / 'out_of_fold_scores.npz', labels=labels, fold_id=fold_id,
                                    **{f'{arm}_{view}': values for arm, logits in all_scores.items()
                                       for view, values in logits.items()})
                audit_tables(output, all_scores, labels, scripts)
        else:
            report[name] = entry
            progress.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            report['template_cpu_benchmark'] = cpu_benchmark(args, model, groups, labels, scripts, check,
                                                             prepared_native[name, 'outer'],
                                                             calibration_result['chosen_margin'], scores)
        progress.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: selective perturbed {entry['delta']['selective']['perturbed']:+.5f}, "
              f"always {entry['delta']['always_dictionary']['perturbed']:+.5f}", flush=True)
    report['pooled'] = {arm: route_metrics(logits, labels, scripts) for arm, logits in all_scores.items()}
    report['five_fold_means'] = {arm: {metric: float(np.mean([fold['metrics'][arm][metric] for fold in report['folds']]))
                                     for metric in report['pooled'][arm]} for arm in ARMS}
    report['deltas'] = {arm: metric_delta(report['five_fold_means'][arm], report['five_fold_means']['base']) for arm in ARMS[1:]}
    latency = report['template_cpu_benchmark']['latency']
    report['checks'] = {arm: {
        'mean_perturbed_at_least_0.002': report['deltas'][arm]['perturbed'] >= .002,
        'four_fold_wins': sum(fold['delta'][arm]['perturbed'] > 0 for fold in report['folds']) >= 4,
        'exported_template_gain_at_least_0.002': report['template_cpu_benchmark']['delta'][arm]['perturbed'] >= .002,
        'roman_asr_preserved': min(report['deltas'][arm]['roman_asr'], report['template_cpu_benchmark']['delta'][arm]['roman_asr']) >= -.001,
        'clean_and_class55_preserved': all(
            delta[arm]['overall'] >= -.003 and delta[arm]['class55_f1'] >= -.02
            for delta in (report['deltas'], report['template_cpu_benchmark']['delta'])),
        'p95_at_most_30ms': latency[arm]['p95_ms'] <= 30,
        'max_under_50ms': latency[arm]['max_ms'] < 50,
    } for arm in ARMS[1:]}
    report['status'] = 'complete'
    progress.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: report[name] for name in ('five_fold_means', 'deltas', 'template_cpu_benchmark', 'checks')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path, help='New audit directory. Omit to cache base embeddings only.')
    args = parser.parse_args()
    if min(args.threads, args.cpu_threads, args.batch_size) < 1 or (args.output and args.output.exists()):
        parser.error('Provide positive thread counts and batch size, and a new output directory')
    args.phrases = args.counterfactuals = True
    args.rewrites = None
    args.cache.mkdir(parents=True, exist_ok=True)
    groups, labels, folds, challenge, frames, scripts, config, _, variants, saved = prepare(
        args, include_reference_vectors=False)
    groups['asr_tta'] = [tta_variant(text) for text in groups['asr']]
    views = {name: groups[name] for name in (*BASE_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    vectors, parameters = cached_vectors(args, views, 'base')
    print(json.dumps({'encoder_parameters': parameters, 'rows': len(labels), 'cached_views': list(vectors)}), flush=True)
    if args.output:
        run(args, groups, labels, folds, challenge, frames, scripts, config, variants, vectors, parameters)


if __name__ == '__main__':
    main()
