import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import DEVANAGARI, clean, phonetic_token, roman_token, transliterate
from sprint.augment import tta_variant
from sprint.data import SEED, load_training
from sprint.encoder import variant_groups
from sprint.indicxlit import IndicXlit
from sprint.runtime import Runtime
from sprint.score_bhasha import embed
from sprint.score_data import splits
from sprint.score_native import ENGLISH, native_view, reverse_dictionary
from sprint.score_push import fit_candidate, metrics, options, prepare, score
from sprint.score_retrieval import rare_roman
from sprint.train import EVAL_VIEWS, take


def parsed(text, dictionary):
    spelling, phonetic = dictionary
    tokens = clean(text).split()
    mapped = []
    unknown = []
    eligible = 0
    for index, token in enumerate(tokens):
        if not token.isascii() or not token.isalpha() or token in ENGLISH:
            continue
        eligible += 1
        key = roman_token(token)
        word = spelling.get(key)
        if word is None and len(key) >= 4:
            word = phonetic.get(phonetic_token(key))
        if word:
            mapped.append((index, word))
        elif 3 <= len(token) <= 20:
            unknown.append((index, token))
    if len(mapped) < (2 if eligible > 2 else 1) or 2 * len(mapped) < eligible:
        return None
    return tokens, mapped, unknown


def convert_rows(rows, dictionary, engine):
    segments = [parsed(text, dictionary) for text in rows]
    unique = list(dict.fromkeys(token for segment in segments if segment for _, token in segment[2]))
    candidates = engine.translate(unique)
    mapping = {word: output for word, output in zip(unique, candidates)
               if DEVANAGARI.search(output) and all(DEVANAGARI.fullmatch(char) for char in output)
               and phonetic_token(roman_token(transliterate(output))) == phonetic_token(roman_token(word))}
    rewritten = []
    for original, segment in zip(rows, segments):
        if segment is None:
            rewritten.append(original)
            continue
        tokens, known, unknown = segment
        for position, word in known:
            tokens[position] = word
        for position, word in unknown:
            tokens[position] = mapping.get(word, word)
        rewritten.append(' '.join(tokens))
    return rewritten


def benchmark(args, queries, dictionary, engine):
    runtime = Runtime(args.runtime, args.threads)

    def neural(text):
        regular = [text, tta_variant(text)]
        native = convert_rows(regular, dictionary, engine)
        unique = list(dict.fromkeys(regular + native))
        logits = dict(zip(unique, runtime.raw(unique)))
        final = [.65 * logits[a] + .35 * logits[b] for a, b in zip(regular, native)]
        return .65 * final[0] + .35 * final[1]

    summary = {}
    def batched(text):
        views = list(dict.fromkeys((text, tta_variant(text))))
        logits = runtime.raw(views)
        return .65 * logits[0] + .35 * logits[-1]

    def two_views(text, learned):
        normalized = tta_variant(text)
        converted = (convert_rows([normalized], dictionary, engine)[0] if learned
                     else native_view(normalized, *dictionary))
        views = list(dict.fromkeys((text, converted)))
        logits = runtime.raw(views)
        return .65 * logits[0] + .35 * logits[-1]

    routes = (('bhasha', lambda query: runtime.score([query])), ('bhasha_batched', batched))
    if args.two_view:
        routes += (('two_views_rules', lambda query: two_views(query, False)),
                   ('two_views_indicxlit', lambda query: two_views(query, True)))
    else:
        routes += (('with_indicxlit', neural),)
    for name, route in routes:
        for query in queries[:12]:
            route(query)
        durations = []
        for query in queries:
            start = perf_counter()
            route(query)
            durations.append(1000 * (perf_counter() - start))
        summary[name] = {'p95_ms': float(np.percentile(durations, 95)), 'max_ms': max(durations)}
    return summary


def run(args):
    groups, labels, original, challenge, _, scripts, config, _, variants, _ = prepare(args)
    engine = IndicXlit(args.xlit, threads=2)
    views = {name: groups[name] for name in (*EVAL_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    records = {}
    if args.finish:
        report = json.loads(args.output.read_text(encoding='utf-8'))
        plan = [(f'fold{i + 1}', split, SEED + i) for i, split in enumerate(original) if f'fold{i + 1}' not in report]
    else:
        report = {}
        plan = [('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)]
    for name, (fit, check), seed in plan:
        dictionary = reverse_dictionary(take(groups['raw'], fit))
        rows = {key: take(groups[key], check) for key in EVAL_VIEWS}
        for key, texts in rows.items():
            if args.two_view and key in ('raw', 'natural'):
                continue
            views[f'{name}_{key}_rules'] = [native_view(text, *dictionary) for text in texts]
            views[f'{name}_{key}_indicxlit'] = convert_rows(texts, dictionary, engine)
        records[name] = fit, check, seed, rows, dictionary
    if records:
        vectors, _ = embed(views, args.encoder, 'xpu', 128)
    for name, (fit, check, seed, rows, dictionary) in records.items():
        with threadpool_limits(limits=args.threads):
            router, _ = fit_candidate(config, groups, labels, vectors, variants, fit, check, seed)
            raw = score(router, groups, vectors, check)
            predictions = {'bhasha': raw}
            for alternative in ('rules', 'indicxlit'):
                updated = {key: prediction.copy() for key, prediction in raw.items()}
                for key, originals in rows.items():
                    if args.two_view and key in ('raw', 'natural'):
                        continue
                    view = f'{name}_{key}_{alternative}'
                    changed = np.array([before != after for before, after in zip(originals, views[view])])
                    if changed.any():
                        selected = np.flatnonzero(changed)
                        scores = router.score(take(views[view], selected), vectors[view][selected])
                        weight = 1. if args.two_view and key.endswith('tta') else .35
                        updated[key][selected] = (1. - weight) * raw[key][selected] + weight * scores
                predictions[alternative] = updated
        low_coverage = rare_roman(groups, scripts, fit, check)
        values = {key: metrics(value, labels[check], scripts[check]) for key, value in predictions.items()}
        report[name] = {
            'metrics': values,
            'indicxlit_vs_rules': {key: values['indicxlit'][key] - values['rules'][key] for key in values['rules']},
            'indicxlit_vs_bhasha': {key: values['indicxlit'][key] - values['bhasha'][key] for key in values['bhasha']},
            'low_coverage_roman': {'rows': int(low_coverage.sum()), **{
                arm: metrics({key: scores[low_coverage] for key, scores in arm_views.items()},
                             labels[check][low_coverage], scripts[check][low_coverage])
                for arm, arm_views in predictions.items()}},
        }
        print(f"{name} IndicXlit vs Bhasha: perturbed {report[name]['indicxlit_vs_bhasha']['perturbed']:+.5f}, "
              f"Roman ASR {report[name]['indicxlit_vs_bhasha']['roman_asr']:+.5f}", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.finish:
        folds = [report[f'fold{i + 1}'] for i in range(len(original))]
        summary = {}
        for arm in ('bhasha', 'rules', 'indicxlit'):
            summary[arm] = {metric: float(np.mean([fold['metrics'][arm][metric] for fold in folds]))
                            for metric in folds[0]['metrics'][arm]}
        report['five_fold_means'] = summary
        deltas = [fold['indicxlit_vs_bhasha']['perturbed'] for fold in folds]
        report['mean_perturbed_gain'] = float(np.mean(deltas))
        report['paired_perturbed_wins'] = int(sum(delta > 0 for delta in deltas))
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return
    del vectors
    _, check, _, _, dictionary = records['template_holdout']
    rng = np.random.default_rng(SEED + 7)
    chosen = rng.choice(check, size=200, replace=False)
    queries = take(groups['raw'], chosen) + take(groups['natural'], chosen[:150])
    if not args.skip_latency:
        report['latency'] = benchmark(args, queries, dictionary, engine)
    base = json.loads((args.runtime / 'runtime.json').read_text(encoding='utf-8'))['parameters']
    xlit = json.loads((args.xlit / 'parameters.json').read_text(encoding='utf-8'))['parameters']
    report['parameters'] = {'bhasha': base, 'indicxlit': xlit, 'total': base + xlit}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def latency_only(args):
    texts, labels = load_training(args.data / 'train.csv')
    _, (fit, check), _, _ = splits(texts, labels)
    groups = variant_groups(texts, [])
    dictionary = reverse_dictionary(take(texts, fit))
    engine = IndicXlit(args.xlit, threads=2)
    rng = np.random.default_rng(SEED + 7)
    chosen = rng.choice(check, size=200, replace=False)
    queries = take(texts, chosen) + take(groups['natural'], chosen[:150])
    result = benchmark(args, queries, dictionary, engine)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


def train_only(args):
    prior = json.loads(args.reference_report.read_text(encoding='utf-8'))
    groups, labels, original, challenge, _, scripts, config, _, variants, _ = prepare(args)
    engine = IndicXlit(args.xlit, threads=2)
    views = {name: groups[name] for name in (*EVAL_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    additions = {}
    plan = [('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)]
    for name, (fit, _), _ in plan:
        originals = take(groups['raw'], fit)
        dictionary = reverse_dictionary(originals)
        rewritten = convert_rows(originals, dictionary, engine)
        changed = np.array([a != b for a, b in zip(originals, rewritten)])
        sources = fit[changed]
        key = f'{name}_xlit_augmentation'
        texts = [text for text, keep in zip(rewritten, changed) if keep]
        views[key] = texts
        additions[name] = (key, texts, sources, None, labels[sources])
    vectors, _ = embed(views, args.encoder, 'xpu', 128)
    report = {}
    for name, (fit, check), seed in plan:
        with threadpool_limits(limits=args.threads):
            router, training = fit_candidate(config, groups, labels, vectors, [*variants, additions[name]],
                                             fit, check, seed)
            result = metrics(score(router, groups, vectors, check), labels[check], scripts[check])
        base = prior[name]['metrics']['bhasha']
        delta = {metric: result[metric] - base[metric] for metric in result}
        report[name] = {'candidate': result, 'baseline': base, 'delta': delta, 'training': training}
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f"{name} training-only IndicXlit: perturbed {delta['perturbed']:+.5f}, "
              f"Roman ASR {delta['roman_asr']:+.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    options(parser)
    parser.set_defaults(phrases=True, counterfactuals=True)
    parser.add_argument('--xlit', type=Path, required=True)
    parser.add_argument('--encoder', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--finish', action='store_true')
    actions.add_argument('--latency-only', action='store_true')
    actions.add_argument('--train-only', action='store_true')
    parser.add_argument('--two-view', action='store_true')
    parser.add_argument('--skip-latency', action='store_true')
    parser.add_argument('--reference-report', type=Path,
                        default=Path('sprint_results/score_push/indicxlit.json'))
    args = parser.parse_args()
    if not args.latency_only and not args.encoder:
        parser.error('Validation requires the local Bhasha checkpoint')
    if args.latency_only:
        latency_only(args)
    elif args.train_only:
        train_only(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
