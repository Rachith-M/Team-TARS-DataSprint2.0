import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from intent_model import DEVANAGARI, clean, phonetic_token, roman_token, transliterate
from sprint.data import SEED
from sprint.encoder import Encoder
from sprint.score_data import ALIASES
from sprint.score_push import compare, fit_candidate, metrics, prepare, score
from sprint.score_retrieval import rare_roman
from sprint.train import EVAL_VIEWS, fit_fold, score_view


ENGLISH = {word for _, phrase in ALIASES for word in phrase.split()} | set(
    'the is to at for on of in i we you he she it this that can and or with have my me an a will was'.split())


def unambiguous(counts, minimum, agreement):
    mappings = {}
    for key, options in counts.items():
        word, count = options.most_common(1)[0]
        if count >= minimum and count >= agreement * options.total():
            mappings[key] = word
    return mappings


def reverse_dictionary(texts):
    counts = Counter(word for text in texts for word in clean(text).split() if DEVANAGARI.search(word))
    spelling = defaultdict(Counter)
    phonetic = defaultdict(Counter)
    for word, count in counts.items():
        latin = roman_token(transliterate(word))
        if latin.isascii() and latin.isalpha():
            spelling[latin][word] += count
            if len(latin) >= 4:
                phonetic[phonetic_token(latin)][word] += count
    return unambiguous(spelling, 3, .85), unambiguous(phonetic, 5, .96)


def native_view(text, spelling, phonetic):
    tokens = clean(text).split()
    converted = []
    eligible = 0
    changed = 0
    for token in tokens:
        word = token
        if token.isascii() and token.isalpha() and token not in ENGLISH:
            eligible += 1
            key = roman_token(token)
            word = spelling.get(key, token)
            if word == token and len(key) >= 4:
                word = phonetic.get(phonetic_token(key), token)
        converted.append(word)
        changed += word != token
    if changed < (2 if eligible > 2 else 1) or changed * 2 < eligible:
        return text
    return ' '.join(converted)


def run(args):
    groups, labels, original, challenge, _, scripts, config, vectors, variants, saved = prepare(args)
    encoder = Encoder(args.cache / 'e5-small', args.threads)
    results = {}
    for name, (fit, check), seed in (
            ('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)):
        spelling, phonetic = reverse_dictionary(groups['raw'][i] for i in fit)
        model = fit_fold(config, groups, labels, vectors, fit, seed, check=check)
        baseline = {view: score_view(model, groups, vectors, view, check) for view in EVAL_VIEWS}
        if name == 'fold1':
            for view in EVAL_VIEWS:
                np.testing.assert_allclose(baseline[view], saved[view][check], rtol=0, atol=.004)
        native = {view: [native_view(groups[view][i], spelling, phonetic) for i in check]
                  for view in EVAL_VIEWS}
        changed = {view: np.array([text != groups[view][i] for text, i in zip(native[view], check)])
                   for view in EVAL_VIEWS}
        if args.train_native:
            pairs = [(native_view(groups['raw'][i], spelling, phonetic), i) for i in fit]
            pairs = [(text, i) for text, i in pairs if text != groups['raw'][i]]
            train_native = [text for text, _ in pairs]
            sources = np.array([i for _, i in pairs], dtype=int)
        else:
            train_native = []
        unique = list(dict.fromkeys(train_native + [text for view in EVAL_VIEWS
                                    for text, active in zip(native[view], changed[view]) if active]))
        encoded = dict(zip(unique, encoder.encode(unique, name=name))) if unique else {}
        training = len(fit)
        if train_native:
            extra = ('native', train_native, sources, None, labels[sources])
            combined_vectors = {**vectors, 'native': np.stack([encoded[text] for text in train_native])}
            model, training = fit_candidate(config, groups, labels, combined_vectors, [*variants, extra],
                                            fit, check, seed)
            candidate = {view: score_view(model, groups, vectors, view, check) for view in EVAL_VIEWS}
        else:
            candidate = {view: logits.copy() for view, logits in baseline.items()}
        for view in EVAL_VIEWS:
            texts = [text for text, active in zip(native[view], changed[view]) if active]
            if texts:
                new_scores = model.score(texts, np.stack([encoded[text] for text in texts]))
                candidate[view][changed[view]] = .65 * candidate[view][changed[view]] + .35 * new_scores
        entry = compare(candidate, baseline, check, labels, scripts, training)
        subset = rare_roman(groups, scripts, fit, check)
        entry['low_coverage_roman'] = compare({v: candidate[v][subset] for v in EVAL_VIEWS},
                                              {v: baseline[v][subset] for v in EVAL_VIEWS},
                                              check[subset], labels, scripts, len(fit))
        entry['changed_rows'] = {view: int(changed[view].sum()) for view in EVAL_VIEWS}
        results[name] = entry
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: perturbed {entry['delta']['perturbed']:+.5f}, Roman ASR "
              f"{entry['delta']['roman_asr']:+.5f}, low-coverage Roman "
              f"{entry['low_coverage_roman']['delta']['overall']:+.5f}", flush=True)


def run_contextual(args):
    from threadpoolctl import threadpool_limits
    from sprint.score_bhasha import embed

    groups, labels, original, challenge, _, scripts, config, _, variants, _ = prepare(args)
    views = {view: groups[view] for view in (*EVAL_VIEWS, 'aug')}
    views.update({name: rows for name, rows, *_ in variants})
    plan = [('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)]
    if args.all_folds:
        plan.extend((f'fold{i + 1}', split, SEED + i) for i, split in enumerate(original) if i)
    additions = {}
    for name, (fit, check), _ in plan:
        dictionary = reverse_dictionary(groups['raw'][i] for i in fit)
        train = [(i, native_view(groups['raw'][i], *dictionary)) for i in fit]
        train = [(i, text) for i, text in train if text != groups['raw'][i]]
        origins, converted = zip(*train)
        entry = (f'{name}_fit', list(converted), np.array(origins, dtype=int), None, labels[list(origins)])
        additions[name] = entry
        views[entry[0]] = entry[1]
        for view in EVAL_VIEWS:
            views[f'{name}_{view}'] = [native_view(groups[view][i], *dictionary) for i in check]
    vectors, parameters = embed(views, args.encoder_checkpoint, 'xpu', 128)
    frozen = json.loads((args.reference.parent / 'score_push/bhasha.json').read_text(encoding='utf-8'))
    report = {}
    for name, (fit, check), seed in plan:
        subset = rare_roman(groups, scripts, fit, check)
        plain = None
        with threadpool_limits(limits=args.threads):
            if args.encoder_control:
                control, _ = fit_candidate(config, groups, labels, vectors, variants, fit, check, seed)
                plain = score(control, groups, vectors, check)
            model, counts = fit_candidate(config, groups, labels, vectors, [*variants, additions[name]],
                                          fit, check, seed)
            mixed = score(model, groups, vectors, check)
            for view in EVAL_VIEWS:
                key = f'{name}_{view}'
                active = np.array([text != groups[view][i] for text, i in zip(views[key], check)])
                if active.any():
                    changed = [text for text, use in zip(views[key], active) if use]
                    mixed[view][active] = .65 * mixed[view][active] + .35 * model.score(changed, vectors[key][active])
        result = metrics(mixed, labels[check], scripts[check])
        baseline = frozen[name]['candidate']
        delta = {metric: result[metric] - baseline[metric] for metric in result}
        report[name] = {'candidate': result, 'delta_vs_frozen': delta, 'training': counts}
        if plain is not None:
            raw = metrics(plain, labels[check], scripts[check])
            report[name].update({
                'encoder_parameters': parameters,
                'parameters_with_lexical_and_head': parameters + model.head.coef_.size
                + model.head.intercept_.size + model.features.parameter_count(),
                'raw_encoder': raw,
                'raw_delta_vs_frozen': {metric: raw[metric] - baseline[metric] for metric in raw},
                'low_coverage_roman': {
                    'rows': int(subset.sum()),
                    'raw_encoder': metrics({key: value[subset] for key, value in plain.items()},
                                           labels[check][subset], scripts[check][subset]),
                    'native_view': metrics({key: value[subset] for key, value in mixed.items()},
                                           labels[check][subset], scripts[check][subset]),
                    'bhasha': frozen[name]['low_coverage_roman']['candidate'],
                },
            })
            print(f"{name} raw encoder: perturbed {raw['perturbed'] - baseline['perturbed']:+.5f}, "
                  f"Roman ASR {raw['roman_asr'] - baseline['roman_asr']:+.5f}", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: perturbed {delta['perturbed']:+.5f}, Roman ASR {delta['roman_asr']:+.5f}", flush=True)
        if name == 'template_holdout' and any(r['delta_vs_frozen']['perturbed'] < .004 for r in report.values()):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/score_push/native.json'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--train-native', action='store_true')
    parser.add_argument('--encoder-checkpoint', '--bhasha-checkpoint', type=Path)
    parser.add_argument('--encoder-control', action='store_true')
    parser.add_argument('--all-folds', action='store_true')
    parser.set_defaults(rewrites=None)
    args = parser.parse_args()
    args.phrases = args.counterfactuals = args.train_native or args.encoder_checkpoint is not None
    run_contextual(args) if args.encoder_checkpoint else run(args)


if __name__ == '__main__':
    main()
