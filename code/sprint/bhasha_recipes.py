"""Compare fold-local recipe contrast pairs against the Bhasha dictionary baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from intent_model import PARAMETER_CAP, Views
from sprint.augment import tta_variant
from sprint.bhasha_audit import (BASE_VIEWS, REQUEST_VIEWS, cached_vectors, dictionary_inputs,
                                 inner_partition, metric_delta, route_metrics, score_parts)
from sprint.bhasha_bias import int8_head
from sprint.data import SEED
from sprint.romansetu_distill import perturbation_interval
from sprint.score_data import normalized_words, word_pattern
from sprint.score_push import fit_candidate, fingerprint, prepare, samples
from sprint.train import take


RECIPE = 9
TECHNIQUE = 36
RECIPE_CLASSES = (RECIPE, TECHNIQUE)
# Each constraint is a plausible, specific change to the general request for the same dish.
DISHES = (
    ('dosa', 'डोसा', 'tawa se aasani se nikle', 'तवे से आसानी से उठ जाए'),
    ('samosa', 'समोसा', 'talte waqt khule nahi', 'तलते वक्त खुल न जाए'),
    ('palak paneer', 'पालक पनीर', 'palak ka rang feeka na pade', 'पालक का रंग फीका न पड़े'),
    ('paneer tikka', 'पनीर टिक्का', 'grill karne par sukha na ho', 'सेंकने पर सूखा न हो'),
    ('sabudana khichdi', 'साबूदाना खिचड़ी', 'dane alag alag rahein', 'दाने अलग-अलग रहें'),
    ('dahi vada', 'दही वडा', 'vada dahi dalne ke baad bhi na toote', 'वड़ा दही डालने के बाद भी न टूटे'),
    ('rajma', 'राजमा', 'gravy patli na rahe', 'ग्रेवी पतली न रहे'),
    ('gulab jamun', 'गुलाब जामुन', 'chasni andar tak pahunche', 'चाशनी अंदर तक पहुँचे'),
    ('kheer', 'खीर', 'doodh tale me na lage', 'दूध तले में न लगे'),
    ('masala chai', 'मसाला चाय', 'zyada der ubaalne se kadwi na ho', 'ज़्यादा देर उबालने से कड़वी न हो'),
    ('egg curry', 'अंडा करी', 'gravy pakate hue tel alag na dikhe', 'ग्रेवी पकाते हुए तेल अलग न दिखे'),
    ('dal makhani', 'दाल मखनी', 'dal neeche na jale', 'दाल नीचे न जले'),
    ('biryani', 'बिरयानी', 'chawal dum dete waqt zyada na gal jaye', 'चावल दम देते वक्त ज़्यादा न गल जाएँ'),
    ('cake', 'केक', 'beech ka hissa kaccha na rahe', 'बीच का हिस्सा कच्चा न रहे'),
    ('paratha', 'पराठा', 'belte waqt bharavan bahar na nikle', 'बेलते वक्त भरावन बाहर न निकले'),
    ('kadhi', 'कढ़ी', 'ubalte hue phate nahi', 'उबलते हुए फटे नहीं'),
    ('bread roll', 'ब्रेड रोल', 'talne se pehle bharavan geela na rahe', 'तलने से पहले भरावन गीला न रहे'),
    ('chhole', 'छोले', 'masala dane ke andar tak jaye', 'मसाला दाने के अंदर तक जाए'),
)


def recipe_variants(groups, labels, fit, check, parts):
    sources = [(i, normalized_words(groups['raw'][i])) for i in fit if labels[i] == TECHNIQUE]
    blocked = [text for view in BASE_VIEWS for text in take(groups[view], check)]
    blocked.extend(text for _, rows, indices, _, _ in parts for text in take(rows, indices))
    known = set(Views.from_texts(blocked).phonetic)
    texts, origins = [], []
    for dish, hindi, constraint, hindi_constraint in DISHES:
        pattern = word_pattern((dish, hindi))
        source = next((i for i, text in sources if pattern.search(text)), None)
        if source is None:
            continue
        pairs = (
            (f'{dish} banane ka tareeka shuru se batao',
             f'{dish} banate waqt kya karun taaki {constraint}'),
            (f'{dish} ki full recipe ingredients ke saath share karo',
             f'{dish} ki recipe me kya badlun taaki {constraint}'),
            (f'{hindi} बनाने की पूरी विधि बता दो',
             f'{hindi} बनाते समय क्या करूँ ताकि {hindi_constraint}'),
            (f'{hindi} बनाने के लिए सामग्री और सारे चरण बताओ',
             f'{hindi} बनाने के तरीके में क्या बदलाव करूँ ताकि {hindi_constraint}'),
        )
        for pair in pairs:
            keys = set(Views.from_texts(pair).phonetic)
            if len(keys) == 2 and keys.isdisjoint(known):
                texts.extend(pair)
                origins.extend((source, source))
                known.update(keys)
    if not origins:
        raise ValueError('No eligible recipe contrast pairs')
    targets = np.tile(RECIPE_CLASSES, len(texts) // 2)
    return 'recipe_pairs', texts, np.asarray(origins), None, targets


def recipe_metrics(labels, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=RECIPE_CLASSES, zero_division=0)
    return {'classes': {str(label): {'precision': p, 'recall': r, 'f1': f, 'support': int(n)}
                        for label, p, r, f, n in zip(RECIPE_CLASSES, precision, recall, f1, support)},
            '36_to_9': int(np.sum((labels == TECHNIQUE) & (predictions == RECIPE))),
            '9_to_36': int(np.sum((labels == RECIPE) & (predictions == TECHNIQUE)))}


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(args):
    groups, labels, folds, challenge, frames, scripts, config, _, variants, _ = prepare(
        args, include_reference_vectors=False)
    groups['asr_tta'] = list(map(tta_variant, groups['asr']))
    base_views = {view: groups[view] for view in (*BASE_VIEWS, 'aug')}
    base_views.update({name: texts for name, texts, *_ in variants})
    vectors, encoder_parameters = cached_vectors(args, base_views, 'base')
    phonetic = np.asarray(Views.from_texts(groups['raw']).phonetic)
    plan = [(f'fold{i + 1}', fit, check, SEED + i, phonetic) for i, (fit, check) in enumerate(folds)]
    plan.append(('template_holdout', *challenge, SEED + 9, frames))

    # Include cached inner views in the same order as the bias experiment to reuse its embeddings.
    native, converted, pairs = {}, {}, {}
    for name, fit, check, seed, keys in plan:
        inner_fit, inner_check = inner_partition(fit, labels, scripts, keys, seed)
        for role, train, heldout in (('inner', inner_fit, inner_check), ('outer', fit, check)):
            dictionary, changed, texts = dictionary_inputs(groups, train, heldout)
            converted.update({f'{name}_{role}_{view}': rows for view, rows in texts.items()})
        native[name] = dictionary, changed, texts
        parts = samples(config, groups, labels, fit, check, variants, seed)
        pairs[name] = recipe_variants(groups, labels, fit, check, parts)
    native_vectors, _ = cached_vectors(args, converted, 'dictionary')
    pair_vectors, _ = cached_vectors(args, {name: variant[1] for name, variant in pairs.items()}, 'recipe-pairs')

    arms = ('baseline', 'recipe_pairs')
    oof = {arm: {view: np.full(len(labels), -1, dtype=np.int32) for view in REQUEST_VIEWS} for arm in arms}
    report = {'status': 'running', 'train_sha256': fingerprint(args.data / 'train.csv'), 'folds': [],
              'method': 'Exploratory training-only contrast pairs, judged with simulated int8 heads',
              'pair_source': 'A class 36 row mentioning the dish must be present in each fit partition.',
              'pair_weight': .55, 'uses_test_csv': False}
    for name, fit, check, seed, keys in [plan[-1], *plan[:-1]]:
        dictionary, changed, texts = native[name]
        embeddings = {view: native_vectors[f'{name}_outer_{view}'] for view in REQUEST_VIEWS}
        extra = pairs[name]
        rows = labels[check]
        measured, diagnostic, predicted, parameter_counts = {}, {}, {}, {}
        for arm in arms:
            print(f'{name}: train and score {arm}', flush=True)
            augmented = arm == 'recipe_pairs'
            with threadpool_limits(limits=args.threads):
                model, counts = fit_candidate(config, groups, labels, {**vectors, 'recipe_pairs': pair_vectors[name]},
                                              [*variants, extra] if augmented else variants, fit, check, seed)
                if augmented and counts.get(arm) != len(extra[1]):
                    raise ValueError('A generated recipe pair was dropped during fitting')
                logits = score_parts(int8_head(model), groups, vectors, check, changed, texts, embeddings)[1]
            parameter_counts[arm] = (encoder_parameters + model.head.coef_.size + model.head.intercept_.size
                                     + 2 * len(model.classes) + 8 + model.features.parameter_count()
                                     + sum(map(len, dictionary)))
            if parameter_counts[arm] > PARAMETER_CAP:
                raise ValueError('Estimated runtime parameters exceed the cap')
            measured[arm] = route_metrics(logits, rows, scripts[check])
            predicted[arm] = {view: scores.argmax(1) for view, scores in logits.items()}
            diagnostic[arm] = {view: recipe_metrics(rows, prediction) for view, prediction in predicted[arm].items()}
        delta = metric_delta(measured['recipe_pairs'], measured['baseline'])
        result = {'split': name, 'evaluation_rows': len(check), 'added_rows': len(extra[1]),
                  'recipe_pair_rows': [{'query': query, 'source_row': int(source), 'label': int(target)}
                                       for query, source, target in zip(extra[1], extra[2], extra[4])],
                  'metrics': measured, 'delta': delta, 'recipe_metrics': diagnostic,
                  'estimated_parameters_with_dictionary': parameter_counts}
        if name == 'template_holdout':
            result['perturbed_paired_group_interval_95pct'] = perturbation_interval(
                rows, predicted['baseline']['perturbed'], predicted['recipe_pairs']['perturbed'], keys[check])
            report[name] = result
        else:
            for arm in arms:
                for view in REQUEST_VIEWS:
                    oof[arm][view][check] = predicted[arm][view]
            report['folds'].append(result)
        write_report(args.output, report)
        print(f"{name}: {len(extra[1])} examples, perturbed {delta['perturbed']:+.5f}", flush=True)

    if any(np.any(scores < 0) for variants_by_view in oof.values() for scores in variants_by_view.values()):
        raise ValueError('Incomplete out-of-fold predictions')
    mean = {arm: {metric: float(np.mean([r['metrics'][arm][metric] for r in report['folds']]))
                  for metric in measured[arm]} for arm in arms}
    report['five_fold_mean'] = mean
    report['five_fold_delta'] = metric_delta(mean['recipe_pairs'], mean['baseline'])
    report['oof_recipe_metrics'] = {arm: {view: recipe_metrics(labels, prediction) for view, prediction in views.items()}
                                    for arm, views in oof.items()}
    report['perturbed_paired_group_interval_95pct'] = perturbation_interval(
        labels, oof['baseline']['perturbed'], oof['recipe_pairs']['perturbed'], phonetic)
    report['status'] = 'complete'
    write_report(args.output, report)
    print('Five-fold delta', report['five_fold_delta'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    for name in ('cache', 'checkpoint', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or min(args.threads, args.batch_size) < 1:
        parser.error('Provide a new output path and positive thread and batch counts')
    args.phrases = args.counterfactuals = True
    args.rewrites = None
    run(args)


if __name__ == '__main__':
    main()
