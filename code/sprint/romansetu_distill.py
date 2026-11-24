"""Compare gold labels and RomanSetu soft targets with a frozen Bhasha encoder."""

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from intent_model import LABELS, PARAMETER_CAP, Views, clean, script
from sprint.augment import romanize
from sprint.data import SEED, dataset_key, folds, load_training
from sprint.encoder import embed_bhasha, variant_groups
from sprint.postprocess import average_tta, logit_scale
from sprint.score_push import WEIGHT, metrics
from sprint.teacher_probe import EMBEDDING_SLOTS, extract, stratified_sample
from sprint.train import EVAL_VIEWS, Router, augmentation_rows, take


ROMANSETU = 'ai4bharat/romansetu-cpt-roman-300m@961fb408409c1b91b5d2d88b477a48c8d3064cc9'
BHASHA = 'AkshitaS/bhasha-embed-v0@64b330ff6703ffebdfc55c412a6a8aaf1b2728c8'
DISTILL_WEIGHT = .35
TEMPERATURE = 2.
RIDGE_ALPHA = 1.


def cached_embeddings(path, texts, compute):
    fingerprint = dataset_key(texts)
    if path.exists():
        with np.load(path) as cache:
            if cache['fingerprint'].item() != fingerprint:
                raise ValueError(f'Cached input fingerprint does not match at {path}')
            return cache['vectors'], int(cache['parameters']), float(cache['seconds'])
    started = monotonic()
    vectors, parameters = compute(texts)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError('Encoder returned an invalid feature matrix')
    seconds = monotonic() - started
    temporary = path.with_suffix('.partial.npz')
    np.savez(temporary, fingerprint=fingerprint, vectors=vectors, parameters=parameters, seconds=seconds)
    temporary.replace(path)
    return vectors, parameters, seconds


def teacher_predictions(fit, validation, labels, weights, groups, texts):
    targets = np.eye(len(LABELS), dtype=np.float32)[labels]
    crossfit = np.full_like(targets, np.nan)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    for number, (train, held) in enumerate(splitter.split(fit, labels, groups), 1):
        # Different source strings can map to the same Romanized query.
        train = train[~np.isin(texts[train], texts[held])]
        if not np.array_equal(np.unique(labels[train]), LABELS):
            raise ValueError('An inner teacher fold is missing a class')
        model = Ridge(alpha=RIDGE_ALPHA, solver='cholesky')
        model.fit(fit[train], targets[train], sample_weight=weights[train])
        crossfit[held] = model.predict(fit[held])
        print(f'Teacher cross-fit {number}/4 scored {len(held):,} rows', flush=True)
    if not np.isfinite(crossfit).all():
        raise ValueError('Missing out-of-fold teacher scores')
    teacher = Ridge(alpha=RIDGE_ALPHA, solver='cholesky').fit(fit, targets, sample_weight=weights)
    return crossfit, {view: teacher.predict(matrix) for view, matrix in validation.items()}


def student_heads(router, fit_texts, fit_vectors, weights, targets, val_texts, val_vectors):
    lexical = router.lexical(fit_texts, fit=True)
    embedding_weight = router.config['embedding_weight']
    dense = fit_vectors * embedding_weight
    # Weighted kernel ridge with a shared factorization and an unpenalized intercept.
    kernel = (lexical @ lexical.T).toarray(order='F')
    kernel += dense @ dense.T
    normalized = weights / weights.sum()
    row_mean = kernel @ normalized
    kernel -= row_mean[:, None]
    kernel -= row_mean[None, :]
    kernel += row_mean @ normalized
    root_weight = np.sqrt(weights)
    kernel *= root_weight[:, None]
    kernel *= root_weight[None, :]
    kernel.ravel(order='F')[::len(kernel) + 1] += RIDGE_ALPHA
    target_mean = normalized @ targets
    dual = cho_solve(cho_factor(kernel, overwrite_a=True), root_weight[:, None] * (targets - target_mean))
    dual *= root_weight[:, None]
    lexical_weights = lexical.T @ dual
    dense_weights = dense.T @ dual
    intercept = target_mean - (lexical.T @ normalized) @ lexical_weights - (normalized @ dense) @ dense_weights
    scores = {view: router.lexical(texts) @ lexical_weights +
              (val_vectors[view] * embedding_weight) @ dense_weights + intercept
              for view, texts in val_texts.items()}
    return scores, lexical.shape[1] + dense.shape[1], router.features.parameter_count()


def perturbation_interval(labels, baseline, candidate, groups):
    _, group_ids = np.unique(groups, return_inverse=True)
    group_count = int(group_ids.max() + 1)
    entries = len(LABELS)
    indices = [labels * entries + predictions for predictions in (baseline, candidate)]

    def macro(rows, weight):
        matrix = np.bincount(rows, weights=weight, minlength=entries ** 2).reshape(entries, entries)
        totals = matrix.sum(0) + matrix.sum(1)
        return np.divide(2 * matrix.diagonal(), totals, out=np.zeros(entries), where=totals > 0).mean()

    rng = np.random.default_rng(SEED + 991)
    differences = []
    for _ in range(2000):
        sampled = np.bincount(rng.integers(group_count, size=group_count), minlength=group_count)
        weight = sampled[group_ids]
        differences.append(macro(indices[1], weight) - macro(indices[0], weight))
    return np.quantile(differences, [.025, .975]).tolist()


def run(args):
    texts, labels = load_training(args.train)
    outer_fit, outer_check = folds(texts, labels)[args.fold - 1]
    rng = np.random.default_rng(SEED + args.fold - 1)
    fit = stratified_sample(outer_fit, labels, args.train_per_class, rng)
    check = stratified_sample(outer_check, labels, args.validation_per_class, rng)
    settings = json.loads((args.reference / 'metrics.json').read_text(encoding='utf-8'))
    config = settings['config']
    groups = variant_groups(texts, [])
    aug = augmentation_rows(groups, fit, config['augment'], SEED + args.fold - 1, check=outer_check)
    source_rows = np.r_[fit, aug]
    fit_texts = take(groups['raw'], fit) + take(groups['aug'], aug)
    val_texts = {view: take(groups[view], check) for view in EVAL_VIEWS}
    forbidden = {canonical(groups[view][index]) for view in EVAL_VIEWS
                 for index in outer_check for canonical in (clean, romanize)}
    allowed = np.asarray([clean(text) not in forbidden and romanize(text) not in forbidden for text in fit_texts])
    source_rows = source_rows[allowed]
    targets = labels[source_rows]
    fit_texts = take(fit_texts, np.flatnonzero(allowed))
    counts = np.bincount(targets, minlength=len(LABELS))
    if not counts.all():
        raise ValueError('A class is missing from the student fitting set')
    weights = (np.r_[np.ones(len(fit)), np.full(len(aug), .7)][allowed] *
               (counts.mean() / counts[targets]) ** config['balance']).astype(np.float32)
    print(f'Fold {args.fold}: {len(fit_texts):,} fitting queries including augmentation, '
          f'{len(check):,} held-out rows', flush=True)

    def inputs(transform):
        raw = [transform(text) for text in fit_texts]
        validation = {view: [transform(text) for text in batch] for view, batch in val_texts.items()}
        all_text = sorted(dict.fromkeys(raw + [text for batch in validation.values() for text in batch]), key=len)
        index = {text: row for row, text in enumerate(all_text)}
        return all_text, [index[text] for text in raw], {view: [index[text] for text in batch]
                                                      for view, batch in validation.items()}

    args.cache.mkdir(parents=True, exist_ok=True)
    bhasha_texts, bhasha_fit, bhasha_eval = inputs(lambda text: text)
    bhasha_cache = args.cache / f'bhasha-fp32-fold{args.fold}-{dataset_key(bhasha_texts)[:16]}.npz'

    def encode_bhasha(batch):
        import torch
        torch.set_num_threads(args.threads)
        values, parameters = embed_bhasha({'all': batch}, args.bhasha, 'cpu', 64)
        return values['all'], parameters

    bhasha, encoder_parameters, bhasha_seconds = cached_embeddings(bhasha_cache, bhasha_texts, encode_bhasha)
    print(f'Bhasha FP32 embeddings ready for {len(bhasha_texts):,} inputs', flush=True)
    if args.bhasha_only:
        return

    roman_texts, roman_fit, roman_eval = inputs(romanize)
    cache_precision = {'Q8_0': 'q8', 'F16': 'f16'}[args.teacher_precision]
    context_size = 3072 if args.teacher_precision == 'F16' else 6144
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer, local_files_only=True)
    token_ids = tokenizer(roman_texts, add_special_tokens=True, verbose=False)['input_ids']
    longest = max(map(len, token_ids))
    if longest >= context_size // EMBEDDING_SLOTS:
        raise ValueError(f'RomanSetu input length {longest} exceeds the per-slot context')
    token_sequences = [','.join(map(str, ids)) for ids in token_ids]
    roman_cache = args.cache / f'romansetu-{cache_precision}-tokenids-fold{args.fold}-{dataset_key(token_sequences)[:16]}.npz'
    roman, _, roman_seconds = cached_embeddings(
        roman_cache, token_sequences, lambda _: (extract(args, token_ids, context_size=context_size)[0], 6_738_423_808))
    if roman.shape[1] != 4096:
        raise ValueError('Expected 4096-dimensional RomanSetu embeddings')

    original_groups = np.asarray(Views.from_texts(texts).phonetic)
    evaluation_scripts = np.asarray([script(texts[row]) for row in check])
    with threadpool_limits(limits=args.threads):
        oof, teacher_scores = teacher_predictions(roman[roman_fit], {v: roman[i] for v, i in roman_eval.items()},
                                                   targets, weights, original_groups[source_rows],
                                                   np.asarray(roman_texts)[roman_fit])
        teacher_metrics = metrics(teacher_scores, labels[check], evaluation_scripts)
        print(f"RomanSetu teacher perturbed Macro-F1 {teacher_metrics['perturbed']:.5f}", flush=True)
        scale = logit_scale(oof, targets)
        probabilities = softmax(oof * (scale / TEMPERATURE), axis=1)
        one_hot = np.eye(len(LABELS), dtype=np.float32)[targets]
        p_gold = probabilities[np.arange(len(targets)), targets, None]
        uniform_wrong = one_hot * p_gold + (1 - one_hot) * (1 - p_gold) / (len(LABELS) - 1)
        variants = {'gold': one_hot, 'matched_smoothing': (1 - DISTILL_WEIGHT) * one_hot + DISTILL_WEIGHT * uniform_wrong,
                    'distilled': (1 - DISTILL_WEIGHT) * one_hot + DISTILL_WEIGHT * probabilities}
        combined = np.hstack(list(variants.values())).astype(np.float32)
        scores, features, lexical_parameters = student_heads(Router(config), fit_texts, bhasha[bhasha_fit], weights,
                                                               combined, val_texts, {v: bhasha[i] for v, i in bhasha_eval.items()})

    parameters = encoder_parameters + lexical_parameters + (features + 1) * len(LABELS)
    if parameters > PARAMETER_CAP:
        raise ValueError(f'The student exceeds the deployment parameter cap ({parameters:,})')
    student_metrics, predictions = {}, {}
    for i, name in enumerate(variants):
        columns = slice(i * len(LABELS), (i + 1) * len(LABELS))
        by_view = {view: values[:, columns] for view, values in scores.items()}
        student_metrics[name] = metrics(by_view, labels[check], evaluation_scripts)
        predictions[name] = average_tta(by_view['natural'], by_view['natural_tta'], WEIGHT).argmax(1)
    with np.load(args.reference / 'scores.npz') as saved:
        if not np.array_equal(saved['target'], labels) or not np.array_equal(np.flatnonzero(saved['fold_id'] == args.fold - 1), outer_check):
            raise ValueError('The archived E5 fold does not match train.csv')
        archived = metrics({view: saved[view][check] for view in EVAL_VIEWS}, labels[check], evaluation_scripts)
    deltas = {name: {key: student_metrics['distilled'][key] - student_metrics[name][key]
                     for key in teacher_metrics} for name in ('gold', 'matched_smoothing')}
    intervals = {name: perturbation_interval(labels[check], predictions[name], predictions['distilled'], original_groups[check])
                 for name in deltas}
    report = {'teacher': ROMANSETU, 'teacher_precision': f'{args.teacher_precision} GGUF',
              'teacher_pooling': 'last-token on Romanized text',
              'teacher_tokenization': 'exact token IDs from the published Hugging Face tokenizer',
              'teacher_max_input_tokens': longest,
              'student_encoder': BHASHA, 'student_encoder_precision': 'FP32 frozen', 'student_parameters': parameters,
              'train_dataset_key': dataset_key(texts), 'fold': args.fold, 'fit_sources_before_overlap_filtering': len(fit),
              'fit_queries_with_augmentation': len(fit_texts), 'evaluation_rows': len(check), 'inner_teacher_folds': 4,
              'teacher_weight': DISTILL_WEIGHT, 'temperature': TEMPERATURE, 'teacher_logit_scale': scale,
              'ridge_alpha': RIDGE_ALPHA, 'teacher_embedding_seconds': roman_seconds, 'student_embedding_seconds': bhasha_seconds,
              'teacher_metrics': teacher_metrics, 'student_metrics': student_metrics,
              'distilled_deltas': deltas, 'perturbed_delta_group_bootstrap_95pct': intervals,
              'archived_e5_with_full_fold_training': archived}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('server', 'backend', 'model', 'teacher-tokenizer', 'bhasha', 'cache', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--train', type=Path, default=Path('train.csv'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--teacher-precision', choices=('Q8_0', 'F16'), default='Q8_0')
    parser.add_argument('--train-per-class', type=int, default=48)
    parser.add_argument('--validation-per-class', type=int, default=24)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--bhasha-only', action='store_true')
    args = parser.parse_args()
    if min(args.train_per_class, args.validation_per_class, args.threads) < 1 or args.output.exists():
        parser.error('Provide positive sample caps and thread count, and a new output path')
    run(args)


if __name__ == '__main__':
    main()
