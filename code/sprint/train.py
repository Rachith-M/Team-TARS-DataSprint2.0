import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, LABELS, Views, Features, clean, script
from sprint.augment import procedural_oos
from sprint.data import SEED, folds, load_data, macro, slices, write_submission
from sprint.encoder import cache_embeddings, variant_groups
from sprint.postprocess import average_tta, cluster_logits, logit_scale, pseudo_candidates


CONFIG_FILE = Path(__file__).with_name('configs.json')
EVAL_VIEWS = ('raw', 'tta', 'natural', 'natural_tta', 'asr')


def read_configs(path=CONFIG_FILE):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def take(sequence, indices):
    return [sequence[index] for index in indices]


def load_oos(path):
    import csv
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        rows = csv.DictReader(file)
        if rows.fieldnames != ['query_text']:
            raise ValueError('OOS text pack must have a query_text column')
        return list(dict.fromkeys(row['query_text'].strip() for row in rows if row['query_text'].strip()))


class Router:
    def __init__(self, config):
        self.config = config
        self.features = {
            'cross-script': Features,
            'char': lambda: TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), dtype=np.float32),
            None: lambda: None,
        }[config['lexical']]()

    def lexical(self, texts, fit=False):
        if self.config['lexical'] == 'cross-script':
            return self.features.encode(Views.from_texts(texts), fit=fit)
        if self.config['lexical'] == 'char':
            return self.features.fit_transform(texts) if fit else self.features.transform(texts)
        return None

    def matrix(self, texts, embedding, fit=False):
        lexical = self.lexical(texts, fit=fit)
        if embedding is None:
            return lexical
        embedding = embedding * self.config.get('embedding_weight', 1.0)
        return embedding if lexical is None else sparse.hstack((lexical, embedding), format='csr')

    def fit(self, texts, labels, embedding=None, sample_weight=None):
        matrix = self.matrix(texts, embedding, fit=True)
        return self.fit_matrix(matrix, labels, sample_weight)

    def fit_matrix(self, matrix, labels, sample_weight=None):
        counts = np.bincount(labels, minlength=56)
        weights = {label: (counts[counts > 0].mean() / count) ** self.config['balance']
                   for label, count in enumerate(counts) if count}
        self.classes = np.flatnonzero(counts)
        if self.config.get('head') == 'ridge':
            self.head = Ridge(alpha=self.config['c'], solver='cholesky')
            weighted = np.asarray([weights[value] for value in labels])
            if sample_weight is not None:
                weighted *= sample_weight
            one_hot = (labels[:, None] == self.classes).astype(np.float32)
            self.head.fit(matrix, one_hot, sample_weight=weighted)
        else:
            self.head = LinearSVC(C=self.config['c'], class_weight=weights, dual=sparse.issparse(matrix),
                                  random_state=SEED, tol=self.config.get('tol', .004),
                                  max_iter=self.config.get('max_iter', 2500))
            self.head.fit(matrix, labels, sample_weight=sample_weight)
        self.head.coef_ = np.asfortranarray(self.head.coef_, dtype=np.float32)
        self.head.intercept_ = self.head.intercept_.astype(np.float32)
        return self

    def score(self, texts, embedding=None):
        matrix = self.matrix(texts, embedding)
        raw = matrix @ self.head.coef_.T + self.head.intercept_
        logits = np.full((len(texts), len(LABELS)), -6, dtype=np.float32)
        logits[:, self.classes] = raw
        return logits


def augmentation_rows(groups, fit, rate, seed, check=None):
    rng = np.random.default_rng(seed)
    available = np.array([i for i in fit if groups['raw'][i] != groups['aug'][i]], dtype=int)
    chosen = rng.choice(available, min(len(available), round(rate * len(fit))), replace=False)
    if check is not None and len(chosen):
        forbidden = {clean(groups[style][i]) for style in EVAL_VIEWS for i in check}
        chosen = np.array([i for i in chosen if clean(groups['aug'][i]) not in forbidden], dtype=int)
    return chosen


def fit_fold(config, groups, labels, embedding, fit, seed, custom_oos=(), encoder=None, pseudo=None, check=None):
    texts = take(groups['raw'], fit)
    train_labels = labels[fit]
    extra = [procedural_oos(texts, train_labels, config['oos'], seed), list(custom_oos)]
    chosen = augmentation_rows(groups, fit, config['augment'], seed, check)
    augmented = take(groups['aug'], chosen)
    synthetic_oos = extra[0] + extra[1]
    weights = np.r_[np.ones(len(fit)), np.full(len(chosen), .70), np.full(len(extra[0]), .40),
                    np.ones(len(extra[1]))]
    targets = np.r_[train_labels, labels[chosen], np.full(len(synthetic_oos), FALLBACK, dtype=int)]
    vectors = None
    if embedding:
        pieces = [embedding['raw'][fit], embedding['aug'][chosen]]
        if synthetic_oos:
            pieces.append(encoder.encode(synthetic_oos))
        vectors = np.concatenate(pieces)
    train_texts = texts + augmented + synthetic_oos
    if pseudo is not None:
        indices, predictions = pseudo
        train_texts += take(groups['test'], indices)
        targets = np.r_[targets, predictions]
        weights = np.r_[weights, np.full(len(indices), .30)]
        if embedding:
            vectors = np.vstack((vectors, embedding['test'][indices]))
    return Router(config).fit(train_texts, targets, vectors, weights)


def score_view(router, groups, embedding, name, rows=None):
    text = groups[name] if rows is None else take(groups[name], rows)
    vectors = None if embedding is None else (embedding[name] if rows is None else embedding[name][rows])
    return router.score(text, vectors)


def metrics(logits, groups, labels, scripts, fold_id):
    candidates = []
    for weight in (0, .35, .65):
        clean = average_tta(logits['raw'], logits['tta'], weight)
        natural = average_tta(logits['natural'], logits['natural_tta'], weight)
        for cluster_weight in (0., 1.):
            if cluster_weight:
                clean = cluster_logits(groups['raw'], clean, cluster_weight, fold_id)
                natural = cluster_logits(groups['natural'], natural, cluster_weight, fold_id)
            predictions = clean.argmax(axis=1)
            test_like = natural.argmax(axis=1)
            baseline = slices(labels, predictions, scripts)
            natural_f1 = macro(labels, test_like)
            asr_f1 = macro(labels, logits['asr'].argmax(axis=1))
            score = .4 * baseline['overall'] + .6 * (.8 * natural_f1 + .2 * asr_f1)
            candidates.append(dict(**baseline, test_like=natural_f1, asr=asr_f1,
                                   test_like_roman=macro(labels[scripts == 'roman'], test_like[scripts == 'roman']),
                                   tta_weight=weight, cluster_weight=cluster_weight, selection_score=score,
                                   class55_f1=float(f1_score(labels == 55, predictions == 55)),
                                   class55_precision=float(precision_score(labels == 55, predictions == 55, zero_division=0)),
                                   class55_recall=float(recall_score(labels == 55, predictions == 55)),
                                   class55_test_like_f1=float(f1_score(labels == 55, test_like == 55)),
                                   by_fold=[macro(labels[fold_id == fold], predictions[fold_id == fold])
                                            for fold in np.unique(fold_id)]))
    return dict(max(candidates, key=lambda row: row['selection_score'])), candidates


def run(config, directory, cache, output, threads, custom_oos, enable_pseudo=False):
    started = perf_counter()
    texts, labels, test, ids = load_data(directory)
    groups = variant_groups(texts, test)
    cache_paths = cache_embeddings(groups, cache, config['encoder'], threads=threads) if config['encoder'] else None
    embedding = {name: np.load(path, mmap_mode='r') for name, path in cache_paths.items()} if cache_paths else None
    encoder = None
    if embedding and (config['oos'] or custom_oos):
        from sprint.encoder import Encoder
        encoder = Encoder(cache / config['encoder'], threads)
    scripts = np.asarray([script(text) for text in texts])
    oof = {name: np.zeros((len(labels), len(LABELS)), dtype=np.float32) for name in EVAL_VIEWS}
    fold_id = np.empty(len(labels), dtype=np.int8)
    with threadpool_limits(limits=threads):
        for fold, (fit, check) in enumerate(folds(texts, labels)):
            model = fit_fold(config, groups, labels, embedding, fit, SEED + fold, custom_oos, encoder, check=check)
            for name in EVAL_VIEWS:
                oof[name][check] = score_view(model, groups, embedding, name, check)
            fold_id[check] = fold
            clean = macro(labels[check], oof['raw'][check].argmax(axis=1))
            shifted = macro(labels[check], oof['natural'][check].argmax(axis=1))
            print(f'Fold {fold+1}/5 clean {clean:.4f}, test-like {shifted:.4f}', flush=True)
            del model
        report, candidates = metrics(oof, groups, labels, scripts, fold_id)
        best_tta = report['tta_weight']
        scale = logit_scale(average_tta(oof['raw'], oof['tta'], best_tta), labels)
        training = np.arange(len(labels))
        model = fit_fold(config, groups, labels, embedding, training, SEED + 5, custom_oos, encoder)
        test_raw = score_view(model, groups, embedding, 'test')
        test_tta = score_view(model, groups, embedding, 'test_tta')
        pseudo_count = 0
        if enable_pseudo:
            indices, targets, _ = pseudo_candidates(test_raw, test_tta, oof['raw'], labels, scale)
            pseudo_count = len(indices)
            if pseudo_count:
                model = fit_fold(config, groups, labels, embedding, training, SEED + 5, custom_oos, encoder,
                                 (indices, targets))
                test_raw = score_view(model, groups, embedding, 'test')
                test_tta = score_view(model, groups, embedding, 'test_tta')
    scores = average_tta(test_raw, test_tta, best_tta)
    if report['cluster_weight']:
        scores = cluster_logits(test, scores, report['cluster_weight'])
    target = scores.argmax(axis=1)
    report.update({'config': config, 'scale': scale, 'test_fallback_predictions': int(np.sum(target == FALLBACK)),
                   'pseudo_labels': pseudo_count, 'total_seconds': round(perf_counter() - started, 1),
                   'alternatives': candidates})
    np.savez_compressed(output / 'scores.npz', **oof, fold_id=fold_id, target=labels, example_id=np.array(ids),
                        test=test_raw, test_tta=test_tta)
    joblib.dump(model.features, output / 'features.joblib', compress=3)
    np.savez_compressed(output / 'head.npz', weights=model.head.coef_.T, bias=model.head.intercept_)
    write_submission(output / 'submission.csv', ids, target)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'alternatives'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--config', type=Path, default=CONFIG_FILE)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--team-oos', type=Path)
    parser.add_argument('--pseudo-label', action='store_true')
    args = parser.parse_args()
    configurations = read_configs(args.config)
    custom_oos = load_oos(args.team_oos) if args.team_oos else []
    for name in args.names:
        path = args.output / (name + ('_pseudo' if args.pseudo_label else ''))
        path.mkdir(parents=True, exist_ok=True)
        print(f'Running {name}', flush=True)
        run(configurations[name], args.data, args.cache, path, args.threads, custom_oos, args.pseudo_label)


if __name__ == '__main__':
    main()
