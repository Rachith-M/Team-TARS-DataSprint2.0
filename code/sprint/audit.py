import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from intent_model import LABELS, Views, clean, script
from sprint.augment import FILLERS, cluster_key, romanize
from sprint.data import SEED, load_data, macro, read_csv, slices


LATIN_TOKEN = re.compile(r'[a-z]+')


def lexical_model(fit_texts, labels, check_texts):
    features = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), dtype=np.float32)
    train = features.fit_transform(fit_texts)
    model = LinearSVC(C=.5, class_weight='balanced', random_state=SEED, dual=True, tol=.001)
    model.fit(train, labels)
    matrices = [features.transform(texts) for texts in check_texts]
    return [model.decision_function(matrix) for matrix in matrices]


def oov(texts, vocabulary):
    counts = Counter(word for text in texts for word in LATIN_TOKEN.findall(text.casefold()))
    return sum(count for word, count in counts.items() if word not in vocabulary) / sum(counts.values())


def token_vocabulary(texts):
    return {word for text in texts for word in LATIN_TOKEN.findall(text.casefold())}


def divergence(a, b):
    terms = a.keys() | b.keys()
    p = np.array([a[k] for k in terms], dtype=float)
    q = np.array([b[k] for k in terms], dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return float(sum(np.sum(x[x > 0] * np.log2(x[x > 0] / m[x > 0])) for x in (p, q)) / 2)


def grams(texts):
    return Counter(text[i:i + 3] for text in texts for i in range(len(text) - 2))


def profile(texts):
    lengths = [len(text.split()) for text in texts]
    filler = [set(LATIN_TOKEN.findall(text.casefold())) & (FILLERS | {'na'}) for text in texts]
    return {
        'scripts': dict(Counter(map(script, texts))),
        'digits_any_unicode': sum(any(char.isdecimal() for char in text) for text in texts),
        'digits_ascii': sum(bool(re.search(r'[0-9]', text)) for text in texts),
        'max_whitespace_words': max(lengths), 'median_whitespace_words': float(np.median(lengths)),
        'filler_rows': sum(bool(row) for row in filler),
        'filler_counts_by_row': dict(Counter(token for row in filler for token in sorted(row))),
        'long_vowel_rows_in_roman': sum(bool(re.search(r'aa|ee|oo', text.casefold()))
                                        for text in texts if script(text) == 'roman'),
    }


def audit(directory):
    texts, labels, test, ids = load_data(directory)
    scripts = np.array(list(map(script, texts)))
    train_roman = [text for text, value in zip(texts, scripts) if value == 'roman']
    train_deva = [text for text, value in zip(texts, scripts) if value == 'devanagari']
    test_roman = [text for text in test if script(text) == 'roman']
    latin_vocab = token_vocabulary(train_roman)
    romanized = [romanize(text) for text in train_deva]
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(texts, labels))
    logits = np.zeros((len(labels), len(LABELS)))
    fold_scores, oov_rates, transfer = [], [], {}
    with threadpool_limits(limits=2):
        for fit, check in splits:
            train_text = [texts[index] for index in fit]
            val_text = [texts[index] for index in check]
            logits[check], = lexical_model(train_text, labels[fit], [val_text])
            fold_scores.append(macro(labels[check], logits[check].argmax(axis=1)))
            vocab = token_vocabulary([text for text in train_text if script(text) == 'roman'])
            oov_rates.append(oov([text for text in val_text if script(text) == 'roman'], vocab))
            print(f'Audit baseline fold {len(fold_scores)}/5', flush=True)
        fitted, test_logits = lexical_model(texts, labels, [texts, test])
        for source, destination in (('devanagari', 'roman'), ('roman', 'devanagari')):
            fit_idx = np.flatnonzero(scripts == source)
            val_idx = np.flatnonzero(scripts == destination)
            raw, = lexical_model([texts[i] for i in fit_idx], labels[fit_idx], [[texts[i] for i in val_idx]])
            aligned, = lexical_model([romanize(texts[i]) for i in fit_idx], labels[fit_idx],
                                     [[romanize(texts[i]) for i in val_idx]])
            transfer[f'{source}_to_{destination}'] = {'raw': macro(labels[val_idx], raw.argmax(axis=1)),
                                                        'romanized': macro(labels[val_idx], aligned.argmax(axis=1))}

    predicted = logits.argmax(axis=1)
    class_f1 = f1_score(labels, predicted, labels=LABELS, average=None, zero_division=0)
    confusion = confusion_matrix(labels, predicted, labels=LABELS)
    np.fill_diagonal(confusion, 0)
    top_errors = sorted(np.ndindex(confusion.shape), key=lambda pair: -confusion[pair])[:10]
    groups = defaultdict(list)
    for row, text in enumerate(test):
        groups[cluster_key(text)].append(row)
    clusters = [group for key, group in groups.items() if key and len(group) > 1]
    test_predictions = test_logits.argmax(axis=1)
    class55_roman = [romanize(text) for text, label in zip(texts, labels) if label == 55]
    markers = ('alarm', 'alaarm', 'podcast', 'podakaast', 'kofi', 'kophee', 'coffee', 'ludo', 'loodo',
               'shataranj', 'chess', 'pabjee', 'olly', 'ollyy', 'khel', 'khelnaa')
    raw_overlap = len(set(texts) & set(test))
    cleaned_train = {clean(text) for text in texts}
    phonetic = set(Views.from_texts(texts).phonetic)
    fit_max, oof_max, test_max = fitted.max(axis=1), logits.max(axis=1), test_logits.max(axis=1)
    template = read_csv(Path(directory) / 'sample_submission.csv', ('example_id', 'target'))
    counts = np.bincount(labels, minlength=56)
    return {
        'train_rows': len(labels), 'test_rows': len(ids), 'empty_train_queries': sum(not t.strip() for t in texts),
        'empty_test_queries': sum(not t.strip() for t in test),
        'duplicate_train_queries': len(texts) - len(set(texts)), 'raw_train_test_overlap': raw_overlap,
        'normalized_train_test_overlap_rows': sum(clean(text) in cleaned_train for text in test),
        'phonetic_train_test_overlap_rows': sum(text in phonetic for text in Views.from_texts(test).phonetic),
        'target_counts': counts.tolist(), 'target_median': float(np.median(counts)),
        'template_placeholders': dict(Counter(row['target'] for row in template)),
        'train': profile(texts), 'test': profile(test),
        'oov': {'test_roman': oov(test_roman, latin_vocab), 'train_roman_cv': float(np.mean(oov_rates)),
                'transliterated_train_deva': oov(romanized, latin_vocab),
                'test_roman_with_transliterated_vocab': oov(test_roman, latin_vocab | token_vocabulary(romanized))},
        'train_deva_train_roman_trigram_jsd': {'raw': divergence(grams(train_deva), grams(train_roman)),
                                              'romanized': divergence(grams(romanized), grams(train_roman))},
        'cross_script': transfer,
        'baseline': {'macro_f1_by_fold': fold_scores, 'oof': slices(labels, predicted, scripts),
                     'worst_classes': [[int(k), float(class_f1[k])] for k in np.argsort(class_f1)[:7]],
                     'top_errors': [[i, j, int(confusion[i,j])] for i, j in top_errors],
                     'test_fallback_rate': float(np.mean(test_predictions == 55)),
                     'test_fallback_by_script': {k: float(np.mean(test_predictions[np.array(list(map(script, test))) == k] == 55))
                                                 for k in ('roman', 'devanagari', 'mixed')},
                     'median_max_margin': {'train_fitted': float(np.median(fit_max)), 'train_oof': float(np.median(oof_max)),
                                           'test': float(np.median(test_max))},
                     'test_below_train_p05': {'train_fitted': float(np.mean(test_max < np.percentile(fit_max, 5))),
                                              'train_oof': float(np.mean(test_max < np.percentile(oof_max, 5)))}},
        'normalized_test_clusters': len(clusters), 'clustered_test_rows': sum(map(len, clusters)),
        'clusters_with_baseline_disagreement': sum(len(set(test_predictions[group])) > 1 for group in clusters),
        'class55_topic_markers': {word: sum(word in tokenize for tokenize in map(set, map(str.split, class55_roman)))
                                  for word in markers},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('sprint_results/audit.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.data)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
