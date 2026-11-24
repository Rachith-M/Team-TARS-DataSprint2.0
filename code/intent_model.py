"""Local Hindi/Hinglish intent routing using sparse lexical models."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


LABELS = np.arange(56)
FALLBACK = 55
PARAMETER_CAP = 300_000_000

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
LATIN = re.compile(r"[a-z]", re.IGNORECASE | re.ASCII)
SPACE = re.compile(r"\s+")
MARKS = re.compile(r"[^\w\s\u0900-\u0963\u0971-\u097f]", re.UNICODE)
ROMAN_WORD = re.compile(r"[a-z0-9]+")
REPEATS = re.compile(r"([a-z])\1{2,}")
ROMAN_LONG_VOWELS = re.compile(r"aa+|ee+|ii+|oo+|uu+|ph|qu|q|w")
ROMAN_CONSONANT_FOLD = re.compile(r"([kgtjdcbps])h|([a-z])\2+|[aeiouzvj]")

VOWELS = dict(zip("अआइईउऊऋॠऌएऐओऔऑऍ", (
    "a", "aa", "i", "ee", "u", "oo", "ri", "ri", "li", "e", "ai", "o", "au", "o", "e",
)))
MATRAS = dict(zip("ािीुूृॄॢेैोौॉॅ", (
    "aa", "i", "ee", "u", "oo", "ri", "ri", "li", "e", "ai", "o", "au", "o", "e",
)))
CONSONANTS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ", (
    "k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n",
    "t", "th", "d", "dh", "n", "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h", "l",
)))
NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}

VOWEL_FOLDS = {"aa": "a", "ee": "i", "ii": "i", "oo": "u", "uu": "u", "ph": "f", "qu": "k", "q": "k", "w": "v"}
PHONETIC_FOLDS = {"z": "s", "v": "b", "j": "y"}


def script(text: str) -> str:
    if DEVANAGARI.search(text):
        return "mixed" if LATIN.search(text) else "devanagari"
    return "roman"


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u200d", "").replace("\u200c", "")
    text = "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in text)
    return SPACE.sub(" ", MARKS.sub(" ", text)).strip()


def transliterate(text: str) -> str:
    """Map Devanagari to Roman text and remove common implicit schwas."""
    output = []
    for index, char in enumerate(text):
        if char in CONSONANTS:
            following = text[index + 1:index + 2]
            consonant = CONSONANTS[char]
            if following == "़":
                consonant = NUKTA.get(char, consonant)
                following = text[index + 2:index + 3]
            output.append(consonant)
            if following and DEVANAGARI.fullmatch(following) and following not in MATRAS and following != "्":
                output.append("a")
        elif char in MATRAS:
            output.append(MATRAS[char])
        elif char in VOWELS:
            output.append(VOWELS[char])
        elif char in "ंँ":
            following = text[index + 1:index + 2]
            output.append("m" if following in {"प", "फ", "ब", "भ", "म"} else "n")
        elif char == "ः":
            output.append("h")
        elif char not in "़्":
            output.append(char)
    return "".join(output)


def roman_token(token: str) -> str:
    token = REPEATS.sub(r"\1\1", token)
    return ROMAN_LONG_VOWELS.sub(lambda match: VOWEL_FOLDS[match[0][:2]], token)


def phonetic_token(token: str) -> str:
    token = token.replace("chh", "ch").replace("sh", "s").replace("zh", "z")
    token = ROMAN_CONSONANT_FOLD.sub(lambda match: match[1] or match[2] or PHONETIC_FOLDS.get(match[0], ""), token)
    return token or "a"


@dataclass
class Views:
    raw: list[str]
    roman: list[str]
    phonetic: list[str]

    @classmethod
    def from_texts(cls, texts):
        raw = [clean(text) for text in texts]
        tokens = [[roman_token(word) for word in ROMAN_WORD.findall(transliterate(text))] for text in raw]
        return cls(raw, [" ".join(words) for words in tokens],
                   [" ".join(map(phonetic_token, words)) for words in tokens])

    def take(self, indices):
        return Views(*([getattr(self, name)[i] for i in indices] for name in ("raw", "roman", "phonetic")))


# Weights apply to L2-normalized view vectors before joint normalization.
FEATURES = (
    ("raw", "char", (2, 5), 0.95, 85_000),
    ("raw", "word", (1, 2), 0.55, 45_000),
    ("roman", "char", (2, 5), 1.05, 85_000),
    ("roman", "word", (1, 2), 0.75, 45_000),
    ("phonetic", "char", (2, 5), 0.80, 70_000),
    ("phonetic", "word", (1, 2), 0.55, 40_000),
)


class Features:
    def __init__(self):
        self.vectorizers = [TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                            min_df=2, max_features=limit, sublinear_tf=True,
                                            lowercase=False, dtype=np.float32)
                            for _, analyzer, ngram_range, _, limit in FEATURES]

    def encode(self, views: Views, fit=False):
        blocks = []
        for (name, _, _, weight, _), vectorizer in zip(FEATURES, self.vectorizers):
            texts = getattr(views, name)
            block = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
            block.data *= weight
            blocks.append(block)
        return normalize(sparse.hstack(blocks, format="csr"), copy=False)

    def parameter_count(self):
        return sum(len(vectorizer.vocabulary_) for vectorizer in self.vectorizers)


def fit_linear(matrix, labels, c=2.0, balance=0.33):
    counts = Counter(labels)
    avg = len(labels) / len(counts)
    weights = {label: (avg / count) ** balance for label, count in counts.items()}
    return LinearSVC(C=c, class_weight=weights, dual=True, random_state=2026, tol=0.002).fit(matrix, labels)


def cosine_batches(matrix, reference, batch_size=96):
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        similarity = (matrix[start:stop] @ reference).toarray()
        yield start, stop, np.clip(similarity, 0, 1, out=similarity)


def reference_scores(matrix, reference, reference_labels):
    """Return local coverage features in bounded memory."""
    votes = np.zeros((matrix.shape[0], len(LABELS)), dtype=np.float32)
    best = np.zeros_like(votes)
    neighbors = min(96, reference.shape[1])
    for start, stop, similarity in cosine_batches(matrix, reference):
        indices = np.argpartition(similarity, -neighbors, axis=1)[:, -neighbors:]
        scores = np.take_along_axis(similarity, indices, axis=1)
        classes = reference_labels[indices]
        rows = np.broadcast_to(np.arange(start, stop)[:, None], classes.shape)
        np.add.at(votes, (rows, classes), scores ** 5)
        np.maximum.at(best, (rows, classes), scores)
    return votes, best


def fallback_signals(linear, votes, best, hidden=None):
    """Measure confidence and lexical coverage for known intents and class 55."""
    known = linear[:, :FALLBACK]
    known_votes = votes[:, :FALLBACK]
    known_similarity = best[:, :FALLBACK]
    rows = np.arange(len(known))
    if hidden is not None:
        # Mask the correct known intent to probe what happens if it is unsupported.
        known = known.copy()
        known_votes = known_votes.copy()
        known_similarity = known_similarity.copy()
        known[rows, hidden] = -np.inf
        known_votes[rows, hidden] = 0
        known_similarity[rows, hidden] = 0

    top_two = np.partition(known, -2, axis=1)[:, -2:]
    winner = known.argmax(axis=1)
    margin = top_two[:, 1] - top_two[:, 0]
    top_score = top_two[:, 1]
    fallback_score = linear[:, FALLBACK]
    known_support = known_votes.sum(axis=1)
    total_support = known_support + votes[:, FALLBACK]
    return np.column_stack((
        fallback_score - top_score, top_score, margin, fallback_score,
        known_similarity.max(axis=1), known_similarity[rows, winner], best[:, FALLBACK],
        votes[:, FALLBACK] / np.maximum(total_support, 1e-8),
        known_votes[rows, winner] / np.maximum(known_support, 1e-8), np.log1p(known_support),
    ))


@dataclass
class FallbackGate:
    threshold: float
    estimator: object

    def decision_function(self, signals):
        return self.estimator.decision_function(signals) if self.estimator is not None else signals[:, 0]

    def parameter_count(self):
        if self.estimator is None:
            return 1
        scaler, logistic = self.estimator.steps[0][1], self.estimator.steps[1][1]
        return 1 + scaler.mean_.size + scaler.scale_.size + logistic.coef_.size + logistic.intercept_.size


class IntentModel:
    def __init__(self, c=0.8, balance=0.40):
        self.c = c
        self.balance = balance
        self.features = Features()
        self.gate = FallbackGate(0.0, None)

    def fit(self, views, labels, build_reference=True):
        matrix = self.features.encode(views, fit=True)
        self.linear = fit_linear(matrix, labels, self.c, self.balance)
        if not np.array_equal(self.linear.classes_, LABELS):
            raise ValueError("Fitting requires every label from 0 to 55")
        # SciPy's sparse-by-dense multiply reads the transposed weights in C order.
        self.linear.coef_ = np.asfortranarray(self.linear.coef_, dtype=np.float32)
        self.linear.intercept_ = self.linear.intercept_.astype(np.float32)
        self.reference = matrix.T.tocsr() if build_reference else None
        self.reference_labels = np.asarray(labels, dtype=np.int32) if build_reference else None
        if self.parameter_count() > PARAMETER_CAP:
            raise ValueError("Intent model exceeds the competition parameter cap")
        return self

    def score(self, views):
        matrix = self.features.encode(views)
        linear = self.linear.decision_function(matrix)
        votes, best = reference_scores(matrix, self.reference, self.reference_labels)
        return linear, votes, best

    def predict(self, texts):
        matrix = self.features.encode(Views.from_texts(texts))
        linear = self.linear.decision_function(matrix)
        predictions = linear[:, :FALLBACK].argmax(axis=1)
        if self.gate.estimator is None:
            fallback_scores = linear[:, FALLBACK] - linear[np.arange(len(linear)), predictions]
        else:
            votes, best = reference_scores(matrix, self.reference, self.reference_labels)
            fallback_scores = self.gate.decision_function(fallback_signals(linear, votes, best))
        return np.where(fallback_scores >= self.gate.threshold, FALLBACK, predictions)

    def parameter_count(self):
        reference_values = 0 if self.reference is None else self.reference.nnz + self.reference_labels.size
        return self.learned_parameter_count() + reference_values

    def learned_parameter_count(self):
        return (self.features.parameter_count() + self.linear.coef_.size + self.linear.intercept_.size
                + self.gate.parameter_count())
