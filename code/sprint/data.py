import csv
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from intent_model import LABELS, Views, script


SEED = 2046
SCRIPTS = ('roman', 'devanagari', 'mixed')


def read_csv(path, expected_columns):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(expected_columns):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    ids = [row["example_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError(f"Expected nonempty unique example IDs in {path}")
    return rows


def load_training(path):
    rows = read_csv(path, ("example_id", "query_text", "target"))
    texts = [row["query_text"] for row in rows]
    labels = np.array([int(row["target"]) for row in rows], dtype=np.int32)
    if not np.array_equal(np.unique(labels), LABELS) or any(not text.strip() for text in texts):
        raise ValueError("Expected nonempty queries and labels 0 through 55 in training data")
    return texts, labels


def load_data(directory):
    directory = Path(directory)
    texts, labels = load_training(directory / 'train.csv')
    test = read_csv(directory / 'test.csv', ('example_id', 'query_text'))
    template = read_csv(directory / 'sample_submission.csv', ('example_id', 'target'))
    ids = [row['example_id'] for row in test]
    if ids != [row['example_id'] for row in template] or ids != [str(i) for i in range(900001, 904913)]:
        raise ValueError('Expected the 4,912 test IDs in sample-template order')
    return texts, labels, [row['query_text'] for row in test], ids


def dataset_key(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def folds(texts, labels, n_splits=5):
    groups = Views.from_texts(texts).phonetic
    strata = [f'{label}_{script(text) == "roman"}' for text, label in zip(texts, labels)]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(texts, strata, groups))


def macro(labels, predictions):
    return float(f1_score(labels, predictions, labels=LABELS, average='macro', zero_division=0))


def slices(labels, predictions, scripts):
    return {'overall': macro(labels, predictions), **{
        name: macro(labels[scripts == name], predictions[scripts == name]) for name in SCRIPTS
    }}


def write_submission(path, ids, predictions):
    predictions = np.asarray(predictions)
    if len(ids) != 4912 or predictions.shape != (4912,) or predictions.dtype.kind not in 'iu':
        raise ValueError('Expected 4,912 integer predictions')
    if not np.isin(predictions, LABELS).all():
        raise ValueError('Prediction outside 0–55')
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('example_id', 'target'))
        writer.writerows(zip(ids, predictions.tolist()))
    written = read_csv(temporary, ('example_id', 'target'))
    assert [row['example_id'] for row in written] == ids
    assert [int(row['target']) for row in written] == predictions.tolist()
    temporary.replace(path)


