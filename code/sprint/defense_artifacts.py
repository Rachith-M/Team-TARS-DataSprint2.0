"""Identity and integrity checks for the reproducible Bhasha defense release."""

import hashlib
import json
from pathlib import Path


VERSION = '1.1.0'
# v1.1 serves the fine-tuned model behind the scored Kaggle entry (submission.csv in the repo).
SUBMITTED_SHA256 = '92fc3b6f737e462b5aee3bbe55e42f81e2ec3c8161e2025ee2100059cd46d85b'
ASSETS = ('model.int8.onnx', 'model.weights', 'features.joblib', 'tokenizer.json', 'runtime.json', 'dictionary.json')
POLICY = Path('sprint_results/bhasha_defense/policy.json')
MAX_TWO_VIEW_TOKENS = 40


def verify_release(root):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != VERSION or manifest['submission_sha256'] != SUBMITTED_SHA256:
        raise ValueError('Release version or submission fingerprint differs')
    for name, fact in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != fact['bytes']:
            raise ValueError(f'Release file missing or changed: {name}')
        with path.open('rb') as file:
            if hashlib.file_digest(file, 'sha256').hexdigest() != fact['sha256']:
                raise ValueError(f'Release fingerprint differs: {name}')
    return len(manifest['files'])
