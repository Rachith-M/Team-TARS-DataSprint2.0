"""Identity and integrity checks for the reproducible Bhasha defense release."""

import hashlib
import json
from pathlib import Path


VERSION = '1.0.0'
SUBMITTED_SHA256 = 'b57e79a709482c1db53e818570a1942f88a5e5c8288dbfdb003d5a3eb9d2fd21'
ASSETS = ('model.int8.onnx', 'features.joblib', 'tokenizer.json', 'runtime.json', 'dictionary.json')
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
