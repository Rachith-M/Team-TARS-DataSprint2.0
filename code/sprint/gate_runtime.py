import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

from intent_model import FALLBACK
from sprint.augment import stage2_normalize
from sprint.runtime import Runtime


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def rejection(risk, predicted, threshold):
    return (predicted == FALLBACK) | (risk >= (np.inf if threshold is None else threshold))


def actions(risk, predicted, settings, band):
    reject = rejection(risk, predicted, settings['threshold'])
    if not band:
        return np.where(reject, 'reject', 'accept')
    accept_to = settings['accept_below']
    accept = risk < (np.inf if accept_to is None else accept_to)
    return np.where(reject, 'reject', np.where(accept, 'accept', 'clarify'))


class Stage2Runtime:
    def __init__(self, model, policy, bank=None, threads=2):
        self.policy = policy
        self.method = policy['chosen']['candidate']
        self.bank = normalize_vectors(bank) if self.method == 'knn' else None
        self.base = Runtime(model, threads, expose_embedding=self.bank is not None)
        self.settings = self.base.settings

    @classmethod
    def load(cls, model, directory, threads=2):
        directory = Path(directory)
        policy = json.loads((directory / 'operating_point.json').read_text(encoding='utf-8'))
        if not policy['deployment_ready']:
            raise ValueError('The evaluated gate did not meet its deployment criteria')
        for name, expected in policy['model_assets'].items():
            with (Path(model) / name).open('rb') as file:
                if hashlib.file_digest(file, 'sha256').hexdigest() != expected:
                    raise ValueError('A deployed model asset differs from the evaluated gate model')
        bank = None
        if policy['chosen']['candidate'] == 'knn':
            path = directory / 'index.float16.npy'
            with path.open('rb') as file:
                if hashlib.file_digest(file, 'sha256').hexdigest() != policy['index_sha256']:
                    raise ValueError('The deployed neighbor bank differs from the evaluated gate bank')
            bank = np.load(path)
        return cls(model, policy, bank, threads)

    def predict(self, text, live=True):
        if not live:
            return self.base.predict(text, live=False)
        if self.policy.get('normalize_input'):
            text = stage2_normalize(text)
        logits, embedding = (self.base.score([text], with_embedding=True) if self.bank is not None
                             else (self.base.score([text]), None))
        scaled = logits[0].astype(np.float64) * self.settings['scale']
        probabilities = softmax(scaled)
        top = int(scaled.argmax())
        settings = self.policy['chosen']
        if self.method == 'knn':
            similarities = np.einsum('ij,j->i', self.bank, normalize_vectors(embedding[0]), optimize=False)
            k = settings['k']
            best = np.sort(np.partition(similarities, -k)[-k:])[::-1]
            risk = float((1 - np.clip(best, -1, 1)).mean())
        elif self.method == 'energy':
            risk = -logsumexp(scaled)
        else:
            risk = 1 - probabilities[top]
        action = str(actions(np.array([risk]), np.array([top]), settings, self.policy['mode'] == 'band')[0])
        if not text.strip():
            action = 'reject'
        intent = top if action == 'accept' else FALLBACK
        return {'intent_id': intent, 'confidence': float(probabilities[intent]), 'fallback': intent == FALLBACK,
                'top_model_intent': top, 'action': action}
