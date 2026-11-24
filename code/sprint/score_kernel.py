import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from intent_model import LABELS
from sprint.data import SEED
from sprint.score_push import compare, prepare, samples
from sprint.score_retrieval import rare_roman
from sprint.train import EVAL_VIEWS, Router, score_view, take


class KernelHead:
    def fit(self, matrix, labels, weights, seed):
        rng = np.random.default_rng(seed)
        slots = [np.flatnonzero(labels == label) for label in LABELS]
        anchors = np.concatenate([rng.choice(rows, min(len(rows), 18), replace=False) for rows in slots])
        self.anchors = matrix[anchors]
        eigenvalues, basis = eigh(rbf_kernel(self.anchors, gamma=.75))
        self.transform = (basis / np.sqrt(np.maximum(eigenvalues, 1e-5))) @ basis.T
        targets = (labels[:, None] == LABELS).astype(np.float32)
        self.ridge = Ridge(alpha=.4, solver='cholesky').fit(
            self.project(matrix), targets, sample_weight=weights)
        self.coef = self.transform @ self.ridge.coef_.T
        return self

    def project(self, matrix):
        return rbf_kernel(matrix, self.anchors, gamma=.75) @ self.transform

    def score(self, matrix):
        return rbf_kernel(matrix, self.anchors, gamma=.75) @ self.coef + self.ridge.intercept_


def run(args):
    groups, labels, original, challenge, _, scripts, config, vectors, _, saved = prepare(args)
    results = {}
    for name, (fit, check), seed in (
            ('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)):
        parts = samples(config, groups, labels, fit, check, [], seed)
        texts = [item for _, text, ids, _, _ in parts for item in take(text, ids)]
        target = np.concatenate([target for _, _, _, target, _ in parts])
        weights = np.concatenate([np.full(len(ids), weight) for _, _, ids, _, weight in parts])
        embeddings = np.concatenate([vectors[view][ids] for view, _, ids, _, _ in parts])
        router = Router(config)
        with threadpool_limits(limits=args.threads):
            matrix = router.matrix(texts, embeddings, fit=True)
            counts = np.bincount(target)
            kernel = KernelHead().fit(matrix, target, weights * (counts.mean() / counts[target]) ** .38, seed)
            if name == 'fold1':
                baseline = {view: saved[view][check] for view in EVAL_VIEWS}
            else:
                router.fit_matrix(matrix, target, weights)
                baseline = {view: score_view(router, groups, vectors, view, check) for view in EVAL_VIEWS}
            alternative = {view: kernel.score(router.matrix(take(groups[view], check), vectors[view][check]))
                           for view in EVAL_VIEWS}
        candidate = {view: baseline[view] + alternative[view] for view in EVAL_VIEWS}
        entry = compare(candidate, baseline, check, labels, scripts, len(fit))
        subset = rare_roman(groups, scripts, fit, check)
        entry['low_coverage_roman'] = compare({v: candidate[v][subset] for v in EVAL_VIEWS},
                                              {v: baseline[v][subset] for v in EVAL_VIEWS},
                                              check[subset], labels, scripts, len(fit))
        entry['kernel_only'] = compare(alternative, baseline, check, labels, scripts, len(fit))
        results[name] = entry
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: perturbed {entry['delta']['perturbed']:+.5f}, Roman ASR "
              f"{entry['delta']['roman_asr']:+.5f}, low-coverage Roman "
              f"{entry['low_coverage_roman']['delta']['overall']:+.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/score_push/kernel.json'))
    parser.add_argument('--threads', type=int, default=4)
    parser.set_defaults(rewrites=None, phrases=False, counterfactuals=False)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
