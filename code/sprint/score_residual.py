import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from intent_model import script
from sprint.data import SEED
from sprint.score_bhasha import embed
from sprint.score_push import fit_candidate, metrics, options, prepare, samples, score
from sprint.train import EVAL_VIEWS, Router, take


class ScriptRouter(Router):
    def matrix(self, texts, embedding, fit=False):
        shared = super().matrix(texts, embedding, fit)
        native = np.array([{'devanagari': 1., 'mixed': .5, 'roman': 0.}[script(text)] for text in texts],
                          dtype=np.float32)[:, None]
        specific = .72 * self.config['embedding_weight'] * embedding
        return sparse.hstack((shared, specific * native, specific * (1. - native)), format='csr')


def train_head(torch, vectors, baseline, target, weights, seed, epochs):
    torch.manual_seed(seed)
    mean = torch.as_tensor(vectors.mean(axis=0))
    std = torch.as_tensor(vectors.std(axis=0)).clamp_min(.003)
    dense = (torch.as_tensor(vectors) - mean) / std
    prior = torch.as_tensor(5 * baseline)
    target = torch.as_tensor(target, dtype=torch.long)
    counts = torch.bincount(target).float()
    label_weight = (counts.mean() / counts[target]).pow(.35) * torch.as_tensor(weights)
    head = torch.nn.Sequential(
        torch.nn.Linear(dense.shape[1] + prior.shape[1], 320),
        torch.nn.GELU(),
        torch.nn.Dropout(.15),
        torch.nn.Linear(320, 160),
        torch.nn.GELU(),
        torch.nn.Dropout(.15),
        torch.nn.Linear(160, prior.shape[1]),
    )
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    for _ in range(epochs):
        for indices in torch.randperm(len(target)).split(384):
            features = dense[indices]
            known = prior[indices] * torch.empty((len(indices), 1)).uniform_(.58, 1.)
            residual = head(torch.cat((features, known), dim=1))
            scores = known + residual
            error = torch.nn.functional.cross_entropy(scores, target[indices], reduction='none')
            loss = (error * label_weight[indices]).mean() + .012 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    head.eval()
    return head, mean, std


def infer(torch, trained, vectors, base):
    head, mean, std = trained
    with torch.inference_mode():
        vectors = (torch.as_tensor(vectors) - mean) / std
        prediction = torch.as_tensor(5 * base)
        residual = head(torch.cat((vectors, prediction), dim=1)).numpy()
    return base + .50 * residual / 5


def run(args):
    import torch

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    groups, labels, original, challenge, _, scripts, config, _, variants, _ = prepare(args)
    views = {name: groups[name] for name in (*EVAL_VIEWS, 'aug')}
    views.update({name: rows for name, rows, *_ in variants})
    vectors, _ = embed(views, args.checkpoint, args.device, 128)
    if args.device == 'xpu':
        torch.xpu.empty_cache()
    frozen = json.loads((args.reference.parent / 'score_push/bhasha.json').read_text(encoding='utf-8'))
    result = {}
    plan = [('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)]
    if args.all_folds:
        plan.extend((f'fold{i + 1}', split, SEED + i) for i, split in enumerate(original) if i)
    for name, (fit, check), seed in plan:
        pieces = samples(config, groups, labels, fit, check, variants, seed)
        texts = [text for _, source, ids, _, _ in pieces for text in take(source, ids)]
        train_vectors = np.concatenate([vectors[view][ids] for view, _, ids, _, _ in pieces])
        target = np.concatenate([target for _, _, _, target, _ in pieces])
        weights = np.concatenate([np.full(len(ids), weight, dtype=np.float32) for _, _, ids, _, weight in pieces])
        with threadpool_limits(limits=args.threads):
            router, counts = fit_candidate(config, groups, labels, vectors, variants, fit, check, seed)
            expectation = score(router, groups, vectors, check)
            if args.augmentation_only:
                candidate = expectation
            elif args.domain_blocks:
                adapted = ScriptRouter(config).fit(texts, target, train_vectors, weights)
                candidate = score(adapted, groups, vectors, check)
            else:
                base_fit = router.score(texts, train_vectors)
                head = train_head(torch, train_vectors, base_fit, target, weights, seed, args.epochs)
                candidate = {view: infer(torch, head, vectors[view][check], expectation[view]) for view in EVAL_VIEWS}
        old = (frozen[name]['candidate'] if args.augmentation_only
               else metrics(expectation, labels[check], scripts[check]))
        new = metrics(candidate, labels[check], scripts[check])
        delta = {metric: new[metric] - old[metric] for metric in new}
        result[name] = {'baseline': old, 'candidate': new, 'delta': delta, 'training': counts}
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: perturbed {delta['perturbed']:+.5f}, Roman ASR {delta['roman_asr']:+.5f}", flush=True)
        if name == 'template_holdout' and any(r['delta']['perturbed'] < .005 for r in result.values()):
            break


def main():
    parser = argparse.ArgumentParser()
    options(parser)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
    parser.add_argument('--all-folds', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--domain-blocks', action='store_true')
    modes.add_argument('--augmentation-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
