import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK
from sprint.data import SEED
from sprint.depth_partial import encoded
from sprint.muril import accelerator, check_budget, json_file
from sprint.score_bhasha import embed
from sprint.score_neural import PREFIXES, paired_samples, prefix_cache, score
from sprint.score_push import fit_candidate, metrics, options, prepare, samples, score as frozen_score
from sprint.train import EVAL_VIEWS, take


def train(torch, args, prepared, vectors, fit, check, seed, deadline):
    from transformers import AutoModel, AutoTokenizer

    class Adapter(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            options = {'device': base.weight.device, 'bias': False}
            self.down = torch.nn.Linear(base.in_features, 12, **options)
            self.up = torch.nn.Linear(12, base.out_features, **options)
            torch.nn.init.zeros_(self.up.weight)

        def forward(self, hidden):
            return self.base(hidden) + 2 * self.up(self.down(hidden))

    groups, labels, _, _, _, _, config, _, variants, _ = prepared
    router, _ = fit_candidate(config, groups, labels, vectors, variants, fit, check, seed)
    initial = frozen_score(router, groups, vectors, check)
    texts, targets, counts = paired_samples(config, groups, labels, fit, check, variants, seed)
    for name, values, ids, target, _ in samples(config, groups, labels, fit, check, variants, seed):
        if name == 'contrast':
            texts.extend(word for word in take(values, ids) for _ in range(2))
            targets = np.r_[targets, target]
            counts['counterfactual'] = len(ids)
    width = router.features.parameter_count()
    lexical = np.ascontiguousarray(router.head.coef_[:, :width] * 5)
    static_scores = router.lexical(texts) @ lexical.T
    torch.manual_seed(seed)
    getattr(torch, args.device).manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(args.checkpoint, local_files_only=True, add_pooling_layer=False,
                                      attn_implementation='eager').requires_grad_(False).eval().to(args.device)
    split = len(model.encoder.layer) - args.unfrozen
    for layer in model.encoder.layer[split:]:
        attention = layer.attention.self
        attention.query = Adapter(attention.query)
        attention.value = Adapter(attention.value)
        layer.output.dense = Adapter(layer.output.dense)
    head = torch.nn.Linear(model.config.hidden_size, 56).to(args.device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(np.ascontiguousarray(router.head.coef_[:, width:] * 5)))
        head.bias.copy_(torch.from_numpy(router.head.intercept_ * 5))
    frequency = np.bincount(targets, minlength=56)
    batches = prefix_cache(torch, model, tokenizer, texts, split, args, deadline, paired=True)
    device = 'cpu' if args.optimizer_cpu else args.device
    if args.optimizer_cpu:
        model.to(device)
        head.to(device)
        getattr(torch, args.device).empty_cache()
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.encoder_lr},
        {'params': head.parameters(), 'lr': .0003},
    ], foreach=False, fused=False)
    weights = torch.as_tensor((frequency.mean() / frequency) ** .32, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    functional = torch.nn.functional
    for epoch in range(args.epochs):
        for step, index in enumerate(rng.permutation(len(batches)), 1):
            check_budget(deadline)
            batch = batches[index]
            ids = batch[0]
            target = torch.as_tensor(targets[ids // 2].astype(np.int64), device=device)
            lex = torch.from_numpy(static_scores[ids]).to(device)
            sample = torch.rand((len(ids) // 2, 1), device=device).repeat_interleave(2, dim=0)
            lex = lex * torch.where(sample < .25, .65, 1.)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device != 'cpu'):
                embedding = encoded(torch, model, batch, split, device)
                logits = lex + head(config['embedding_weight'] * embedding).float()
            similarity = embedding @ embedding.T / .10
            nonself = ~torch.eye(len(ids), dtype=torch.bool, device=device)
            positives = target[:, None].eq(target[None, :]) & nonself
            logits_normalizer = similarity.masked_fill(~nonself, -torch.inf).logsumexp(dim=1)
            attraction = (similarity * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
            contrastive = ((logits_normalizer - attraction) * (target != FALLBACK)).mean()
            classification = functional.cross_entropy(logits, target, weight=weights, label_smoothing=.02)
            loss = classification + .06 * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            if step % 150 == 0:
                print(f'Epoch {epoch + 1}: {step}/{len(batches)} batches', flush=True)
    router.head.coef_ = np.asfortranarray(np.concatenate((lexical, head.weight.detach().cpu().numpy()), axis=1))
    router.head.intercept_ = head.bias.detach().cpu().numpy().copy()
    if args.optimizer_cpu:
        model.to(args.device)
    return (model, tokenizer, router, split, counts), initial


def run(args):
    import torch

    args.device = accelerator(torch, args.device)
    args.encoder = 'bhasha'
    PREFIXES[args.encoder] = ''
    torch.set_num_threads(args.threads)
    deadline = perf_counter() + args.hours * 3600
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = prepare(args)
    groups, labels, original, challenge, _, scripts, _, _, variants, _ = prepared
    views = {view: groups[view] for view in (*EVAL_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    vectors, _ = embed(views, args.checkpoint, args.device, 128)
    getattr(torch, args.device).empty_cache()
    published = json.loads((args.reference.parent / 'score_push/contrast/results.json').read_text(encoding='utf-8'))
    plan = [('fold1', original[0], SEED, published['folds'][0]['candidate']),
            ('template_holdout', challenge, SEED + 9, published['challenge']['candidate'])]
    if args.all_folds:
        plan.extend((f'fold{i + 1}', split, SEED + i, published['folds'][i]['candidate'])
                    for i, split in enumerate(original) if i)
    report = {'seed': SEED, 'epochs': args.epochs, 'unfrozen_layers': args.unfrozen, 'results': {}}
    for name, (fit, check), seed, prior in plan:
        check_budget(deadline)
        with threadpool_limits(limits=args.threads):
            trained, baseline = train(torch, args, prepared, vectors, fit, check, seed, deadline)
            current = score(torch, args, prepared, trained, check, deadline)
        start = metrics(baseline, labels[check], scripts[check])
        end = metrics(current, labels[check], scripts[check])
        differences = {key: end[key] - start[key] for key in start}
        entry = {'frozen': start, 'adapted': end, 'delta': differences,
                 'delta_vs_contrast': {key: end[key] - prior[key] for key in end}, 'training': trained[-1]}
        report['results'][name] = entry
        json_file(args.output / 'results.json', report)
        print(f"{name}: perturbed {differences['perturbed']:+.5f} vs frozen and "
              f"{entry['delta_vs_contrast']['perturbed']:+.5f} vs submitted", flush=True)
        del trained
        getattr(torch, args.device).empty_cache()
        if name == 'template_holdout':
            gains = [r['delta']['perturbed'] for r in report['results'].values()]
            if min(gains) < -.005 or np.mean(gains) < .006:
                break


def main():
    parser = argparse.ArgumentParser()
    options(parser)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=('auto', 'xpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--unfrozen', type=int, default=4)
    parser.add_argument('--optimizer-cpu', action='store_true')
    parser.add_argument('--encoder-lr', type=float, default=.00012)
    parser.add_argument('--all-folds', action='store_true')
    parser.add_argument('--hours', type=float, default=2.)
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2 or not 1 <= args.unfrozen <= 12:
        parser.error('A paired batch needs an even size and between one and twelve unfrozen layers')
    run(args)


if __name__ == '__main__':
    main()
