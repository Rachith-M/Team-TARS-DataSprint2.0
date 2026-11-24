"""Compare plain and hard-negative token heads alongside the frozen Bhasha classifier."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from intent_model import LABELS, Views
from sprint.bhasha_audit import (BASE_VIEWS, REQUEST_VIEWS, cached_vectors, dictionary_inputs,
                                 inner_partition, metric_delta, route_metrics, score_parts)
from sprint.bhasha_bias import int8_head
from sprint.bhasha_token_cache import TokenBank, parser, pilot_inputs
from sprint.data import SEED, macro
from sprint.postprocess import logit_scale
from sprint.romansetu_distill import perturbation_interval
from sprint.score_push import fit_candidate, prepare
from sprint.train import take


MIXES = (.10, .20, .35)
HEADS = 4
WEIGHT = .35


class AttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(HEADS, hidden_size) * .05)
        self.classifier = nn.Sequential(nn.LayerNorm(HEADS * hidden_size), nn.Dropout(.10),
                                        nn.Linear(HEADS * hidden_size, len(LABELS)))

    def forward(self, tokens, mask):
        similarity = (tokens @ self.queries.T).transpose(1, 2)
        attention = similarity.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
        context = (attention @ tokens).flatten(1)
        raw, alternate = self.classifier(context).reshape(-1, 2, len(LABELS)).unbind(1)
        return (1 - WEIGHT) * raw + WEIGHT * alternate


def token_scores(model, bank, pairs, batch_size):
    model.eval()
    with torch.inference_mode():
        scores = []
        for start in range(0, len(pairs), batch_size):
            states, mask = bank.batch(pairs[start:start + batch_size])
            scores.append(model(torch.from_numpy(states), torch.from_numpy(mask)).numpy())
    return np.concatenate(scores)


def mine(args, labels, keys, scripts, train, groups, vectors, variants, config, seed):
    folds = StratifiedGroupKFold(n_splits=args.mine_folds, shuffle=True, random_state=seed + 603)
    strata = [f'{labels[i]}_{scripts[i] == "roman"}' for i in train]
    oof = np.full((len(train), len(LABELS)), np.nan, dtype=np.float32)
    for i, (fit, check) in enumerate(folds.split(train, strata, keys[train])):
        fitting, heldout = train[fit], train[check]
        print(f'Mining fold {i + 1}/{args.mine_folds} with {len(fitting):,} fit rows', flush=True)
        with threadpool_limits(limits=args.threads):
            model, _ = fit_candidate(config, groups, labels, vectors, variants, fitting, heldout, seed + i)
            oof[check] = int8_head(model).score(take(groups['raw'], heldout), vectors['raw'][heldout])
    if not np.isfinite(oof).all():
        raise ValueError('Missing inner mining predictions')
    gold = labels[train]
    competitors = oof.copy()
    competitors[np.arange(len(train)), gold] = -np.inf
    best_rival = competitors.argmax(1)
    gaps = oof[np.arange(len(train)), gold] - competitors.max(1)
    graph = np.zeros((len(LABELS), len(LABELS)), dtype=np.float32)
    np.add.at(graph, (gold, best_rival), np.where(gaps <= 0, 1., np.where(gaps <= .20, .5, 0.)))
    graph /= np.bincount(gold, minlength=len(LABELS)).clip(min=1)[:, None]
    graph = (graph + graph.T) / 2
    neighbors = np.argsort(-graph, axis=1, kind='stable')[:, :5]
    local_classes = np.eye(len(LABELS), dtype=bool)
    for c, rows in enumerate(neighbors):
        local_classes[c, rows] = graph[c, rows] >= .01
        local_classes[c, c] = True
    scaled = torch.from_numpy(oof * logit_scale(oof, gold)).softmax(-1).numpy()
    weights = 1 + .5 * (1 - scaled[np.arange(len(train)), gold])
    return torch.from_numpy(local_classes), torch.from_numpy(weights), int(local_classes.sum() - len(LABELS))


def cached_second_views(args, groups, labels, folds, challenge, frames, scripts):
    phonetic = np.asarray(Views.from_texts(groups['raw']).phonetic)
    plan = [(str(i + 1), fit, check, SEED + i, phonetic) for i, (fit, check) in enumerate(folds)]
    plan.append(('template', *challenge, SEED + 9, frames))
    converted, current = {}, {}
    for name, fit, check, seed, keys in plan:
        learn, dev = inner_partition(fit, labels, scripts, keys, seed)
        prefix = f'fold{name}' if name != 'template' else 'template_holdout'
        for role, fitting, heldout in (('inner', learn, dev), ('outer', fit, check)):
            _, changed, texts = dictionary_inputs(groups, fitting, heldout)
            converted.update({f'{prefix}_{role}_{view}': rows for view, rows in texts.items()})
            if name == args.split:
                current[role] = prefix, changed, texts
    vectors, _ = cached_vectors(args, converted, 'dictionary')
    return {role: (changed, texts, {v: vectors[f'{prefix}_{role}_{v}'] for v in REQUEST_VIEWS})
            for role, (prefix, changed, texts) in current.items()}


def fit_head(args, bank, pairs, dev_inputs, labels, dev_labels, dev_base,
             offsets, confidence, balance, seed, hard):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = AttentionHead(bank.states.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.01)
    y = torch.as_tensor(labels, dtype=torch.long)
    weights = torch.as_tensor((len(labels) / (len(LABELS) * np.bincount(labels, minlength=len(LABELS)))) ** balance,
                              dtype=torch.float32)
    base_scale = logit_scale(dev_base['overall'], dev_labels)
    best = None
    for epoch in range(args.epochs):
        model.train()
        ordering = rng.permutation(len(y))
        for start in range(0, len(y), args.train_batch_size):
            rows = ordering[start:start + args.train_batch_size]
            states, mask = bank.batch(pairs[rows])
            gold = y[rows]
            scores = model(torch.from_numpy(states), torch.from_numpy(mask))
            per_row = F.cross_entropy(scores, gold, reduction='none')
            if hard:
                local = scores.masked_fill(~offsets[gold], -torch.inf)
                local = local - .3 * F.one_hot(gold, len(LABELS))
                per_row = per_row + .5 * confidence[rows] * F.cross_entropy(local, gold, reduction='none')
            loss = (weights[gold] * per_row).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        raw = token_scores(model, bank, dev_inputs['overall'], args.train_batch_size)
        perturbed = token_scores(model, bank, dev_inputs['perturbed'], args.train_batch_size)
        head_scale = logit_scale(raw, dev_labels)
        trial = [tuple(macro(dev_labels, ((1 - mix) * base_scale * dev_base[view]
                                        + mix * head_scale * scores).argmax(1))
                       for view, scores in (('perturbed', perturbed), ('overall', raw))) for mix in MIXES]
        winner = max(range(len(trial)), key=trial.__getitem__)
        objective = trial[winner][0]
        print(f"{'hard' if hard else 'plain'} seed {seed} epoch {epoch + 1}: dev {objective:.5f}", flush=True)
        if best is None or objective > best['best_dev_perturbed']:
            best = {'epoch': epoch + 1, 'best_dev_perturbed': objective, 'mix_weight': MIXES[winner],
                    'head_scale': head_scale, 'base_scale': base_scale,
                    'checkpoint': {key: value.detach().clone() for key, value in model.state_dict().items()}}
        if epoch + 1 - best['epoch'] >= 3:
            break
    model.load_state_dict(best.pop('checkpoint'))
    return model, best


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(args):
    groups, labels, scripts, keys, seed, fit, check, learn, dev, views = pilot_inputs(args.data, args.split)
    args.phrases = args.counterfactuals = True
    args.rewrites = None
    _, _, folds, challenge, frames, _, config, _, variants, _ = prepare(args, include_reference_vectors=False)
    base_views = {view: groups[view] for view in (*BASE_VIEWS, 'aug')}
    base_views.update({name: texts for name, texts, *_ in variants})
    vectors, _ = cached_vectors(args, base_views, 'base')
    second = cached_second_views(args, groups, labels, folds, challenge, frames, scripts)
    baseline = {}
    for role, fitting, heldout in (('inner', learn, dev), ('outer', fit, check)):
        print(f'Refitting matched Bhasha control for {role}', flush=True)
        with threadpool_limits(limits=args.threads):
            model, _ = fit_candidate(config, groups, labels, vectors, variants, fitting, heldout, seed)
            baseline[role] = score_parts(int8_head(model), groups, vectors, heldout, *second[role])[1]
    base_measure = route_metrics(baseline['outer'], labels[check], scripts[check])
    archived = json.loads(args.control_report.read_text(encoding='utf-8'))
    reference = (archived['template_holdout'] if args.split == 'template' else archived['folds'][int(args.split) - 1])
    if abs(base_measure['perturbed'] - reference['metrics']['base']['perturbed']) > .003:
        raise ValueError('The matched Bhasha control differs from its archived fold score')
    masks, hardness, edges = mine(args, labels, keys, scripts, learn, groups, vectors, variants, config, seed)
    del vectors, second, model
    bank = TokenBank(args, views)
    inputs = {key: bank.indices(value) for key, value in views.items()}
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    report = {'status': 'running', 'split': args.split, 'training_rows': len(learn), 'checkpoint_rows': len(dev),
              'outer_rows': len(check), 'baseline': base_measure, 'mining_folds': args.mine_folds,
              'negative_edges': edges, 'token_state_bytes': bank.states.nbytes, 'seeds': []}
    for offset in range(args.seeds):
        fit_seed = SEED + 760 + offset
        arms = {}
        for name, hard in (('plain', False), ('hard', True)):
            model, settings = fit_head(args, bank, inputs['learn', 'overall'],
                                       {view: inputs['dev', view] for view in REQUEST_VIEWS},
                                       labels[learn], labels[dev], baseline['inner'], masks, hardness,
                                       config['balance'], fit_seed, hard)
            logits = {view: (1 - settings['mix_weight']) * settings['base_scale'] * baseline['outer'][view]
                      + settings['mix_weight'] * settings['head_scale'] * token_scores(model, bank, inputs['outer', view],
                                                                                      args.train_batch_size)
                      for view in REQUEST_VIEWS}
            metrics = route_metrics(logits, labels[check], scripts[check])
            arms[name] = {**settings, 'metrics': metrics, 'delta': metric_delta(metrics, base_measure),
                          'extra_head_parameters': sum(p.numel() for p in model.parameters()),
                          'paired_group_interval_95pct': perturbation_interval(labels[check],
                              baseline['outer']['perturbed'].argmax(1), logits['perturbed'].argmax(1), keys[check])}
        report['seeds'].append({'seed': fit_seed, **arms})
        write_report(args.output, report)
        print(f"Seed {fit_seed}: plain {arms['plain']['delta']['perturbed']:+.5f}, "
              f"hard {arms['hard']['delta']['perturbed']:+.5f}", flush=True)

    for name in ('plain', 'hard'):
        report[f'{name}_mean_delta'] = {metric: float(np.mean([row[name]['delta'][metric] for row in report['seeds']]))
                                        for metric in base_measure}
    report['status'] = 'complete'
    write_report(args.output, report)
    print(json.dumps({name: report[name] for name in ('baseline', 'plain_mean_delta', 'hard_mean_delta')}, indent=2))


def main():
    options = parser()
    options.add_argument('--output', type=Path, required=True)
    options.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    options.add_argument('--control-report', type=Path, default=Path('sprint_results/score_push/bhasha_full_audit_run/nested_bias_comparison.json'))
    options.add_argument('--mine-folds', type=int, default=3)
    options.add_argument('--seeds', type=int, default=3)
    options.add_argument('--epochs', type=int, default=8)
    options.add_argument('--train-batch-size', type=int, default=64)
    args = options.parse_args()
    if args.output.exists() or args.mine_folds < 2 or min(args.seeds, args.epochs, args.train_batch_size, args.threads) < 1:
        options.error('Use a new report path, at least two mining folds and positive run sizes')
    run(args)


if __name__ == '__main__':
    main()
