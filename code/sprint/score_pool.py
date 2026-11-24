import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from sprint.data import SEED
from sprint.encoder import MAX_LEN
from sprint.score_push import metrics, options, prepare, samples
from sprint.train import EVAL_VIEWS, Router, take


def signal_weights(tokenizer, texts, labels):
    tokens = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=MAX_LEN)['input_ids']
    word_ids, targets = [], []
    for row, target in zip(tokens, labels):
        unique = set(row)
        word_ids.extend(unique)
        targets.extend([target] * len(unique))
    counts = np.zeros((len(tokenizer), 56), dtype=np.float32)
    np.add.at(counts, (word_ids, targets), 1)
    prior = np.bincount(labels, minlength=56).astype(np.float32)
    prior /= prior.sum()
    counts += 12 * prior
    counts /= counts.sum(axis=1, keepdims=True)
    information = (counts * np.log(counts / prior)).sum(axis=1)
    weights = 1. + 1.6 * np.sqrt(np.maximum(information, 0)).clip(max=1.75)
    weights[tokenizer.all_special_ids] = .6
    return weights


def extract(torch, model, tokenizer, views, information, args):
    universe = sorted(dict.fromkeys(text for sample in views.values() for text in sample), key=len)
    table = {}
    info = torch.as_tensor(information, device=args.device)
    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
        for offset in range(0, len(universe), args.batch_size):
            chunk = universe[offset:offset + args.batch_size]
            batch = tokenizer(chunk, padding=True, max_length=MAX_LEN, truncation=True,
                              return_tensors='pt').to(args.device)
            hidden = model(**batch).last_hidden_state.float()
            mask = batch['attention_mask'].unsqueeze(-1)
            word_weight = info[batch['input_ids']].unsqueeze(-1)
            normalize = torch.nn.functional.normalize
            average = normalize((hidden * mask).sum(dim=1), dim=1)
            salient = normalize((hidden * mask * word_weight).sum(dim=1), dim=1)
            table.update(zip(chunk, torch.cat((average, .7 * salient), dim=1).cpu().numpy()))
    return {name: np.stack([table[text] for text in rows]) for name, rows in views.items()}


def run(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    groups, labels, original, challenge, _, scripts, config, _, variants, _ = prepare(args)
    frozen = json.loads((args.reference.parent / 'score_push/bhasha.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(args.checkpoint, add_pooling_layer=False,
                                     local_files_only=True, attn_implementation='sdpa').eval().to(args.device)
    plan = [('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)]
    if args.all_folds:
        plan.extend((f'fold{i + 1}', split, SEED + i) for i, split in enumerate(original) if i)
    report = {}
    for name, (fit, check), seed in plan:
        parts = samples(config, groups, labels, fit, check, variants, seed)
        texts = [text for _, source, ids, _, _ in parts for text in take(source, ids)]
        target = np.concatenate([target for _, _, _, target, _ in parts])
        weights = np.concatenate([np.full(len(ids), weight) for _, _, ids, _, weight in parts])
        importance = signal_weights(tokenizer, take(groups['raw'], fit), labels[fit])
        views = {view: take(groups[view], check) for view in EVAL_VIEWS}
        vectors = extract(torch, model, tokenizer, {**views, 'fit': texts}, importance, args)
        with threadpool_limits(limits=args.threads):
            router = Router(config).fit(texts, target, vectors['fit'], weights)
            output = {view: router.score(views[view], vectors[view]) for view in EVAL_VIEWS}
        candidate = metrics(output, labels[check], scripts[check])
        baseline = frozen[name]['candidate']
        delta = {key: candidate[key] - baseline[key] for key in candidate}
        report[name] = {'candidate': candidate, 'frozen': baseline, 'delta': delta, 'training_rows': len(texts)}
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: perturbed {delta['perturbed']:+.5f}, Roman ASR {delta['roman_asr']:+.5f}", flush=True)
        if name == 'template_holdout' and any(r['delta']['perturbed'] < .004 for r in report.values()):
            break


def main():
    parser = argparse.ArgumentParser()
    options(parser)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--all-folds', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', choices=('xpu', 'cuda'), default='xpu')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
