import argparse
import gc
import json
import math
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from intent_model import PARAMETER_CAP, clean, script
from sprint.augment import procedural_oos
from sprint.data import SEED, folds, load_data, macro, write_submission
from sprint.encoder import MAX_LEN, variant_groups
from sprint.postprocess import average_tta, cluster_logits, logit_scale
from sprint.train import EVAL_VIEWS, load_oos, metrics, take


CONFIG_FILE = Path(__file__).with_name('gpu_models.json')
LAYER = re.compile(r'\.layer\.(\d+)\.')


def optimizer_groups(model, lr, decay):
    groups = {}
    depth = model.config.num_hidden_layers
    for name, parameter in model.named_parameters():
        layer = LAYER.search(name)
        level = int(layer[1]) + 1 if layer else (0 if '.embeddings.' in name else depth + 1)
        rate = lr * decay ** (depth + 1 - level)
        weight_decay = 0 if parameter.ndim == 1 or name.endswith('.bias') else .01
        groups.setdefault((rate, weight_decay), []).append(parameter)
    return [dict(params=params, lr=rate, weight_decay=weight_decay)
            for (rate, weight_decay), params in groups.items()]


def token_batch(tokenizer, texts, prefix, device):
    return tokenizer([prefix + text for text in texts], truncation=True, padding=True, max_length=MAX_LEN,
                     return_tensors='pt').to(device)


def train_one(config, groups, labels, rows, args, seed, team_oos, check=None):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    repo = args.checkpoint or config['repo']
    source_options = {'revision': config['revision']} if not args.checkpoint else {'local_files_only': True}
    options = dict(trust_remote_code=False, **source_options)
    tokenizer = AutoTokenizer.from_pretrained(repo, **options)
    model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=56, **options)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count <= PARAMETER_CAP, count
    model.to(args.device)
    train = take(groups['raw'], rows)
    augmented = take(groups['aug'], rows)
    if check is not None:
        forbidden = {clean(groups[style][i]) for style in EVAL_VIEWS for i in check}
        augmented = [raw if clean(variant) in forbidden else variant for raw, variant in zip(train, augmented)]
    novel = procedural_oos(train, labels[rows], args.procedural_oos, seed) + team_oos
    train += novel
    augmented += novel
    targets = np.r_[labels[rows], np.full(len(novel), 55, dtype=np.int32)]
    counts = np.bincount(targets, minlength=56)
    weights = torch.as_tensor((counts.mean() / np.maximum(counts, 1)) ** args.balance,
                              dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config['lr'], args.layer_decay))
    total = args.epochs * math.ceil(len(targets) / args.batch_size)
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(total * .08), total)
    is_cuda = args.device.startswith('cuda')
    dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda and dtype == torch.float16)
    model.train()
    prefix = config.get('prefix', '')
    elapsed = perf_counter()
    steps = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(targets))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            y = torch.as_tensor(targets[indices], dtype=torch.long, device=args.device)
            inputs = token_batch(tokenizer, take(train, indices), prefix, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dtype, enabled=is_cuda):
                one = model(**inputs).logits.float()
                loss = F.cross_entropy(one, y, weight=weights, label_smoothing=args.smoothing)
                if args.rdrop:
                    two = model(**token_batch(tokenizer, take(augmented, indices), prefix, args.device)).logits.float()
                    log_one = F.log_softmax(one, dim=-1)
                    log_two = F.log_softmax(two, dim=-1)
                    kl = (F.kl_div(log_one, log_two, reduction='batchmean', log_target=True) +
                          F.kl_div(log_two, log_one, reduction='batchmean', log_target=True)) / 2
                    loss = (loss + F.cross_entropy(two, y, weight=weights, label_smoothing=args.smoothing)) / 2 + args.rdrop * kl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            steps += 1
            if args.profile_steps and steps >= args.profile_steps:
                if is_cuda:
                    torch.cuda.synchronize()
                seconds = perf_counter() - elapsed
                print(f'{count:,} params, {seconds / steps:.3f}s/step. '
                      f'One {args.epochs}-epoch fold estimate {seconds / steps * total / 60:.1f} min.', flush=True)
                return model, tokenizer, count
        print(f'Epoch {epoch + 1}/{args.epochs}, elapsed {perf_counter() - elapsed:.0f}s', flush=True)
    return model, tokenizer, count


def score(model, tokenizer, texts, prefix, args):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.eval_batch):
            inputs = token_batch(tokenizer, texts[start:start + args.eval_batch], prefix, args.device)
            result.append(model(**inputs).logits.float().cpu().numpy())
    return np.concatenate(result)


def run(config, args):
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use a CUDA notebook or --device cpu for a profile run.')
    texts, labels, test, ids = load_data(args.data)
    groups = variant_groups(texts, test)
    parts = folds(texts, labels)
    oos = load_oos(args.team_oos) if args.team_oos else []
    if args.profile_steps:
        train_one(config, groups, labels, parts[0][0], args, SEED, oos)
        return
    out = args.output / args.model
    out.mkdir(parents=True, exist_ok=True)
    oof = {name: np.zeros((len(labels), 56), dtype=np.float32) for name in EVAL_VIEWS}
    fold_id = np.empty(len(labels), dtype=np.int8)
    prefix = config.get('prefix', '')
    for fold, (train, check) in enumerate(parts):
        model, tokenizer, count = train_one(config, groups, labels, train, args, SEED + fold, oos, check=check)
        for name in EVAL_VIEWS:
            oof[name][check] = score(model, tokenizer, take(groups[name], check), prefix, args)
        fold_id[check] = fold
        print(f'Fold {fold + 1}/5 macro-F1 {macro(labels[check], oof["raw"][check].argmax(1)):.4f}', flush=True)
        del model, tokenizer
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    scripts = np.array(list(map(script, texts)))
    report, alternatives = metrics(oof, groups, labels, scripts, fold_id)
    model, tokenizer, count = train_one(config, groups, labels, np.arange(len(labels)), args, SEED + 5, oos)
    test_raw, test_tta = [score(model, tokenizer, groups[name], prefix, args) for name in ('test', 'test_tta')]
    weight = report['tta_weight']
    report.update({'config': config, 'parameters': count, 'alternatives': alternatives,
                   'scale': logit_scale(average_tta(oof['raw'], oof['tta'], weight), labels)})
    logits = average_tta(test_raw, test_tta, weight)
    if report['cluster_weight']:
        logits = cluster_logits(test, logits, report['cluster_weight'])
    write_submission(out / 'submission.csv', ids, logits.argmax(axis=1))
    np.savez_compressed(out / 'scores.npz', **oof, test=test_raw, test_tta=test_tta,
                        target=labels, fold_id=fold_id, example_id=np.array(ids))
    model.save_pretrained(out / 'checkpoint')
    tokenizer.save_pretrained(out / 'checkpoint')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main():
    configs = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=configs)
    parser.add_argument('--checkpoint')
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('sprint_results'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--layer-decay', type=float, default=.90)
    parser.add_argument('--balance', type=float, default=.40)
    parser.add_argument('--smoothing', type=float, default=.06)
    parser.add_argument('--rdrop', type=float, default=.25)
    parser.add_argument('--team-oos', type=Path)
    parser.add_argument('--procedural-oos', type=int, default=0)
    parser.add_argument('--profile-steps', type=int, default=0)
    args = parser.parse_args()
    run(configs[args.model], args)


if __name__ == '__main__':
    main()
