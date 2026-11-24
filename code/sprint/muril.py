import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, PARAMETER_CAP, Features, Views, fit_linear, script
from sprint.data import SEED, dataset_key, folds, load_training, macro
from sprint.encoder import MAX_LEN, variant_groups
from sprint.postprocess import average_tta, logit_scale
from sprint.runtime import pack_lexical
from sprint.train import augmentation_rows, take
from sprint.train_gpu import CONFIG_FILE, optimizer_groups


VIEWS = ('raw', 'tta', 'natural', 'natural_tta')
LR = 3e-5
LAYER_DECAY = .9
SMOOTHING = .1
AUGMENT = .65
SEMANTIC_WEIGHT = 1.4
MIN_GAIN = .015
MIN_PERTURBED = .8536
MIN_ROMAN = .865
MIN_FALLBACK = .85
MIN_WINS = 4
BUDGET = 7200


class BudgetExpired(Exception):
    pass


def json_file(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def evaluate(logits, labels, roman, weight):
    clean = average_tta(logits['raw'], logits['tta'], weight).argmax(axis=1)
    perturbed = average_tta(logits['natural'], logits['natural_tta'], weight).argmax(axis=1)
    return {'overall': macro(labels, clean), 'roman': macro(labels[roman], clean[roman]),
            'perturbed': macro(labels, perturbed), 'class55_f1': float(f1_score(
                labels == FALLBACK, clean == FALLBACK, zero_division=0))}


def baseline(directory, labels, roman, parts):
    settings = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    if settings['cluster_weight'] != 0:
        raise ValueError('This paired protocol expects the published E5-small hybrid without cluster averaging')
    with np.load(directory / 'scores.npz') as saved:
        ids = np.empty(len(labels), dtype=np.int8)
        for number, (_, check) in enumerate(parts):
            ids[check] = number
        if not np.array_equal(saved['fold_id'], ids) or not np.array_equal(saved['target'], labels):
            raise ValueError('The supplied baseline does not match the training labels and outer folds')
        logits = {name: saved[name] for name in VIEWS}
    weight = settings['tta_weight']
    scores = [evaluate({name: z[check] for name, z in logits.items()}, labels[check], roman[check], weight)
              for _, check in parts]
    return {'tta_weight': weight, 'folds': scores, 'pooled': evaluate(logits, labels, roman, weight),
            'mean_perturbed': float(np.mean([fold['perturbed'] for fold in scores]))}


def stop_after_two(gains):
    if len(gains) < 2:
        return False
    first = gains[:2]
    mean = np.mean(first)
    return bool(mean < 0 or (mean < .005 and max(first) <= .01))


def accuracy_checks(pooled, gains):
    return {'five_folds': len(gains) == 5,
            'mean_paired_gain': bool(np.mean(gains) >= MIN_GAIN),
            'perturbed_floor': pooled['perturbed'] >= MIN_PERTURBED,
            'fold_wins': sum(gain > 0 for gain in gains) >= MIN_WINS,
            'roman': pooled['roman'] >= MIN_ROMAN, 'class55_f1': pooled['class55_f1'] >= MIN_FALLBACK}


def accelerator(torch, requested):
    available = {'cuda': torch.cuda.is_available(),
                 'xpu': hasattr(torch, 'xpu') and torch.xpu.is_available()}
    if requested == 'auto':
        requested = next((name for name in available if available[name]), None)
    if not requested or not available.get(requested):
        raise RuntimeError('This training protocol requires an available CUDA or Intel XPU device')
    return requested


def check_budget(deadline):
    if perf_counter() >= deadline:
        raise BudgetExpired


def init_model(torch, args, seed, warm):
    from transformers import AutoModel, AutoTokenizer

    class Hybrid(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.config = encoder.config
            self.lexical = torch.nn.Embedding(warm.coef_.shape[1], 56)
            self.semantic = torch.nn.Linear(self.config.hidden_size, 56)
            with torch.no_grad():
                self.lexical.weight.copy_(torch.from_numpy(3 * warm.coef_.T.copy()))
                self.semantic.bias.copy_(torch.from_numpy(3 * warm.intercept_))

        def forward(self, tokens, indices, values):
            hidden = self.encoder(**tokens).last_hidden_state.float()
            summed = (hidden * tokens['attention_mask'].unsqueeze(-1)).sum(dim=1)
            embedding = torch.nn.functional.normalize(summed, dim=1)
            lexical = (self.lexical(indices) * values.unsqueeze(-1)).sum(dim=1)
            return lexical + self.semantic(SEMANTIC_WEIGHT * embedding).float()

    torch.manual_seed(seed)
    backend = getattr(torch, args.device)
    backend.manual_seed_all(seed)
    config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))['muril']
    source = args.checkpoint or config['repo']
    options = ({'local_files_only': True} if args.checkpoint else {'revision': config['revision']})
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **options)
    encoder = AutoModel.from_pretrained(source, trust_remote_code=False, add_pooling_layer=False, **options)
    model = Hybrid(encoder)
    count = sum(parameter.numel() for parameter in model.parameters()) + warm.coef_.shape[1]
    if count > PARAMETER_CAP:
        raise ValueError(f'MuRIL hybrid exceeds parameter cap ({count:,})')
    return model.to(args.device), tokenizer, count


def training_data(groups, labels, fit, check, seed):
    chosen = augmentation_rows(groups, fit, AUGMENT, seed, check)
    texts = take(groups['raw'], fit) + take(groups['aug'], chosen)
    targets = np.r_[labels[fit], labels[chosen]]
    augment_weight = np.r_[np.ones(len(fit)), np.full(len(chosen), .7)]
    counts = np.bincount(targets, weights=augment_weight, minlength=56)
    probabilities = augment_weight / counts[targets]
    return texts, targets, probabilities / probabilities.sum()


class Batches:
    def __init__(self, texts, tokenizer, features, matrix=None):
        tokens = tokenizer(texts, max_length=MAX_LEN, truncation=True)
        self.tokenizer = tokenizer
        self.tokens = [dict(zip(tokens, values)) for values in zip(*tokens.values())]
        self.matrix = features.encode(Views.from_texts(texts)) if matrix is None else matrix
        self.lengths = np.array([len(row['input_ids']) for row in self.tokens])

    def get(self, rows, torch, device):
        tokens = self.tokenizer.pad(take(self.tokens, rows), padding=True, pad_to_multiple_of=8,
                                    return_tensors='pt').to(device)
        ids, values = pack_lexical(self.matrix[rows])
        return tokens, torch.from_numpy(ids).to(device), torch.from_numpy(values).to(device)


def score_views(torch, groups, features, model, tokenizer, rows, args, deadline, names=VIEWS):
    model.eval()
    logits = {}
    with torch.inference_mode(), torch.autocast(args.device, dtype=args.precision):
        for name in names:
            check_budget(deadline)
            batches = Batches(take(groups[name], rows), tokenizer, features)
            result = np.empty((len(rows), 56), dtype=np.float32)
            order = np.argsort(batches.lengths, kind='stable')
            for start in range(0, len(order), args.eval_batch):
                check_budget(deadline)
                batch = order[start:start + args.eval_batch]
                result[batch] = model(*batches.get(batch, torch, args.device)).cpu().numpy()
            logits[name] = result
    return logits


def fit(torch, groups, labels, fit_rows, check, args, deadline, seed, weight, inner=None, epochs=None, progress=None):
    check_budget(deadline)
    texts, targets, probabilities = training_data(groups, labels, fit_rows, check, seed)
    features = Features()
    matrix = features.encode(Views.from_texts(texts), fit=True)
    warm = fit_linear(matrix, targets, c=.8, balance=.4)
    check_budget(deadline)
    model, tokenizer, count = init_model(torch, args, seed, warm)
    del warm
    batches = Batches(texts, tokenizer, features, matrix)
    parameters = optimizer_groups(model, LR, LAYER_DECAY)
    if args.optimizer == 'adamw8':
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(parameters, lr=LR)
    elif args.optimizer == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(parameters, lr=LR, scale_parameter=False, relative_step=False)
    else:
        optimizer = torch.optim.AdamW(parameters, fused=True)
    from transformers import get_cosine_schedule_with_warmup
    epochs = epochs or args.epochs
    steps_per_epoch = math.ceil(len(targets) / args.batch_size)
    length = steps_per_epoch * epochs
    schedule = get_cosine_schedule_with_warmup(optimizer, round(length * .08), length)
    scaler = torch.amp.GradScaler(args.device, enabled=args.precision == torch.float16)
    rng = np.random.default_rng(seed)
    first_times = []
    best, best_epoch = -1., 0
    steps = 0
    with TemporaryDirectory(dir=args.output, prefix='inner-') as scratch:
        checkpoint = Path(scratch) / 'best.pt'
        for epoch in range(1, epochs + 1):
            started = perf_counter()
            model.train()
            order = rng.choice(len(targets), len(targets), replace=True, p=probabilities)
            order = order[np.argsort(batches.lengths[order], kind='stable')]
            for batch_index, bucket in enumerate(rng.permutation(steps_per_epoch), start=1):
                check_budget(deadline)
                tick = perf_counter()
                start = bucket * args.batch_size
                batch = order[start:start + args.batch_size]
                inputs = batches.get(batch, torch, args.device)
                target = torch.from_numpy(targets[batch].astype(np.int64)).to(args.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(args.device, dtype=args.precision):
                    logits = model(*inputs)
                    loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing=SMOOTHING)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
                steps += 1
                if steps <= max(32, args.profile_steps):
                    getattr(torch, args.device).synchronize()
                    first_times.append(perf_counter() - tick)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError('Nonfinite training loss in the accelerator profile')
                if args.profile_steps and steps == args.profile_steps:
                    timing = first_times[min(4, len(first_times) // 2):]
                    seconds = float(np.median(timing))
                    return {'steps': steps, 'median_step_seconds': seconds,
                            'steps_per_epoch': steps_per_epoch, 'epoch_minutes_estimate': seconds * steps_per_epoch / 60,
                            'parameters': count, 'loss': float(loss.item()),
                            'peak_gpu_allocation_bytes': getattr(torch, args.device).max_memory_allocated()}
                if steps % 100 == 0:
                    getattr(torch, args.device).synchronize()
                    if progress is not None:
                        progress(epoch, batch_index, steps_per_epoch)
                    print(f'  epoch {epoch} step {batch_index}/{steps_per_epoch}', flush=True)
            if inner is not None:
                logits = score_views(torch, groups, features, model, tokenizer, inner, args, deadline,
                                     ('natural', 'natural_tta'))
                prediction = average_tta(logits['natural'], logits['natural_tta'], weight).argmax(axis=1)
                result = macro(labels[inner], prediction)
                print(f'  epoch {epoch} inner perturbed {result:.4f}, {perf_counter()-started:.0f}s', flush=True)
                if result > best:
                    best, best_epoch = result, epoch
                    torch.save(model.state_dict(), checkpoint)
                if epoch - best_epoch >= args.patience:
                    break
        if inner is not None:
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return model, tokenizer, features, count, best_epoch or epochs


def run(args):
    os.environ.setdefault('USE_TF', '0')
    texts, labels = load_training(args.data / 'train.csv')
    roman = np.array([script(text) == 'roman' for text in texts])
    parts = folds(texts, labels)
    reference = baseline(args.baseline, labels, roman, parts)
    args.output.mkdir(parents=True, exist_ok=True)
    json_file(args.output / 'baseline.json', reference)
    print(json.dumps(reference, indent=2), flush=True)
    if args.baseline_only:
        return

    import torch

    args.device = accelerator(torch, args.device)
    backend = getattr(torch, args.device)
    args.precision = torch.bfloat16 if backend.is_bf16_supported() else torch.float16
    print(f'{args.device}: {backend.get_device_name(0)}', flush=True)
    groups = variant_groups(texts, [])
    protocol = {'version': 2, 'dataset': dataset_key(texts), 'label_hash': hashlib.sha256(labels.tobytes()).hexdigest(),
                'device': args.device, 'device_name': backend.get_device_name(0), 'batch_size': args.batch_size,
                'eval_batch': args.eval_batch, 'epochs': args.epochs, 'patience': args.patience, 'optimizer': args.optimizer,
                'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else 'pinned Google MuRIL-base',
                'baseline': reference, 'budget_seconds': args.budget_seconds}
    path = args.output / 'experiment.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'protocol': protocol, 'folds': [], 'elapsed_seconds': 0., 'status': 'prepared'}
    if report['protocol'] != protocol:
        raise ValueError('Use a new output directory for a different accelerator or training protocol')
    if report['status'] in ('early_stop', 'accuracy_failed', 'budget_expired', 'accuracy_passed',
                            'deployment_failed', 'passed'):
        print(f"Experiment status is already {report['status']}", flush=True)
        return
    started = perf_counter()
    spent = report['elapsed_seconds']
    deadline = started + args.budget_seconds - spent
    weight = reference['tta_weight']

    def progress(epoch, step, steps):
        report['progress'] = {'fold': len(report['folds']) + 1, 'epoch': epoch, 'step': step, 'steps': steps}
        report['elapsed_seconds'] = spent + perf_counter() - started
        json_file(path, report)

    try:
        if len(report['folds']) == 2 and stop_after_two([row['paired_gain'] for row in report['folds']]):
            report['status'] = 'early_stop'
            return
        for number in range(len(report['folds']), len(parts)):
            check_budget(deadline)
            outer, check = parts[number]
            train, inner = folds(take(texts, outer), labels[outer], n_splits=8)[0]
            fit_rows, early = outer[train], outer[inner]
            report['status'] = 'in_progress'
            progress(0, 0, 0)
            print(f'Fold {number+1}/5, {len(fit_rows)} train, {len(early)} inner, {len(check)} outer', flush=True)
            fit_result = fit(torch, groups, labels, fit_rows, np.r_[early, check], args, deadline, SEED + number,
                             weight, inner=early, progress=progress)
            if args.profile_steps:
                report['profile'] = fit_result
                report['status'] = 'profiled'
                print(json.dumps(fit_result, indent=2), flush=True)
                return
            model, tokenizer, features, count, best_epoch = fit_result
            logits = score_views(torch, groups, features, model, tokenizer, check, args, deadline)
            del model, tokenizer, features, fit_result
            backend.empty_cache()
            scores = evaluate(logits, labels[check], roman[check], weight)
            gain = scores['perturbed'] - reference['folds'][number]['perturbed']
            result = {'fold': number + 1, 'candidate': scores, 'baseline': reference['folds'][number],
                      'paired_gain': gain, 'epochs': best_epoch, 'parameters': count}
            out = args.output / f'fold-{number+1}.npz'
            np.savez_compressed(out, rows=check, **logits)
            report['folds'].append(result)
            report.pop('progress', None)
            report['elapsed_seconds'] = spent + perf_counter() - started
            report['status'] = 'in_progress'
            json_file(path, report)
            print(json.dumps(result, indent=2), flush=True)
            if number == 1 and stop_after_two([row['paired_gain'] for row in report['folds']]):
                report['status'] = 'early_stop'
                return

        pooled_logits = {name: np.empty((len(labels), 56), dtype=np.float32) for name in VIEWS}
        for number in range(1, 6):
            with np.load(args.output / f'fold-{number}.npz') as saved:
                for name in VIEWS:
                    pooled_logits[name][saved['rows']] = saved[name]
        pooled = evaluate(pooled_logits, labels, roman, weight)
        gains = [row['paired_gain'] for row in report['folds']]
        checks = accuracy_checks(pooled, gains)
        report.update({'pooled': pooled, 'mean_paired_gain': float(np.mean(gains)),
                       'fold_wins': sum(gain > 0 for gain in gains), 'checks': checks})
        if not all(checks.values()):
            report['status'] = 'accuracy_failed'
            return
        check_budget(deadline)
        final_epochs = int(np.median([row['epochs'] for row in report['folds']]))
        model, tokenizer, features, count, _ = fit(torch, groups, labels, np.arange(len(labels)), None, args,
                                                    deadline, SEED + 5, weight, epochs=final_epochs, progress=progress)
        from sprint.muril_export import save_candidate
        save_candidate(args.output, model, tokenizer, features, count, weight,
                       logit_scale(average_tta(pooled_logits['raw'], pooled_logits['tta'], weight), labels))
        report['status'] = 'accuracy_passed'
    except BudgetExpired:
        report['status'] = 'budget_expired'
    except BaseException as error:
        report['status'] = 'interrupted'
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        if report['status'] not in ('interrupted', 'budget_expired'):
            report.pop('progress', None)
        report['elapsed_seconds'] = spent + perf_counter() - started
        json_file(path, report)
        print(f"{report['status']}, {len(report['folds'])}/5 complete, {report['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--baseline', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/muril_paired'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('auto', 'xpu', 'cuda'), default='auto')
    parser.add_argument('--optimizer', choices=('adamw', 'adamw8', 'adafactor'), default='adamw')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--budget-seconds', type=int, default=BUDGET)
    parser.add_argument('--profile-steps', type=int, default=0)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.eval_batch, args.patience, args.budget_seconds) <= 0 or args.budget_seconds > BUDGET:
        parser.error('Positive batch sizes and patience are required, and the run budget is capped at 7,200 seconds')
    if args.profile_steps < 0:
        parser.error('Profile steps must be nonnegative')
    with threadpool_limits(limits=2):
        run(args)


if __name__ == '__main__':
    main()
