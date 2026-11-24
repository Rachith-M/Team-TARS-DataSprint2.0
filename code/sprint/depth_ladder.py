import argparse
import gc
import json
import math
import os
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import LABELS, PARAMETER_CAP, Features, Views, script
from sprint.augment import augment_batch, tta_variant
from sprint.bhasha_audit import CHECKPOINT_REVISION, dictionary_view, inner_partition
from sprint.data import SEED, dataset_key, folds, load_training, macro
from sprint.defense_artifacts import MAX_TWO_VIEW_TOKENS
from sprint.depth_ladder_report import ARMS, VIEWS, grouped, pilot, tables, template, write_json
from sprint.runtime import pack_lexical
from sprint.score_native import reverse_dictionary
from sprint.score_push import fingerprint


EPOCHS = 8
BATCH = 32
PATIENCE = 2
LR = {'A0': 0., 'A1': 3e-5, 'A2': 2e-5}
HEAD_LR = 1e-3
DECAY = .9
WEIGHT = .35
SEMANTIC_WEIGHT = 1.4
MAX_LEN = 64
FALLBACK_MARGIN = .20
DEVICE = 'cuda'  # 'xpu' for Intel Arc; set before calling fit()
AMP = 'float16'
OPTIMIZER = 'adamw'
CHECKPOINT_HASHES = {
    'model.safetensors': 'e5f3340b73ad5a63e0820d9a7e885ab67c0178b13a1bf74e7951e161f19ce8df',
    'tokenizer.json': '677ed37c781a90998f7095d3951f70f09f0eddf2199f59ba5e0f91cfdbef33df',
}


class Pause(Exception):
    pass


class Diverged(Exception):
    pass


def data_identity(texts, labels):
    return dataset_key(f'{label}\x1f{text}' for text, label in zip(texts, labels))


def split_diagnostics(keys, fit, check):
    count = int(np.count_nonzero(np.isin(keys[check], np.unique(keys[fit]))))
    return {'fit_rows': len(fit), 'validation_rows': len(check), 'phonetic_twins': count,
            'phonetic_twin_rate': count / len(check), 'twin_rate_le_1_5_pct': count / len(check) <= .015}


def register(args, texts, labels, scripts, keys):
    if args.plan.exists():
        raise FileExistsError(args.plan)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    outer_id = np.zeros(len(labels), dtype=np.uint8)
    early_ids = np.zeros((5, len(labels)), dtype=bool)
    summary = {}
    for number, (fit, check) in enumerate(folds(texts, labels), 1):
        inner_fit, early = inner_partition(fit, labels, scripts, keys, SEED + number - 1)
        outer_id[check] = number
        early_ids[number - 1, early] = True
        summary[f'fold{number}'] = {'outer': split_diagnostics(keys, fit, check),
                                    'early_stop': split_diagnostics(keys, inner_fit, early)}
    with args.plan.open('xb') as file:
        np.savez_compressed(file, fold_id=outer_id, early=early_ids,
                            identity=data_identity(texts, labels), revision=CHECKPOINT_REVISION)
    write_json(args.plan.with_suffix('.json'), {'checkpoint_revision': CHECKPOINT_REVISION, 'seed': SEED,
        'pilot_folds': [1, 2], 'template_evaluated': False, 'head_lr': HEAD_LR, 'patience': PATIENCE,
        'encoder_lr': LR, 'weight_decay': .01, 'layer_decay': DECAY, 'warmup_fraction': .10,
        'batch': BATCH, 'max_epochs': EPOCHS, 'divergence_retry_encoder_lr_factor': .5,
        'folds': summary})
    print(json.dumps(summary, indent=2), flush=True)


def read_plan(path, texts, labels, keys):
    with np.load(path) as archive:
        if archive['identity'].item() != data_identity(texts, labels) or archive['revision'].item() != CHECKPOINT_REVISION:
            raise ValueError('The organizer training rows or the pinned checkpoint revision differ from registration')
        ids, early = archive['fold_id'], archive['early']
    if ids.shape != labels.shape or early.shape != (5, len(labels)) or set(ids) != set(range(1, 6)):
        raise ValueError('Invalid frozen fold plan')
    for number in range(1, 6):
        fit = np.flatnonzero(ids != number)
        check = np.flatnonzero(ids == number)
        inner = np.flatnonzero(early[number - 1])
        if not len(inner) or np.any(ids[inner] == number) or np.intersect1d(keys[fit], keys[check]).size:
            raise ValueError('Invalid or overlapping outer and early-stopping partitions')
        if np.intersect1d(keys[fit[~early[number - 1, fit]]], keys[inner]).size:
            raise ValueError('An early-stopping phonetic family overlaps its training partition')
    return ids, early


def atomic_torch(torch, path, state):
    temporary = path.with_suffix('.partial')
    torch.save(state, temporary)
    temporary.replace(path)


class ViewBatches:
    def __init__(self, texts, dictionary, tokenizer, features, fit=False):
        alternate = [dictionary_view(text, tta_variant(text), dictionary) for text in texts]
        flat = [text for pair in zip(texts, alternate) for text in pair]
        tokens = tokenizer(flat, max_length=MAX_LEN, truncation=True, verbose=False)
        self.rows = [dict(zip(tokens, row)) for row in zip(*tokens.values())]
        self.features = features.encode(Views.from_texts(flat), fit=fit)
        self.lengths = np.array([len(row['input_ids']) for row in self.rows]).reshape(-1, 2).max(axis=1)
        self.tokenizer = tokenizer

    def batches(self, seed=None):
        order = np.arange(len(self.lengths))
        if seed is not None:
            rng = np.random.default_rng(seed)
            order = rng.permutation(order)
        order = order[np.argsort(self.lengths[order], kind='stable')]
        chunks = [order[start:start + BATCH] for start in range(0, len(order), BATCH)]
        return chunks if seed is None else [chunks[i] for i in rng.permutation(len(chunks))]

    def inputs(self, torch, indices, device):
        rows = (indices[:, None] * 2 + np.arange(2)).reshape(-1)
        tokens = self.tokenizer.pad([self.rows[i] for i in rows], padding=True, return_tensors='pt').to(device)
        ids, values = pack_lexical(self.features[rows])
        return tokens, torch.from_numpy(ids).to(device), torch.from_numpy(values).to(device), rows


def model_for(torch, checkpoint, dimension, arm, seed, device):
    from transformers import AutoModel

    class Classifier(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.lexical = torch.nn.Embedding(dimension, len(LABELS))
            self.semantic = torch.nn.Linear(encoder.config.hidden_size, len(LABELS))
            torch.nn.init.zeros_(self.lexical.weight)

        def embed(self, tokens):
            hidden = self.encoder(**tokens).last_hidden_state.float()
            masked = (hidden * tokens['attention_mask'].unsqueeze(-1)).sum(1)
            return torch.nn.functional.normalize(masked, dim=1)

        def forward(self, tokens, ids, values, cached=None, live=False):
            embedding = self.embed(tokens) if cached is None else cached
            lexical = (self.lexical(ids) * values.unsqueeze(-1)).sum(1)
            logits = lexical + self.semantic(SEMANTIC_WEIGHT * embedding).float()
            result = combine(logits)
            if live:
                long = tokens['attention_mask'].sum(1).reshape(-1, 2).max(1).values > MAX_TWO_VIEW_TOKENS
                result = torch.where(long[:, None], logits[::2], result)
            return result

        def training_mode(self):
            self.train()
            if arm == 'A0':
                self.encoder.eval()
            else:
                if arm == 'A1':
                    self.encoder.embeddings.eval()
                    for layer in self.encoder.encoder.layer[:-4]:
                        layer.eval()

    torch.manual_seed(seed)
    encoder = AutoModel.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
                                        add_pooling_layer=False, attn_implementation='sdpa')
    layers = encoder.encoder.layer
    if encoder.config.model_type != 'bert' or len(layers) != 12:
        raise ValueError('Registration requires the pinned 12-layer Bhasha BERT encoder')
    encoder.requires_grad_(arm == 'A2')
    if arm == 'A1':
        for layer in layers[-4:]:
            layer.requires_grad_(True)
    if arm != 'A0':
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = Classifier(encoder).to(device)
    if sum(p.numel() for p in model.parameters()) + dimension > PARAMETER_CAP:
        raise ValueError('Model plus lexical vocabulary exceeds 300M parameters')
    return model


def combine(logits):
    return (1 - WEIGHT) * logits[::2] + WEIGHT * logits[1::2]


def optimization(torch, model, arm, lr_factor, batches):
    from transformers import get_linear_schedule_with_warmup

    groups = [{'params': [*model.lexical.parameters(), *model.semantic.parameters()], 'lr': HEAD_LR}]
    blocks = [model.encoder.embeddings, *model.encoder.encoder.layer]
    for depth, block in enumerate(reversed(blocks)):
        values = [p for p in block.parameters() if p.requires_grad]
        if values:
            groups.append({'params': values, 'lr': LR[arm] * lr_factor * DECAY**depth})
    if OPTIMIZER == 'adafactor':
        from transformers.optimization import Adafactor
        optimizer = Adafactor(groups, lr=HEAD_LR, weight_decay=.01, scale_parameter=False, relative_step=False,
                              warmup_init=False)
    else:
        optimizer = torch.optim.AdamW(groups, lr=HEAD_LR, weight_decay=.01, foreach=False)
    steps = EPOCHS * len(batches.batches())
    schedule = get_linear_schedule_with_warmup(optimizer, int(math.ceil(.10 * steps)), steps)
    return optimizer, schedule, torch.amp.GradScaler(DEVICE, enabled=AMP == 'float16')


def embedding_cache(torch, batches, model, device):
    result = torch.empty((2 * len(batches.lengths), model.encoder.config.hidden_size), device=device)
    model.eval()
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=getattr(torch, AMP)):
        for batch in batches.batches():
            tokens, _, _, rows = batches.inputs(torch, batch, device)
            result[rows] = model.embed(tokens)
    return result


def score(torch, model, batches, device, live=False):
    logits = np.empty((len(batches.lengths), len(LABELS)), dtype=np.float32)
    model.eval()
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=getattr(torch, AMP)):
        for batch in batches.batches():
            tokens, ids, values, _ = batches.inputs(torch, batch, device)
            logits[batch] = model(tokens, ids, values, live=live).cpu().numpy()
    if not np.isfinite(logits).all():
        raise FloatingPointError('Validation logits are nonfinite')
    return logits


def training_loss(torch, logits, targets, fallback_penalty, weights=None, label_smoothing=0.):
    if weights is None and not label_smoothing:
        loss = torch.nn.functional.cross_entropy(logits, targets)
    else:
        each = torch.nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
        loss = (each * weights).sum() / weights.sum() if weights is not None else each.mean()
    if fallback_penalty:
        fallback = logits.float().softmax(dim=1)[:, 55]
        excess = (fallback - FALLBACK_MARGIN).clamp_min(0)
        loss = loss + fallback_penalty * (excess.square() * (targets != 55)).mean()
    return loss


def fit(torch, args, arm, folder, raw, labels, fit_rows, validation, seed, lr_factor, fixed_epochs=None,
        *, fallback_penalty=0., clean_validation=None, best_logits=None, sample_weight=None,
        label_smoothing=0., dictionary_rows=None):
    from transformers import AutoTokenizer

    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'{DEVICE}:0')
    dictionary = reverse_dictionary([raw[i] for i in (fit_rows if dictionary_rows is None else dictionary_rows)])
    features = Features()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    with threadpool_limits(limits=2):
        train = ViewBatches([raw[i] for i in fit_rows], dictionary, tokenizer, features, fit=True)
        valid = (ViewBatches(validation[0], dictionary, tokenizer, features) if validation is not None else None)
        clean = (ViewBatches(clean_validation, dictionary, tokenizer, features)
                 if clean_validation is not None else None)
    model = model_for(torch, args.checkpoint, features.parameter_count(), arm, seed, device)
    optimizer, scheduler, scaler = optimization(torch, model, arm, lr_factor, train)
    resume = folder / 'resume.pt'
    epoch, next_batch, best_epoch, best, stalled, consecutive_skips = 0, 0, 0, -1., 0, 0
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=False)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        epoch, next_batch, best_epoch, best, stalled, consecutive_skips = state['progress']
        torch.set_rng_state(state['rng'])
        getattr(torch, DEVICE).set_rng_state(state['cuda_rng'])
        del state
    last_save = monotonic()

    def checkpoint():
        atomic_torch(torch, resume, {'model': {key: value for key, value in model.state_dict().items() if key in trainable},
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'progress': [epoch, next_batch, best_epoch, best, stalled, consecutive_skips],
                     'rng': torch.get_rng_state(), 'cuda_rng': getattr(torch, DEVICE).get_rng_state()})

    cache = embedding_cache(torch, train, model, device) if arm == 'A0' else None
    target = torch.from_numpy(labels[fit_rows].astype(np.int64)).to(device)
    weight = (None if sample_weight is None else
              torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).to(device))
    if weight is not None and len(weight) != len(fit_rows):
        raise ValueError('sample_weight must align with fit_rows')
    steps = fixed_epochs if fixed_epochs is not None else EPOCHS
    while epoch < steps and (fixed_epochs is not None or stalled < PATIENCE):
        model.training_mode()
        for batch_number, batch in enumerate(train.batches(seed + epoch * 101), 1):
            if batch_number <= next_batch:
                continue
            if monotonic() >= args.deadline:
                checkpoint()
                raise Pause
            tokens, ids, values, rows = train.inputs(torch, batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=getattr(torch, AMP)):
                logits = model(tokens, ids, values, cached=cache[rows] if cache is not None else None)
                loss = training_loss(torch, logits, target[batch], fallback_penalty,
                                     None if weight is None else weight[batch], label_smoothing)
            if not torch.isfinite(loss).item():
                raise Diverged('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale:
                scheduler.step()
                consecutive_skips = 0
            else:
                consecutive_skips += 1
                if consecutive_skips >= 32:
                    raise Diverged('32 consecutive skipped fp16 updates')
            next_batch = batch_number
            if monotonic() - last_save >= 300:
                checkpoint()
                last_save = monotonic()
        epoch += 1
        next_batch = 0
        if valid is not None:
            try:
                inner_logits = score(torch, model, valid, device)
                f1 = macro(validation[1], inner_logits.argmax(1))
                if f1 > best and best_logits is not None:
                    clean_logits = score(torch, model, clean, device)
                    temporary = best_logits.with_suffix('.partial.npz')
                    np.savez_compressed(temporary, epoch=epoch, overall=clean_logits, perturbed=inner_logits)
                    temporary.replace(best_logits)
            except FloatingPointError as error:
                raise Diverged(str(error)) from error
            if f1 > best:
                best, best_epoch, stalled = f1, epoch, 0
            else:
                stalled += 1
            print(f'{folder.parent.parent.name} epoch {epoch} inner perturbed {f1:.5f}', flush=True)
        checkpoint()
    if fixed_epochs is not None:
        return model, tokenizer, features, dictionary, resume
    return best_epoch, best, resume


def publish_weights(model, tokenizer, features, dictionary, folder):
    target = folder / 'checkpoint'
    target.mkdir(exist_ok=True)
    model.encoder.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    np.savez(target / 'head.npz', lexical=model.lexical.weight.detach().cpu().numpy(),
             semantic=model.semantic.weight.detach().cpu().numpy(), bias=model.semantic.bias.detach().cpu().numpy())
    joblib.dump(features, target / 'features.joblib', compress=3)
    write_json(target / 'dictionary.json', list(dictionary))


def validation_views(texts):
    return {'overall': texts, 'perturbed': augment_batch(texts, SEED + 2, 'natural'),
            'asr': augment_batch(texts, SEED + 3, 'asr')}


def job(torch, args, name, arm, raw, labels, train, check, early, seed, keep_weights=False, final_epochs=None):
    root = args.output / f'{name}-{arm}'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        return True
    if (root / 'failed.json').exists():
        return False
    for factor in ((1.,) if arm == 'A0' else (1., .5)):
        folder = root / str(factor)
        if (folder / 'failed.json').exists():
            continue
        try:
            if final_epochs is None:
                inner_fit = np.setdiff1d(train, early)
                chosen = folder / 'selection.json'
                if not chosen.exists():
                    perturbed = args.views['perturbed']
                    validation = ([perturbed[i] for i in early], labels[early])
                    chosen.parent.mkdir(parents=True, exist_ok=True)
                    epoch, val, path = fit(torch, args, arm, folder / 'inner', raw, labels, inner_fit, validation,
                                           seed, factor)
                    write_json(chosen, {'epochs': epoch, 'inner_perturbed': val, 'encoder_lr_factor': factor})
                    path.unlink(missing_ok=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                epochs = json.loads(chosen.read_text(encoding='utf-8'))['epochs']
            else:
                epochs = final_epochs
            model, tokenizer, features, dictionary, resume = fit(torch, args, arm, folder / 'refit', raw, labels,
                                                                 train, None, seed, factor, epochs)
            saved_logits = root / 'logits.npz'
            if len(check) and not saved_logits.exists():
                device = torch.device('cuda:0')
                logits = {}
                for view in VIEWS:
                    batches = ViewBatches([args.views[view][i] for i in check], dictionary, tokenizer, features)
                    logits[view] = score(torch, model, batches, device)
                    if view == 'overall':
                        logits['live'] = score(torch, model, batches, device, live=True)
                temporary = root / 'logits.partial.npz'
                np.savez_compressed(temporary, rows=check, **logits)
                temporary.replace(saved_logits)
            if keep_weights:
                publish_weights(model, tokenizer, features, dictionary, root)
            count = sum(p.numel() for p in model.parameters()) + features.parameter_count()
            write_json(root / 'complete.json', {'epochs': epochs, 'train_rows': len(train), 'validation_rows': len(check),
                                               'seed': seed, 'encoder_lr_factor': factor, 'parameters': count})
            resume.unlink(missing_ok=True)
            print(f'{name}-{arm} complete', flush=True)
            return True
        except Diverged as error:
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / 'failed.json', {'error': str(error), 'encoder_lr_factor': factor})
            for phase in ('inner', 'refit'):
                (folder / phase / 'resume.pt').unlink(missing_ok=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    write_json(root / 'failed.json', {'error': 'All pre-registered attempts diverged'})
    print(f'{name}-{arm} diverged on all pre-registered attempts', flush=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('register', 'pilot', 'promotion', 'refit'))
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--minutes', type=float, default=300)
    args = parser.parse_args()
    raw, labels = load_training(args.train)
    keys = np.asarray(Views.from_texts(raw).phonetic)
    scripts = np.asarray([script(text) for text in raw])
    if args.stage == 'register':
        register(args, raw, labels, scripts, keys)
        return
    if args.checkpoint is None or args.output is None or args.minutes <= 0:
        parser.error('Training requires --checkpoint, --output and a positive time budget')
    ids, early = read_plan(args.plan, raw, labels, keys)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise SystemExit('Training untested. Select an NVIDIA CUDA GPU in Kaggle or Colab.')
    if any(fingerprint(args.checkpoint / name) != digest for name, digest in CHECKPOINT_HASHES.items()):
        raise ValueError('The Bhasha weights or tokenizer do not match the pinned files')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    gpu = torch.cuda.get_device_properties(0)
    print(f'{gpu.name}, {gpu.total_memory / 2**30:.2f} GiB, CUDA {torch.version.cuda}, fp16 AMP', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'plan_identity': data_identity(raw, labels), 'plan_sha256': fingerprint(args.plan),
              'trainer_sha256': fingerprint(Path(__file__)), 'report_sha256': fingerprint(Path(tables.__code__.co_filename)),
              'revision': CHECKPOINT_REVISION, 'seed': SEED,
              'layers': 12, 'arms': LR, 'head_lr': HEAD_LR, 'max_epochs': EPOCHS, 'batch': BATCH,
              'patience': PATIENCE, 'max_tokens': MAX_LEN}
    marker = args.output / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != config:
        raise ValueError('The output folder belongs to a different experiment')
    write_json(marker, config)
    args.deadline = monotonic() + args.minutes * 60
    args.views = validation_views(raw)
    try:
        if args.stage == 'pilot':
            plan = [(i, arm) for i in (1, 2) for arm in ARMS]
        else:
            decision = pilot(args.output, labels, scripts, ids)
            winner = decision['winner']
            if decision['status'] != 'go':
                raise SystemExit('Pilot did not pass; later stages are locked')
            plan = [(i, arm) for i in (3, 4, 5) for arm in ('A0', winner)] if args.stage == 'promotion' else []
        failed = []
        for number, arm in plan:
            complete = job(torch, args, f'fold{number}', arm, raw, labels, np.flatnonzero(ids != number),
                           np.flatnonzero(ids == number), np.flatnonzero(early[number - 1]), SEED + number - 1)
            if not complete:
                failed.append(f'fold{number}-{arm}')
        if args.stage == 'pilot':
            write_json(args.output / 'pilot.json', pilot(args.output, labels, scripts, ids))
            return
        if args.stage == 'promotion':
            if failed:
                write_json(args.output / 'grouped.json', {'passed': False, 'failed_runs': failed})
                return
            decision = grouped(args.output, labels, scripts, ids, keys, winner)
            write_json(args.output / 'grouped.json', decision)
            tables(args.output, decision['folds'])
            if not decision['passed']:
                return
            import sklearn
            from sprint.score_data import splits

            if sklearn.__version__ != '1.9.1':
                raise ValueError('The locked template split requires scikit-learn 1.9.1')

            _, (train, check), _, _ = splits(raw, labels)
            seed = SEED + 9
            inner, early_template = inner_partition(train, labels, scripts, keys, seed)
            diagnostics = split_diagnostics(keys, inner, early_template)
            if not diagnostics['twin_rate_le_1_5_pct']:
                raise ValueError('The template early-stop split exceeds the phonetic twin budget')
            for arm in ('A0', winner):
                if not job(torch, args, 'template', arm, raw, labels, train, check, early_template, seed, keep_weights=True):
                    write_json(args.output / 'template.json', {'passed': False, 'failed_run': f'template-{arm}'})
                    return
            report = template(args.output, labels, scripts, check, winner)
            report['early_stop_grouping'] = diagnostics
            write_json(args.output / 'template.json', report)
            tables(args.output, [*decision['folds'], report])
        else:
            report = json.loads((args.output / 'template.json').read_text(encoding='utf-8'))
            if not report['passed'] or not report['task2_floor']:
                raise SystemExit('Task 1 did not reach the template continuation gate')
            epochs = [json.loads((args.output / f'fold{i}-{winner}' / 'complete.json').read_text())['epochs']
                      for i in range(1, 6)]
            if not job(torch, args, 'final', winner, raw, labels, np.arange(len(labels)), np.array([], dtype=int),
                       None, SEED + 5, keep_weights=True, final_epochs=int(np.median(epochs))):
                raise SystemExit('The final refit failed')
    except Pause:
        print('Time budget reached. Restart the same command to resume from the saved update.', flush=True)


if __name__ == '__main__':
    main()
