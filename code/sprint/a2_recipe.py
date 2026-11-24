"""A2 (all 12 Bhasha layers trainable) plus the submitted recipe's train-only extra rows.

Extra rows follow the submitted Bhasha recipe (sprint.score_push.samples): 65% augmented copies at
weight 0.7, phrase views and counterfactual contrasts at weight 0.55, never Class 55 phrase/contrast
rows, and nothing whose cleaned text matches any evaluation view of the held-out rows. The dictionary
is fitted on organizer rows only. Per pilot fold: inner early stopping picks the epoch count, then
a refit on the full outer-training split scores the outer fold with the pilot's measures().
Folds 3-5 and the locked template holdout are not touched.
"""
import argparse
import gc
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from intent_model import FALLBACK, Views, clean, script
from sprint.augment import augment_batch, tta_variant
from sprint.data import SEED, load_training, macro
import sprint.depth_ladder as ladder
from sprint.depth_ladder_report import VIEWS, measures, write_json
from sprint.englishize import englishize_batch
from sprint.score_data import counterfactuals, phrase_views
from sprint.score_push import fingerprint

AUGMENT_RATE = .65
WEIGHTS = {'raw': 1., 'aug': .7, 'phrases': .55, 'contrast': .55, 'english': .7, 'llm': .7}
TESTLIKE_RATE = .45  # share of rewritten rows that brings English-word share near the test set's 0.195


def extra_rows(groups, labels, fit, check, seed):
    """Return (texts, labels, weights, origin) for train-only rows tied to `fit`, blocked against `check`."""
    blocked = {clean(groups[view][i]) for view in ('raw', 'tta', 'natural', 'natural_tta', 'asr', 'english_view', 'llm_view')
               if view in groups for i in check}
    seen = {clean(groups['raw'][i]) for i in fit}
    rng = np.random.default_rng(seed)
    available = np.array([i for i in fit if groups['raw'][i] != groups['aug'][i]], dtype=int)
    chosen = rng.choice(available, min(len(available), round(AUGMENT_RATE * len(fit))), replace=False)
    texts, targets, weights, origin = [], [], [], []
    for i in chosen:
        if clean(groups['aug'][i]) not in blocked:
            texts.append(groups['aug'][i]); targets.append(labels[i]); weights.append(WEIGHTS['aug']); origin.append('aug')
    present = np.zeros(len(labels), dtype=bool)
    present[fit] = True
    for name, text, sources, target in (('phrases', groups['phrases'], np.arange(len(labels)), labels),
                                        ('contrast', *groups['contrast'])):
        for i, source in enumerate(sources):
            if not present[source] or labels[source] == FALLBACK:
                continue
            key = clean(text[i])
            if key and key not in blocked and key not in seen:
                seen.add(key)
                texts.append(text[i]); targets.append(int(target[i])); weights.append(WEIGHTS[name]); origin.append(name)
    # English-spelled copies go last so they never displace the submitted recipe's rows.
    if 'english' in groups:
        for i in fit:
            key = clean(groups['english'][i])
            if groups['english'][i] != groups['raw'][i] and key and key not in blocked and key not in seen:
                seen.add(key)
                texts.append(groups['english'][i]); targets.append(labels[i]); weights.append(WEIGHTS['english']); origin.append('english')
    if 'llm' in groups:
        for i in fit:
            key = clean(groups['llm'][i])
            if groups['llm'][i] != groups['raw'][i] and key and key not in blocked and key not in seen:
                seen.add(key)
                texts.append(groups['llm'][i]); targets.append(labels[i]); weights.append(WEIGHTS['llm']); origin.append('llm')
    return texts, np.asarray(targets, dtype=labels.dtype), np.asarray(weights, dtype=np.float32), origin


def main():
    parser = argparse.ArgumentParser()
    for option in ('train', 'checkpoint', 'plan', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--label-smoothing', default=.1, type=float)
    parser.add_argument('--minutes', default=650., type=float)
    parser.add_argument('--dry-run', action='store_true', help='build and count extra rows, no training')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'xpu'))
    parser.add_argument('--amp', default='float16', choices=('float16', 'bfloat16'))
    parser.add_argument('--optimizer', default='adamw', choices=('adamw', 'adafactor'))
    parser.add_argument('--folds', default='1,2', help='pilot folds to evaluate, e.g. 1')
    parser.add_argument('--final-epochs', type=int, help='skip folds: train on all rows for N epochs, save weights, predict --test')
    parser.add_argument('--test', type=Path)
    parser.add_argument('--llm-rewrites', type=Path, help='jsonl from sprint.llm_rewrite: adds rewritten train copies and llm/testlike eval views')
    parser.add_argument('--llm-eval-only', action='store_true', help='score the llm views without adding rewritten training copies')
    parser.add_argument('--englishize', default='off', choices=('off', 'eval', 'train'),
                        help='eval: score an English-spelled view of held-out rows; train: also add English-spelled training copies')
    args = parser.parse_args()
    raw, labels = load_training(args.train)
    ids, early = ladder.read_plan(args.plan, raw, labels, np.asarray(Views.from_texts(raw).phonetic))
    natural = augment_batch(raw, SEED + 2, 'natural')
    groups = {'raw': raw, 'aug': augment_batch(raw, SEED + 1), 'natural': natural,
              'asr': augment_batch(raw, SEED + 3, 'asr'), 'tta': list(map(tta_variant, raw)),
              'natural_tta': list(map(tta_variant, natural)), 'phrases': phrase_views(raw),
              'contrast': counterfactuals(raw, labels)}
    if args.englishize != 'off':
        groups['english_view'] = englishize_batch(raw, SEED + 5, rate=1.)
    if args.englishize == 'train':
        groups['english'] = englishize_batch(raw, SEED + 4)
    views = {'overall': raw, 'perturbed': natural, 'asr': groups['asr']}
    extra_views = ('english_view',) if args.englishize != 'off' else ()
    if args.llm_rewrites:
        import re
        rewrites = {}
        for line in args.llm_rewrites.open(encoding='utf-8'):
            item = json.loads(line)
            if item['rewrite'] and not re.search('[ऀ-ॿ]', item['rewrite']):
                rewrites[item['source']] = item['rewrite']
        groups['llm_view'] = [rewrites.get(text, text) for text in raw]
        mix = np.random.default_rng(SEED + 6).random(len(raw)) < TESTLIKE_RATE
        groups['testlike_view'] = [groups['llm_view'][i] if mix[i] else raw[i] for i in range(len(raw))]
        if not args.llm_eval_only:
            groups['llm'] = groups['llm_view']
        extra_views += ('llm_view', 'testlike_view')
        print('llm rewrites available for', sum(text in rewrites for text in raw), 'of', len(raw), 'rows', flush=True)
    scripts = np.array([script(text) for text in raw])
    plan = []
    for fold in ([] if args.final_epochs else [int(f) for f in args.folds.split(',')]):
        outer_train, outer_check = np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
        inner_fit, inner_check = np.flatnonzero((ids != fold) & ~early[fold - 1]), np.flatnonzero(early[fold - 1])
        plan.append((fold, 'inner', inner_fit, inner_check, extra_rows(groups, labels, inner_fit, inner_check, SEED + fold)))
        plan.append((fold, 'outer', outer_train, outer_check, extra_rows(groups, labels, outer_train, outer_check, SEED + fold)))
    for fold, phase, fit_rows, check, (texts, _, weights, origin) in plan:
        print(fold, phase, 'raw', len(fit_rows), 'check', len(check),
              {name: origin.count(name) for name in ('aug', 'phrases', 'contrast', 'english', 'llm')}, flush=True)
    if args.dry_run:
        return

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch

    accelerator = getattr(torch, args.device)
    if not accelerator.is_available():
        raise SystemExit(f'No {args.device} device available')
    ladder.DEVICE, ladder.AMP, ladder.OPTIMIZER = args.device, args.amp, args.optimizer
    if any(fingerprint(args.checkpoint / name) != digest for name, digest in ladder.CHECKPOINT_HASHES.items()):
        raise ValueError('The Bhasha weights or tokenizer do not match the pinned files')
    torch.use_deterministic_algorithms(True, warn_only=args.device != 'cuda' or torch.version.hip is not None)
    torch.set_num_threads(2 if args.device == 'cuda' else 4)
    ladder.EPOCHS = args.epochs
    ladder.LR = {**ladder.LR, 'A2': args.lr}
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'trainer': fingerprint(Path(ladder.__file__)), 'runner': fingerprint(Path(__file__)),
              'plan': fingerprint(args.plan), 'epochs': args.epochs, 'lr': args.lr,
              'label_smoothing': args.label_smoothing, 'augment_rate': AUGMENT_RATE, 'weights': WEIGHTS,
              'device': args.device, 'amp': args.amp, 'optimizer': args.optimizer, 'final_epochs': args.final_epochs,
              **({'englishize': args.englishize} if args.englishize != 'off' else {}),
              **({'llm_rewrites': fingerprint(args.llm_rewrites), 'llm_eval_only': args.llm_eval_only} if args.llm_rewrites else {})}
    marker = args.output / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != json.loads(json.dumps(config)):
        raise ValueError('The output folder belongs to a different experiment')
    write_json(marker, config)
    args.deadline = monotonic() + args.minutes * 60
    if args.final_epochs:
        everything_rows = np.arange(len(raw))
        texts, targets, weights, origin = extra_rows(groups, labels, everything_rows, np.array([], dtype=int), SEED)
        print('final rows', len(raw), {name: origin.count(name) for name in ('aug', 'phrases', 'contrast', 'english', 'llm')}, flush=True)
        root = args.output / 'final'
        try:
            model, tokenizer, features, dictionary, resume = ladder.fit(
                torch, args, 'A2', root / 'refit', list(raw) + texts, np.concatenate([labels, targets]),
                np.concatenate([everything_rows, len(raw) + np.arange(len(texts))]), None, SEED, 1., args.final_epochs,
                sample_weight=np.concatenate([np.ones(len(raw), dtype=np.float32), weights]),
                label_smoothing=args.label_smoothing, dictionary_rows=everything_rows)
        except ladder.Pause:
            print('Time budget reached. Rerun the same command to resume.', flush=True)
            return
        ladder.publish_weights(model, tokenizer, features, dictionary, root)
        if args.test is not None:
            import csv
            with args.test.open(encoding='utf-8', newline='') as handle:
                test_rows = list(csv.DictReader(handle))
            column = 'query_text' if 'query_text' in test_rows[0] else [k for k in test_rows[0] if k != 'example_id'][0]
            batches = ladder.ViewBatches([row[column] for row in test_rows], dictionary, tokenizer, features)
            logits = ladder.score(torch, model, batches, torch.device(f'{args.device}:0'))
            np.save(root / 'test_logits.npy', logits)
            with (root / 'submission.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['example_id', 'target'])
                writer.writerows((row['example_id'], int(label)) for row, label in zip(test_rows, logits.argmax(1)))
            print('FINAL_SUBMISSION', root / 'submission.csv', len(test_rows), flush=True)
        resume.unlink(missing_ok=True)
        return
    report = {'config': config, 'inner': {}, 'outer': {}}
    previous = args.output / 'a2_recipe.json'
    if previous.exists():
        report.update({k: v for k, v in json.loads(previous.read_text(encoding='utf-8')).items() if k in ('inner', 'outer')})
    for fold, phase, fit_rows, check, (texts, targets, weights, _) in plan:
        key = f'fold{fold}'
        if key in report[phase]:
            continue
        everything = list(raw) + texts
        all_labels = np.concatenate([labels, targets])
        rows = np.concatenate([fit_rows, len(raw) + np.arange(len(texts))])
        sample_weight = np.concatenate([np.ones(len(fit_rows), dtype=np.float32), weights])
        root = args.output / f'fold{fold}-{phase}'
        try:
            if phase == 'inner':
                validation = ([natural[i] for i in check], labels[check])
                epoch, score, resume = ladder.fit(
                    torch, args, 'A2', root / 'inner' / 'checkpoints', everything, all_labels, rows, validation,
                    SEED + fold - 1, 1., clean_validation=[raw[i] for i in check], best_logits=root / 'inner_logits.npz',
                    sample_weight=sample_weight, label_smoothing=args.label_smoothing, dictionary_rows=fit_rows)
                with np.load(root / 'inner_logits.npz') as saved:
                    clean_f1 = macro(labels[check], saved['overall'].argmax(1))
                report['inner'][key] = {'selected_epoch': epoch, 'perturbed_macro_f1': score,
                                        'clean_macro_f1': clean_f1, 'train_rows': len(rows)}
            else:
                epochs = report['inner'][key]['selected_epoch']
                model, tokenizer, features, dictionary, resume = ladder.fit(
                    torch, args, 'A2', root / 'refit', everything, all_labels, rows, None, SEED + fold - 1, 1.,
                    epochs, sample_weight=sample_weight, label_smoothing=args.label_smoothing,
                    dictionary_rows=fit_rows)
                device = torch.device(f'{args.device}:0')
                logits = {}
                for view in VIEWS:
                    batches = ladder.ViewBatches([views[view][i] for i in check], dictionary, tokenizer, features)
                    logits[view] = ladder.score(torch, model, batches, device)
                for view in extra_views:
                    batches = ladder.ViewBatches([groups[view][i] for i in check], dictionary, tokenizer, features)
                    logits[view] = ladder.score(torch, model, batches, device)
                np.savez_compressed(root / 'outer_logits.npz', rows=check, **logits)
                result = measures(labels[check], scripts[check], logits)
                result.pop('per_class')
                result.update({view: macro(labels[check], logits[view].argmax(1)) for view in extra_views})
                result.update({'epochs': epochs, 'train_rows': len(rows)})
                report['outer'][key] = result
                del model
        except ladder.Pause:
            print('Time budget reached. Attach this output and rerun to resume.', flush=True)
            return
        resume.unlink(missing_ok=True)
        gc.collect()
        accelerator.empty_cache()
        write_json(previous, report)
        print('RECIPE_RESULT', phase, key, json.dumps(report[phase][key]), flush=True)


if __name__ == '__main__':
    main()
