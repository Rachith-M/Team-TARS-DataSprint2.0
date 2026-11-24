import argparse
import gc
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from intent_model import Views
from sprint.augment import augment_batch
from sprint.class55_recall import counts
from sprint.data import SEED, load_training, macro
from sprint.depth_ladder import (
    CHECKPOINT_HASHES, FALLBACK_MARGIN, Diverged, Pause, fit, read_plan,
)
from sprint.depth_ladder_report import write_json
from sprint.score_push import fingerprint


PENALTIES = (0., .5, 1., 2.)
MIN_RECALL = .80
MACRO_TOLERANCE = .002


def candidate(rows, parity):
    if len(rows) != 8 or not all(row['matched_original'] for row in parity):
        return None
    control = {row['fold']: row for row in rows if row['lambda'] == 0}
    eligible = []
    for penalty in PENALTIES:
        pair = [row for row in rows if row['lambda'] == penalty]
        if all(row['clean_class55']['recall'] >= MIN_RECALL and
               row['perturbed_macro_f1'] >= control[row['fold']]['perturbed_macro_f1'] - MACRO_TOLERANCE
               for row in pair):
            eligible.append((np.mean([row['clean_class55']['f1'] for row in pair]), -penalty))
    return -max(eligible)[1] if eligible else None


def main():
    parser = argparse.ArgumentParser()
    for option in ('train', 'checkpoint', 'plan', 'output', 'original'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    parser.add_argument('--minutes', default=280., type=float)
    args = parser.parse_args()
    if args.minutes <= 0:
        parser.error('--minutes must be positive')
    raw, labels = load_training(args.train)
    ids, early = read_plan(args.plan, raw, labels, np.asarray(Views.from_texts(raw).phonetic))
    refs = {}
    for fold in (1, 2):
        original = args.original / f'fold{fold}-A2'
        finished = json.loads((original / 'complete.json').read_text(encoding='utf-8'))
        factor = finished['encoder_lr_factor']
        selection = json.loads((original / str(factor) / 'selection.json').read_text(encoding='utf-8'))
        refs[fold] = {'epochs': finished['epochs'], 'inner_perturbed': selection['inner_perturbed'],
                      'encoder_lr_factor': factor}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise SystemExit('Select an NVIDIA CUDA GPU')
    if any(fingerprint(args.checkpoint / name) != digest for name, digest in CHECKPOINT_HASHES.items()):
        raise ValueError('The Bhasha weights or tokenizer do not match the pinned files')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    gpu = torch.cuda.get_device_properties(0)
    print(f'{gpu.name}, {gpu.total_memory / 2**30:.2f} GiB, CUDA {torch.version.cuda}, fp16 AMP', flush=True)
    print('Original A2 selection:', json.dumps(refs), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'trainer': fingerprint(Path(fit.__code__.co_filename)), 'runner': fingerprint(Path(__file__)),
              'plan': fingerprint(args.plan), 'original_a2': refs, 'margin': FALLBACK_MARGIN,
              'lambdas': list(PENALTIES), 'min_clean_recall_each_fold': MIN_RECALL,
              'max_perturbed_macro_drop_each_fold': MACRO_TOLERANCE}
    marker = args.output / 'protocol.json'
    serialized = json.loads(json.dumps(config))
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != serialized:
        raise ValueError('The output folder belongs to a different experiment')
    write_json(marker, config)
    args.deadline = monotonic() + args.minutes * 60
    perturbed = augment_batch(raw, SEED + 2, 'natural')
    rows, parity = [], []
    for penalty in PENALTIES:
        for fold in (1, 2):
            root = args.output / f'fold{fold}-A2-lambda-{penalty:g}'
            result = root / 'selection.json'
            archive = root / 'inner_logits.npz'
            fit_rows = np.flatnonzero((ids != fold) & ~early[fold - 1])
            check = np.flatnonzero(early[fold - 1])
            if not result.exists():
                folder = root / 'inner' / 'checkpoints'
                validation = ([perturbed[i] for i in check], labels[check])
                try:
                    epoch, score, resume = fit(torch, args, 'A2', folder, raw, labels, fit_rows,
                                               validation, SEED + fold - 1, refs[fold]['encoder_lr_factor'],
                                               fallback_penalty=penalty, clean_validation=[raw[i] for i in check],
                                               best_logits=archive)
                except Pause:
                    print('Time budget reached. Rerun the same cell to resume.', flush=True)
                    return
                except Diverged as error:
                    row = {'fold': fold, 'lambda': penalty, 'error': str(error)}
                    write_json(args.output / 'failed.json', row)
                    raise
                with np.load(archive) as logits:
                    clean = counts(labels[check], logits['overall'].argmax(1))
                    repro = macro(labels[check], logits['perturbed'].argmax(1))
                    if logits['epoch'].item() != epoch or repro != score:
                        raise ValueError('Saved inner logits do not match early stopping')
                row = {'fold': fold, 'lambda': penalty, 'selected_refit_epochs': epoch,
                       'perturbed_macro_f1': score, 'clean_class55': clean,
                       'train_rows': len(fit_rows), 'validation_rows': len(check)}
                write_json(result, row)
                resume.unlink(missing_ok=True)
                gc.collect()
                torch.cuda.empty_cache()
            row = json.loads(result.read_text(encoding='utf-8'))
            rows.append(row)
            if penalty == 0:
                ref = refs[fold]
                parity.append({'fold': fold, 'matched_original': row['selected_refit_epochs'] == ref['epochs']
                               and row['perturbed_macro_f1'] == ref['inner_perturbed']})
            write_json(args.output / 'class55_penalty_inner.json',
                       {'margin': FALLBACK_MARGIN, 'complete': len(rows) == 8,
                        'pilot_evaluated': False, 'template_evaluated': False, 'rows': rows,
                        'baseline_parity': parity, 'proposed_lambda': candidate(rows, parity)})
            print('INNER_RESULT', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
