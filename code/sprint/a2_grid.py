"""Inner-fold epoch / learning-rate grid for the fully fine-tuned Bhasha arm (A2).

Reuses the frozen grouped inner splits and the depth-ladder trainer unchanged, overriding only
the module-level schedule constants. Outer folds and the locked template holdout are not touched.
"""
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
import sprint.depth_ladder as ladder
from sprint.depth_ladder_report import write_json
from sprint.score_push import fingerprint


GRID = ((12, 2e-5), (12, 3e-5), (12, 5e-5))


def main():
    parser = argparse.ArgumentParser()
    for option in ('train', 'checkpoint', 'plan', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    parser.add_argument('--minutes', default=600., type=float)
    args = parser.parse_args()
    raw, labels = load_training(args.train)
    ids, early = ladder.read_plan(args.plan, raw, labels, np.asarray(Views.from_texts(raw).phonetic))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise SystemExit('Select an NVIDIA CUDA GPU')
    if any(fingerprint(args.checkpoint / name) != digest for name, digest in ladder.CHECKPOINT_HASHES.items()):
        raise ValueError('The Bhasha weights or tokenizer do not match the pinned files')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    gpu = torch.cuda.get_device_properties(0)
    print(f'{gpu.name}, {gpu.total_memory / 2**30:.2f} GiB, CUDA {torch.version.cuda}, fp16 AMP', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'trainer': fingerprint(Path(ladder.__file__)), 'runner': fingerprint(Path(__file__)),
              'plan': fingerprint(args.plan), 'grid': [list(g) for g in GRID], 'decay': ladder.DECAY,
              'patience': ladder.PATIENCE, 'head_lr': ladder.HEAD_LR}
    marker = args.output / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != json.loads(json.dumps(config)):
        raise ValueError('The output folder belongs to a different experiment')
    write_json(marker, config)
    args.deadline = monotonic() + args.minutes * 60
    perturbed = augment_batch(raw, SEED + 2, 'natural')
    base_lr = dict(ladder.LR)
    rows = []
    for epochs, rate in GRID:
        for fold in (1, 2):
            ladder.EPOCHS = epochs
            ladder.LR = {**base_lr, 'A2': rate}
            root = args.output / f'fold{fold}-A2-e{epochs}-lr{rate:g}'
            result = root / 'selection.json'
            archive = root / 'inner_logits.npz'
            fit_rows = np.flatnonzero((ids != fold) & ~early[fold - 1])
            check = np.flatnonzero(early[fold - 1])
            if not result.exists():
                validation = ([perturbed[i] for i in check], labels[check])
                try:
                    epoch, score, resume = ladder.fit(
                        torch, args, 'A2', root / 'inner' / 'checkpoints', raw, labels, fit_rows, validation,
                        SEED + fold - 1, 1., clean_validation=[raw[i] for i in check], best_logits=archive)
                except ladder.Pause:
                    print('Time budget reached. Attach this output and rerun to resume.', flush=True)
                    return
                with np.load(archive) as logits:
                    clean = logits['overall'].argmax(1)
                    if logits['epoch'].item() != epoch or macro(labels[check], logits['perturbed'].argmax(1)) != score:
                        raise ValueError('Saved inner logits do not match early stopping')
                row = {'fold': fold, 'max_epochs': epochs, 'encoder_lr': rate, 'selected_epoch': epoch,
                       'perturbed_macro_f1': score, 'clean_macro_f1': macro(labels[check], clean),
                       'clean_class55': counts(labels[check], clean),
                       'train_rows': len(fit_rows), 'validation_rows': len(check)}
                write_json(result, row)
                resume.unlink(missing_ok=True)
                gc.collect()
                torch.cuda.empty_cache()
            row = json.loads(result.read_text(encoding='utf-8'))
            rows.append(row)
            print('GRID_RESULT', json.dumps(row), flush=True)
            means = {}
            for e, r in GRID:
                pair = [x for x in rows if x['max_epochs'] == e and x['encoder_lr'] == r]
                if len(pair) == 2:
                    means[f'e{e}-lr{r:g}'] = {k: float(np.mean([x[k] for x in pair]))
                                              for k in ('perturbed_macro_f1', 'clean_macro_f1')}
            write_json(args.output / 'a2_grid_inner.json',
                       {'complete': len(rows) == 2 * len(GRID), 'outer_evaluated': False, 'rows': rows,
                        'means': means})


if __name__ == '__main__':
    main()
