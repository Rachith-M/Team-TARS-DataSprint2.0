"""Audit the Stage 2 fallback protocol on the locked, organizer-labeled template holdout."""

import json

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer

from intent_model import FALLBACK
from sprint.augment import tta_variant
from sprint.bhasha_audit import BASE_VIEWS, REQUEST_VIEWS, cached_vectors, inner_partition, score_parts
from sprint.bhasha_bias import int8_head
from sprint.bhasha_token_cache import parser
from sprint.bhasha_token_pilot import cached_second_views
from sprint.data import SEED
from sprint.defense_artifacts import MAX_TWO_VIEW_TOKENS
from sprint.defense_policy import evaluate
from sprint.encoder import MAX_LEN
from sprint.gate_experiment import specificity_threshold
from sprint.postprocess import logit_scale
from sprint.score_push import fingerprint, fit_candidate, prepare
from sprint.train import take


def with_length_guard(parts, groups, indices, second, tokenizer):
    _, dictionary_logits, _, _, raw = parts
    changed, native, _ = second
    results, forced = {}, {}
    for view, (first, normalized) in REQUEST_VIEWS.items():
        orig = take(groups[first], indices)
        alternate = take(groups[normalized], indices)
        for i, text in zip(changed[view], native[view]):
            alternate[i] = text
        lengths = np.fromiter((len(tokens.ids) for tokens in tokenizer.encode_batch(orig + alternate)), int)
        width = np.maximum(*np.split(lengths, 2))
        guard = (width > MAX_TWO_VIEW_TOKENS) & np.array([a != b for a, b in zip(orig, alternate)])
        results[view] = np.where(guard[:, None], raw[first], dictionary_logits[view])
        forced[view] = int(guard.sum())
    return results, forced


def main():
    from pathlib import Path

    options = parser()
    options.set_defaults(split='template')
    options.add_argument('--output', type=Path, required=True)
    options.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    args = options.parse_args()
    if args.split != 'template' or args.output.exists():
        options.error('Use the locked template split and a new output path')
    args.phrases = args.counterfactuals = True
    args.rewrites = None
    groups, labels, folds, challenge, frames, scripts, config, _, variants, _ = prepare(
        args, include_reference_vectors=False)
    groups['asr_tta'] = list(map(tta_variant, groups['asr']))
    fit, heldout = challenge
    seed = SEED + 9
    train, calibrate = inner_partition(fit, labels, scripts, frames, seed)
    if (set(frames[heldout]) & set(frames[fit]) or set(frames[train]) & set(frames[calibrate])):
        raise ValueError('Template groups overlap between a fit and evaluation partition')
    views = {view: groups[view] for view in (*BASE_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    vectors, _ = cached_vectors(args, views, 'base')
    second = cached_second_views(args, groups, labels, folds, challenge, frames, scripts)
    tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    tokenizer.enable_truncation(MAX_LEN)
    tokenizer.no_padding()
    results, length_guard, unguarded = {}, {}, {}
    for name, fitting, blocked, check in (('inner', train, np.r_[calibrate, heldout], calibrate),
                                          ('outer', fit, heldout, heldout)):
        print(f'Fitting {name} Bhasha model on {len(fitting):,} organizer-labeled queries', flush=True)
        with threadpool_limits(limits=args.threads):
            model, _ = fit_candidate(config, groups, labels, vectors, variants, fitting, blocked, seed)
            parts = score_parts(int8_head(model), groups, vectors, check, *second[name])
        unguarded[name] = parts[1]
        results[name], length_guard[name] = with_length_guard(parts, groups, check, second[name], tokenizer)
    clean = results['inner']['overall']
    y = labels[calibrate]
    scale = logit_scale(clean, y)
    correct = (clean.argmax(1) == y) & (y != FALLBACK)
    risk = 1 - softmax(clean[correct].astype(np.float64) * scale, axis=1).max(1)
    direct = np.zeros(len(risk), dtype=bool)
    rule = {'threshold': specificity_threshold(risk, direct, .03),
            'accept_below': specificity_threshold(risk, direct, .07)}
    baseline = {'threshold': None, 'accept_below': None}
    test_labels = labels[heldout]
    groups_to_report = {'all': np.ones(len(heldout), dtype=bool),
                        'roman': scripts[heldout] == 'roman',
                        'devanagari': scripts[heldout] == 'devanagari',
                        'mixed': scripts[heldout] == 'mixed'}
    score_report = {view: {group: {'direct_class55': evaluate(matrix, test_labels, mask, scale, baseline),
                                  'live_fallback': evaluate(matrix, test_labels, mask, scale, rule)}
                           for group, mask in groups_to_report.items()}
                    for view, matrix in results['outer'].items()}
    report = {'protocol': 'Locked template-group holdout. Scale and confidence bands fit on a separate '
                          'inner partition of the outer fit rows. Matched Bhasha with simulated export int8 head.',
              'train_sha256': fingerprint(args.data / 'train.csv'), 'outer_fit_rows': len(fit),
              'inner_fit_rows': len(train), 'calibration_rows': len(calibrate), 'holdout_rows': len(heldout),
              'holdout_template_groups': len(set(frames[heldout])), 'scale': scale, 'rule': rule,
              'max_two_view_tokens': MAX_TWO_VIEW_TOKENS, 'length_guard_rows': length_guard,
              'changed_holdout_argmax_due_to_length_guard': {view: int((matrix.argmax(1) !=
                   unguarded['outer'][view].argmax(1)).sum()) for view, matrix in results['outer'].items()},
              'holdout': score_report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps({name: views['all'] for name, views in score_report.items()}, indent=2))


if __name__ == '__main__':
    main()
