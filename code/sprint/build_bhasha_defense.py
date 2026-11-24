"""Rebuild and verify the exact Bhasha dictionary submission model for the live defense."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import PARAMETER_CAP
from sprint.augment import tta_variant
from sprint.bhasha_audit import BASE_VIEWS, CHECKPOINT_REVISION, cached_vectors, dictionary_view, score_cpu
from sprint.data import SEED, load_data, read_csv, write_submission
from sprint.defense_artifacts import ASSETS, SUBMITTED_SHA256
from sprint.postprocess import average_tta
from sprint.runtime import Runtime
from sprint.score_bhasha import bundle, runtime_scores
from sprint.score_native import reverse_dictionary
from sprint.score_push import WEIGHT, fingerprint, fit_candidate, prepare
from sprint.train import take


SUBMISSION = Path('sprint_results/score_push/bhasha_dictionary_cpu/bhasha_dictionary_submission.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'AnveshanNLP'
    parser.add_argument('--cache', type=Path, default=root / 'bhasha-full-audit')
    parser.add_argument('--checkpoint', type=Path, default=root / 'bhasha-embed-v0-64b330ff')
    parser.add_argument('--output', type=Path, default=root / 'stage2-bhasha')
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--submission', type=Path, default=SUBMISSION)
    args = parser.parse_args()
    args.phrases = args.counterfactuals = True
    args.rewrites = None
    args.threads = 8
    args.batch_size = 64
    encoder = args.cache / f'bhasha-{CHECKPOINT_REVISION[:8]}-encoder.fp32.onnx'
    if args.output.exists():
        raise FileExistsError(args.output)
    if not encoder.is_file():
        raise FileNotFoundError(encoder)
    if fingerprint(args.submission) != SUBMITTED_SHA256:
        raise ValueError('The reference CSV differs from the submitted Bhasha CSV')

    groups, labels, _, _, _, _, config, _, variants, _ = prepare(args, include_reference_vectors=False)
    groups['asr_tta'] = list(map(tta_variant, groups['asr']))
    views = {name: groups[name] for name in (*BASE_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    vectors, _ = cached_vectors(args, views, 'base')
    print(f'Fitting the submitted recipe on {len(labels):,} labeled rows', flush=True)
    with threadpool_limits(limits=args.threads):
        model, counts = fit_candidate(config, groups, labels, vectors, variants, np.arange(len(labels)),
                                      np.array([], dtype=int), SEED + 5)
    del vectors
    dictionary = reverse_dictionary(groups['raw'])
    entries = sum(map(len, dictionary))
    _, _, test, ids = load_data(args.data)
    normalized = list(map(tta_variant, test))
    second = [dictionary_view(text, variant, dictionary) for text, variant in zip(test, normalized)]
    unique = list(dict.fromkeys(test + second))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix='bhasha-defense-build-') as temp:
        target = Path(temp) / 'model'
        parameters = bundle(target, encoder, model, args.checkpoint, 5.)
        if parameters + entries > PARAMETER_CAP:
            raise ValueError('The rebuilt bundle exceeds the parameter cap')
        del model
        runtime = Runtime(target, args.threads)
        print(f'Verifying all {len(test):,} submission predictions', flush=True)
        scores = dict(zip(unique, runtime_scores(runtime, unique)))
        logits = average_tta(np.stack([scores[text] for text in test]),
                             np.stack([scores[text] for text in second]), WEIGHT)
        predictions = logits.argmax(1)
        expected = read_csv(args.submission, ('example_id', 'target'))
        if ids != [row['example_id'] for row in expected]:
            raise ValueError('The rebuilt prediction order differs from the submitted CSV')
        np.testing.assert_array_equal(predictions, [int(row['target']) for row in expected])
        subset = np.random.default_rng(SEED + 48).choice(len(test), size=40, replace=False)
        live_logits = np.concatenate([score_cpu(runtime, [test[i]], [normalized[i]], dictionary, np.inf)
                                      for i in subset])
        np.testing.assert_array_equal(live_logits.argmax(1), predictions[subset])
        del runtime

        regenerated = Path(temp) / 'parity.csv'
        write_submission(regenerated, ids, predictions)
        csv_hash = fingerprint(regenerated)
        if csv_hash != SUBMITTED_SHA256 or fingerprint(args.submission) != SUBMITTED_SHA256:
            raise ValueError('The reproduced CSV bytes differ from the submitted CSV')
        (target / 'dictionary.json').write_text(json.dumps(dictionary, ensure_ascii=False, sort_keys=True) + '\n',
                                               encoding='utf-8')
        report = {'submission_sha256': csv_hash, 'submission_rows': len(ids), 'matching_predictions': len(ids),
                  'matching_batch_one_predictions': len(subset), 'model_parameters': parameters,
                  'dictionary_entries': entries, 'threads': args.threads, 'checkpoint_revision': CHECKPOINT_REVISION,
                  'training_rows': counts, 'train_sha256': fingerprint(args.data / 'train.csv'),
                  'assets': {name: fingerprint(target / name) for name in ASSETS}}
        (target / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        target.rename(args.output)
    print(f'Published locally at {args.output}. Verified {len(ids):,} matches and SHA-256 {csv_hash}.', flush=True)


if __name__ == '__main__':
    main()
