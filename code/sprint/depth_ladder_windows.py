import argparse
import json
import math
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
from scipy.special import softmax

from intent_model import FALLBACK, PARAMETER_CAP, Views, script
from sprint.augment import tta_variant
from sprint.bhasha_audit import dictionary_view
from sprint.data import SEED, load_training, macro, read_csv
from sprint.defense import MAX_CHARS, MODEL, POLICY, WEB, DefenseRuntime
from sprint.defense_artifacts import MAX_TWO_VIEW_TOKENS
from sprint.defense_policy import AUDIT
from sprint.defense_rehearsal import timing
from sprint.depth_ladder import SEMANTIC_WEIGHT, WEIGHT, ViewBatches, combine, model_for, read_plan, validation_views
from sprint.depth_ladder_report import write_json
from sprint.encoder import variant_groups
from sprint.export import attach_head
from sprint.gate_experiment import specificity_threshold
from sprint.gate_runtime import actions
from sprint.muril_export import encoder_onnx
from sprint.postprocess import logit_scale
from sprint.runtime import Runtime
from sprint.score_push import fingerprint
from sprint.stage2_fixed import supplied


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export(checkpoint, output):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output.mkdir()
    head = np.load(checkpoint / 'head.npz')
    lexical_dim = len(head['lexical'])
    weights = np.concatenate((head['lexical'], head['semantic'].T))
    target = output / 'model.int8.onnx'
    with TemporaryDirectory(dir=output) as directory:
        float_encoder = Path(directory) / 'encoder.onnx'
        quant_encoder = float_encoder.with_name('encoder.int8.onnx')
        tokenizer = encoder_onnx(checkpoint, float_encoder)
        quantize_dynamic(str(float_encoder), str(quant_encoder), weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul', 'Gemm'], per_channel=True)
        graph = attach_head(onnx.load(quant_encoder), weights, head['bias'], lexical_dim, SEMANTIC_WEIGHT, True)
        tensor_count = sum(math.prod(tensor.dims) for tensor in graph.graph.initializer)
        onnx.checker.check_model(graph)
        onnx.save(graph, target)
    head.close()
    tokenizer.backend_tokenizer.save(str(output / 'tokenizer.json'))
    for name in ('features.joblib', 'dictionary.json'):
        shutil.copyfile(checkpoint / name, output / name)
    counts = {'onnx_initializer_elements': tensor_count, 'lexical_vocabulary_entries': lexical_dim,
              'dictionary_entries': sum(map(len, read_json(output / 'dictionary.json')))}
    counts['complete_pipeline_count'] = sum(counts.values())
    if counts['complete_pipeline_count'] > PARAMETER_CAP:
        raise ValueError(f'The int8 pipeline exceeds the parameter cap: {counts}')
    write_json(output / 'runtime.json', {'parameters': counts['complete_pipeline_count'], 'lexical': 'cross-script',
               'prefix': '', 'tta_weight': WEIGHT, 'cluster_weight': 0., 'scale': 1., 'count_breakdown': counts})
    return counts


def float_predictions(checkpoint, texts):
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    features = joblib.load(checkpoint / 'features.joblib')
    dictionary = read_json(checkpoint / 'dictionary.json')
    batches = ViewBatches(texts, dictionary, tokenizer, features)
    model = model_for(torch, checkpoint, features.parameter_count(), 'A0', SEED, 'cpu').eval()
    predictions = np.empty(len(texts), dtype=np.int64)
    with np.load(checkpoint / 'head.npz') as head, torch.inference_mode():
        model.lexical.weight.copy_(torch.from_numpy(head['lexical']))
        model.semantic.weight.copy_(torch.from_numpy(head['semantic']))
        model.semantic.bias.copy_(torch.from_numpy(head['bias']))
        for batch in batches.batches():
            tokens, ids, values, _ = batches.inputs(torch, batch, 'cpu')
            predictions[batch] = model(tokens, ids, values).argmax(1).numpy()
    return predictions


class CandidateDemo(DefenseRuntime):
    def __init__(self, output, policy, threads):
        self.base = Runtime(output, threads)
        self.policy = policy
        self.dictionary = read_json(output / 'dictionary.json')
        self.names = read_json(WEB / 'intent_names.json')

    def route(self, text):
        result = self.predict(text)
        return {'intent_id': result['intent_id'], 'confidence': result['top_confidence'], 'action': result['action']}


def batch_logits(candidate, texts, live=False):
    result = np.empty((len(texts), FALLBACK + 1), dtype=np.float32)
    for start in range(0, len(texts), 16):
        original = texts[start:start + 16]
        alternate = [dictionary_view(text, tta_variant(text), candidate.dictionary) for text in original]
        flat = [text for pair in zip(original, alternate) for text in pair]
        encoded = candidate.base.encoder_inputs(flat)
        logits = candidate.base.raw(flat, encoded=encoded)
        averaged = combine(logits)
        if live:
            long = encoded['attention_mask'].sum(1).reshape(-1, 2).max(1) > MAX_TWO_VIEW_TOKENS
            averaged[long] = logits[::2][long]
        result[start:start + len(original)] = averaged
    return result


def risks(logits, policy):
    return 1 - softmax(logits.astype(np.float64) * policy['model_logit_scale'], axis=1).max(1)


def route_logits(logits, policy):
    return actions(risks(logits, policy), logits.argmax(1), policy['rule'], True)


def shares(routed):
    return {action: float(np.mean(routed == action)) for action in ('reject', 'clarify', 'accept')}


def assess(oof, test, unknown, labels, scripts, held, candidate_policy, baseline):
    correct = held & (labels != FALLBACK) & (oof.argmax(1) == labels)
    routed = route_logits(oof, candidate_policy)
    known = {tag: {'correct_rows': int(mask.sum()), **shares(routed[mask])}
             for tag, mask in [('overall', correct), *((tag, correct & (scripts == tag)) for tag in np.unique(scripts))]
             if mask.any()}
    test_shares = shares(route_logits(test, candidate_policy))
    unknown_shares = shares(route_logits(unknown, candidate_policy))
    checks = {'supported_correct_reject_0_015_to_0_045': .015 <= known['overall']['reject'] <= .045,
              'supported_correct_reject_per_script_le_0_05': len(known) == 1 + len(np.unique(scripts)) and
              all(value['reject'] <= .05 for tag, value in known.items() if tag != 'overall'),
              'test_reject_increase_le_0_03': test_shares['reject'] - baseline['test']['reject'] <= .03,
              'unsupported_reject_clarify_within_0_05': abs(unknown_shares['accept'] - baseline['unsupported']['accept']) <= .05}
    return {'supported_correct': known, 'test': test_shares, 'unsupported': unknown_shares,
            'checks': checks, 'passed': all(checks.values())}


def gate_checks(args, winner, labels, scripts, ids, test, unknown):
    baseline = DefenseRuntime(args.reference_model, args.policy, args.threads)
    reference = {'test': shares(np.asarray([baseline.predict(text)['action'] for text in test])),
                 'unsupported': shares(np.asarray([baseline.predict(text)['action'] for text in unknown]))}
    policy = baseline.policy
    del baseline
    oof = np.empty((len(labels), FALLBACK + 1), dtype=np.float32)
    for fold in range(1, 6):
        rows = np.flatnonzero(ids == fold)
        with np.load(args.results / f'fold{fold}-{winner}' / 'logits.npz') as archive:
            if not np.array_equal(archive['rows'], rows):
                raise ValueError('Candidate OOF files disagree with registration')
            oof[rows] = archive['live']
    with np.load(args.reference_oof) as archive:
        if not np.array_equal(archive['labels'], labels) or not np.array_equal(archive['fold_id'] + 1, ids):
            raise ValueError('Reference OOF files disagree with registration')
        reference_oof = archive['always_dictionary_overall']
    candidate = CandidateDemo(args.output / 'final', policy, args.threads)
    test_scores = batch_logits(candidate, test, live=True)
    unknown_scores = batch_logits(candidate, unknown, live=True)
    held = ids > 3
    correct = ~held & (labels != FALLBACK) & (labels == oof.argmax(1))
    reference_correct = ~held & (labels != FALLBACK) & (labels == reference_oof.argmax(1))
    median = float(np.median(1 - risks(oof[correct], policy)))
    reference_median = float(np.median(1 - risks(reference_oof[reference_correct], policy)))
    report = {'baseline': reference, 'candidate_correct_calibration_median': median,
              'reference_correct_calibration_median': reference_median, 'sharpened': median > reference_median,
              'fixed': assess(oof, test_scores, unknown_scores, labels, scripts, held, policy, reference),
              'recalibrated': None}
    if report['sharpened']:
        scale = logit_scale(oof[~held], labels[~held])
        risk = risks(oof[correct], {'model_logit_scale': scale})
        direct = np.zeros(len(risk), dtype=bool)
        candidate.policy = {'model_logit_scale': scale, 'rule': {'threshold': specificity_threshold(risk, direct, .03),
                            'accept_below': specificity_threshold(risk, direct, .07)}}
        report['recalibrated'] = assess(oof, test_scores, unknown_scores, labels, scripts, held, candidate.policy, reference)
    report['passed'] = (report['recalibrated'] if report['sharpened'] else report['fixed'])['passed']
    return candidate, report


def latency(candidate, raw):
    variants = variant_groups(raw, [])
    chosen = np.random.default_rng(SEED + 7).choice(len(raw), size=200, replace=False)
    queries = [variants[name][i] for name, size in (('raw', 200), ('natural', 150), ('asr', 50)) for i in chosen[:size]]
    for query in queries[:24]:
        candidate.route(query)
    rng = np.random.default_rng(SEED + 48)
    times = []
    for _ in range(2):
        for i in rng.permutation(len(queries)):
            start = perf_counter()
            candidate.route(queries[i])
            times.append((perf_counter() - start) * 1000)
    return timing(times, 45.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('sprint_results/depth_ladder/plan.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=Path, default=Path('train.csv'))
    parser.add_argument('--test', type=Path, default=Path('test.csv'))
    parser.add_argument('--unsupported', type=Path, default=Path('unsupported_eval_queries.csv'))
    parser.add_argument('--reference-model', type=Path, default=MODEL)
    parser.add_argument('--reference-oof', type=Path, default=AUDIT)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    pilot = read_json(args.results / 'pilot.json')
    grouped = read_json(args.results / 'grouped.json')
    template = read_json(args.results / 'template.json')
    winner = pilot['winner']
    if pilot['status'] != 'go' or not grouped['passed'] or not template['passed'] or not template['final_floor']:
        raise SystemExit('The candidate has not cleared full promotion and the 0.885 locked-template floor')
    if winner != template['arm'] or fingerprint(args.plan) != read_json(args.results / 'protocol.json')['plan_sha256']:
        raise SystemExit('The promotion files disagree with registration')
    final = args.results / f'final-{winner}' / 'checkpoint'
    holdout = args.results / f'template-{winner}'
    if not (final.parent / 'complete.json').is_file():
        raise SystemExit('The promoted candidate needs its completed all-training refit')
    raw, labels = load_training(args.train)
    test = [row['query_text'] for row in read_csv(args.test, ('example_id', 'query_text'))]
    proxy_text, _, _, _, proxy_split, proxy_sha = supplied(args.unsupported)
    unknown = [text for text, split in zip(proxy_text, proxy_split) if split == 'evaluation']
    if len(test) != 4912 or len(unknown) != 80 or any(len(text) > MAX_CHARS for text in test + unknown):
        raise ValueError('The evaluation files do not match the registered row counts or length bounds')
    keys = np.asarray(Views.from_texts(raw).phonetic)
    scripts = np.asarray([script(text) for text in raw])
    ids, _ = read_plan(args.plan, raw, labels, keys)
    args.output.mkdir(parents=True)
    import torch

    torch.set_num_threads(args.threads)
    policy = read_json(args.policy)
    counts = export(final, args.output / 'final')
    export(holdout / 'checkpoint', args.output / 'template')
    with np.load(holdout / 'logits.npz') as report:
        check = report['rows']
    perturbed = validation_views(raw)['perturbed']
    queries = [perturbed[i] for i in check]
    template_model = CandidateDemo(args.output / 'template', policy, args.threads)
    fp32_score = macro(labels[check], float_predictions(holdout / 'checkpoint', queries))
    int8_score = macro(labels[check], batch_logits(template_model, queries).argmax(1))
    del template_model
    candidate, gate = gate_checks(args, winner, labels, scripts, ids, test, unknown)
    agreement = float(np.mean(float_predictions(final, test) == batch_logits(candidate, test).argmax(1)))
    speed = latency(candidate, raw)
    checks = {'parameters_le_300m': counts['complete_pipeline_count'] <= PARAMETER_CAP,
              'cpu_p95_and_max_le_45_ms': speed['p95_ms'] <= 45 and speed['max_ms'] <= 45,
              'test_agreement_ge_0_99': agreement >= .99, 'perturbed_macro_f1_within_0_003': abs(int8_score - fp32_score) <= .003,
              'fixed_or_recalibrated_gate_passed': gate['passed']}
    output = {'checks': checks, 'passed': all(checks.values()), 'arm': winner, 'gate': gate,
              'template_fp32_macro_f1': fp32_score, 'template_int8_macro_f1': int8_score,
              'test_agreement': agreement, 'test_rows': len(test), 'unsupported_rows': len(unknown),
              'unsupported_proxy_sha256': proxy_sha, 'parameters': counts,
              'latency': speed, 'threads': args.threads}
    write_json(args.output / 'windows_checks.json', output)
    if output['passed']:
        write_json(args.output / 'final' / 'candidate_policy.json', candidate.policy)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
