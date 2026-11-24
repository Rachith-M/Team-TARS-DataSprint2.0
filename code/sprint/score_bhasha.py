import argparse
import json
import math
import re
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import PARAMETER_CAP
from sprint.augment import augment_batch
from sprint.data import SEED, load_data, write_submission
from sprint.encoder import Encoder, tokenized
from sprint.encoder import embed_bhasha as embed
from sprint.export import attach_head, batch_submission_scores, benchmark
from sprint.runtime import Runtime
from sprint.score_push import compare, fit_candidate, metrics, prepare, score
from sprint.score_retrieval import rare_roman
from sprint.train import EVAL_VIEWS, fit_fold, take


class BhashaOnnx(Encoder):
    def encode_batch(self, texts):
        inputs = tokenized(texts, self.tokenizer, self.names, prefixed=False)
        hidden = self.runtime.run(None, inputs)[0]
        mask = inputs['attention_mask'][:, :, None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)


def embed_onnx(directory, views, threads):
    encoder = BhashaOnnx(directory, threads)
    texts = list(dict.fromkeys(text for view in views.values() for text in view))
    vectors = encoder.encode(texts, batch_size=64, name='int8')
    positions = {text: index for index, text in enumerate(texts)}
    return {view: vectors[[positions[text] for text in queries]] for view, queries in views.items()}


def run(args):
    groups, labels, original, challenge, _, scripts, config, reference, variants, saved = prepare(args)
    extra = {name: texts for name, texts, _, _, _ in variants}
    views = {name: groups[name] for name in (*EVAL_VIEWS, 'aug')}
    vectors, parameters = embed({**views, **extra}, args.checkpoint, args.device, args.batch_size)
    contrast = json.loads(args.contrast_report.read_text(encoding='utf-8'))
    if args.finish:
        results = json.loads(args.output.read_text(encoding='utf-8'))
        assert results['encoder_parameters'] == parameters and results['device'] == args.device
        plan = [(f'fold{index + 1}', split, SEED + index, contrast['folds'][index])
                for index, split in enumerate(original) if f'fold{index + 1}' not in results]
    else:
        results = {'encoder_parameters': parameters, 'device': args.device}
        plan = [('fold1', original[0], SEED, contrast['folds'][0]),
                ('template_holdout', challenge, SEED + 9, contrast['challenge'])]
    for name, (fit, check), seed, previous in plan:
        with threadpool_limits(limits=args.threads):
            model, training = fit_candidate(config, groups, labels, vectors, variants, fit, check, seed)
            candidate = score(model, groups, vectors, check)
            if name != 'template_holdout':
                baseline = {view: saved[view][check] for view in EVAL_VIEWS}
            else:
                control = fit_fold(config, groups, labels, reference, fit, seed, check=check)
                baseline = score(control, groups, reference, check)
        entry = compare(candidate, baseline, check, labels, scripts, training)
        entry['total_parameters'] = parameters + model.head.coef_.size + model.head.intercept_.size
        assert entry['total_parameters'] <= 300_000_000
        np.testing.assert_allclose(list(entry['baseline'].values()), list(previous['baseline'].values()), atol=.0005,
                                   rtol=0)
        entry['delta_vs_contrast'] = {metric: value - previous['candidate'][metric]
                                      for metric, value in entry['candidate'].items()}
        subset = rare_roman(groups, scripts, fit, check)
        entry['low_coverage_roman'] = compare({v: candidate[v][subset] for v in EVAL_VIEWS},
                                              {v: baseline[v][subset] for v in EVAL_VIEWS},
                                              check[subset], labels, scripts, len(fit))
        results[name] = entry
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(f"{name} vs submitted recipe: perturbed {entry['delta_vs_contrast']['perturbed']:+.5f}, "
              f"Roman ASR {entry['delta_vs_contrast']['roman_asr']:+.5f}", flush=True)
    if args.finish:
        folded = [results[f'fold{index + 1}'] for index in range(len(original))]
        averages = {key: {metric: float(np.mean([fold[key][metric] for fold in folded]))
                          for metric in folded[0][key]}
                    for key in ('baseline', 'candidate', 'delta_vs_contrast')}
        gains = [fold['delta_vs_contrast']['perturbed'] for fold in folded]
        results['five_fold_means'] = averages
        results['criteria'] = {
            'mean_perturbed_gain_ge_0.010': float(np.mean(gains)) >= .010,
            'four_perturbed_wins': sum(gain > 0 for gain in gains) >= 4,
            'worst_perturbed_gain_ge_minus_0.005': min(gains) >= -.005,
            'mean_class55_f1_ge_0.85': averages['candidate']['class55_f1'] >= .85,
        }
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


def bundle(directory, source, router, checkpoint, scale):
    import onnx

    directory.mkdir()
    width = router.features.parameter_count()
    graph = attach_head(onnx.load(source), router.head.coef_.T, router.head.intercept_,
                        width, router.config['embedding_weight'], True)
    count = sum(math.prod(weight.dims) for weight in graph.graph.initializer) + width
    if count > PARAMETER_CAP:
        raise ValueError(f'Exported model contains {count:,} parameters')
    onnx.checker.check_model(graph)
    onnx.save(graph, directory / 'model.int8.onnx')
    joblib.dump(router.features, directory / 'features.joblib', compress=3)
    copyfile(checkpoint / 'tokenizer.json', directory / 'tokenizer.json')
    settings = {'config_name': 'bhasha', 'parameters': count, 'prefix': '', 'lexical': 'cross-script',
                'scale': scale, 'tta_weight': .35, 'cluster_weight': 0.}
    (directory / 'runtime.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    return count


def runtime_scores(runtime, texts):
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    scores = np.empty((len(texts), 56), dtype=np.float32)
    for offset in range(0, len(order), 32):
        rows = order[offset:offset + 32]
        scores[rows] = runtime.raw(take(texts, rows))
    return scores


def float_control(args):
    import torch
    from sprint.muril_export import encoder_onnx

    torch.set_num_threads(args.threads)
    groups, labels, _, (fit, check), _, scripts, config, _, variants, _ = prepare(args)
    views = {name: groups[name] for name in (*EVAL_VIEWS, 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    reference, _ = embed(views, args.checkpoint, args.device, args.batch_size)
    with threadpool_limits(limits=args.threads):
        model, _ = fit_candidate(config, groups, labels, reference, variants, fit, check, SEED + 9)
        expected = score(model, groups, reference, check)
    previous = json.loads(args.contrast_report.read_text(encoding='utf-8'))['challenge']['candidate']
    torch_metrics = metrics(expected, labels[check], scripts[check])
    print(f"Matched PyTorch perturbation Macro-F1 {torch_metrics['perturbed']:.5f}", flush=True)
    rng = np.random.default_rng(SEED + 7)
    with TemporaryDirectory(prefix='sprint-bhasha-control-') as folder:
        scratch = Path(folder)
        encoder_path = scratch / 'encoder.onnx'
        encoder_onnx(args.checkpoint, encoder_path)
        copyfile(args.checkpoint / 'tokenizer.json', scratch / 'tokenizer.json')
        bundle(scratch / 'validation', encoder_path, model, args.checkpoint, 5.)
        warm = rng.choice(check, size=300, replace=False)
        sample = np.sort(rng.choice(len(check), size=512, replace=False))
        positions = check[sample]
        texts = take(groups['natural'], positions)
        plain = BhashaOnnx(scratch, args.threads)
        enc = plain.encode(texts, batch_size=32)
        del plain
        runtime = Runtime(scratch / 'validation', args.threads)
        inputs = take(groups['raw'], warm)
        latency = benchmark(runtime, inputs + augment_batch(inputs[:50], SEED + 37, 'asr'))
        joined = runtime_scores(runtime, texts)
        python_head = model.score(texts, enc)
        torch_head = expected['natural'][sample]
        summary = {
            'onnx_encoder_cosine_p05': float(np.quantile((enc * reference['natural'][positions]).sum(axis=1), .05)),
            'onnx_encoder_argmax_agreement': float(np.mean(python_head.argmax(axis=1) == torch_head.argmax(axis=1))),
            'int8_head_argmax_agreement': float(np.mean(joined.argmax(axis=1) == python_head.argmax(axis=1))),
        }
        result = {'torch': torch_metrics, 'latency': latency, 'diagnostic_rows': len(sample),
                  'parity': summary}
        print(json.dumps(result), flush=True)
        argsets = {view: take(groups[view], check) for view in EVAL_VIEWS}
        unique = list(dict.fromkeys(text for texts in argsets.values() for text in texts))
        combined = dict(zip(unique, runtime_scores(runtime, unique)))
        logits = {view: np.stack([combined[text] for text in texts]) for view, texts in argsets.items()}
        measured = metrics(logits, labels[check], scripts[check])
        result['onnx'] = measured
        result['delta_vs_torch'] = {metric: measured[metric] - torch_metrics[metric] for metric in measured}
        result['delta_vs_contrast'] = {metric: measured[metric] - previous[metric] for metric in measured}
        args.parity_report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        del runtime


def build_float(args):
    import torch
    from sprint.muril_export import encoder_onnx

    evidence = json.loads(args.float_evidence.read_text(encoding='utf-8'))
    if (evidence['delta_vs_contrast']['perturbed'] < .010
            or evidence['delta_vs_torch']['perturbed'] < -.005
            or evidence['latency']['p95_ms'] > 30 or evidence['latency']['max_ms'] >= 50
            or evidence['onnx']['class55_f1'] < .85):
        raise ValueError('The unquantized encoder control did not meet the deployment checks')
    torch.set_num_threads(args.threads)
    groups, labels, _, _, _, _, config, _, variants, _ = prepare(args)
    views = {name: groups[name] for name in ('raw', 'aug')}
    views.update({name: texts for name, texts, *_ in variants})
    vectors, _ = embed(views, args.checkpoint, args.device, args.batch_size)
    if args.device == 'xpu':
        torch.xpu.empty_cache()
    with threadpool_limits(limits=args.threads):
        model, counts = fit_candidate(config, groups, labels, vectors, variants, np.arange(len(labels)),
                                      np.array([], dtype=int), SEED + 5)
    del vectors
    args.build_float.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix='sprint-bhasha-build-') as folder:
        encoder_path = Path(folder) / 'encoder.onnx'
        encoder_onnx(args.checkpoint, encoder_path)
        location = args.build_float / 'candidate'
        count = bundle(location, encoder_path, model, args.checkpoint, 5.)
    runtime = Runtime(location, args.threads)
    rng = np.random.default_rng(SEED + 7)
    texts = take(groups['raw'], rng.choice(len(labels), size=300, replace=False))
    times = benchmark(runtime, texts + augment_batch(texts[:50], SEED + 37, 'asr'))
    report = {'parameters': count, 'precision': {'encoder': 'fp32', 'head': 'int8'},
              'latency': times, 'training': counts, 'validation_source': args.float_evidence.as_posix(),
              'passed': times['p95_ms'] <= 30 and times['max_ms'] < 50}
    (args.build_float / 'deployment.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    if report['passed']:
        _, _, test, ids = load_data(args.data)
        output = batch_submission_scores(runtime, test).argmax(axis=1)
        write_submission(args.build_float / 'bhasha_candidate_submission.csv', ids, output)


def quantize_embedding_rows(source, output):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(source)
    graph = model.graph
    table = max((weight for weight in graph.initializer if len(weight.dims) == 2),
                key=lambda weight: math.prod(weight.dims))
    users = [(index, node) for index, node in enumerate(graph.node) if table.name in node.input]
    if len(users) != 1 or users[0][1].op_type != 'Gather':
        raise ValueError('Expected a single word embedding gather')
    index, gather = users[0]
    values = numpy_helper.to_array(table)
    rows = np.empty(values.shape, dtype=np.int8)
    scales = np.empty((len(rows), 1), dtype=np.float32)
    for offset in range(0, len(rows), 2048):
        block = slice(offset, offset + 2048)
        scales[block] = np.maximum(np.abs(values[block]).max(axis=1, keepdims=True), 1e-8) / 127
        rows[block] = np.rint(values[block] / scales[block]).clip(-127, 127).astype(np.int8)
    indices, original = gather.input[1], gather.output[0]
    graph.initializer.remove(table)
    graph.initializer.extend((numpy_helper.from_array(rows, 'sprint.token_rows'),
                              numpy_helper.from_array(scales, 'sprint.token_scales')))
    gather.input[0] = 'sprint.token_rows'
    gather.output[0] = 'sprint.token_quantized'
    graph.node.insert(index + 1, helper.make_node('Gather', ['sprint.token_scales', indices],
                                                 ['sprint.token_scale'], axis=0))
    graph.node.insert(index + 2, helper.make_node('Cast', ['sprint.token_quantized'],
                                                 ['sprint.token_float'], to=TensorProto.FLOAT))
    graph.node.insert(index + 3, helper.make_node('Mul', ['sprint.token_float', 'sprint.token_scale'], [original]))
    onnx.checker.check_model(model)
    onnx.save(model, output)


def export(args):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sprint.muril_export import encoder_onnx
    import onnx

    prior = json.loads(args.output.read_text(encoding='utf-8'))
    if not all(prior['criteria'].values()):
        raise ValueError('The matched five-fold accuracy checks did not pass')
    args.export_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    groups, labels, _, (fit, check), _, scripts, config, reference, variants, _ = prepare(args)
    source_views = {name: groups[name] for name in ('raw', 'aug')}
    views = {**source_views, **{name: texts for name, texts, *_ in variants}}
    if not args.refit_int8:
        vectors, _ = embed(views, args.checkpoint, args.device, args.batch_size)
    seed = SEED + 9
    with threadpool_limits(limits=args.threads):
        control = fit_fold(config, groups, labels, reference, fit, seed, check=check)
        baseline = score(control, groups, reference, check)
    previous = json.loads(args.contrast_report.read_text(encoding='utf-8'))['challenge']['candidate']
    with TemporaryDirectory(prefix='sprint-bhasha-') as temp:
        scratch = Path(temp)
        encoder_onnx(args.checkpoint, scratch / 'fp32.onnx')
        compressed = scratch / ('layers_int8.onnx' if args.embedding_rows else 'encoder.onnx')
        operators = ['MatMul', 'Gemm'] if args.embedding_rows else ['MatMul', 'Gemm', 'Gather']
        selected = None
        if args.mixed_precision:
            graph = onnx.load(scratch / 'fp32.onnx').graph
            weights = {tensor.name for tensor in graph.initializer}
            selected = [node.name for node in graph.node if node.op_type in ('MatMul', 'Gemm')
                        and any(name in weights for name in node.input)
                        and re.search(r'layer[./][0-3](?:/|$)', node.name)]
            del graph
            if not selected:
                raise ValueError('Cannot identify the first four BERT layers in the exported graph')
            operators = ['MatMul', 'Gemm']
        quantize_dynamic(str(scratch / 'fp32.onnx'), str(compressed), weight_type=QuantType.QInt8,
                         per_channel=True, op_types_to_quantize=operators, nodes_to_quantize=selected,
                         reduce_range=args.refit_int8 or args.mixed_precision)
        encoded_path = scratch / 'encoder.onnx'
        if args.embedding_rows:
            quantize_embedding_rows(compressed, encoded_path)
        if args.refit_int8:
            copyfile(args.checkpoint / 'tokenizer.json', scratch / 'tokenizer.json')
            vectors = embed_onnx(scratch, views, max(4, args.threads))
        with threadpool_limits(limits=args.threads):
            model, training = fit_candidate(config, groups, labels, vectors, variants, fit, check, seed)
        count = bundle(scratch / 'validation', encoded_path, model, args.checkpoint, 5.)
        runtime = Runtime(scratch / 'validation', args.threads)
        rng = np.random.default_rng(SEED + 7)
        inputs = take(groups['raw'], rng.choice(check, size=300, replace=False))
        performance = benchmark(runtime, inputs + augment_batch(inputs[:50], SEED + 37, 'asr'))
        print(f"Quantized CPU latency p95 {performance['p95_ms']:.2f} ms", flush=True)
        quantized = {view: runtime_scores(runtime, take(groups[view], check)) for view in EVAL_VIEWS}
        del runtime
        comparison = compare(quantized, baseline, check, labels, scripts, training)
        measured = comparison['candidate']
        checks = {'p95_le_30_ms': performance['p95_ms'] <= 30,
                  'perturbed_gain_ge_0.010': measured['perturbed'] - previous['perturbed'] >= .010,
                  'quantization_loss_le_0.005': measured['perturbed'] >=
                  prior['template_holdout']['candidate']['perturbed'] - .005,
                  'class55_f1_ge_0.85': measured['class55_f1'] >= .85}
        result = {'parameters': count, 'latency': performance, 'validation': comparison,
                  'embedding_quantization': ('float32' if args.mixed_precision else
                                             'per_row' if args.embedding_rows else 'per_tensor'),
                  'mixed_precision': args.mixed_precision,
                  'head_fit_on_int8': args.refit_int8,
                  'delta_vs_contrast': {key: measured[key] - previous[key] for key in measured},
                  'checks': checks}
        report = args.export_dir / 'deployment.json'
        report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        if all(checks.values()):
            with threadpool_limits(limits=args.threads):
                full, _ = fit_candidate(config, groups, labels, vectors, variants, np.arange(len(labels)),
                                        np.array([], dtype=int), SEED + 5)
            bundle(args.export_dir / 'candidate', encoded_path, full, args.checkpoint, 5.)
            deployed = Runtime(args.export_dir / 'candidate', args.threads)
            _, _, test, ids = load_data(args.data)
            predictions = batch_submission_scores(deployed, test).argmax(axis=1)
            write_submission(args.export_dir / 'bhasha_candidate_submission.csv', ids, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--contrast-report', type=Path, default=Path('sprint_results/score_push/contrast/results.json'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/score_push/bhasha.json'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--finish', action='store_true')
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--parity-report', type=Path)
    parser.add_argument('--build-float', type=Path)
    parser.add_argument('--float-evidence', type=Path,
                        default=Path('sprint_results/score_push/bhasha_float_control.json'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--embedding-rows', action='store_true')
    modes.add_argument('--refit-int8', action='store_true')
    modes.add_argument('--mixed-precision', action='store_true')
    parser.set_defaults(rewrites=None, phrases=True, counterfactuals=True)
    args = parser.parse_args()
    if args.build_float:
        build_float(args)
    elif args.parity_report:
        float_control(args)
    elif args.export_dir:
        export(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
