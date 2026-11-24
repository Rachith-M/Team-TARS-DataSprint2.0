import argparse
import json
import math
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from intent_model import PARAMETER_CAP
from sprint.augment import tta_variant
from sprint.data import SEED, load_data, write_submission
from sprint.encoder import fetch
from sprint.postprocess import average_tta, cluster_logits
from sprint.runtime import Runtime


def quantized(weights):
    scales = np.maximum(np.max(np.abs(weights), axis=0), 1e-8) / 127
    return np.round(weights / scales).clip(-127, 127).astype(np.int8), scales.astype(np.float32)


def attach_head(model, weights, bias, lexical_dim, embedding_weight, encoder):
    graph = model.graph
    opset = next(item.version for item in model.opset_import if item.domain in ('', 'ai.onnx'))
    axes = {}

    def constant(name, array):
        graph.initializer.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def node(op, inputs, output, **options):
        graph.node.append(helper.make_node(op, inputs, [output], **options))
        return output

    def with_axis(op, value, axis, output, **options):
        if opset < 13:
            return node(op, [value], output, axes=[axis], **options)
        if axis not in axes:
            axes[axis] = constant(f'sprint.axis{axis}', np.array([axis], dtype=np.int64))
        return node(op, [value, axes[axis]], output, **options)

    terms = []
    if lexical_dim:
        quant, scale = quantized(weights[:lexical_dim])
        graph.input.extend((helper.make_tensor_value_info('feature_ids', TensorProto.INT64, ['batch', 'nonzero']),
                            helper.make_tensor_value_info('feature_values', TensorProto.FLOAT, ['batch', 'nonzero'])))
        constant('sprint.lex_q', quant)
        constant('sprint.lex_scale', scale)
        q = node('Gather', ['sprint.lex_q', 'feature_ids'], 'sprint.active_q', axis=0)
        active = node('Cast', [q], 'sprint.active', to=TensorProto.FLOAT)
        values = with_axis('Unsqueeze', 'feature_values', 2, 'sprint.values')
        updates = node('Mul', [active, values], 'sprint.updates')
        partial = with_axis('ReduceSum', updates, 1, 'sprint.partial', keepdims=0)
        terms.append(node('Mul', [partial, 'sprint.lex_scale'], 'sprint.lex_logits'))
    if encoder:
        hidden = graph.output[0].name
        quant, scale = quantized(weights[lexical_dim:] * embedding_weight)
        constant('sprint.dense_q', quant)
        constant('sprint.dense_scale', scale)
        constant('sprint.eps', np.array(1e-8, dtype=np.float32))
        mask = with_axis('Unsqueeze', 'attention_mask', 2, 'sprint.mask3')
        mask = node('Cast', [mask], 'sprint.mask', to=TensorProto.FLOAT)
        masked = node('Mul', [hidden, mask], 'sprint.masked')
        pooled = with_axis('ReduceSum', masked, 1, 'sprint.pooled', keepdims=0)
        square = node('Mul', [pooled, pooled], 'sprint.square')
        sum_square = with_axis('ReduceSum', square, 1, 'sprint.sum_square')
        norm = node('Sqrt', [sum_square], 'sprint.norm')
        safe_norm = node('Max', [norm, 'sprint.eps'], 'sprint.safe_norm')
        normalized = node('Div', [pooled, safe_norm], 'sprint.normalized')
        dense = node('Cast', ['sprint.dense_q'], 'sprint.dense', to=TensorProto.FLOAT)
        products = node('MatMul', [normalized, dense], 'sprint.products')
        terms.append(node('Mul', [products, 'sprint.dense_scale'], 'sprint.dense_logits'))
    total = node('Add', terms, 'sprint.features_logits') if len(terms) == 2 else terms[0]
    constant('sprint.bias', bias)
    node('Add', [total, 'sprint.bias'], 'logits')
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 56]))
    return model


def export_cpu(source, output, config, cache):
    feature_vectors = joblib.load(source / 'features.joblib')
    lexical_dim = 0
    if config['lexical']:
        shutil.copyfile(source / 'features.joblib', output / 'features.joblib')
        lexical_dim = (feature_vectors.parameter_count() if config['lexical'] == 'cross-script'
                       else len(feature_vectors.vocabulary_))
    encoder = config['encoder']
    if encoder:
        registry = fetch(cache, encoder)
        model = onnx.load(registry / 'encoder.onnx')
        shutil.copyfile(registry / 'tokenizer.json', output / 'tokenizer.json')
    else:
        model = helper.make_model(helper.make_graph([], 'intent-routing', [], []),
                                  opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    with np.load(source / 'head.npz') as head:
        model = attach_head(model, head['weights'], head['bias'], lexical_dim,
                            config.get('embedding_weight', 1.), encoder)
    count = sum(math.prod(tensor.dims) for tensor in model.graph.initializer) + lexical_dim
    assert count <= PARAMETER_CAP, count
    onnx.checker.check_model(model)
    onnx.save(model, output / 'model.int8.onnx')
    return count


def export_gpu(source, output):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = source / 'checkpoint'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False).eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count <= PARAMETER_CAP
    inputs = tokenizer('warmup', return_tensors='pt')
    names = list(inputs.keys())

    class Logits(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, *values):
            return self.policy(**dict(zip(names, values))).logits

    target = output / 'model.int8.onnx'
    temporary = output / 'model.float.onnx'
    dynamic = {name: {0: 'batch', 1: 'tokens'} for name in names}
    dynamic['logits'] = {0: 'batch'}
    with torch.inference_mode():
        torch.onnx.export(Logits(model), tuple(inputs.values()), str(temporary), input_names=names,
                          output_names=['logits'], dynamic_axes=dynamic, opset_version=17, dynamo=False)
    quantize_dynamic(str(temporary), str(target), weight_type=QuantType.QInt8,
                     op_types_to_quantize=['MatMul', 'Gemm', 'Gather'], per_channel=True)
    onnx.checker.check_model(str(target))
    temporary.unlink()
    tokenizer.backend_tokenizer.save(str(output / 'tokenizer.json'))
    return count


def benchmark(runtime, queries):
    for text in queries[:10]:
        runtime.predict(text)
    durations = []
    for text in queries:
        started = perf_counter()
        runtime.predict(text)
        durations.append(1000 * (perf_counter() - started))
    p50, p95, p99, maximum = np.percentile(durations, [50, 95, 99, 100])
    return {'queries': len(queries), 'p50_ms': float(p50), 'p95_ms': float(p95),
            'p99_ms': float(p99), 'max_ms': float(maximum)}


def batch_submission_scores(runtime, texts, batch_size=32):
    def encode(variants, logits):
        unique = sorted(dict.fromkeys(text for text in variants if text not in logits), key=len)
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            logits.update(zip(batch, runtime.raw(batch)))

    logits = {}
    encode(texts, logits)
    raw = np.asarray([logits[text] for text in texts])
    weight = runtime.settings['tta_weight']
    if not weight:
        return raw
    variants = [tta_variant(text) for text in texts]
    encode(variants, logits)
    return average_tta(raw, np.asarray([logits[text] for text in variants]), weight)


def run(source, output, data, cache, threads):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
    config = report['config']
    gpu = (source / 'checkpoint').is_dir()
    count = export_gpu(source, output) if gpu else export_cpu(source, output, config, cache)
    settings = {'parameters': count, 'lexical': None if gpu else config['lexical'],
                'prefix': config.get('prefix', '') if gpu else ('query: ' if config['encoder'] else ''),
                'tta_weight': report['tta_weight'], 'cluster_weight': report['cluster_weight'],
                'scale': report['scale'], 'config_name': source.name}
    (output / 'runtime.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    if (source / 'live.json').is_file():
        shutil.copyfile(source / 'live.json', output / 'live.json')
    runtime = Runtime(output, threads=threads)
    texts, _, test, ids = load_data(data)
    rng = np.random.default_rng(SEED)
    from sprint.augment import augment_batch
    typical = [texts[i] for i in rng.choice(len(texts), size=260, replace=False)]
    shifted = augment_batch(typical[:60], SEED + 96, 'natural')
    long = sorted(test, key=len, reverse=True)[:30]
    latency = benchmark(runtime, typical + shifted + long)

    quantized_logits = batch_submission_scores(runtime, test)
    float_scores = np.load(source / 'scores.npz')
    original = average_tta(float_scores['test'], float_scores['test_tta'], report['tta_weight'])
    if report['cluster_weight']:
        quantized_logits = cluster_logits(test, quantized_logits, report['cluster_weight'])
        original = cluster_logits(test, original, report['cluster_weight'])
    predictions = quantized_logits.argmax(axis=1)
    agreement = float(np.mean(predictions == original.argmax(axis=1)))
    write_submission(output / 'submission.csv', ids, predictions)
    validation = {'config': source.name, 'parameters': count, 'latency': latency,
                  'onnx_size_bytes': (output / 'model.int8.onnx').stat().st_size,
                  'float_int8_agreement': agreement, 'test_fallback_predictions': int(np.sum(predictions == 55)),
                  'test_rows': len(test), 'cpu_threads': threads}
    (output / 'export_report.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(validation, indent=2))
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.source, args.output, args.data, args.cache, args.threads)


if __name__ == '__main__':
    main()
