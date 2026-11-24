import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from intent_model import PARAMETER_CAP
from sprint.augment import augment_batch
from sprint.data import SEED, load_data, write_submission
from sprint.runtime import Runtime


def save_candidate(directory, model, tokenizer, features, parameters, weight, scale):
    candidate = directory / 'candidate'
    candidate.mkdir(parents=True, exist_ok=True)
    model.cpu()
    model.encoder.save_pretrained(candidate / 'checkpoint')
    tokenizer.save_pretrained(candidate / 'checkpoint')
    joblib.dump(features, candidate / 'features.joblib', compress=3)
    np.savez_compressed(candidate / 'head.npz', lexical=model.lexical.weight.detach().numpy(),
                        semantic=model.semantic.weight.detach().numpy().T, bias=model.semantic.bias.detach().numpy())
    settings = {'config_name': 'muril_paired', 'lexical': 'cross-script', 'prefix': '', 'cluster_weight': 0.,
                'parameters': parameters, 'tta_weight': weight, 'scale': scale}
    (candidate / 'runtime.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')


def encoder_onnx(checkpoint, filename):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(checkpoint, add_pooling_layer=False, attn_implementation='eager',
                                      local_files_only=True, trust_remote_code=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    inputs = tokenizer('warmup', return_tensors='pt')
    names = list(inputs)

    class HiddenStates(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, *values):
            return self.encoder(**dict(zip(names, values))).last_hidden_state

    dynamic = {name: {0: 'batch', 1: 'tokens'} for name in names + ['hidden_states']}
    with torch.inference_mode():
        torch.onnx.export(HiddenStates(model), tuple(inputs.values()), str(filename), input_names=names,
                          output_names=['hidden_states'], dynamic_axes=dynamic, opset_version=17, dynamo=False)
    return tokenizer


def run(experiment, data, threads):
    from shutil import copyfile

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from sprint.export import attach_head, batch_submission_scores, benchmark
    from sprint.muril import SEMANTIC_WEIGHT, json_file

    report_path = experiment / 'experiment.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['status'] not in ('accuracy_passed', 'deployment_failed', 'passed') or not all(report['checks'].values()):
        raise RuntimeError('All five-fold accuracy checks must pass before exporting a candidate')
    source = experiment / 'candidate'
    out = experiment / 'int8'
    out.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=experiment, prefix='onnx-') as temporary:
        fp32_model = Path(temporary) / 'encoder.onnx'
        int8_model = Path(temporary) / 'encoder.int8.onnx'
        tokenizer = encoder_onnx(source / 'checkpoint', fp32_model)
        quantize_dynamic(str(fp32_model), str(int8_model), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=['MatMul', 'Gemm', 'Gather'])
        model = onnx.load(int8_model)
        with np.load(source / 'head.npz') as head:
            width = head['lexical'].shape[0]
            weights = np.concatenate([head['lexical'], head['semantic']])
            model = attach_head(model, weights, head['bias'], width, SEMANTIC_WEIGHT, True)
        count = sum(math.prod(value.dims) for value in model.graph.initializer) + width
        if count > PARAMETER_CAP:
            raise ValueError(f'Exported MuRIL hybrid exceeds parameter cap ({count:,})')
        onnx.checker.check_model(model)
        onnx.save(model, out / 'model.int8.onnx')
        tokenizer.backend_tokenizer.save(str(out / 'tokenizer.json'))
    del model
    copyfile(source / 'features.joblib', out / 'features.joblib')
    settings = json.loads((source / 'runtime.json').read_text(encoding='utf-8'))
    settings['parameters'] = count
    json_file(out / 'runtime.json', settings)
    runtime = Runtime(out, threads)
    texts, _, test, ids = load_data(data)
    rng = np.random.default_rng(SEED)
    queries = [texts[i] for i in rng.choice(len(texts), size=260, replace=False)]
    queries += augment_batch(queries[:60], SEED + 96, 'natural') + sorted(test, key=len, reverse=True)[:30]
    performance = benchmark(runtime, queries)
    passes = performance['p95_ms'] <= 30 and performance['max_ms'] < 50
    report['deployment'] = {'parameters': count, 'cpu_threads': threads, 'latency': performance,
                            'passed': passes}
    if passes:
        logits = batch_submission_scores(runtime, test)
        write_submission(out / 'submission.csv', ids, logits.argmax(axis=1))
    report['status'] = 'passed' if passes else 'deployment_failed'
    json_file(report_path, report)
    print(json.dumps(report['deployment'], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.experiment, args.data, args.threads)


if __name__ == '__main__':
    main()
