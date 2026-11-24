import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.special import softmax

from intent_model import FALLBACK, PARAMETER_CAP, Views
from sprint.augment import tta_variant
from sprint.encoder import MAX_LEN, session, tokenized
from sprint.live import route


def pack_lexical(matrix):
    sizes = np.diff(matrix.indptr)
    shape = (matrix.shape[0], max(1, int(sizes.max())))
    ids = np.zeros(shape, dtype=np.int64)
    values = np.zeros(shape, dtype=np.float32)
    for row, (start, stop) in enumerate(zip(matrix.indptr[:-1], matrix.indptr[1:])):
        ids[row, :sizes[row]] = matrix.indices[start:stop]
        values[row, :sizes[row]] = matrix.data[start:stop]
    return ids, values


class Runtime:
    def __init__(self, directory, threads=2, expose_embedding=False):
        directory = Path(directory)
        self.settings = json.loads((directory / 'runtime.json').read_text(encoding='utf-8'))
        assert self.settings['parameters'] <= PARAMETER_CAP
        live = directory / 'live.json'
        self.gate = json.loads(live.read_text(encoding='utf-8'))['gate'] if live.exists() else None
        model = directory / 'model.int8.onnx'
        if expose_embedding:
            import onnx
            graph = onnx.load(model)
            graph.graph.output.append(onnx.helper.make_tensor_value_info(
                'sprint.normalized', onnx.TensorProto.FLOAT, ['batch', 384],
            ))
            onnx.checker.check_model(graph)
            model = graph.SerializeToString()
        self.runtime = session(model, threads)
        self.input_names = [item.name for item in self.runtime.get_inputs()]
        self.tokenizer = None
        self.features = None
        if 'input_ids' in self.input_names:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
            self.tokenizer.enable_truncation(max_length=MAX_LEN)
            pad_id = next((self.tokenizer.token_to_id(token) for token in ('<pad>', '[PAD]')
                           if self.tokenizer.token_to_id(token) is not None), 0)
            self.tokenizer.enable_padding(pad_id=pad_id)
        if 'feature_ids' in self.input_names:
            self.features = joblib.load(directory / 'features.joblib')

    def encoder_inputs(self, texts):
        if self.tokenizer is None:
            return {}
        names = [name for name in self.input_names if name in ('input_ids', 'attention_mask', 'token_type_ids')]
        prefix = self.settings.get('prefix', '')
        return tokenized([prefix + text for text in texts], self.tokenizer, names, prefixed=False)

    def raw(self, texts, with_embedding=False, *, encoded=None):
        inputs = self.encoder_inputs(texts) if encoded is None else dict(encoded)
        if self.features is not None:
            matrix = (self.features.encode(Views.from_texts(texts)) if self.settings['lexical'] == 'cross-script'
                      else self.features.transform(texts))
            inputs['feature_ids'], inputs['feature_values'] = pack_lexical(matrix)
        if with_embedding:
            return tuple(self.runtime.run(['logits', 'sprint.normalized'], inputs))
        return self.runtime.run(['logits'], inputs)[0]

    def score(self, texts, tta=True, with_embedding=False):
        raw, embedding = self.raw(texts, with_embedding=True) if with_embedding else (self.raw(texts), None)
        weight = self.settings['tta_weight'] if tta else 0
        if not weight:
            return (raw, embedding) if with_embedding else raw
        variants = [tta_variant(text) for text in texts]
        changed = [index for index, (text, variant) in enumerate(zip(texts, variants)) if text != variant]
        if changed:
            raw[changed] = (1 - weight) * raw[changed] + weight * self.raw([variants[i] for i in changed])
        return (raw, embedding) if with_embedding else raw

    def predict(self, text, live=True):
        logits = self.score([text])
        probabilities = softmax(logits[0] * self.settings['scale'])
        chosen = int(route(logits, self.gate if live else None)[0]) if text.strip() else FALLBACK
        high = int(probabilities.argmax())
        return {'intent_id': chosen, 'confidence': float(probabilities[chosen]),
                'fallback': chosen == FALLBACK, 'top_model_intent': high}


def response(runtime, text, live):
    start = perf_counter()
    result = runtime.predict(text, live)
    result['latency_ms'] = round(1000 * (perf_counter() - start), 2)
    if result.get('action') == 'clarify':
        result['message'] = 'Could you rephrase that request?'
    elif result['fallback']:
        result['message'] = "I don't know how to do that."
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--query')
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--stage2-gate', type=Path)
    args = parser.parse_args()
    if args.kaggle and args.stage2_gate:
        parser.error('--stage2-gate is for the live demo')
    if args.stage2_gate:
        from sprint.gate_runtime import Stage2Runtime
        runtime = Stage2Runtime.load(args.directory, args.stage2_gate, args.threads)
    else:
        runtime = Runtime(args.directory, args.threads)
    runtime.predict('warmup')
    if args.query is not None:
        response(runtime, args.query, not args.kaggle)
        return
    print(f"Router loaded with {runtime.settings['parameters']:,} parameters. Enter /exit to close.")
    while True:
        try:
            text = input('Query> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip() == '/exit':
            break
        response(runtime, text, not args.kaggle)


if __name__ == '__main__':
    main()
