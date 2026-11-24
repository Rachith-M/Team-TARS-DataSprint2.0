import argparse
import json
import math
from pathlib import Path

import numpy as np


LANGUAGES = 'en as bn brx gom gu hi kn ks mai ml mni mr ne or pa sa sd si ta te ur'.split()
SPECIAL = ['<s>', '<blank>', '</s>', '<unk>']


def convert(source, output):
    import torch
    from ctranslate2.specs import common_spec, transformer_spec

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(source / 'transformer/indicxlit.pt', map_location='cpu', weights_only=True)
    config = checkpoint['cfg']['model']
    task = checkpoint['cfg']['task']
    assert task.encoder_langtok == 'tgt' and not task.decoder_langtok
    assert config.activation_fn == 'gelu' and config.encoder_normalize_before and config.decoder_normalize_before
    state = checkpoint['model']
    spec = transformer_spec.TransformerSpec.from_config(
        (config.encoder_layers, config.decoder_layers), config.encoder_attention_heads,
        pre_norm=True, layernorm_embedding=True, activation=common_spec.Activation.GELU)

    def linear(target, *names):
        target.weight = np.concatenate([state[f'{name}.weight'].numpy() for name in names], axis=0)
        if f'{names[0]}.bias' in state:
            target.bias = np.concatenate([state[f'{name}.bias'].numpy() for name in names])

    def normalize(target, name):
        target.gamma = state[f'{name}.weight'].numpy()
        target.beta = state[f'{name}.bias'].numpy()

    def attention(target, name, self_attention):
        query, key, value = (f'{name}.{part}_proj' for part in ('q', 'k', 'v'))
        if self_attention:
            linear(target.linear[0], query, key, value)
        else:
            linear(target.linear[0], query)
            linear(target.linear[1], key, value)
        linear(target.linear[-1], f'{name}.out_proj')
        normalize(target.layer_norm, f'{name}_layer_norm')

    for title, target, vocab_name in (('encoder', spec.encoder, 'en'), ('decoder', spec.decoder, 'mlt')):
        embeddings = state[f'{title}.embed_tokens.weight'].numpy()
        half = embeddings.shape[1] // 2
        positions = np.arange(2, 1026, dtype=np.float32)[:, None]
        angle = positions * np.exp(np.arange(half, dtype=np.float32) * (-math.log(10000) / (half - 1)))
        target.position_encodings.encodings = np.concatenate((np.sin(angle), np.cos(angle)), axis=1)
        target.scale_embeddings = math.sqrt(embeddings.shape[1])
        embedding = target.embeddings[0] if isinstance(target.embeddings, list) else target.embeddings
        embedding.weight = embeddings
        normalize(target.layernorm_embedding, f'{title}.layernorm_embedding')
        normalize(target.layer_norm, f'{title}.layer_norm')
        for index, layer in enumerate(target.layer):
            prefix = f'{title}.layers.{index}'
            attention(layer.self_attention, f'{prefix}.self_attn', True)
            if title == 'decoder':
                attention(layer.attention, f'{prefix}.encoder_attn', False)
            linear(layer.ffn.linear_0, f'{prefix}.fc1')
            linear(layer.ffn.linear_1, f'{prefix}.fc2')
            normalize(layer.ffn.layer_norm, f'{prefix}.final_layer_norm')
        tokens = (source / f'corpus-bin/dict.{vocab_name}.txt').read_text(encoding='utf-8').splitlines()
        vocab = SPECIAL + [line.rsplit(maxsplit=1)[0] for line in tokens] + [f'__{lang}__' for lang in LANGUAGES]
        if len(vocab) != embeddings.shape[0] or len(set(vocab)) != len(vocab):
            raise ValueError('The checkpoint vocabulary has an unexpected size or duplicate tokens')
        register = spec.register_source_vocabulary if title == 'encoder' else spec.register_target_vocabulary
        register(vocab)
    linear(spec.decoder.projection, 'decoder.output_projection')
    spec.config.decoder_start_token = spec.config.eos_token
    spec.config.add_source_eos = True
    spec.validate()
    spec.optimize()
    output.mkdir(parents=True, exist_ok=False)
    spec.save(str(output))
    parameters = sum(value.numel() for name, value in state.items() if not name.endswith(('.version', '._float_tensor')))
    (output / 'parameters.json').write_text(json.dumps({'parameters': parameters}) + '\n', encoding='utf-8')


class IndicXlit:
    def __init__(self, model, threads=2, beam=1):
        import ctranslate2

        self.engine = ctranslate2.Translator(str(model), intra_threads=threads, compute_type='float32')
        self.beam = beam

    def translate(self, words):
        if not words:
            return []
        inputs = [['__hi__', *word] for word in words]
        result = self.engine.translate_batch(inputs, beam_size=self.beam, max_batch_size=64,
                                             max_decoding_length=32)
        return [''.join(item.hypotheses[0]) for item in result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
    model = IndicXlit(args.output)
    expected = ['\u0928\u092e\u0938\u094d\u0924\u0947', '\u092e\u094c\u0938\u092e', '\u092a\u093e\u0928\u0940']
    if model.translate(['namaste', 'mausam', 'paani']) != expected:
        raise ValueError('IndicXlit smoke check did not match the expected Hindi words')
    print('IndicXlit conversion and smoke check passed', flush=True)


if __name__ == '__main__':
    main()
