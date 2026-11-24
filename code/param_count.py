"""Count every parameter loaded at inference for a checkpoint written by sprint.a2_recipe.

Usage (from code/):  python param_count.py ../runs/final/final/checkpoint
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from safetensors import safe_open

CAP = 300_000_000


def main(folder):
    folder = Path(folder)
    with safe_open(folder / 'model.safetensors', 'np') as weights:
        encoder = sum(int(np.prod(weights.get_slice(name).get_shape())) for name in weights.keys())
    with np.load(folder / 'head.npz') as head:
        heads = sum(head[name].size for name in head.files)
    features = joblib.load(folder / 'features.joblib')
    lexical_vocabulary = sum(len(vectorizer.vocabulary_) for vectorizer in features.vectorizers)
    dictionary = sum(map(len, json.loads((folder / 'dictionary.json').read_text(encoding='utf-8'))))
    neural = encoder + heads
    total = neural + lexical_vocabulary + dictionary
    print(f'Encoder weights            {encoder:>13,}')
    print(f'Classifier heads           {heads:>13,}')
    print(f'Neural parameters          {neural:>13,}   (= sum(p.numel()) over the loaded model)')
    print(f'TF-IDF vocabulary entries  {lexical_vocabulary:>13,}')
    print(f'Dictionary entries         {dictionary:>13,}')
    print(f'Conservative total         {total:>13,}   cap {CAP:,}  ->  {"OK" if total <= CAP else "OVER"}')
    assert total <= CAP


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv[1] if len(sys.argv) > 1 else '../runs/final/final/checkpoint')
