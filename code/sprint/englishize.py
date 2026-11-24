"""Rewrite queries toward English-spelled Hinglish (kailendar -> calendar, कैलेंडर -> calendar).

The test set writes loanwords in English spelling far more often than train.csv does. The map in
sprint_results/loanmap.json was built offline from an English word list, IndicXlit (en->hi) and
sprint.augment.romanize, keeping only forms that occur in train.csv; no test text was used.
"""
import json
from pathlib import Path

import numpy as np

from intent_model import script
from sprint.augment import romanize

MAP = json.loads((Path(__file__).resolve().parent.parent / 'sprint_results' / 'loanmap.json').read_text(encoding='utf-8'))


def englishize(text, rng=None, rate=1.):
    """Romanize non-roman text, then swap mapped loanwords for English (each with probability `rate`)."""
    words = text.split()
    if script(text) != 'roman':
        words = [MAP['deva'].get(word, word) if rng is None or rng.random() < rate else word for word in words]
        words = romanize(' '.join(words)).split()
    return ' '.join(MAP['roman'].get(word.lower(), word) if rng is None or rng.random() < rate else word
                    for word in words)


def englishize_batch(texts, seed, rate=.8):
    rng = np.random.default_rng(seed)
    return [englishize(text, rng, rate) for text in texts]
