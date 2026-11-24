import re

import numpy as np

from intent_model import MATRAS, clean, script, transliterate
from sprint.data import SEED


FILLERS = frozenset('bhai yaar jaldi please plz hey google zara bro'.split())
HINDI_FILLERS = frozenset(('भाई', 'यार', 'प्लीज', 'ज़रा', 'जरा'))
NUMBERS = {
    'shoonya': 0, 'zero': 0, 'ek': 1, 'do': 2, 'teen': 3, 'char': 4, 'chaar': 4,
    'paanch': 5, 'panch': 5, 'chhe': 6, 'chhah': 6, 'chhahh': 6, 'saat': 7, 'sat': 7,
    'aath': 8, 'ath': 8, 'nau': 9, 'das': 10, 'gyarah': 11, 'baarah': 12, 'terah': 13,
    'chaudah': 14, 'pandrah': 15, 'solah': 16, 'satrah': 17, 'atharah': 18,
    'unnees': 19, 'bees': 20, 'tees': 30, 'chalees': 40, 'pachaas': 50, 'saath': 60,
    'sattar': 70, 'assee': 80, 'nabbe': 90, 'ek sau': 100, 'एक': 1, 'दो': 2,
    'तीन': 3, 'चार': 4, 'पाँच': 5, 'पांच': 5, 'छह': 6, 'सात': 7, 'आठ': 8,
    'नौ': 9, 'दस': 10, 'बीस': 20, 'तीस': 30, 'चालीस': 40, 'पचास': 50, 'साठ': 60,
    'सत्तर': 70, 'अस्सी': 80, 'नब्बे': 90,
}
SPOKEN = ('shoonya ek do teen chaar paanch chhah saat aath nau das gyaarah baarah terah '
          'chaudah pandrah solah satrah athaarah unnees bees').split()
SPOKEN_EXTRA = {30: 'tees', 40: 'chaalees', 50: 'pachaas', 60: 'saath', 70: 'sattar',
                80: 'assee', 90: 'nabbe', 100: 'ek sau'}
UNITS = r'baje|ghante?|minute?s?|pratishat|percent|प्रतिशत|बजे|घंटे|मिनट|%​?'
SPELLED = re.compile(r'(?<!\S)(' + '|'.join(map(re.escape, sorted(NUMBERS, key=len, reverse=True)))
                     + r')(?=\s+(?:' + UNITS + r')(?:\s|$))')
DIGITS = re.compile(r'(?<!\d)([0-9]{1,3})(?=\s*(?:' + UNITS + r')(?:\s|$))')
TIME = re.compile(r'(?<!\d)([0-9]{1,2}):([0-9]{2})(?!\d)')
ENGLISH = {
    'samay': 'time', 'tithi': 'date', 'mausam': 'weather', 'baarish': 'rain',
    'gaana': 'song', 'sangeet': 'music', 'khabar': 'news', 'samachaar': 'news',
    'batti': 'light', 'battiyaan': 'lights', 'roshni': 'brightness', 'dhvani': 'sound',
    'taiksee': 'taxi', 'railgaadee': 'train', 'relgaadee': 'train', 'tikat': 'ticket',
    'kharidna': 'buy', 'kharido': 'buy', 'hatao': 'remove', 'jodo': 'add',
    'dukaan': 'store', 'khojo': 'search', 'suchi': 'list', 'soochi': 'list',
    'sanpark': 'contact', 'imel': 'email', 'eemel': 'email', 'sandesh': 'message',
    'kailendar': 'calendar', 'aaj': 'today', 'subah': 'morning', 'shaam': 'evening',
    'shuroo': 'start', 'badhao': 'increase', 'mitao': 'delete', 'ghanta': 'hour',
    'rasoi': 'kitchen', 'kamra': 'room', 'khareedna': 'buy', 'alaarm': 'alarm',
}
PHONETIC_NOISE = {
    'mujhe': 'mjhe', 'hai': 'h', 'kya': 'kia', 'aap': 'ap', 'main': 'mai', 'mein': 'me',
    'nahi': 'nai', 'liye': 'lie', 'bahut': 'bohot', 'chahiye': 'chahie',
    'kripaya': 'kripya', 'dijiye': 'dijie', 'bataiye': 'bataie', 'gaana': 'gana',
    'chaahie': 'chahie', 'rahaa': 'raha', 'dekhnaa': 'dekhna', 'hamen': 'hame',
}


def romanize(text):
    return transliterate(clean(text))


def strip_fillers(text, leading_na=False):
    parts = text.split()
    leading = FILLERS | HINDI_FILLERS | ({'na', 'ना'} if leading_na else set())
    while len(parts) > 2 and parts[0].casefold().strip(',.!?') in leading:
        parts.pop(0)
    while len(parts) > 2 and parts[-1].casefold().strip(',.!?') in FILLERS | HINDI_FILLERS | {'na', 'ना'}:
        parts.pop()
    return ' '.join(parts)


def spoken(number):
    number = int(number)
    return SPOKEN[number] if 0 <= number < len(SPOKEN) else SPOKEN_EXTRA.get(number, str(number))


def normalize_numbers(text):
    text = TIME.sub(lambda m: f'{spoken(m[1])} baje {spoken(m[2])} minute', text)
    return DIGITS.sub(lambda m: spoken(m[1]), text)


def digitize(text):
    return SPELLED.sub(lambda m: str(NUMBERS[m[1]]), text)


def replace_some(text, mapping, rng, limit):
    matches = [m for m in re.finditer(r'(?<!\S)[a-z]+(?!\S)', text) if m[0] in mapping]
    if not matches:
        return text
    selected = rng.choice(len(matches), size=min(limit, len(matches)), replace=False)
    for match in sorted((matches[i] for i in selected), key=lambda m: -m.start()):
        text = text[:match.start()] + mapping[match[0]] + text[match.end():]
    return text


def naturalize(text, rng):
    text = romanize(text)
    text = digitize(replace_some(text, ENGLISH, rng, 2))
    text = replace_some(text, PHONETIC_NOISE, rng, 2)
    if rng.random() < .5:
        return f'{rng.choice(sorted(FILLERS))} {text}'
    return f'{text} {rng.choice(sorted(FILLERS | {"na"}))}'


def asr_variant(text, rng):
    text = clean(text)
    if script(text) == 'devanagari':
        positions = [index for index, char in enumerate(text) if char in MATRAS]
    else:
        changed = replace_some(text, PHONETIC_NOISE, rng, 2)
        if changed != text:
            return changed
        positions = [m.start() for m in re.finditer(r'(?<=[a-z]{2})\s(?=[a-z]{2})', text)]
    if not positions:
        return text
    index = int(rng.choice(positions))
    return text[:index] + text[index + 1:]


def augment_batch(texts, seed=SEED, style='train'):
    rng = np.random.default_rng(seed)
    if style == 'asr':
        return [asr_variant(text, rng) for text in texts]
    if style == 'natural':
        return [naturalize(text, rng) for text in texts]
    result = []
    for text in texts:
        draw = rng.random()
        if draw < .57:
            result.append(naturalize(text, rng))
        elif draw < .82 and script(text) != 'roman':
            result.append(romanize(text))
        else:
            result.append(asr_variant(text, rng))
    return result


def tta_variant(text):
    return normalize_numbers(strip_fillers(text))


def stage2_normalize(text):
    return normalize_numbers(strip_fillers(text, leading_na=True).translate(str.maketrans('०१२३४५६७८९', '0123456789')))


def cluster_key(text):
    return clean(tta_variant(text))


def procedural_oos(texts, labels, size, seed=SEED):
    rng = np.random.default_rng(seed)
    by_label = {label: [text for text, value in zip(texts, labels) if value == label]
                for label in np.unique(labels) if label != 55}
    possible = sorted(by_label)
    outliers = []
    for _ in range(size):
        label = int(rng.choice(possible))
        source = str(rng.choice(by_label[label])).split()
        if len(source) > 3 and rng.random() < .65:
            rng.shuffle(source)
            outliers.append(' '.join(source[:max(2, len(source) - 2)]))
        else:
            other_label = int(rng.choice([item for item in possible if item != label]))
            other = str(rng.choice(by_label[other_label])).split()
            outliers.append(' '.join(source[:max(1, len(source) // 2)] + other[-max(1, len(other) // 2):]))
    return outliers
