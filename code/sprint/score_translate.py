import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, clean, roman_token, script
from sprint.augment import romanize
from sprint.data import SEED, load_training
from sprint.score_bhasha import embed
from sprint.score_data import NEGATION, frame_groups
from sprint.score_push import fit_candidate, metrics, options, prepare, score
from sprint.train import EVAL_VIEWS


REVISION = 'a7d96a16729f812578bd55e7366147beda625d86'
ENGLISH_NEGATION = re.compile(r"\b(?:no|not|never|cannot|without|\w+n't)\b", re.I)
DIGITS = re.compile(r'\d+')
STOPWORDS = set('a an the i me my we our you your it its of to in on at by from for with or is are be do does can '
                'could would should please no not never yes all this that there here what how when where which '
                'and only some every most one two three ten many there have has will'.split())


def checksum(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def translate(args):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    path = args.data / 'train.csv'
    texts, labels = load_training(path)
    families = frame_groups(texts, labels)
    rng = np.random.default_rng(SEED + 25)
    candidates = defaultdict(list)
    for index in rng.permutation(len(labels)):
        if labels[index] != FALLBACK and script(texts[index]) == 'devanagari':
            candidates[labels[index]].append(int(index))
    chosen = []
    for rows in candidates.values():
        keys = set()
        quota = min(args.per_class, round(len(rows) * .85))
        for index in rows:
            family = families[index]
            if family not in keys and len(keys) < quota:
                chosen.append(index)
                keys.add(family)
    chosen.sort(key=lambda i: len(texts[i]))
    tokenizer = AutoTokenizer.from_pretrained(args.translator, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.translator, local_files_only=True).eval()
    translated = {}
    output_labels = defaultdict(set)
    for offset in range(0, len(chosen), 32):
        indices = chosen[offset:offset + 32]
        inputs = tokenizer([texts[i] for i in indices], truncation=True, max_length=96, padding=True,
                           return_tensors='pt')
        with torch.inference_mode():
            queries = model.generate(**inputs, max_new_tokens=52, num_beams=2)
        for i, wording in zip(indices, tokenizer.batch_decode(queries, skip_special_tokens=True)):
            source = romanize(texts[i])
            key = clean(wording)
            if (not 3 <= len(key.split()) <= 25 or not key.isascii() or key == source
                    or (NEGATION.search(source) and not ENGLISH_NEGATION.search(wording))
                    or not set(DIGITS.findall(source)) <= set(DIGITS.findall(key))):
                continue
            translated[i] = wording
            output_labels[key].add(int(labels[i]))
        if (offset + 32) % 1024 == 0:
            print(f'Translated {offset + 32}/{len(chosen)} training rows', flush=True)
    rows = [[i, wording] for i, wording in sorted(translated.items()) if len(output_labels[clean(wording)]) == 1]
    counts = Counter(int(labels[i]) for i, _ in rows)
    args.pack.write_text(json.dumps({'train_sha256': checksum(path), 'checkpoint': REVISION, 'rows': rows,
                                     'class_counts': counts}, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Kept {len(rows)} English variants for {len(counts)} training classes', flush=True)


def switch_words(texts, pairs, fit, stem):
    sides = []
    pairs_seen = Counter()
    source_count = Counter()
    target_count = Counter()
    present = set(fit)
    for source, english in pairs:
        if source not in present:
            continue
        original = romanize(texts[source]).split()
        latin = [roman_token(word) for word in original]
        left = {word for word in latin if len(word) >= 3 and word.isalpha() and word not in STOPWORDS
                and not NEGATION.fullmatch(word)}
        right = {word for word in clean(english).split() if len(word) >= 3 and word.isalpha()
                 and word not in STOPWORDS and not ENGLISH_NEGATION.fullmatch(word)}
        source_count.update(left)
        target_count.update(right)
        pairs_seen.update((word, translation) for word in left for translation in right)
        sides.append((source, original, latin, left, right))
    forward = defaultdict(list)
    backward = defaultdict(list)
    for (word, translation), count in pairs_seen.items():
        if count >= 4 and count >= .44 * source_count[word] and count >= .14 * target_count[translation]:
            score = 2 * count / (source_count[word] + target_count[translation])
            forward[word].append((score, translation))
            backward[translation].append((score, word))
    links = {}
    for word, candidates in forward.items():
        ranked = sorted(candidates, reverse=True)
        strength, translation = ranked[0]
        if (len(ranked) == 1 or strength >= 1.10 * ranked[1][0]) and max(backward[translation])[1] == word:
            links[word] = (translation, strength)
    changed = []
    indices = []
    for source, original, latin, left, right in sides:
        edits = sorted(((links[word][1], word, links[word][0]) for word in left if word in links
                        and links[word][0] in right and roman_token(links[word][0]) != word), reverse=True)
        replacements = {word: english for _, word, english in edits[:2]}
        if replacements:
            changed.append(' '.join(replacements.get(word, token) for word, token in zip(latin, original)))
            indices.append(source)
    return stem, changed, np.asarray(indices, dtype=int)


def assess(args):
    groups, labels, original, challenge, _, scripts, config, _, variants, _ = prepare(args)
    pack = json.loads(args.pack.read_text(encoding='utf-8'))
    if pack['train_sha256'] != checksum(args.data / 'train.csv') or pack['checkpoint'] != REVISION:
        raise ValueError('Translation source or checkpoint does not match')
    plan = [('fold1', original[0], SEED), ('template_holdout', challenge, SEED + 9)]
    if args.all_folds:
        plan.extend((f'fold{i + 1}', split, SEED + i) for i, split in enumerate(original) if i)
    bank = {}
    if args.code_switch:
        for name, (fit, _), _ in plan:
            view, texts, origins = switch_words(groups['raw'], pack['rows'], fit, f'switch_{name}')
            bank[name] = (view, texts, origins, None, labels[origins])
    else:
        ids, english = zip(*pack['rows'])
        origins = np.asarray(ids, dtype=int)
        entry = ('english', list(english), origins, None, labels[origins])
        bank = {name: entry for name, _, _ in plan}
    views = {view: groups[view] for view in (*EVAL_VIEWS, 'aug')}
    views.update({name: text for name, text, *_ in [*variants, *bank.values()]})
    vectors, _ = embed(views, args.bhasha, args.device, 128)
    frozen = json.loads((args.reference.parent / 'score_push/bhasha.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, (fit, check), seed in plan:
        with threadpool_limits(limits=args.threads):
            model, counts = fit_candidate(config, groups, labels, vectors, [*variants, bank[name]], fit, check, seed)
            values = metrics(score(model, groups, vectors, check), labels[check], scripts[check])
        if name not in frozen:
            raise ValueError(f'Missing frozen comparison for {name}')
        baseline = frozen[name]['candidate']
        delta = {key: values[key] - baseline[key] for key in values}
        result[name] = {'candidate': values, 'delta_vs_frozen': delta, 'training': counts}
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f"{name}: perturbed {delta['perturbed']:+.5f}, Roman ASR {delta['roman_asr']:+.5f}", flush=True)
        if name == 'template_holdout' and any(x['delta_vs_frozen']['perturbed'] < .004 for x in result.values()):
            break


def main():
    parser = argparse.ArgumentParser()
    options(parser)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--translator', type=Path)
    parser.add_argument('--bhasha', type=Path)
    parser.add_argument('--per-class', type=int, default=120)
    parser.add_argument('--device', choices=('cpu', 'xpu'), default='xpu')
    parser.add_argument('--all-folds', action='store_true')
    parser.add_argument('--code-switch', action='store_true')
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument('--generate', action='store_true')
    stages.add_argument('--assess', action='store_true')
    args = parser.parse_args()
    if (args.generate and not args.translator) or (args.assess and not args.bhasha):
        parser.error('The requested stage needs a local checkpoint')
    translate(args) if args.generate else assess(args)


if __name__ == '__main__':
    main()
