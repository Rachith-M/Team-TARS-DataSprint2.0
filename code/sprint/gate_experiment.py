import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, Views
from sprint.augment import cluster_key
from sprint.data import SEED, SCRIPTS, dataset_key, read_csv
from sprint.diagnostics import score_columns, wilson, write_rows
from sprint.gate_runtime import Stage2Runtime, actions, normalize_vectors, rejection
from sprint.runtime import Runtime


EXPERIMENT = 'team_unsupported_category_holdout_v1'
SECONDS = 4 * 60 * 60
K_VALUES = (1, 5, 15, 30)
BOOTSTRAPS = 5000
METHODS = ('max_softmax', 'energy', 'knn')


@dataclass
class Scores:
    predicted: np.ndarray
    maximum: np.ndarray
    energy: np.ndarray
    distances: np.ndarray
    script: np.ndarray
    group: np.ndarray
    actual: np.ndarray

    def risk(self, method, k=None):
        if method == 'knn':
            return self.distances[:, K_VALUES.index(k)].astype(np.float64)
        return self.energy if method == 'energy' else 1 - self.maximum

    def take(self, mask):
        return Scores(*(column[mask] for column in vars(self).values()))


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def json_file(path, record, exclusive=False):
    if exclusive:
        with path.open('x', encoding='utf-8') as file:
            json.dump(record, file, indent=2, allow_nan=False)
        return
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def time_check(deadline):
    if perf_counter() >= deadline:
        raise TimeoutError('The four-hour calibration and evaluation time cap was reached')


def supplied(path):
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    if reader.fieldnames != ['query_text', 'category', 'script']:
        raise ValueError('Expected query_text,category,script in that order')
    queries, categories, scripts = [], [], []
    for number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'Malformed unsupported CSV record {number}')
        text, category, script = row['query_text'].strip(), row['category'].strip().casefold(), row['script'].strip().casefold()
        if not text or not category or script not in SCRIPTS:
            raise ValueError(f'Unsupported CSV record {number} has missing text/category or an invalid script tag')
        queries.append(text)
        categories.append(category)
        scripts.append(script)
    families = {}
    for text, category in zip(queries, categories):
        key = cluster_key(text)
        if families.setdefault(key, category) != category:
            raise ValueError('A normalized query family spans unsupported categories')
    if len(set(categories)) < 2:
        raise ValueError('A category holdout requires at least two categories')
    return queries, np.asarray(categories), np.asarray(scripts), hashlib.sha256(content).hexdigest()


def category_split(categories):
    names = np.unique(categories)
    order = np.random.default_rng(SEED).permutation(names)
    return sorted(order[:len(names)//2].tolist()), sorted(order[len(names)//2:].tolist())


def workspace_rows(args):
    train_path = args.data / 'train.csv'
    train = read_csv(train_path, ('example_id','query_text','target'))
    oof = read_csv(args.oof, ('example_id','fold','script','true_class','predicted_class','max_softmax','margin','energy'))
    if [row['example_id'] for row in train] != [row['example_id'] for row in oof] or any(
        row['target'] != diagnostic['true_class'] for row, diagnostic in zip(train, oof)
    ):
        raise ValueError('OOF diagnostic rows do not match training IDs and labels')
    text = [row['query_text'] for row in train]
    vector_path = args.cache / 'e5-small' / f'raw-{dataset_key(text)[:16]}.npy'
    vectors = np.load(vector_path, mmap_mode='r')
    if vectors.shape != (len(train), 384):
        raise ValueError('The cached encoder dimensions do not match the train rows')
    split = np.asarray([int(row['fold']) for row in oof])
    script = np.asarray([row['script'] for row in oof])
    actual = np.asarray([int(row['true_class']) for row in oof])
    if set(split) != set(range(1, 6)) or set(script) != set(SCRIPTS):
        raise ValueError('The five folds and three script tags must be present')
    for selected in (split <= 3, split >= 4):
        if set(script[selected & (actual != FALLBACK)]) != set(SCRIPTS):
            raise ValueError('Both supported splits need all three script groups')
    _, group = np.unique(Views.from_texts(text).phonetic, return_inverse=True)
    bank_rows = (split <= 3) & (actual != FALLBACK)
    bank = np.asarray(vectors[bank_rows], dtype=np.float16)
    files = [train_path, args.oof, vector_path, *(args.model / name for name in (
        'model.int8.onnx','runtime.json','tokenizer.json','features.joblib'))]
    keys = {file.name: digest(file) for file in files}
    return text, oof, vectors, split, script, actual, group, bank, bank_rows, keys


def distances(queries, bank, deadline, query_groups=None, bank_groups=None):
    index = normalize_vectors(bank)
    result = np.empty((len(queries), len(K_VALUES)), dtype=np.float32)
    for start in range(0, len(queries), 96):
        time_check(deadline)
        stop = min(start + 96, len(queries))
        similarity = normalize_vectors(queries[start:stop]) @ index.T
        if query_groups is not None:
            similarity[query_groups[start:stop, None] == bank_groups[None, :]] = -np.inf
        top = np.partition(similarity, -max(K_VALUES), axis=1)[:, -max(K_VALUES):]
        if not np.isfinite(top).all():
            raise ValueError('The neighbor bank has too few independent reference groups')
        nearest = 1 - np.clip(np.sort(top, axis=1)[:, ::-1], -1., 1.)
        result[start:stop] = np.column_stack([nearest[:, :k].mean(axis=1) for k in K_VALUES])
    return result


def known_scores(data, mask, deadline):
    _, oof, vectors, _, scripts, labels, group, bank, bank_rows, _ = data
    rows = np.flatnonzero(mask)
    near = distances(vectors[rows], bank, deadline, group[rows], group[bank_rows])
    return Scores(np.asarray([int(oof[i]['predicted_class']) for i in rows]),
                  np.asarray([float(oof[i]['max_softmax']) for i in rows]),
                  np.asarray([float(oof[i]['energy']) for i in rows]), near, scripts[rows],
                  np.full(len(rows), ''), labels[rows])


def unsupported_scores(model, dataset, mask, bank, deadline, threads, return_logits=False):
    runtime = Runtime(model, threads, expose_embedding=True)
    text, categories, scripts, _ = dataset
    logits, embeddings = [], []
    for row in np.flatnonzero(mask):
        time_check(deadline)
        z, e = runtime.score([text[row]], with_embedding=True)
        logits.append(z[0])
        embeddings.append(e[0])
    z = np.asarray(logits)
    maximum, _, energy = score_columns(z, runtime.settings['scale'])
    result = Scores(z.argmax(axis=1), maximum, energy, distances(np.asarray(embeddings), bank, deadline),
                    scripts[mask], categories[mask], np.full(len(z), -1))
    return (result, z) if return_logits else result


def bootstrap_recall(flags, groups):
    names, group = np.unique(groups, return_inverse=True)
    counts = np.bincount(group)
    positives = np.bincount(group, weights=flags)
    rng = np.random.default_rng(SEED + 1)
    selected = rng.integers(0, len(names), (BOOTSTRAPS, len(names)))
    chosen_counts = counts[selected]
    simulated = rng.binomial(chosen_counts, (positives / counts)[selected])
    interval = np.quantile(simulated.sum(axis=1) / chosen_counts.sum(axis=1), [.025, .975])
    return interval.tolist()


def metrics(known, unknown, specification):
    method, k, threshold = (specification[key] for key in ('candidate', 'k', 'threshold'))
    known_flags = rejection(known.risk(method, k), known.predicted, threshold)
    unknown_flags = rejection(unknown.risk(method, k), unknown.predicted, threshold)
    supported = known.actual != FALLBACK
    fallback = ~supported
    groups = {script: float(known_flags[supported & (known.script == script)].mean()) for script in SCRIPTS}
    return {'frr': float(known_flags[supported].mean()), 'recall': float(unknown_flags.mean()),
            'recall_ci': bootstrap_recall(unknown_flags, unknown.group), 'script_frr': groups,
            'supported_n': int(supported.sum()), 'unsupported_n': len(unknown_flags),
            'known55_n': int(fallback.sum()), 'known55_rejected': int(known_flags[fallback].sum())}


def targets(metrics_row):
    return {'frr_le_5pct': metrics_row['frr'] <= .05, 'recall_ge_80pct': metrics_row['recall'] >= .8,
            'recall_ci_lower_ge_65pct': metrics_row['recall_ci'][0] >= .65,
            **{f'{script}_frr_le_8pct': metrics_row['script_frr'][script] <= .08 for script in SCRIPTS},
            'minimum_frr_le_8pct_and_recall_ge_70pct': metrics_row['frr'] <= .08 and metrics_row['recall'] >= .7}


def specificity_threshold(risk, direct, budget):
    remaining = int(np.floor(budget * len(risk))) - int(direct.sum())
    pool = risk[~direct]
    if remaining < 0 or not len(pool):
        return np.inf
    if remaining >= len(pool):
        return -np.inf
    edge = np.partition(pool, len(pool)-1-remaining)[len(pool)-1-remaining]
    return float(np.nextafter(np.float64(edge), np.inf))


def calibrate_rule(known, unknown, method, k, budget):
    risk, novel = known.risk(method, k), unknown.risk(method, k)
    supported = known.actual != FALLBACK
    direct = known.predicted == FALLBACK
    cut = specificity_threshold(risk[supported], direct[supported], budget)
    for group in SCRIPTS:
        rows = supported & (known.script == group)
        cut = max(cut, specificity_threshold(risk[rows], direct[rows], .08))
    captures = novel[(unknown.predicted != FALLBACK) & (novel >= cut)]
    cut = float(captures.min()) if len(captures) else np.inf
    potentially_accepted = np.sort(novel[unknown.predicted != FALLBACK])
    allowed = int(.20 * len(unknown.predicted))
    accept = float(potentially_accepted[allowed]) if allowed < len(potentially_accepted) else np.inf
    accept = min(accept, cut)
    return {'candidate': method, 'k': k, 'threshold': cut if np.isfinite(cut) else None,
            'accept_below': accept if np.isfinite(accept) else None, 'calibration_frr_budget': budget}


def rank(entry):
    cal = entry['calibration']
    checks = targets(cal)
    minimum = checks.pop('minimum_frr_le_8pct_and_recall_ge_70pct')
    return all(checks.values()), minimum, cal['recall'], -cal['frr'], -(entry['rule']['k'] or 0)


def calibration_rules(known, unknown):
    result = []
    for method in METHODS:
        options = []
        for k in K_VALUES if method == 'knn' else (None,):
            for budget in (.05, .08):
                rule = calibrate_rule(known, unknown, method, k, budget)
                options.append({'rule': rule, 'calibration': metrics(known, unknown, rule)})
        result.append(max(options, key=rank))
    return result


def latency(runtime, texts, deadline, base=False):
    times = []
    for query in texts[:12]:
        runtime.predict(query, live=not base)
    for text in texts:
        time_check(deadline)
        start = perf_counter()
        runtime.predict(text, live=not base)
        times.append(1000 * (perf_counter() - start))
    return np.asarray(times)


def benchmark_candidates(candidates, model, bank, data, deadline, threads):
    texts, _, _, split, script, actual, _, _, _, _ = data
    rng = np.random.default_rng(SEED + 2)
    indices = []
    for name in SCRIPTS:
        rows = np.flatnonzero((split <= 3) & (actual != FALLBACK) & (script == name))
        indices.extend(rng.choice(rows, min(50, len(rows)), replace=False))
    queries = [texts[index] for index in rng.permutation(indices)]
    ungated = latency(Runtime(model, threads), queries, deadline, base=True)
    for item in candidates:
        staged = Stage2Runtime(model, {'chosen': item['rule'], 'mode': 'band'}, bank, threads)
        gated = latency(staged, queries, deadline)
        del staged
        total = float(np.quantile(gated, .95))
        added = max(0., float(np.quantile(gated - ungated, .95)))
        item['latency'] = {'queries': len(queries), 'baseline_p95_ms': float(np.quantile(ungated, .95)),
                           'total_p95_ms': total, 'added_p95_ms': added, 'met': total < 50 and added <= 5}


def cell_rows(name, phase, known, unknown, rule):
    method, k, threshold = (rule[key] for key in ('candidate','k','threshold'))
    kp = rejection(known.risk(method, k), known.predicted, threshold)
    up = rejection(unknown.risk(method, k), unknown.predicted, threshold)
    supported = known.actual != FALLBACK
    for population, scores, flags, grouping, values in (
        ('supported', known.take(supported), kp[supported], 'script', SCRIPTS),
        ('oof_true55', known.take(~supported), kp[~supported], 'script', SCRIPTS),
        ('unsupported', unknown, up, 'script', SCRIPTS),
        ('unsupported', unknown, up, 'category', np.unique(unknown.group)),
    ):
        for value in values:
            mask = (scores.script if grouping == 'script' else scores.group) == value
            n, count = int(mask.sum()), int(flags[mask].sum())
            yield name, phase, population, grouping, value, n, count, (count/n if n else None), *wilson(count, n)


def band_report(known, unknown, rule):
    method, k = rule['candidate'], rule['k']
    mask = known.actual != FALLBACK
    known = known.take(mask)
    supported = actions(known.risk(method, k), known.predicted, rule, True)
    unsupported = actions(unknown.risk(method, k), unknown.predicted, rule, True)
    records = []
    for action in ('accept', 'clarify', 'reject'):
        n_supported, n_unknown = int(np.sum(supported == action)), int(np.sum(unsupported == action))
        n = n_supported + n_unknown
        incorrect = int(np.sum((supported == 'accept') & (known.predicted != known.actual))) if action == 'accept' else None
        error = (n_unknown+incorrect) / n if action == 'accept' and n else None
        if action == 'reject' and n:
            error = n_supported / n
        records.append((action, n, n / (len(supported)+len(unsupported)), n_supported,
                        n_supported/len(supported), n_unknown, n_unknown/len(unsupported), incorrect, error))
    return records


def run(args):
    start = perf_counter()
    dataset = supplied(args.unsupported)
    data = workspace_rows(args)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'calibration.json'
    identity = {'protocol': EXPERIMENT, 'seed': SEED, 'unsupported_sha256': dataset[-1], 'artifacts': data[-1],
                'threads': args.threads}
    if args.phase == 'calibrate':
        cal, hold = category_split(dataset[1])
        if set(map(cluster_key, dataset[0])) & set(map(cluster_key, data[0])):
            raise ValueError('The supplied unsupported set overlaps the labeled training text after normalization')
        identity.update({'calibration_categories': cal, 'evaluation_categories': hold})
        json_file(args.output / 'calibration_started.json', identity, exclusive=True)
        deadline = start + SECONDS
        known = known_scores(data, data[3] <= 3, deadline)
        unknown = unsupported_scores(args.model, dataset, np.isin(dataset[1], cal), data[7], deadline, args.threads)
        candidates = calibration_rules(known, unknown)
        benchmark_candidates(candidates, args.model, data[7], data, deadline, args.threads)
        selected = max(candidates, key=lambda item: (item['latency']['met'], *rank(item)))
        record = {'identity': identity, 'candidates': candidates, 'chosen': selected['rule']['candidate'],
                  'unsupported_total': len(dataset[0]), 'work_seconds': perf_counter()-start}
        write_rows(args.output / 'calibration_groups.csv', GROUP_COLUMNS, (row for item in candidates for row in
                   cell_rows(item['rule']['candidate'], 'calibration', known, unknown, item['rule'])))
        json_file(path, record, exclusive=True)
        print(json.dumps({'selected_on_calibration': record['chosen'], 'unsupported_calibration_rows': len(unknown.actual),
                          'unsupported_target_120_met': len(dataset[0]) >= 120}, indent=2))
        return

    plan = json.loads(path.read_text(encoding='utf-8'))
    expected = plan['identity']
    if any(expected[key] != value for key, value in identity.items()):
        raise ValueError('The input file, model, protocol, or measured thread count differs from calibration')
    json_file(args.output / 'evaluation_started.json', expected, exclusive=True)
    deadline = start + SECONDS - plan['work_seconds']
    known = known_scores(data, data[3] >= 4, deadline)
    unknown = unsupported_scores(args.model, dataset, np.isin(dataset[1], expected['evaluation_categories']),
                                  data[7], deadline, args.threads)
    for item in plan['candidates']:
        item['evaluation'] = metrics(known, unknown, item['rule'])
        item['targets'] = targets(item['evaluation'])
    selected = next(item for item in plan['candidates'] if item['rule']['candidate'] == plan['chosen'])
    minimum = selected['targets']['minimum_frr_le_8pct_and_recall_ge_70pct']
    record = {'mode': 'binary' if minimum else 'band', 'chosen': selected['rule'],
              'targets': {**selected['targets'], 'latency': selected['latency']['met']},
              'deployment_ready': selected['latency']['met'], 'latency': selected['latency'],
              'calibration': selected['calibration'], 'evaluation': selected['evaluation'],
              'model_assets': {name: expected['artifacts'][name] for name in (
                  'model.int8.onnx', 'runtime.json', 'tokenizer.json', 'features.joblib')}}
    write_rows(args.output / 'evaluation_groups.csv', GROUP_COLUMNS, (row for item in plan['candidates'] for row in
               cell_rows(item['rule']['candidate'], 'evaluation', known, unknown, item['rule'])))
    write_rows(args.output / 'candidate_results.csv',
               ('candidate','k','threshold','calibration_frr','calibration_recall','evaluation_frr','evaluation_recall',
                'evaluation_recall_ci_lower','evaluation_recall_ci_upper','added_p95_ms','total_p95_ms'),
               ((item['rule']['candidate'], item['rule']['k'], item['rule']['threshold'], item['calibration']['frr'],
                 item['calibration']['recall'], item['evaluation']['frr'], item['evaluation']['recall'],
                 *item['evaluation']['recall_ci'], item['latency']['added_p95_ms'], item['latency']['total_p95_ms'])
                for item in plan['candidates']))
    if record['mode'] == 'band':
        write_rows(args.output / 'confidence_bands.csv',
                   ('action','rows','coverage','supported_n','supported_coverage','unsupported_n','unsupported_coverage',
                    'supported_wrong_intent','conditional_action_error'), band_report(known, unknown, selected['rule']))
    if selected['rule']['candidate'] == 'knn' and record['deployment_ready']:
        index_path = args.output / 'index.float16.npy'
        np.save(index_path, data[7])
        record['index_sha256'] = digest(index_path)
    json_file(args.output / 'evaluation.json', {'candidates': plan['candidates'], 'chosen': plan['chosen'],
              'mode': record['mode'], 'unsupported_evaluation_categories': len(np.unique(unknown.group)),
              'work_seconds': plan['work_seconds'] + perf_counter()-start}, exclusive=True)
    json_file(args.output / 'operating_point.json', record, exclusive=True)
    print(json.dumps(record, indent=2))


GROUP_COLUMNS = ('candidate','phase','population','grouping','group','rows','rejected','rate',
                 'wilson_lower','wilson_upper')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('calibrate','evaluate'))
    parser.add_argument('--unsupported', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--oof', type=Path, default=Path('sprint_results/diagnostics/oof_diagnostics.csv'))
    parser.add_argument('--model', type=Path, default=Path('sprint_results/final'))
    parser.add_argument('--cache', type=Path, default=Path('.sprint_cache'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/stage2_team_gate'))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=2, user_api='blas'):
        run(args)


if __name__ == '__main__':
    main()
