import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import confusion_matrix

from intent_model import FALLBACK, LABELS, script
from sprint.augment import cluster_key
from sprint.data import SCRIPTS, load_data, read_csv
from sprint.export import batch_submission_scores
from sprint.postprocess import average_tta
from sprint.runtime import Runtime


HIST_EDGES = np.linspace(0, 1, 11)
PROB_EDGES = (0., .5, .75, 1.)
MARGIN_EDGES = (0., .25, .5, 1.)


def write_rows(path, header, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def score_columns(logits, scale):
    scaled = np.asarray(logits, dtype=np.float64) * scale
    probabilities = softmax(scaled, axis=1)
    two = np.sort(np.partition(probabilities, -2, axis=1)[:, -2:], axis=1)
    return two[:, 1], two[:, 1] - two[:, 0], -logsumexp(scaled, axis=1)


def oof_logits(directory, labels, weight):
    with np.load(directory / 'scores.npz') as saved:
        if not np.array_equal(saved['target'], labels):
            raise ValueError('OOF labels differ from the supplied training CSV')
        fold = saved['fold_id'].astype(int) + 1
        if not np.array_equal(np.unique(fold), np.arange(1, 6)):
            raise ValueError('OOF fold coverage is incomplete')
        return fold, average_tta(saved['raw'], saved['tta'], weight)


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def class_rows(labels, predictions, scripts):
    for name in ('overall', *SCRIPTS):
        indices = np.ones(len(labels), dtype=bool) if name == 'overall' else scripts == name
        matrix = confusion_matrix(labels[indices], predictions[indices], labels=LABELS)
        correct = matrix.diagonal()
        actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
        yield from zip([name] * len(LABELS), LABELS, actual, predicted, correct,
                       ratio(correct, predicted), ratio(correct, actual), ratio(2*correct, actual+predicted))


def confusion_cost(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    correct = matrix.diagonal()
    denom = matrix.sum(axis=1) + matrix.sum(axis=0)
    f1 = ratio(2 * correct, denom)
    pairs = []
    for source, destination in zip(*np.where(matrix > 0)):
        if source == destination:
            continue
        count = int(matrix[source, destination])
        src_new = 2 * (correct[source]+count) / (denom[source]+count)
        dst_new = 2 * correct[destination] / (denom[destination]-count) if denom[destination] > count else 0
        gain = (src_new + dst_new - f1[source] - f1[destination]) / len(LABELS)
        pairs.append((int(source), int(destination), count, float(gain)))
    return sorted(pairs, key=lambda item: (-item[3], -item[2], item[0], item[1]))[:15]


def hist_rows(cohorts):
    for name, (p, gap) in cohorts.items():
        for score, values in (('max_softmax', p), ('margin', gap)):
            counts, _ = np.histogram(values, bins=HIST_EDGES)
            for low, high, count in zip(HIST_EDGES[:-1], HIST_EDGES[1:], counts):
                yield name, score, round(low, 2), round(high, 2), int(count), float(count / len(values))


def draw_histograms(path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    with path.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    styles = [('test_pred55', 'Test predicted 55', '#193b6b'),
              ('oof_true55_pred55', 'OOF true 55 predicted 55', '#28745e'),
              ('oof_non55_pred55', 'OOF non-55 predicted 55', '#af512d')]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, score, label in zip(axes, ('max_softmax', 'margin'), ('Max softmax', 'Top-two probability gap')):
        for cohort, legend, color in styles:
            bins = [row for row in rows if row['cohort'] == cohort and row['score'] == score]
            axis.stairs([float(row['cohort_fraction']) for row in bins], HIST_EDGES, color=color,
                        linewidth=2.4, label=f'{legend} (n={sum(int(row["count"]) for row in bins)})')
        axis.set(xlim=(0, 1), xlabel=label)
        axis.grid(axis='y', alpha=.18)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Share of cohort')
    figure.suptitle('Confidence distributions for predictions of class 55', fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    figure.tight_layout(rect=(0, .085, 1, .955))
    figure.savefig(path.with_suffix('.png'), dpi=190)
    plt.close(figure)


def wilson(successes, total):
    if not total:
        return None, None
    z = 1.95996398454
    rate = successes / total
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    radius = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return max(0., float(center-radius)), min(1., float(center+radius))


def risk_grid(cohorts):
    tp = np.histogram2d(*cohorts['oof_true55_pred55'], bins=(PROB_EDGES, MARGIN_EDGES))[0].astype(int)
    fp = np.histogram2d(*cohorts['oof_non55_pred55'], bins=(PROB_EDGES, MARGIN_EDGES))[0].astype(int)
    test = np.histogram2d(*cohorts['test_pred55'], bins=(PROB_EDGES, MARGIN_EDGES))[0].astype(int)
    floor = max(.25, 2 * fp.sum() / (tp.sum()+fp.sum()))
    eligible = 0
    rows = []
    naive_low_conf = 0
    for p, m in np.ndindex(tp.shape):
        real, false, test_count = int(tp[p,m]), int(fp[p,m]), int(test[p,m])
        low_confidence = p == 0 or m == 0
        total = real + false
        qualified = bool(low_confidence and total >= 12 and false >= 3 and false/total >= floor)
        if low_confidence:
            naive_low_conf += test_count
        if qualified:
            eligible += test_count
        rows.append((PROB_EDGES[p], PROB_EDGES[p+1], MARGIN_EDGES[m], MARGIN_EDGES[m+1], real, false,
                     false/total if total else None, *wilson(false, total), test_count, low_confidence, qualified))
    summary = {'threshold_false_share': float(floor), 'min_oof_predicted55': 12, 'min_oof_false55': 3,
               'low_confidence': 'max_softmax<0.50 or margin<0.25',
               'test_pred55': int(test.sum()), 'test_low_confidence': naive_low_conf,
               'test_in_qualified_region': eligible, 'test_fraction_in_qualified_region': eligible / int(test.sum()),
               'task3_trigger': bool(eligible > test.sum() / 2)}
    return rows, summary


def run(data, model, reference, output, expected):
    texts, labels, test, ids = load_data(data)
    training = read_csv(data / 'train.csv', ('example_id', 'query_text', 'target'))
    output.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(model)
    if runtime.settings['cluster_weight'] != 0:
        raise ValueError('Expected the selected zero-clustering model')
    scale, weight = runtime.settings['scale'], runtime.settings['tta_weight']
    pred_test_logits = batch_submission_scores(runtime, test)
    pred_test = pred_test_logits.argmax(axis=1)
    submitted = read_csv(expected, ('example_id', 'target'))
    if [row['example_id'] for row in submitted] != ids or not np.array_equal(
        pred_test, np.array([int(row['target']) for row in submitted]),
    ):
        raise ValueError('The diagnostic predictions do not reproduce the selected submission')
    fold, logits = oof_logits(reference, labels, weight)
    pred = logits.argmax(axis=1)
    oof_p, oof_gap, oof_energy = score_columns(logits, scale)
    test_p, test_gap, test_energy = score_columns(pred_test_logits, scale)
    scripts = np.array(list(map(script, texts)))
    test_scripts = np.array(list(map(script, test)))
    family_keys = [cluster_key(query) or example_id for query, example_id in zip(test, ids)]
    index = {key: f'F{number:05d}' for number, key in enumerate(dict.fromkeys(family_keys), start=1)}
    family_sizes = Counter(family_keys)
    grouped = Counter()
    family_labels = {}
    for key, label in zip(family_keys, pred_test):
        if family_sizes[key] > 1:
            grouped[key] += 1
            family_labels.setdefault(key, set()).add(int(label))
    disagreement = [key for key, values in family_labels.items() if len(values) > 1]
    write_rows(output / 'test_diagnostics.csv',
               ('example_id','max_softmax','margin','energy','predicted_class','script','filler_family_id'),
               zip(ids, test_p, test_gap, test_energy, pred_test, test_scripts, (index[key] for key in family_keys)))
    write_rows(output / 'oof_diagnostics.csv',
               ('example_id','fold','script','true_class','predicted_class','max_softmax','margin','energy'),
               zip((row['example_id'] for row in training), fold, scripts, labels, pred, oof_p, oof_gap, oof_energy))
    classes = list(class_rows(labels, pred, scripts))
    write_rows(output / 'oof_per_class.csv',
               ('script','class_id','support','predicted_count','true_positives','precision','recall','f1'), classes)
    pairs = confusion_cost(labels, pred)
    write_rows(output / 'oof_confusion_cost.csv', ('rank','true_class','predicted_class','errors','macro_f1_cost'),
               ((rank, *row) for rank, row in enumerate(pairs, start=1)))

    oof55, test55 = pred == FALLBACK, pred_test == FALLBACK
    cohorts = {name: (p[mask], gap[mask]) for name, mask, p, gap in (
        ('test_pred55', test55, test_p, test_gap),
        ('oof_true55_pred55', oof55 & (labels == FALLBACK), oof_p, oof_gap),
        ('oof_non55_pred55', oof55 & (labels != FALLBACK), oof_p, oof_gap),
    )}
    write_rows(output / 'class55_histograms.csv',
               ('cohort','score','lower_edge','upper_edge','count','cohort_fraction'), hist_rows(cohorts))
    cells, decision = risk_grid(cohorts)
    write_rows(output / 'class55_risk_grid.csv',
               ('max_softmax_lower','max_softmax_upper','margin_lower','margin_upper','oof_true55','oof_non55',
                'oof_false_share','oof_false_share_wilson_low','oof_false_share_wilson_high','test_pred55_count',
                'low_confidence','qualified_region'), cells)
    descriptions = {name: {'count': len(p), 'max_softmax_median': float(np.median(p)),
                           'margin_median': float(np.median(gap))} for name, (p, gap) in cohorts.items()}
    by_script = {name: {'predicted55': int(np.sum(test55 & (test_scripts == name))),
                        'test_rows': int(np.sum(test_scripts == name))} for name in SCRIPTS}
    by_script_text = ', '.join(f'{name} {x["predicted55"]}/{x["test_rows"]}' for name,x in by_script.items())
    weakest = sorted(classes[:len(LABELS)], key=lambda row: (row[-1], row[1]))[:5]
    cost_text = ', '.join(f'{a}→{b} ({n}, +{gain:.5f})' for a,b,n,gain in pairs[:5])
    cohort_text = [f'- {name.replace("_", " ")} has {row["count"]} rows, median probability '
                   f'{row["max_softmax_median"]:.3f} and median margin {row["margin_median"]:.3f}.'
                   for name, row in descriptions.items()]
    decision.update({'cohorts': descriptions, 'test55_by_script': by_script, 'seed_fixed': True,
                     'scale': scale, 'tta_weight': weight, 'test_rows': len(test), 'oof_rows': len(labels),
                     'filler_families': {'groups': len(grouped), 'rows': sum(grouped.values()),
                                         'disagreement_groups': len(disagreement),
                                         'disagreement_rows': sum(grouped[key] for key in disagreement),
                                         'disagreement_groups_with_55': sum(FALLBACK in family_labels[key] for key in disagreement)}})
    (output / 'summary.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    summary = ['# Selected model diagnostics','',
               '- Test scores come from the released int8 batch path. All hard labels match the submission.',
               '- OOF scores reuse grouped-fold float head logits and the released 0.35 TTA and scale.',
               f'- Class 55 accounts for {test55.sum()}/{len(test)} test predictions. By script, {by_script_text}.',
               f'- {sum(family_sizes[key] == 1 for key, is55 in zip(family_keys, test55) if is55)} class 55 predictions are filler-family singletons.',
               f'- The released filler/number key yields {len(grouped)} families with {sum(grouped.values())} rows. The {len(disagreement)} disagreeing families contain no class 55 predictions.',
               *cohort_text,
               f'- {decision["test_low_confidence"]}/{test55.sum()} class 55 predictions have max softmax <0.50 or margin <0.25.',
               f'- {decision["test_in_qualified_region"]}/{test55.sum()} ({decision["test_fraction_in_qualified_region"]:.1%}) fall in low-confidence cells with enough OOF observations and elevated false-55 share.',
               '- A cell qualifies with at least 12 OOF predictions, at least 3 false-55s, and false share at least 25% and twice the overall OOF false-55 share.',
               '- Margins are probability gaps and energy is negative logsumexp of scaled logits. Confidence overlap measures similarity to labeled OOF errors. Test correctness remains unknown.',
               '- Lowest overall OOF class F1: ' + ', '.join(f'{row[1]} ({row[-1]:.3f}, n={row[2]})' for row in weakest) + '.',
               f'- Highest confusion costs, measured as the macro-F1 gain from repairing just that pair: {cost_text}.',
               f'- Task 3 majority trigger is {"met" if decision["task3_trigger"] else "not met"}.']
    (output / 'SUMMARY.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path.cwd())
    parser.add_argument('--model', type=Path, default=Path('sprint_results/final'))
    parser.add_argument('--reference', type=Path, default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--output', type=Path, default=Path('sprint_results/diagnostics'))
    parser.add_argument('--submission', type=Path, default=Path('submission.csv'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    run(args.data, args.model, args.reference, args.output, args.submission)
    if args.plot:
        draw_histograms(args.output / 'class55_histograms.csv')


if __name__ == '__main__':
    main()
