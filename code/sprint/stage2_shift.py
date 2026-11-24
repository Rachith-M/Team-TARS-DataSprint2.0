import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, clean, script
from sprint.augment import FILLERS, HINDI_FILLERS, asr_variant, cluster_key, digitize, stage2_normalize, tta_variant
from sprint.data import SEED, SCRIPTS, dataset_key, folds, load_data
from sprint.diagnostics import score_columns, write_rows
from sprint.encoder import Encoder, variant_groups
from sprint.gate_experiment import Scores, digest, json_file, latency, time_check
from sprint.gate_runtime import Stage2Runtime, actions
from sprint.postprocess import average_tta
from sprint.runtime import Runtime
from sprint.stage2_fixed import GROUP_COLUMNS, band_rows, group_rows, rule, summarize, supplied
from sprint.train import fit_fold, take


BUDGET = 2 * 3600
NORMALIZATION_LIMIT = -.003


def perturb(texts, rows):
    shifted = []
    for i in rows:
        raw = texts[i]
        rng = np.random.default_rng([SEED, int(i)])
        words = HINDI_FILLERS | {'ना'} if script(raw) == 'devanagari' else FILLERS | {'na'}
        filler = rng.choice(sorted(words))
        altered = asr_variant(digitize(clean(raw)), rng)
        shifted.append(f'{filler} {altered}' if rng.random() < .5 else f'{altered} {filler}')
    return shifted


def score_pair(raw, score):
    normalized = list(map(stage2_normalize, raw))
    unique = list(dict.fromkeys([*raw, *map(tta_variant, raw), *normalized, *map(tta_variant, normalized)]))
    logits = tuple(score(unique))
    if len(logits) != len(unique):
        raise ValueError('Query alignment changed during scoring')
    lookup = dict(zip(unique, logits))
    def get(texts):
        return np.stack([lookup[text] for text in texts])
    return {name: average_tta(get(texts), get(list(map(tta_variant, texts))), .35)
            for name, texts in (('raw', raw), ('normalized', normalized))}


def scores(logits, scale, scripts, group, actual):
    maximum, _, energy = score_columns(logits, scale)
    return Scores(logits.argmax(axis=1), maximum, energy, np.empty((len(logits), 0)), scripts, group, actual)


def fold_scores(data, numbers, scale, deadline):
    texts, labels, groups, parts, base, vectors, encoder, config = data
    across = defaultdict(list)
    changed = Counter()
    for number in numbers:
        time_check(deadline)
        fit, check = parts[number]
        head = fit_fold(config, groups, labels, vectors, fit, SEED+number, check=check)
        raw = head.score(take(texts, check), vectors['raw'][check])
        if not np.allclose(raw, base['raw'][check], atol=1e-4, rtol=1e-4):
            raise ValueError('The refit head did not reproduce the archived selected fold')
        known = check[labels[check] != FALLBACK]
        shifted = perturb(texts, known)
        scripts = np.array([script(texts[i]) for i in known])
        for origin, variant in zip(take(texts, known), shifted):
            changed[('rows', script(origin))] += 1
            changed[('digit_form_changed', script(origin))] += digitize(clean(origin)) != clean(origin)
            changed[('asr_changed', script(origin))] += not any(
                text == digitize(clean(origin)) for text in (variant.partition(' ')[2], variant.rpartition(' ')[0]))
        def score(text):
            return head.score(text, encoder.encode(text, batch_size=48))
        for name, logits in score_pair(shifted, score).items():
            across[name].append(scores(logits, scale, scripts, np.full(len(known), number+1), labels[known]))
        print(f'Perturbed supported fold {number+1}: {len(known)} rows scored', flush=True)
    return {name: Scores(*(np.concatenate(columns) for columns in zip(*(vars(part).values() for part in pieces))))
            for name, pieces in across.items()}, {name: dict((s, changed[(name, s)]) for s in SCRIPTS)
                                                    for name in sorted({key for key,_ in changed})}


def deployed_score(runtime, deadline):
    def score(texts):
        batches = []
        for i in range(0, len(texts), 48):
            time_check(deadline)
            batches.append(runtime.raw(texts[i:i+48]))
        return np.concatenate(batches)
    return score


def route(s, gate):
    return actions(1-s.maximum, s.predicted, gate, True)


def paired(known, current):
    raw, norm = known['raw'], known['normalized']
    rejected = [route(s, current)=='reject' for s in (raw,norm)]
    result = {}
    for name, mask in [('overall', np.ones(len(raw.actual), dtype=bool)), *((s,raw.script==s) for s in SCRIPTS)]:
        a,b = (s.predicted[mask]==s.actual[mask] for s in (raw,norm))
        result[name] = {'rows':int(mask.sum()),'raw_accuracy':float(a.mean()),'normalized_accuracy':float(b.mean()),
                        'accuracy_delta':float(b.mean()-a.mean()),'wrong_to_right':int((~a & b).sum()),
                        'right_to_wrong':int((a & ~b).sum()),'raw_median':float(np.median(raw.maximum[mask])),
                        'normalized_median':float(np.median(norm.maximum[mask])),
                        'median_paired_confidence_delta':float(np.median(norm.maximum[mask]-raw.maximum[mask])),
                        'raw_rejection':float(rejected[0][mask].mean()),'normalized_rejection':float(rejected[1][mask].mean()),
                        'rejection_delta':float(rejected[1][mask].mean()-rejected[0][mask].mean())}
    return result


def unknown_scores(runtime, supplied_data, phase, deadline):
    queries, cats, scripts, tiers, split, _ = supplied_data
    mask = split == phase
    logits = score_pair([queries[i] for i in np.flatnonzero(mask)], deployed_score(runtime, deadline))
    return {name:scores(z,runtime.settings['scale'],scripts[mask],cats[mask],np.full(mask.sum(),-1))
            for name,z in logits.items()}, tiers[mask]


def test_results(runtime, test, ids, archive, gates, current, deadline):
    scale = runtime.settings['scale']
    if archive['example_id'].tolist() != ids:
        raise ValueError('Test diagnostics are out of order')
    pairs = score_pair(test,deployed_score(runtime,deadline))
    scripts = np.array(list(map(script,test)))
    groups = defaultdict(list)
    for i, text in enumerate(test):
        groups[cluster_key(text)].append(i)
    families = [rows for key,rows in groups.items() if key and len(rows)>1]
    family_rows = np.array([i for rows in families for i in rows])
    arms = {name:scores(z,scale,scripts,np.full(len(test),''),np.full(len(test),-1)) for name,z in pairs.items()}
    original, norm = (arms[name] for name in ('raw','normalized'))
    diag = {'families':len(families),'rows':len(family_rows),
            'raw_median':float(np.median(original.maximum[family_rows])),
            'normalized_median':float(np.median(norm.maximum[family_rows])),
            'median_paired_confidence_delta':float(np.median(norm.maximum[family_rows]-original.maximum[family_rows])),
            'label_agreement':float(np.mean(original.predicted[family_rows]==norm.predicted[family_rows])),
            'unanimous_raw_families':sum(len(set(original.predicted[ids]))==1 for ids in families),
            'unanimous_normalized_families':sum(len(set(norm.predicted[ids]))==1 for ids in families)}
    rows = []
    for name, source, gate in [('current','raw',current), *[(name,name,gate) for name,gate in gates.items()]]:
        action = route(arms[source],gate)
        for tag in ('overall',*SCRIPTS):
            mask = np.ones(len(test),dtype=bool) if tag=='overall' else scripts==tag
            rows.append((name,tag,int(mask.sum()),*(float(np.mean(action[mask]==kind)) for kind in ('reject','clarify','accept'))))
    return rows, diag


def write_policy_table(output, summaries, test_rows):
    test = {name:(reject,clarify) for name,script,_,reject,clarify,_ in test_rows if script=='overall'}
    rows = []
    for item in summaries:
        s = item['summary']
        name, phase = item['policy'], item['phase']
        rows.append((phase,name,s['all_supported']['rate'],s['correct_supported']['rate'],
                     *(s['scripts'][tag]['correct_supported']['rate'] for tag in SCRIPTS),s['unsupported']['rate'],
                     item['deflected'],item['accepted_accuracy'],item['accepted_supported_accuracy'],
                     *(test.get(name,(None,None)) if phase=='evaluation' else (None,None))))
    write_rows(output/'policy_results.csv',('phase','policy','all_supported_rejection','correct_supported_rejection',
               *(f'{tag}_correct_supported_rejection' for tag in SCRIPTS),'unsupported_rejection',
               'unsupported_rejected_or_clarified','accepted_accuracy','accepted_supported_accuracy',
               'test_rejection','test_clarification'),rows)


def run(args):
    start = perf_counter()
    deadline = start+BUDGET
    old = json.loads((args.current/'operating_point.json').read_text(encoding='utf-8'))
    config = json.loads((args.reference/'metrics.json').read_text(encoding='utf-8'))['config']
    supplied_data = supplied(args.unsupported)
    plan = {'seed':SEED,'normalize_loss_limit':NORMALIZATION_LIMIT,'budget_seconds':BUDGET,'known_rejection_budget':.03,
            'files':{str(path):digest(path) for path in (args.unsupported,args.data/'train.csv',args.data/'test.csv',
                     args.reference/'scores.npz',args.current/'operating_point.json')},
            'normalize_selection':'calibration accuracy delta >= -0.003, then one held-out safety check',
            'clarify':'same <=20% unsupported calibration acceptance rule as original',
            'script':'source script before perturbation or normalization'}
    args.output.mkdir(parents=True,exist_ok=True)
    json_file(args.output/'started.json',plan,exclusive=True)
    texts, labels, test, ids = load_data(args.data)
    parts, groups = folds(texts,labels), variant_groups(texts,[])
    archive = np.load(args.reference/'scores.npz')
    base = {key:archive[key] for key in ('raw','fold_id','target')}
    if not np.array_equal(labels,base['target']) or any(
            not np.all(base['fold_id'][check]==i) for i,(_,check) in enumerate(parts)):
        raise ValueError('The archived folds do not match train.csv')
    paths = args.cache/'e5-small'
    vectors = {name:np.load(paths/f'{name}-{dataset_key(groups[name])[:16]}.npy',mmap_mode='r') for name in ('raw','aug')}
    data = texts,labels,groups,parts,base,vectors,Encoder(paths,args.threads),config
    runtime = Runtime(args.model,args.threads)
    scale = runtime.settings['scale']
    if runtime.settings['tta_weight'] != .35 or old['chosen']['candidate'] != 'max_softmax':
        raise ValueError('The selected runtime or current gate changed')
    current = old['chosen']
    calibration, cal_meta = fold_scores(data,range(3),scale,deadline)
    cal_unknown, cal_tiers = unknown_scores(runtime,supplied_data,'calibration',deadline)
    gates = {arm:rule(calibration[arm],cal_unknown[arm],'max_softmax',None) for arm in calibration}
    cal_pairs = paired(calibration,current)
    primary = 'normalized' if cal_pairs['overall']['accuracy_delta'] >= NORMALIZATION_LIMIT else 'raw'
    json_file(args.output/'calibration.json',{'primary':primary,'gates':gates,'paired':cal_pairs,
              'perturbation':cal_meta},exclusive=True)
    json_file(args.output/'evaluation_started.json',{'primary':primary},exclusive=True)
    evaluation, eval_meta = fold_scores(data,range(3,5),scale,deadline)
    eval_unknown, eval_tiers = unknown_scores(runtime,supplied_data,'evaluation',deadline)
    eval_pairs = paired(evaluation,current)
    results, bands, summaries = [], [], []
    for phase, known, unknown, tiers in [('calibration',calibration,cal_unknown,cal_tiers),
                                        ('evaluation',evaluation,eval_unknown,eval_tiers)]:
        for name, arm, gate in [('current','raw',current),('normalized_current','normalized',current),
                                *((arm,arm,gate) for arm,gate in gates.items())]:
            summary = summarize(known[arm],unknown[arm],tiers,gate)
            results.extend(group_rows(name,phase,summary))
            band = list(band_rows(known[arm],unknown[arm],gate,phase))
            bands.extend((name,*row) for row in band)
            novel = route(unknown[arm],gate)
            summaries.append({'phase':phase,'policy':name,'summary':summary,'deflected':float(np.mean(novel!='accept')),
                              'accepted_accuracy':band[0][9],'accepted_supported_accuracy':band[0][10]})
    test_rows, family = test_results(runtime,test,ids,archive,gates,current,deadline)
    write_policy_table(args.output,summaries,test_rows)
    write_rows(args.output/'test_routing.csv',('policy','script','rows','reject','clarify','accept'),test_rows)
    write_rows(args.output/'groups.csv',GROUP_COLUMNS,results)
    write_rows(args.output/'bands.csv',('policy','phase','action','rows','coverage','supported','unsupported',
               'all_supported_action_share','correct_supported_action_share','unsupported_action_share',
               'accepted_accuracy','accepted_supported_accuracy','all_supported_rejection','correct_supported_rejection'),bands)
    pair_rows = [(phase,tag,*values.values()) for phase,rows in [('calibration',cal_pairs),('evaluation',eval_pairs)]
                 for tag,values in rows.items()]
    write_rows(args.output/'normalization.csv',('phase','script',*eval_pairs['overall']),pair_rows)
    by_policy = {row['policy']:row for row in summaries if row['phase']=='evaluation'}
    chosen = by_policy[primary]
    r = chosen['summary']
    operating = {**old,'chosen':gates[primary],'normalize_input':primary=='normalized'}
    staged = Stage2Runtime(args.model,operating,threads=args.threads)
    rng = np.random.default_rng(SEED)
    sample = [i for n in range(3) for i in parts[n][1] if labels[i]!=FALLBACK]
    benchmark = perturb(texts,rng.choice(sample,150,replace=False))
    times = latency(staged,benchmark,deadline)
    p95 = float(np.quantile(times,.95))
    passed = {'correct_only_overall':.015 <= r['correct_supported']['rate'] <= .045,
              **{f'correct_only_{tag}':r['scripts'][tag]['correct_supported']['rate']<=.05 for tag in SCRIPTS},
              'test_rejection_below_12pct':next(row[3] for row in test_rows if row[:2]==(primary,'overall')) < .12,
              'unsupported_deflected_ge_75pct':chosen['deflected']>=.75,
              'normalization_accuracy':eval_pairs['overall']['accuracy_delta']>=NORMALIZATION_LIMIT if primary=='normalized' else None,
              'cpu_p95_le_50ms':p95<=50}
    ready = all(value for value in passed.values() if value is not None)
    operating.update({'deployment_ready':ready,'targets':passed,'latency':{'queries':len(times),'total_p95_ms':p95}})
    json_file(args.output/'operating_point.json',operating,exclusive=True)
    summary = {'primary':primary,'criteria':passed,'ship':ready,'test_filler_families':family,
               'normalization':eval_pairs,'evaluation_perturbation':eval_meta,'summaries':summaries,
               'latency_p95_ms':p95,'elapsed_seconds':perf_counter()-start}
    json_file(args.output/'results.json',summary,exclusive=True)
    print(json.dumps({key:summary[key] for key in ('primary','criteria','ship','test_filler_families','latency_p95_ms')},
                     indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path.cwd())
    parser.add_argument('--unsupported',type=Path,default=Path('unsupported_eval_queries.csv'))
    parser.add_argument('--current',type=Path,default=Path('sprint_results/stage2_provided_gate'))
    parser.add_argument('--model',type=Path,default=Path('sprint_results/final'))
    parser.add_argument('--reference',type=Path,default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--cache',type=Path,default=Path('.sprint_cache'))
    parser.add_argument('--output',type=Path,default=Path('sprint_results/stage2_shift'))
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__=='__main__':
    main()
