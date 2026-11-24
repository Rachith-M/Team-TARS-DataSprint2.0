import json

import numpy as np
from sklearn.cluster import KMeans

from intent_model import FALLBACK
from sprint.data import SEED, SCRIPTS, dataset_key, folds, read_csv
from sprint.diagnostics import score_columns, wilson, write_rows
from sprint.encoder import variant_groups
from sprint.gate_experiment import BOOTSTRAPS, K_VALUES, bootstrap_recall, distances, json_file, known_scores, time_check
from sprint.gate_runtime import actions, rejection
from sprint.live import novelty_score, probe
from sprint.postprocess import average_tta


def paired_interval(candidate, baseline, groups):
    _, groups = np.unique(groups, return_inverse=True)
    n = np.bincount(groups)
    good = np.bincount(groups, weights=candidate & ~baseline)
    bad = np.bincount(groups, weights=baseline & ~candidate)
    rng = np.random.default_rng(SEED+7)
    clusters = rng.integers(0,len(n),(BOOTSTRAPS,len(n)))
    draws = rng.binomial(n[clusters],(good/n)[clusters])
    bad_probability = np.divide(bad,n-good,out=np.zeros_like(bad),where=n!=good)
    errors = rng.binomial(n[clusters]-draws,bad_probability[clusters])
    return np.quantile((draws-errors).sum(axis=1)/n[clusters].sum(axis=1),[.025,.975]).tolist()


def p1(args, data, chosen, legacy, old_gate, deadline, settings):
    text, _, _, split, _, labels, group, bank, bank_rows, _ = data
    history = json.loads((args.reference/'live.json').read_text(encoding='utf-8'))
    selected = history['test_withheld_classes']
    config = json.loads((args.reference/'metrics.json').read_text(encoding='utf-8'))['config']
    groups = variant_groups(text, [])
    keys = ('raw','aug','tta','natural','natural_tta')
    embeddings = {name:np.load(args.cache/'e5-small'/f'{name}-{dataset_key(groups[name])[:16]}.npy',mmap_mode='r')
                  for name in keys}
    parts = folds(text,labels)
    if any(not np.all(split[check]==i+1) for i,(_,check) in enumerate(parts)):
        raise ValueError('Historical probe folds differ from the stored OOF folds')
    actual, scores, distance = [], [], []
    print('Replaying the two historical withheld-intent scoring folds',flush=True)
    for fold, removed in enumerate(selected,start=2):
        time_check(deadline)
        style = 'raw' if fold % 2 == 0 else 'natural'
        raw, tta, targets, withheld = probe(config,groups,labels,embeddings,parts[fold],np.array(removed),style=style)
        logits = average_tta(raw,tta,settings['tta_weight'])[withheld]
        actual.append(targets[withheld])
        scores.append(logits)
        if chosen['rule']['candidate']=='knn':
            references = ~np.isin(labels[bank_rows],removed) & (split[bank_rows]!=fold+1)
            held = parts[fold][1][withheld]
            distance.append(distances(embeddings[style][held],bank[references],deadline,group[held],group[bank_rows][references]))
    logits = np.concatenate(scores)
    actual = np.concatenate(actual)
    if len(logits) != history['test_with_gate']['novel_rows'] or len(logits) != 649:
        raise ValueError('The historical withheld-intent count did not reproduce')
    pred = logits.argmax(axis=1)
    max_softmax,_,energy = score_columns(logits,settings['scale'])
    old_risk = -novelty_score(logits,old_gate['measure'],old_gate['scale'])
    baseline = rejection(old_risk,pred,legacy['threshold'])
    original = (pred==FALLBACK) | (old_risk>-old_gate['threshold'])
    method, k, cut = (chosen['rule'][key] for key in ('candidate','k','threshold'))
    risk = {'max_softmax':1-max_softmax,'energy':energy}.get(method)
    if method=='knn':
        risk = np.concatenate(distance)[:,K_VALUES.index(k)].astype(np.float64)
    pred_flags = rejection(risk,pred,cut)
    delta = float(pred_flags.mean()-baseline.mean())
    return {'rows':len(pred), 'historical_threshold_recall_reproduced':float(original.mean()),
            'historical_threshold_recall_reported':history['test_with_gate']['novel_recall'],
            'existing_at_3pct_recall':float(baseline.mean()),'existing_at_3pct_ci':bootstrap_recall(baseline,actual),
            'chosen_recall':float(pred_flags.mean()),'chosen_recall_ci':bootstrap_recall(pred_flags,actual),
            'paired_gain':delta,'paired_ci':paired_interval(pred_flags,baseline,actual),'target_ge_5pp':delta>=.05}


def p2(data, rule, deadline, context, output):
    mask = data[5]==FALLBACK
    scores = known_scores(data,mask,deadline)
    groups = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit_predict(data[2][mask])
    baseline = scores.predicted==FALLBACK
    gate = rejection(scores.risk(rule['candidate'],rule['k']),scores.predicted,rule['threshold'])
    rows = []
    for name, rows_mask in [('overall',np.ones(len(groups),dtype=bool)), *[(f'T{i+1:02d}',groups==i) for i in range(6)]]:
        n = int(rows_mask.sum())
        initial = int((baseline & rows_mask).sum())
        rescued = int((gate & ~baseline & rows_mask).sum())
        rows.append((name,n,initial,rescued,rescued/n,*wilson(rescued,n),rescued/(n-initial) if n>initial else None,
                     *wilson(rescued,n-initial),*context))
    write_rows(output/'p2_known55_clusters.csv',('topic_cluster','rows','baseline_rejections','additional_rejections',
               'additional_recall','wilson_low','wilson_high','share_of_baseline_misses_rejected','miss_ci_low','miss_ci_high',
               'context_all_supported_rejection','context_correct_supported_rejection'),rows)
    return {'rows':rows[0][1],'baseline_rejections':rows[0][2],'additional_rejections':rows[0][3],
            'additional_recall':rows[0][4],'additional_recall_ci':list(rows[0][5:7])}


def p3(args, rule, bank, deadline, context, band):
    path = args.oof.with_name('test_diagnostics.csv')
    rows = read_csv(path,('example_id','max_softmax','margin','energy','predicted_class','script','filler_family_id'))
    with np.load(args.reference/'scores.npz') as saved:
        if [row['example_id'] for row in rows]!=saved['example_id'].tolist():
            raise ValueError('The frozen test diagnostic order differs from the selected archive')
    pred = np.array([int(row['predicted_class']) for row in rows])
    script = np.array([row['script'] for row in rows])
    if rule['candidate']=='knn':
        vectors = np.load(args.cache/'e5-small/test-71b25af51d20f472.npy',mmap_mode='r')
        if vectors.shape != (len(rows),384):
            raise ValueError('The frozen test embedding dimensions differ from the diagnostic rows')
        risk = distances(vectors,bank,deadline)[:,K_VALUES.index(rule['k'])].astype(np.float64)
    elif rule['candidate']=='energy':
        risk = np.array([float(row['energy']) for row in rows])
    else:
        risk = 1-np.array([float(row['max_softmax']) for row in rows])
    action = actions(risk,pred,rule,band)
    summary = []
    for name in ('overall',*SCRIPTS):
        mask = np.ones(len(rows),dtype=bool) if name=='overall' else script==name
        total = int(mask.sum())
        rejected = int(((action=='reject')&mask).sum())
        clarified = int(((action=='clarify')&mask).sum())
        summary.append((name,total,rejected,rejected/total,clarified,clarified/total,*context))
    write_rows(args.output/'p3_test_rejection.csv',('script','rows','rejected','rejected_share','clarified','clarified_share',
               'context_all_supported_rejection','context_correct_supported_rejection'),summary)
    return {'rows':summary[0][1],'rejected':summary[0][2],'rejected_share':summary[0][3],
            'clarified':summary[0][4],'clarified_share':summary[0][5]}


def report_proxies(args, data, chosen, legacy, old_gate, deadline, band):
    primary = chosen['evaluation']
    context = primary['all_supported']['rate'],primary['correct_supported']['rate']
    settings = json.loads((args.model/'runtime.json').read_text(encoding='utf-8'))
    known55 = p2(data,chosen['rule'],deadline,context,args.output)
    unknown_test = p3(args,chosen['rule'],data[7],deadline,context,band)
    withheld = p1(args,data,chosen,legacy,old_gate,deadline,settings)
    json_file(args.output/'proxies.json',{'P1':withheld,'P2':known55,'P3':unknown_test,
              'context_all_supported_rejection':context[0],'context_correct_supported_rejection':context[1]},exclusive=True)
