import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK
from sprint.augment import cluster_key
from sprint.data import SEED, SCRIPTS
from sprint.diagnostics import oof_logits, wilson, write_rows
from sprint.gate_experiment import (BOOTSTRAPS, K_VALUES, METHODS, benchmark_candidates, bootstrap_recall,
                                    digest, json_file, known_scores, specificity_threshold,
                                    time_check, unsupported_scores, workspace_rows)
from sprint.gate_runtime import Stage2Runtime, actions, rejection
from sprint.live import novelty_score


PROTOCOL = 'provided_category_split_correct_only_budget_v1'
SECONDS = 4 * 3600
BUDGET = .03


def supplied(path):
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    columns = ['query_text','category','script','style','tier','split']
    if reader.fieldnames != columns:
        raise ValueError('Unexpected team CSV columns')
    rows, cats, families = [], {}, {}
    for number, row in enumerate(reader, start=2):
        if None in row or any(row[name] is None for name in columns):
            raise ValueError(f'Malformed CSV record {number}')
        text = row['query_text'].strip()
        category, script, style, tier, split = (row[name].strip().casefold() for name in columns[1:])
        if (not text or not category or script not in SCRIPTS or style not in ('deva','train_style','natural','mixed')
                or tier not in ('far','near') or split not in ('calibration','evaluation')):
            raise ValueError(f'Missing text or invalid metadata on record {number}')
        if cats.setdefault(category, (tier, split)) != (tier, split):
            raise ValueError('A category spans tiers or splits')
        if families.setdefault(cluster_key(text), split) != split:
            raise ValueError('A normalized evaluation family leaks across splits')
        rows.append((text,category,script,tier,split))
    counts = Counter(row[1] for row in rows)
    if len(rows) != 160 or len(cats) != 16 or set(counts.values()) != {10} or Counter(cats.values()) != {
        ('far','calibration'):4, ('far','evaluation'):4, ('near','calibration'):4, ('near','evaluation'):4,
    }:
        raise ValueError('The supplied dataset differs from the preregistered 160-row category balance')
    queries,categories,scripts,tiers,splits = zip(*rows)
    return list(queries), np.array(categories), np.array(scripts), np.array(tiers), np.array(splits), hashlib.sha256(data).hexdigest()


def threshold(risk, known):
    correct = (known.actual != FALLBACK) & (known.actual == known.predicted)
    if not np.any(correct):
        raise ValueError('No correctly classified supported calibration queries')
    cut = specificity_threshold(risk[correct], np.zeros(correct.sum(), dtype=bool), BUDGET)
    return cut if np.isfinite(cut) else None


def rule(known, unknown, method, k):
    cut = threshold(known.risk(method,k), known)
    novel = np.sort(unknown.risk(method,k)[unknown.predicted != FALLBACK])
    allowed = int(.20 * len(unknown.predicted))
    accept = float(novel[allowed]) if allowed < len(novel) else np.inf
    accept = min(accept, np.inf if cut is None else cut)
    return {'candidate':method,'k':k,'threshold':cut,'accept_below':accept if np.isfinite(accept) else None}


def counts(flags, mask):
    n, positive = int(mask.sum()), int(flags[mask].sum())
    return {'n':n,'count':positive,'rate':positive/n if n else None, 'wilson_ci':wilson(positive,n)}


def risks(known, unknown, setting, legacy=None):
    if legacy is not None:
        return legacy
    return known.risk(setting['candidate'],setting['k']), unknown.risk(setting['candidate'],setting['k'])


def summarize(known, unknown, tiers, setting, legacy=None):
    known_risk, unknown_risk = risks(known,unknown,setting,legacy)
    kp = rejection(known_risk, known.predicted, setting['threshold'])
    up = rejection(unknown_risk, unknown.predicted, setting['threshold'])
    supported = known.actual != FALLBACK
    correct = supported & (known.actual == known.predicted)
    sgroups = {name: {'all_supported': counts(kp,supported & (known.script==name)),
                     'correct_supported': counts(kp,correct & (known.script==name)),
                     'known55': counts(kp,~supported & (known.script==name)),
                     'unsupported': counts(up,unknown.script==name)} for name in SCRIPTS}
    outcome = {'all_supported':counts(kp,supported),'correct_supported':counts(kp,correct),'known55':counts(kp,~supported),
               'unsupported':counts(up,np.ones(len(up),dtype=bool)), 'recall_ci':bootstrap_recall(up,unknown.group),
               'scripts':sgroups,'tiers':{name:counts(up,tiers==name) for name in ('far','near')},
               'categories':{name:counts(up,unknown.group==name) for name in np.unique(unknown.group)}}
    return outcome


def checks(summary, latency=None):
    all_frr = summary['all_supported']['rate']
    correct_frr = summary['correct_supported']['rate']
    detail = {'correct_frr_in_range': .015 <= correct_frr <= .045, 'all_frr_le_8pct': all_frr <= .08,
              **{f'{name}_correct_frr_le_5pct': summary['scripts'][name]['correct_supported']['rate'] <= .05
                 for name in SCRIPTS},
              'recall_ci_lower_ge_60pct': summary['recall_ci'][0] >= .6,
              'recall_ge_70pct': summary['unsupported']['rate'] >= .7,
              'recall_ge_80pct': summary['unsupported']['rate'] >= .8}
    if latency is not None:
        detail['latency'] = latency['met']
    return detail


def enough(detail):
    return all(value for name,value in detail.items() if name != 'recall_ge_80pct')


def rank(summary):
    detail = checks(summary)
    return enough(detail), summary['unsupported']['rate'], -summary['all_supported']['rate']


def candidate_rules(known, unknown, tiers):
    candidates = []
    for method in METHODS:
        alternatives = []
        for k in K_VALUES if method == 'knn' else (None,):
            chosen = rule(known, unknown, method, k)
            alternatives.append({'rule':chosen, 'calibration':summarize(known,unknown,tiers,chosen)})
        candidates.append(max(alternatives, key=lambda row: (*rank(row['calibration']), -(row['rule']['k'] or 0))))
    return candidates


def group_rows(method, phase, summary):
    all_rate, correct_rate = (summary[name]['rate'] for name in ('all_supported','correct_supported'))
    for population in ('all_supported','correct_supported','known55','unsupported'):
        row = summary[population]
        yield method,phase,'overall','all',population,row['n'],row['count'],row['rate'],*row['wilson_ci'],all_rate,correct_rate
    for name,populations in summary['scripts'].items():
        for population,row in populations.items():
            yield (method,phase,'script',name,population,row['n'],row['count'],row['rate'],*row['wilson_ci'],
                   populations['all_supported']['rate'],populations['correct_supported']['rate'])
    for label in ('tiers','categories'):
        for name,row in summary[label].items():
            yield method,phase,label,name,'unsupported',row['n'],row['count'],row['rate'],*row['wilson_ci'],all_rate,correct_rate


GROUP_COLUMNS = ('candidate','phase','grouping','group','population','rows','rejected','rate','wilson_lower','wilson_upper',
                 'context_all_supported_rejection','context_correct_supported_rejection')


def band_rows(known, unknown, setting, phase):
    ka = actions(known.risk(setting['candidate'],setting['k']),known.predicted,setting,True)
    ua = actions(unknown.risk(setting['candidate'],setting['k']),unknown.predicted,setting,True)
    support = known.actual != FALLBACK
    correct = support & (known.actual == known.predicted)
    for action in ('accept','clarify','reject'):
        ks,us = ka==action, ua==action
        known_count, novel_count = int((ks & support).sum()), int(us.sum())
        total = known_count + novel_count
        yield (phase,action,total,total/(support.sum()+len(ua)),known_count,novel_count,
               known_count/int(support.sum()),int((ks & correct).sum())/int(correct.sum()),novel_count/len(ua),
               int((ks & correct).sum())/total if total and action == 'accept' else None,
               int((ks & correct).sum())/known_count if known_count and action == 'accept' else None,
               float(np.mean(ka[support]=='reject')),float(np.mean(ka[correct]=='reject')))


def smoke(args, operating, bank):
    runtime = (Stage2Runtime.load(args.model, args.output, args.threads) if operating['deployment_ready']
               else Stage2Runtime(args.model, operating, bank, args.threads))
    rows = []
    for line in args.demo.read_text(encoding='utf-8').splitlines():
        if not line.startswith('| ') or '`' not in line:
            continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        expected = int(cells[2])
        got = runtime.predict(cells[1].strip('`'))
        rows.append((len(rows)+1, 'pass' if got['intent_id']==expected else 'fail'))
    if len(rows) != 10:
        raise ValueError('The existing demo checklist must contain ten smoke cases')
    write_rows(args.output/'demo_smoke.csv',('case','status'),rows)
    return [status for _,status in rows]


def run(args):
    start = perf_counter()
    queries,categories,scripts,tiers,splits,sha = supplied(args.unsupported)
    data = workspace_rows(args)
    old_gate = json.loads((args.model/'live.json').read_text(encoding='utf-8'))['gate']
    settings = json.loads((args.model/'runtime.json').read_text(encoding='utf-8'))
    _, score = oof_logits(args.reference,data[5],settings['tta_weight'])
    if not np.array_equal(score.argmax(axis=1), np.array([int(row['predicted_class']) for row in data[1]])):
        raise ValueError('The OOF archive differs from the released diagnostic labels')
    old_score = -novelty_score(score,old_gate['measure'],old_gate['scale'])
    legacy_digest = {**data[-1],'live.json':digest(args.model/'live.json'), 'scores.npz':digest(args.reference/'scores.npz')}
    links = {'protocol':PROTOCOL,'seed':SEED,'threads':args.threads,'unsupported_sha256':sha,'artifacts':legacy_digest}
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path = args.output/'calibration.json'
    mask = splits==('calibration' if args.phase=='calibrate' else 'evaluation')
    train_mask = data[3]<=3 if args.phase=='calibrate' else data[3]>=4
    overlap = set(map(cluster_key,data[0]))
    normalized_overlap = sum(cluster_key(queries[i]) in overlap for i in np.flatnonzero(mask))
    if args.phase == 'calibrate':
        deadline = start+SECONDS
    else:
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if links!=plan['identity']:
            raise ValueError('Calibration and evaluation input fingerprints differ')
        deadline = start+SECONDS-plan['work_seconds']
    json_file(args.output/f'{args.phase}_started.json',links,exclusive=True)

    known = known_scores(data,train_mask,deadline)
    unknown, unknown_logits = unsupported_scores(args.model, (queries,categories,scripts,sha), mask,
                                                  data[7],deadline,args.threads,return_logits=True)
    old_values = old_score[train_mask], -novelty_score(unknown_logits,old_gate['measure'],old_gate['scale'])
    if args.phase=='calibrate':
        candidates = candidate_rules(known,unknown,tiers[mask])
        benchmark_candidates(candidates,args.model,data[7],data,deadline,args.threads)
        picked = max(candidates,key=lambda row:(row['latency']['met'],*rank(row['calibration'])))
        legacy_rule = {'candidate':'existing_at_3pct','threshold':threshold(old_values[0],known)}
        legacy = summarize(known,unknown,tiers[mask],legacy_rule,old_values)
        plan = {'identity':links,'candidates':candidates,'chosen':picked['rule']['candidate'],
                'legacy_rule':legacy_rule,'legacy_calibration':legacy,'normalized_train_overlap':normalized_overlap,
                'work_seconds':perf_counter()-start}
        json_file(plan_path,plan,exclusive=True)
        print(json.dumps({'chosen_before_evaluation':plan['chosen'], 'calibration_rows':int(mask.sum())}),flush=True)
        return

    candidates = plan['candidates']
    for item in candidates:
        item['evaluation'] = summarize(known,unknown,tiers[mask],item['rule'])
        item['targets'] = checks(item['evaluation'],item['latency'])
    existing_eval = summarize(known,unknown,tiers[mask],plan['legacy_rule'],old_values)
    chosen = next(item for item in candidates if item['rule']['candidate']==plan['chosen'])
    mode = 'binary' if enough(chosen['targets']) else 'band'
    operating = {'mode':mode,'chosen':chosen['rule'],'deployment_ready':chosen['latency']['met'],
                 'targets':chosen['targets'],'latency':chosen['latency'],
                 'model_assets':{name:links['artifacts'][name] for name in (
                     'model.int8.onnx','runtime.json','tokenizer.json','features.joblib')}}
    if chosen['rule']['candidate']=='knn' and operating['deployment_ready']:
        bank_path = args.output/'index.float16.npy'
        np.save(bank_path,data[7])
        operating['index_sha256'] = digest(bank_path)

    methods = candidates + [{'rule':plan['legacy_rule'],'calibration':plan['legacy_calibration'],'evaluation':existing_eval}]
    write_rows(args.output/'groups.csv', GROUP_COLUMNS,
               (row for item in methods for phase in ('calibration','evaluation')
                for row in group_rows(item['rule']['candidate'],phase,item[phase])))
    header = ('candidate','phase','k','threshold','all_supported_rejection','correct_supported_rejection',
              'unsupported_recall','far_recall','near_recall','ci_lower','ci_upper','gate_added_p95_ms','total_p95_ms')
    write_rows(args.output/'candidate_results.csv',header,
               ((item['rule']['candidate'],phase,item['rule'].get('k'),item['rule']['threshold'],
                 item[phase]['all_supported']['rate'],item[phase]['correct_supported']['rate'],
                 item[phase]['unsupported']['rate'],*(item[phase]['tiers'][tier]['rate'] for tier in ('far','near')),
                 *item[phase]['recall_ci'],item.get('latency',{}).get('added_p95_ms'),item.get('latency',{}).get('total_p95_ms'))
                for item in methods for phase in ('calibration','evaluation')))
    action = actions(unknown.risk(chosen['rule']['candidate'],chosen['rule']['k']),unknown.predicted,chosen['rule'],mode=='band')
    write_rows(args.output/'evaluation_category_actions.csv',
               ('category','tier','rows','accepted_as_supported','clarified','rejected','context_all_supported_rejection',
                'context_correct_supported_rejection'),
               ((name,tiers[mask][unknown.group==name][0],int((unknown.group==name).sum()),
                 *(int(((unknown.group==name)&(action==value)).sum()) for value in ('accept','clarify','reject')),
                 chosen['evaluation']['all_supported']['rate'],chosen['evaluation']['correct_supported']['rate'])
                for name in np.unique(unknown.group)))
    if mode=='band':
        write_rows(args.output/'confidence_bands.csv',('phase','action','rows','coverage','supported','unsupported',
                    'all_supported_action_share','correct_supported_action_share','unsupported_action_share',
                    'accepted_accuracy','accepted_supported_accuracy','all_supported_rejection','correct_supported_rejection'),
                   band_rows(known,unknown,chosen['rule'],'evaluation'))
    json_file(args.output/'operating_point.json',operating,exclusive=True)
    smoke_status = smoke(args, operating, data[7])
    report = {'candidates':methods,'chosen':plan['chosen'],'operating_point':operating,'smoke_status':smoke_status,
              'normalized_train_overlap_by_split':{'calibration':plan['normalized_train_overlap'],
                                                    'evaluation':normalized_overlap},
              'unsupported_source':'provided LLM-written proxy',
              'elapsed_seconds':plan['work_seconds']+perf_counter()-start}
    json_file(args.output/'evaluation.json',report,exclusive=True)
    from sprint.stage2_proxies import report_proxies
    report_proxies(args,data,chosen,plan['legacy_rule'],old_gate,deadline,mode=='band')
    print(json.dumps({'chosen':plan['chosen'],'mode':mode,'targets':chosen['targets'],'smoke_status':smoke_status},indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('calibrate','evaluate'))
    parser.add_argument('--unsupported',type=Path,default=Path('unsupported_eval_queries.csv'))
    parser.add_argument('--data',type=Path,default=Path.cwd())
    parser.add_argument('--oof',type=Path,default=Path('sprint_results/diagnostics/oof_diagnostics.csv'))
    parser.add_argument('--model',type=Path,default=Path('sprint_results/final'))
    parser.add_argument('--reference',type=Path,default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--cache',type=Path,default=Path('.sprint_cache'))
    parser.add_argument('--output',type=Path,default=Path('sprint_results/stage2_provided_gate'))
    parser.add_argument('--demo',type=Path,default=Path('sprint/DEMO.md'))
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):
        run(args)


if __name__=='__main__':
    main()
