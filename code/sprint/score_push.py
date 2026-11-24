import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK, clean
from sprint.data import SEED, dataset_key, load_training, macro
from sprint.encoder import Encoder, cache_embeddings, variant_groups
from sprint.muril import json_file
from sprint.postprocess import average_tta, logit_scale
from sprint.score_data import MODEL, counterfactuals, phrase_views, splits
from sprint.train import EVAL_VIEWS, Router, augmentation_rows, fit_fold, take


WEIGHT = .35
PROMOTION_GAIN = .025
PROMOTION_CHALLENGE = .010


def read_rewrites(path, train, labels):
    if path is None:
        return [],[],[]
    digest = fingerprint(train)
    texts, origins, contexts = [], [], []
    models = set()
    with path.open(encoding='utf-8') as file:
        for line in file:
            row = json.loads(line)
            sources = tuple(row['anchors'])
            if row['train_sha256']!=digest or row['model']!=MODEL or not sources:
                raise ValueError('The rewrite provenance does not match this train.csv and model')
            models.add(row['model_digest'])
            if any(not 0<=i<len(labels) for i in sources):
                raise ValueError('A rewrite context is invalid')
            for i,text in row['pairs']:
                if i not in sources:
                    raise ValueError('A rewrite is missing its source')
                texts.append(text)
                origins.append(i)
                contexts.append(sources)
    if len(models)!=1:
        raise ValueError('The rewrite corpus must come from one local model digest')
    return texts,np.asarray(origins,dtype=int),contexts


def fingerprint(path):
    if path is None:
        return None
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def metrics(logits, labels, scripts):
    raw = average_tta(logits['raw'],logits['tta'],WEIGHT).argmax(1)
    shift = average_tta(logits['natural'],logits['natural_tta'],WEIGHT).argmax(1)
    asr = logits['asr'].argmax(1)
    is_roman = scripts=='roman'
    return {'overall':macro(labels,raw),'perturbed':macro(labels,shift),'asr':macro(labels,asr),
            'roman':macro(labels[is_roman],raw[is_roman]),'roman_asr':macro(labels[is_roman],asr[is_roman]),
            'class55_f1':float(f1_score(labels==FALLBACK,raw==FALLBACK,zero_division=0))}


def prepare(args, *, include_reference_vectors=True):
    train = args.data/'train.csv'
    text, labels = load_training(train)
    original, challenge, frames, scripts = splits(text,labels)
    settings = json.loads((args.reference/'metrics.json').read_text(encoding='utf-8'))
    if settings['tta_weight']!=WEIGHT or settings['cluster_weight']:
        raise ValueError('The reference settings changed')
    groups = variant_groups(text,[])
    with np.load(args.reference/'scores.npz') as saved:
        expected_folds = np.empty(len(labels),dtype=int)
        for number,(_,check) in enumerate(original):
            expected_folds[check] = number
        if not np.array_equal(saved['target'],labels) or not np.array_equal(saved['fold_id'],expected_folds):
            raise ValueError('The saved baseline does not match train.csv or the prescribed folds')
        saved_scores = {v:saved[v] for v in EVAL_VIEWS}
    teacher, origins, contexts = read_rewrites(args.rewrites,train,labels)
    if args.rewrites is not None and not teacher:
        raise ValueError('The specified rewrite corpus has no accepted examples')
    encoder_groups = {name:groups[name] for name in (*EVAL_VIEWS,'aug')}
    variants = []
    if args.phrases:
        encoder_groups['phrases'] = phrase_views(text)
        variants.append(('phrases',encoder_groups['phrases'],np.arange(len(text)),None,labels))
    if teacher:
        encoder_groups['teacher'] = teacher
        variants.append(('teacher',teacher,origins,contexts,labels[origins]))
    if args.counterfactuals:
        contrast,source,target = counterfactuals(text,labels)
        encoder_groups['contrast'] = contrast
        variants.append(('contrast',contrast,source,None,target))
    cached = (cache_embeddings(encoder_groups,args.cache,'e5-small',threads=args.threads)
              if include_reference_vectors else {})
    vectors = {name:np.load(path,mmap_mode='r') for name,path in cached.items()}
    return groups,labels,original,challenge,frames,scripts,settings['config'],vectors,variants,saved_scores


def samples(config,groups,labels,fit,check,variants,seed):
    chosen = augmentation_rows(groups,fit,config['augment'],seed,check)
    present = np.zeros(len(labels),dtype=bool)
    present[fit] = True
    blocked = {clean(groups[view][i]) for view in EVAL_VIEWS for i in check}
    known = defaultdict(set)
    for i in fit:
        known[clean(groups['raw'][i])].add(int(labels[i]))
    seen = set(known)
    parts = [('raw',groups['raw'],fit,labels[fit],1.),('aug',groups['aug'],chosen,labels[chosen],.7)]
    for name,text,origins,context,targets in variants:
        accepted = []
        for i,source in enumerate(origins):
            if not present[source] or labels[source]==FALLBACK or (context is not None and not np.all(present[list(context[i])])):
                continue
            key = clean(text[i])
            if key and key not in blocked and key not in seen:
                accepted.append(i)
                seen.add(key)
        accepted = np.asarray(accepted,dtype=int)
        if len(accepted):
            parts.append((name,text,accepted,targets[accepted],.55))
    return parts


def fit_candidate(config,groups,labels,vectors,variants,fit,check,seed):
    parts = samples(config,groups,labels,fit,check,variants,seed)
    texts = [item for _,text,ids,_,_ in parts for item in take(text,ids)]
    target = np.concatenate([labels for _,_,_,labels,_ in parts])
    weights = np.concatenate([np.full(len(ids),weight) for _,_,ids,_,weight in parts])
    embeddings = np.concatenate([vectors[name][ids] for name,_,ids,_,_ in parts])
    return Router(config).fit(texts,target,embeddings,weights),{name:len(ids) for name,_,ids,_,_ in parts}


def score(model,groups,vectors,check):
    return {v:model.score(take(groups[v],check),vectors[v][check]) for v in EVAL_VIEWS}


def promote(pooled, base, paired, challenge):
    gains = [row['delta']['perturbed'] for row in paired]
    return {'perturbed_gain':pooled['perturbed']-base['perturbed']>=PROMOTION_GAIN,
            'fold_wins':sum(value>0 for value in gains)>=4,
            'clean':pooled['overall']>=base['overall']-.005,
            'class55':pooled['class55_f1']>=max(.85,base['class55_f1']-.02),
            'challenge_roman_asr':challenge['delta']['roman_asr']>=PROMOTION_CHALLENGE}


def compare(candidate, baseline, rows, labels, scripts, training):
    old = metrics(baseline,labels[rows],scripts[rows])
    new = metrics(candidate,labels[rows],scripts[rows])
    return {'baseline':old,'candidate':new,'delta':{name:new[name]-old[name] for name in old},
            'evaluation_rows':len(rows),'training_rows':training}


def run(args):
    start = perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    prepared = prepare(args)
    groups,labels,original,challenge,frames,scripts,config,vectors,variants,saved = prepared
    report = {'seed':SEED,'folds':[],'sources':[x[0] for x in variants],'rewrite_sha256':fingerprint(args.rewrites),
              'template_groups':int(len(np.unique(frames))),'largest_template_group':int(np.bincount(frames).max())}
    all_scores = {v:np.empty_like(saved[v]) for v in EVAL_VIEWS}
    for number,(fit,check) in enumerate(original):
        model,training = fit_candidate(config,groups,labels,vectors,variants,fit,check,SEED+number)
        current = score(model,groups,vectors,check)
        entry = compare(current,{v:saved[v][check] for v in EVAL_VIEWS},check,labels,scripts,training)
        report['folds'].append({'fold':number+1,**entry})
        for v in EVAL_VIEWS:
            all_scores[v][check] = current[v]
        print(f"Fold {number+1}: perturbed Δ {entry['delta']['perturbed']:+.5f}, Roman ASR Δ {entry['delta']['roman_asr']:+.5f}",flush=True)
        json_file(args.output/'results.json',report)
    fit,check = challenge
    model,training = fit_candidate(config,groups,labels,vectors,variants,fit,check,SEED+9)
    reference = fit_fold(config,groups,labels,vectors,fit,SEED+9,check=check)
    report['challenge'] = compare(score(model,groups,vectors,check),score(reference,groups,vectors,check),
                                  check,labels,scripts,training)
    report['pooled'] = compare(all_scores,saved,np.arange(len(labels)),labels,scripts,None)
    report['checks'] = promote(report['pooled']['candidate'],report['pooled']['baseline'],report['folds'],report['challenge'])
    report['elapsed_seconds'] = perf_counter()-start
    json_file(args.output/'results.json',report)
    print(json.dumps({k:report[k] for k in ('pooled','challenge','checks')},indent=2),flush=True)
    if all(report['checks'].values()):
        from sprint.score_deploy import export_frozen
        model,_ = fit_candidate(config,groups,labels,vectors,variants,np.arange(len(labels)),np.array([],dtype=int),SEED+5)
        export_frozen(args,model,groups['raw'],logit_scale(average_tta(all_scores['raw'],all_scores['tta'],WEIGHT),labels))


def options(parser):
    parser.add_argument('--data',type=Path,default=Path.cwd())
    parser.add_argument('--cache',type=Path,default=Path('.sprint_cache'))
    parser.add_argument('--reference',type=Path,default=Path('sprint_results/hybrid_small'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rewrites',type=Path)
    parser.add_argument('--phrases',action='store_true')
    parser.add_argument('--counterfactuals',action='store_true')
    parser.add_argument('--threads',type=int,default=3)


def export_validated(args):
    report = json.loads((args.output/'results.json').read_text(encoding='utf-8'))
    if len(report['folds'])!=5 or report['pooled']['delta']['perturbed']<=0:
        raise ValueError('Export requires a completed five-fold improvement')
    groups,labels,_,_,_,_,config,vectors,variants,_ = prepare(args)
    if report['sources'] != [x[0] for x in variants] or report.get('rewrite_sha256')!=fingerprint(args.rewrites):
        raise ValueError('This run was evaluated with a different set of augmentations')
    model,_ = fit_candidate(config,groups,labels,vectors,variants,np.arange(len(labels)),np.array([],dtype=int),SEED+5)
    from sprint.score_deploy import export_frozen
    reference = json.loads((args.reference/'metrics.json').read_text(encoding='utf-8'))
    export_frozen(args,model,groups['raw'],reference['scale'])


def main():
    parser = argparse.ArgumentParser()
    options(parser)
    parser.add_argument('--export-validated',action='store_true')
    args = parser.parse_args()
    if args.rewrites is None and not args.phrases and not args.counterfactuals:
        parser.error('Select at least one train-only augmentation source')
    with threadpool_limits(limits=args.threads):
        (export_validated if args.export_validated else run)(args)


if __name__=='__main__':
    main()
