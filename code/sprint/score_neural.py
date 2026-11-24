import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from intent_model import FALLBACK
from sprint.data import SEED
from sprint.depth_partial import encoded
from sprint.encoder import MAX_LEN
from sprint.muril import accelerator, check_budget, json_file
from sprint.postprocess import average_tta, logit_scale
from sprint.score_push import WEIGHT, compare, fit_candidate, options, prepare, promote, samples
from sprint.train import EVAL_VIEWS, fit_fold, take


PREFIXES = {'e5-small':'query: ', 'hingbert':''}
NEIGHBOR_LABELS = ((0,37),(1,18,30,39,40),(2,41),(3,16,19),(4,10),(5,12),(6,22),
                   (7,20,27,35,42,44,53),(8,23,33),(9,36),(11,51),(14,28,34,45),
                   (15,17),(21,50,54),(25,47),(26,46),(29,31,48,49),(32,43))


def paired_samples(config,groups,labels,fit,check,variants,seed):
    parts = samples(config,groups,labels,fit,check,variants,seed)
    origins = {name:source for name,_,source,_,_ in variants}
    paraphrases = {}
    for name,text,ids,_,_ in parts[1:]:
        if name=='contrast':
            continue
        source_rows = ids if name=='aug' else origins[name][ids]
        for i,source in zip(ids,source_rows):
            if name!='phrases' or (source+seed)%4!=0:
                paraphrases[int(source)] = (text[i],name)
    pair = [paraphrases.get(int(i),(groups['raw'][i],'duplicate')) for i in fit]
    text = [view for i,(other,_) in zip(fit,pair) for view in (groups['raw'][i],other)]
    counts = {kind:sum(name==kind for _,name in pair) for kind in ('teacher','phrases','aug','duplicate')}
    return text,labels[fit],counts


def prefix_cache(torch,model,tokenizer,texts,split,args,deadline,paired=False):
    tokens = tokenizer([PREFIXES[args.encoder]+t for t in texts],max_length=MAX_LEN,truncation=True,verbose=False)
    length = np.asarray([len(ids) for ids in tokens['input_ids']])
    if paired:
        order = np.argsort(length.reshape(-1,2).max(1),kind='stable')
        order = np.column_stack((2*order,2*order+1)).ravel()
    else:
        order = np.argsort(length,kind='stable')
    result = []
    with torch.no_grad(), torch.autocast(args.device,dtype=torch.bfloat16):
        for start in range(0,len(order),args.batch_size):
            check_budget(deadline)
            ids = order[start:start+args.batch_size]
            items = [{key:col[i] for key,col in tokens.items()} for i in ids]
            batch = tokenizer.pad(items,padding=True,pad_to_multiple_of=8,return_tensors='pt').to(args.device)
            h = model.embeddings(input_ids=batch['input_ids'],token_type_ids=batch.get('token_type_ids'))
            mask = batch['attention_mask']
            attention = model.get_extended_attention_mask(mask,mask.shape,dtype=h.dtype)
            for layer in model.encoder.layer[:split]:
                h = layer(h,attention_mask=attention)[0]
            result.append((ids,h.to(torch.bfloat16).cpu(),mask.cpu()))
    return result


def train(torch,args,inputs,fit,check,seed,deadline):
    from transformers import Adafactor, AutoModel, AutoTokenizer

    groups,labels,_,_,_,_,config,vectors,variants,_ = inputs
    warm,_ = fit_candidate(config,groups,labels,vectors,variants,fit,check,seed)
    text,target,counts = paired_samples(config,groups,labels,fit,check,variants,seed)
    width = warm.features.parameter_count()
    lex_coef = np.ascontiguousarray(warm.head.coef_[:,:width]*5)
    lex = warm.lexical(text) @ lex_coef.T
    torch.manual_seed(seed)
    getattr(torch,args.device).manual_seed_all(seed)
    options = {'local_files_only':True,'trust_remote_code':False}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,**options)
    model = AutoModel.from_pretrained(args.checkpoint,add_pooling_layer=False,attn_implementation='eager',**options)
    if model.config.model_type!='bert':
        raise ValueError('This paired training path expects a BERT-compatible encoder')
    model.requires_grad_(False).eval().to(args.device)
    depth = model.config.num_hidden_layers
    split = depth-4
    head = torch.nn.Linear(model.config.hidden_size,56).to(args.device)
    with torch.no_grad():
        head.bias.copy_(torch.from_numpy(warm.head.intercept_*5).to(args.device))
        if args.encoder=='e5-small':
            head.weight.copy_(torch.from_numpy(np.ascontiguousarray(warm.head.coef_[:,width:]*5)).to(args.device))
        else:
            head.weight.zero_()
    encoder_parameters = []
    for i in range(split,depth):
        encoder_parameters.append({'params':model.encoder.layer[i].requires_grad_(True).parameters(),
                                   'lr':1.5e-5*.9**(depth-i-1)})
    encoder_parameters.append({'params':head.parameters(),'lr':.001})
    optimizer = Adafactor(encoder_parameters,lr=1.5e-5,scale_parameter=False,relative_step=False)
    frequency = np.bincount(target,minlength=56)
    class_weight = (frequency.mean()/np.maximum(frequency,1))**.45
    class_weight = torch.as_tensor(class_weight,dtype=torch.float32,device=args.device)
    close = np.zeros((56,56),dtype=bool)
    for group in NEIGHBOR_LABELS:
        close[np.ix_(group,group)] = True
    np.fill_diagonal(close,False)
    has_rival = torch.as_tensor(close.any(1),device=args.device)
    close = torch.as_tensor(close,device=args.device)
    pool = prefix_cache(torch,model,tokenizer,text,split,args,deadline,True)
    F = torch.nn.functional
    rng = np.random.default_rng(seed)
    for epoch in range(args.epochs):
        for b,j in enumerate(rng.permutation(len(pool)),1):
            check_budget(deadline)
            part = pool[j]
            ids = part[0]
            y = torch.as_tensor(target[ids[::2]//2].astype(np.int64),device=args.device)
            yt = y.repeat_interleave(2)
            values = torch.from_numpy(lex[ids]).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(args.device,dtype=torch.bfloat16):
                logits = values+head(config['embedding_weight']*encoded(torch,model,part,split,args.device)).float()
            ce = F.cross_entropy(logits,yt,weight=class_weight,label_smoothing=.03)
            linked = logits.reshape(-1,2,56)
            logp = F.log_softmax(linked,dim=-1)
            p = logp.detach().exp()
            consistency = -(p.flip(dims=[1])*logp).sum(-1).mean(1)
            consistency = (consistency*(y!=FALLBACK)).mean()
            mean = linked.mean(1)
            correct = mean.gather(1,y[:,None]).squeeze(1)
            rival = mean.masked_fill(F.one_hot(y,num_classes=56).bool(),float('-inf')).max(1).values
            domain_rival = mean.masked_fill(~close[y],float('-inf')).max(1).values
            domain = torch.where(has_rival[y],F.relu(.85+domain_rival-correct),0.).mean()
            loss = ce+.18*consistency+.10*F.relu(.60+rival-correct).mean()+.12*domain
            loss.backward()
            optimizer.step()
            if b%150==0:
                print(f'{args.encoder} epoch {epoch+1}: {b}/{len(pool)} paired batches',flush=True)
    del pool
    warm.head.coef_ = np.asfortranarray(np.concatenate((lex_coef,head.weight.detach().cpu().numpy()),axis=1))
    warm.head.intercept_ = head.bias.detach().cpu().numpy().copy()
    return model,tokenizer,warm,split,counts


def score(torch,args,inputs,trained,rows,deadline):
    groups = inputs[0]
    model,tokenizer,router,split,_ = trained
    output = {}
    for name in EVAL_VIEWS:
        texts = take(groups[name],rows)
        vectors = np.empty((len(texts),model.config.hidden_size),dtype=np.float32)
        batches = prefix_cache(torch,model,tokenizer,texts,split,args,deadline)
        with torch.no_grad(), torch.autocast(args.device,dtype=torch.bfloat16):
            for part in batches:
                check_budget(deadline)
                vectors[part[0]] = encoded(torch,model,part,split,args.device).cpu().numpy()
        output[name] = router.score(texts,vectors)
    return output


def run(args):
    import torch

    args.device = accelerator(torch,args.device)
    torch.set_num_threads(args.threads)
    start = perf_counter()
    deadline = start+args.hours*3600
    args.output.mkdir(parents=True,exist_ok=False)
    inputs = prepare(args)
    groups,labels,original,challenge,frames,scripts,config,vectors,_,saved = inputs
    report = {'seed':SEED,'encoder':args.encoder,'folds':[],'status':'running'}
    all_scores = {v:np.empty_like(saved[v]) for v in EVAL_VIEWS}
    for number,(fit,check) in enumerate(original):
        trained = train(torch,args,inputs,fit,check,SEED+number,deadline)
        current = score(torch,args,inputs,trained,check,deadline)
        result = compare(current,{k:z[check] for k,z in saved.items()},check,labels,scripts,trained[-1])
        report['folds'].append({'fold':number+1,**result})
        print(f"{args.encoder} fold {number+1}: perturbed Δ {result['delta']['perturbed']:+.5f}, Roman ASR Δ {result['delta']['roman_asr']:+.5f}",flush=True)
        del trained
        getattr(torch,args.device).empty_cache()
        for name,z in current.items():
            all_scores[name][check] = z
        if len(report['folds'])==2 and np.mean([row['delta']['perturbed'] for row in report['folds']])<.012:
            report['status'] = 'dropped_after_two'
        json_file(args.output/'results.json',report)
        if report['status']=='dropped_after_two':
            return
    fit,check = challenge
    trained = train(torch,args,inputs,fit,check,SEED+9,deadline)
    from sprint.score_push import score as frozen_score
    base = fit_fold(config,groups,labels,vectors,fit,SEED+9,check=check)
    report['challenge'] = compare(score(torch,args,inputs,trained,check,deadline),frozen_score(base,groups,vectors,check),
                                  check,labels,scripts,trained[-1])
    report['pooled'] = compare(all_scores,saved,np.arange(len(labels)),labels,scripts,None)
    report['checks'] = promote(report['pooled']['candidate'],report['pooled']['baseline'],report['folds'],report['challenge'])
    report['status'] = 'complete'
    report['elapsed_seconds'] = perf_counter()-start
    json_file(args.output/'results.json',report)
    if all(report['checks'].values()):
        from sprint.score_deploy import export_tuned
        del trained
        model,tokenizer,router,_,_ = train(torch,args,inputs,np.arange(len(labels)),np.array([],dtype=int),SEED+5,deadline)
        scale = logit_scale(average_tta(all_scores['raw'],all_scores['tta'],WEIGHT),labels)
        export_tuned(args,model,tokenizer,router,groups['raw'],scale,PREFIXES[args.encoder])


def main():
    parser = argparse.ArgumentParser()
    options(parser)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--encoder',choices=tuple(PREFIXES),default='e5-small')
    parser.add_argument('--device',choices=('auto','xpu','cuda'),default='auto')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=48)
    parser.add_argument('--hours',type=float,default=1.)
    args = parser.parse_args()
    if args.batch_size<2 or args.batch_size%2:
        parser.error('Paired batches require a positive even batch size')
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__=='__main__':
    main()
