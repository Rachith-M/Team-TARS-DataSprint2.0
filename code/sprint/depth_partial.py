import json
from collections import Counter
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from intent_model import PARAMETER_CAP
from sprint.data import SEED, macro
from sprint.depth import BUDGET, MIN_TWO, choose_depth, fold_result, inner_split, report
from sprint.encoder import MAX_LEN
from sprint.muril import BudgetExpired, VIEWS, accelerator, check_budget, evaluate
from sprint.postprocess import average_tta, logit_scale
from sprint.train import augmentation_rows, fit_fold, take


EPOCHS = 2
BATCH = 48
PREFIX = 'query: '
RATE = 1e-5
BLOCKS = {'D2':4, 'D3':8}


class PrefixBatches:
    def __init__(self, torch, model, tokenizer, texts, split, device, deadline):
        self.rows = len(texts)
        self.parts = []
        tokens = tokenizer([PREFIX+text for text in texts], max_length=MAX_LEN, truncation=True, verbose=False)
        order = np.argsort([len(ids) for ids in tokens['input_ids']], kind='stable')
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for start in range(0,len(order),BATCH):
                check_budget(deadline)
                indices = order[start:start+BATCH]
                items = [{name:column[i] for name,column in tokens.items()} for i in indices]
                batch = tokenizer.pad(items, padding=True, pad_to_multiple_of=8, return_tensors='pt').to(device)
                hidden = model.embeddings(input_ids=batch['input_ids'], token_type_ids=batch.get('token_type_ids'))
                mask = batch['attention_mask']
                attention = model.get_extended_attention_mask(mask, mask.shape, dtype=hidden.dtype)
                for layer in model.encoder.layer[:split]:
                    hidden = layer(hidden, attention_mask=attention)[0]
                self.parts.append((indices,hidden.to(torch.bfloat16).cpu(),mask.cpu()))


def encoded(torch, model, part, split, device):
    _, hidden, mask = part
    hidden = hidden.to(device).float()
    mask = mask.to(device)
    attention = model.get_extended_attention_mask(mask, mask.shape, dtype=hidden.dtype)
    for layer in model.encoder.layer[split:]:
        hidden = layer(hidden, attention_mask=attention)[0]
    pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1)
    return torch.nn.functional.normalize(pooled,dim=1)


def all_embeddings(torch, model, batches, split, device, deadline):
    vectors = np.empty((batches.rows,model.config.hidden_size),dtype=np.float32)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for part in batches.parts:
            check_budget(deadline)
            vectors[part[0]] = encoded(torch,model,part,split,device).cpu().numpy()
    return vectors


def train_epoch(torch, model, router, prepared, optimizer, split, args, deadline, seed):
    batches, lexical, targets, weights = prepared
    width = lexical.shape[1]
    head = router.head
    with threadpool_limits(limits=args.threads):
        lexical_scores = lexical @ head.coef_[:,:width].T + head.intercept_
    semantic = torch.from_numpy(np.ascontiguousarray(head.coef_[:,width:].T)).to(args.device)
    counts = np.bincount(targets,minlength=56)
    positive_weights = (counts.mean()/counts)**router.config['balance']
    order = np.random.default_rng(seed).permutation(len(batches.parts))
    for step,j in enumerate(order,1):
        check_budget(deadline)
        part = batches.parts[j]
        ids = part[0]
        labels = torch.from_numpy(targets[ids].astype(np.int64)).to(args.device)
        base = torch.from_numpy(lexical_scores[ids]).to(args.device)
        target = torch.nn.functional.one_hot(labels,num_classes=56)
        positive = torch.from_numpy(positive_weights[targets[ids]].astype(np.float32)).to(args.device)
        sample_weight = torch.from_numpy(weights[ids]).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(args.device,dtype=torch.bfloat16):
            vector = encoded(torch,model,part,split,args.device)
            logits = base + (router.config['embedding_weight'] * vector) @ semantic
        error = (1-(2*target-1)*logits.float()).clamp_min(0).square()
        loss = ((error * (1 + target*(positive[:,None]-1))).sum(dim=1) * sample_weight).mean()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f'Encoder updates {step}/{len(order)}',flush=True)


def views(torch, model, tokenizer, groups, rows, names, split, args, deadline):
    return {name:PrefixBatches(torch,model,tokenizer,take(groups[name],rows),split,args.device,deadline) for name in names}


def view_scores(torch, model, router, prepared, groups, rows, split, args, deadline):
    return {name:router.score(take(groups[name],rows),all_embeddings(torch,model,batches,split,args.device,deadline))
            for name,batches in prepared.items()}


def fit(torch, args, inputs, config, fit_rows, blocked, seed, top, deadline, valid=None, epochs=EPOCHS):
    from transformers import Adafactor, AutoModel, AutoTokenizer

    groups, labels, _, _, base, _, vectors = inputs
    torch.manual_seed(seed)
    getattr(torch,args.device).manual_seed_all(seed)
    settings = {'local_files_only':True,'trust_remote_code':False}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,**settings)
    model = AutoModel.from_pretrained(args.checkpoint,add_pooling_layer=False,attn_implementation='eager',**settings)
    model.requires_grad_(False).eval().to(args.device)
    split = model.config.num_hidden_layers - top
    params = []
    for i in range(split,model.config.num_hidden_layers):
        layer = model.encoder.layer[i].requires_grad_(True)
        params.append({'params':list(layer.parameters()),'lr':RATE * .9**(model.config.num_hidden_layers-1-i)})
    optimizer = Adafactor(params,lr=RATE,scale_parameter=False,relative_step=False)
    chosen = augmentation_rows(groups,fit_rows,config['augment'],seed,blocked)
    train_text = take(groups['raw'],fit_rows)+take(groups['aug'],chosen)
    targets = np.r_[labels[fit_rows],labels[chosen]]
    weights = np.r_[np.ones(len(fit_rows),dtype=np.float32),np.full(len(chosen),.70,dtype=np.float32)]
    router = fit_fold(config,groups,labels,vectors,fit_rows,seed,check=blocked)
    lexical = router.lexical(train_text)
    rows = PrefixBatches(torch,model,tokenizer,train_text,split,args.device,deadline)
    reference = (views(torch,model,tokenizer,groups,valid,('natural','natural_tta'),split,args,deadline)
                 if valid is not None else None)
    parameters = sum(p.numel() for p in model.parameters()) + router.head.coef_.size + 56 + lexical.shape[1]
    if parameters>PARAMETER_CAP:
        raise ValueError('The training model exceeds the competition parameter cap')
    curve = []
    for epoch in range(1,epochs+1):
        started = perf_counter()
        train_epoch(torch,model,router,(rows,lexical,targets,weights),optimizer,split,args,deadline,seed+epoch)
        vectors = all_embeddings(torch,model,rows,split,args.device,deadline)
        matrix = sparse.hstack((lexical,vectors*config['embedding_weight']),format='csr')
        router.fit_matrix(matrix,targets,weights)
        if reference is not None:
            z = view_scores(torch,model,router,reference,groups,valid,split,args,deadline)
            predicted = average_tta(z['natural'],z['natural_tta'],base['tta_weight']).argmax(axis=1)
            curve.append(macro(labels[valid],predicted))
        print(f'Unfrozen blocks {top}, epoch {epoch}/{epochs}, {perf_counter()-started:.1f}s',flush=True)
    return model, tokenizer, router, split, curve, parameters


def run_partial(args, inputs):
    import torch

    args.device = accelerator(torch,args.device)
    args.threads = max(1,args.threads)
    torch.set_num_threads(args.threads)
    start = perf_counter()
    deadline = start+BUDGET-args.previous_gpu_seconds
    groups, labels, roman, parts, base, config, vectors = inputs
    states = {name:{'rows':[],'logits':{v:np.zeros((len(labels),56),dtype=np.float32) for v in VIEWS},'status':'running'}
              for name in BLOCKS}
    d1_path = args.output/'D1.json'
    d1 = json.loads(d1_path.read_text(encoding='utf-8')) if d1_path.is_file() else {}
    d1_rows = {row['fold']:row['selected'] for row in d1.get('folds',[])}
    if d1.get('deployment',{}).get('passed'):
        for name in BLOCKS:
            report(args,name,base,[],status='untested_D1_met_all_targets')
        return
    try:
        for number,(outer,check) in enumerate(parts):
            if not any(state['status']=='running' for state in states.values()):
                break
            seed = SEED+number
            train, valid = inner_split(groups,labels,outer)
            selection = d1_rows.get(number+1)
            if selection is None:
                selection,_ = choose_depth(groups,labels,roman,vectors,config,outer,check,(train,valid),base['tta_weight'],seed)
            for name,top in BLOCKS.items():
                state = states[name]
                if state['status'] != 'running':
                    continue
                check_budget(deadline)
                print(f'{name} fold {number+1}: grouped inner selection',flush=True)
                inner = fit(torch,args,inputs,selection,train,np.r_[check,valid],seed,top,deadline,valid)
                curve = inner[4]
                epoch = 2 if curve[1] > curve[0]+.0001 else 1
                del inner
                getattr(torch,args.device).empty_cache()
                print(f'{name} fold {number+1}: outer refit for {epoch} epoch(s)',flush=True)
                model,tokenizer,router,split,_,parameters = fit(torch,args,inputs,selection,outer,check,seed,top,deadline,epochs=epoch)
                prepared = views(torch,model,tokenizer,groups,check,VIEWS,split,args,deadline)
                logits = view_scores(torch,model,router,prepared,groups,check,split,args,deadline)
                del prepared,model,router,tokenizer
                getattr(torch,args.device).empty_cache()
                for view,values in logits.items():
                    state['logits'][view][check] = values
                row = fold_result(number,logits,check,labels,roman,base,selected=selection,chosen_epochs=epoch,
                                  inner_perturbed=curve,parameter_count=parameters)
                state['rows'].append(row)
                if len(state['rows'])==2 and np.mean([r['deltas']['perturbed'] for r in state['rows']])<MIN_TWO:
                    state['status'] = 'dropped_after_two'
                elif len(state['rows'])==5:
                    state['status'] = 'complete'
                print(json.dumps({'variant':name,**row}),flush=True)
                report(args,name,base,state['rows'],status=state['status'],elapsed_seconds=perf_counter()-start)
    except BudgetExpired:
        for state in states.values():
            if state['status']=='running':
                state['status']='gpu_time_cap'
    qualified = []
    for name,state in states.items():
        pooled = evaluate(state['logits'],labels,roman,base['tta_weight']) if len(state['rows'])==5 else None
        record = report(args,name,base,state['rows'],pooled,state['status'],elapsed_seconds=perf_counter()-start,
                        device=args.device)
        checks = record['checks']
        if checks['five_folds'] and all(checks[key] for key in checks if key not in ('int8_p95_le_30_ms','parameters_le_300m')):
            qualified.append((record['mean_paired_perturbed_gain'],name,record))
    if qualified:
        _,name,result = max(qualified)
        state = states[name]
        histogram = Counter((row['selected']['max_iter'],row['selected']['tol'],row['chosen_epochs']) for row in state['rows'])
        iterations,tol,epoch = min(histogram,key=lambda key:(-histogram[key],key[0],key[2]))
        try:
            model,tokenizer,router,_,_,count = fit(torch,args,inputs,{**config,'max_iter':iterations,'tol':tol},
                                                     np.arange(len(labels)),None,SEED+5,BLOCKS[name],deadline,epochs=epoch)
            from sprint.depth_export import export_partial
            raw = average_tta(state['logits']['raw'],state['logits']['tta'],base['tta_weight'])
            export_partial(args, name, result, groups['raw'], model,tokenizer,router,count,logit_scale(raw,labels))
        except BudgetExpired:
            report(args,name,base,state['rows'],result['pooled'],'gpu_time_cap_before_export',elapsed_seconds=perf_counter()-start)
