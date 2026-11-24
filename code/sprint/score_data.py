import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from intent_model import FALLBACK, Views, clean, roman_token, script
from sprint.augment import digitize, romanize, strip_fillers
from sprint.data import SEED, folds, load_training


MODEL = 'hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M'
ENDPOINT = 'http://127.0.0.1:11434/api/chat'
CHALLENGE_SEED = 2947
NEGATION = re.compile(r"\b(nahi|nahin|nheen|naheen|mat|not|never|don't|dont|don t)\b")
ALIASES = (
    ('raushanee', 'brightness'), ('roshanee','brightness'), ('braaitanes','brightness'),
    ('volyoom', 'volume'), ('volyum','volume'), ('speekar','speaker'), ('myoot','mute'),
    ('aavaaz','sound'), ('dhvani','sound'), ('prakash','light'), ('battiyaan','lights'),
    ('laaits','lights'), ('soket','socket'), ('plag','plug'), ('smaart','smart'), ('paavar','power'),
    ('kelendar','calendar'), ('kailendar','calendar'), ('baithak','meeting'), ('alaarm','alarm'),
    ('yaad dilaaen','remind me'), ('yaad dilaao','remind me'), ('radd karo','cancel'),
    ('hataa den','remove'), ('hataaen','remove'), ('hataao','remove'), ('jod den','add'),
    ('joren','add'), ('barhaaen','increase'), ('barhaao','increase'), ('ghataaen','decrease'),
    ('band kar do','turn off'), ('band karo','turn off'), ('chaaloo kar do','turn on'),
    ('chaaloo karo','turn on'), ('set kar do','set'), ('set karo','set'),
    ('samparkon','contacts'), ('sanparkon','contacts'), ('sampark','contact'), ('sanpark','contact'),
    ('eemel','email'), ('imel','email'), ('inaboks','inbox'), ('javaab bhejo','reply'),
    ('soochee','list'), ('soocheeyaan','lists'), ('aaitam','item'), ('aid','add'),
    ('relagaaree','train'), ('tren','train'), ('tikat','ticket'), ('buk karo','book'),
    ('taiksee','taxi'), ('taiksi','taxi'), ('kaib','cab'), ('yaataayaat','traffic'),
    ('traiphik','traffic'), ('retaurant','restaurant'), ('restorent','restaurant'),
    ('nikatatam','nearest'), ('aas paas','nearby'), ('pijjaa','pizza'), ('pizzaa','pizza'),
    ('dileevaree','delivery'), ('ordar','order'), ('tekave','takeaway'), ('resipee','recipe'),
    ('sangeet','music'), ('gaanaa','song'), ('gaane','songs'), ('plelist','playlist'),
    ('chalaao','play'), ('bajaaen','play'), ('shaphal','shuffle'), ('doharaaen','repeat'),
    ('ripeet','repeat'), ('rediyo','radio'), ('steshan','station'), ('chainal','channel'),
    ('odiyobuk','audiobook'), ('podakaast','podcast'), ('episod','episode'), ('chutakulaa','joke'),
    ('philmen','movies'), ('philm','movie'), ('sinemaaghar','cinema'), ('taareekh','date'),
    ('mausam','weather'), ('poorvaanumaan','forecast'), ('baarish','rain'), ('samay','time'),
    ('samay kshetr','time zone'), ('samaachaar','news'), ('surkhiyaan','headlines'),
    ('sheyar','share'), ('maarket','market'), ('vinimay dar','exchange rate'), ('mudraa','currency'),
    ('soshal meediyaa','social media'), ('tveet','tweet'), ('phesabuk','facebook'),
    ('kaaryakram','event'), ('saptaahaant','weekend'), ('paribhaashaa','definition'),
    ('paribhaashit karo','define'), ('matlab','meaning'), ('vaikyoom','vacuum'),
    ('saphaaee','cleaning'), ('shuroo karo','start'), ('kaise banaate hain','how to make'),
    ('kaise banaaen','how to make'), ('kitane baje','what time'), ('gunaa','times'),
    ('vibhaajit','divided'), ('vargamool','square root'),
)


def aliases():
    mapping = {' '.join(map(roman_token, src.split())): dest for src,dest in ALIASES}
    pattern = r'(?<!\S)(?:' + '|'.join(re.escape(s) for s in sorted(mapping,key=lambda key:(-len(key),key))) + r')(?!\S)'
    return mapping, re.compile(pattern)


TRANSLATION, PHRASES = aliases()
SWITCH = ((('band','off','bujhaao','bujhaaen'),('chaaloo','on','jalaao','jalaaen')))
LEVEL = ((('barhaao','barhaaen','badhaao','badhaaen','increase','raise','up'),
          ('ghataao','ghataaen','kam karo','kam kar do','decrease','lower','down')))
OPPOSITES = (
    (8,23,('soket','plag','plug','socket','vemo','wemo'),SWITCH),
    (39,40,('batti','battiyaan','laait','laaits','lights','light','laimp','lamp'),SWITCH),
    (18,30,('roshanee','prakaash','roshni','brightness','light','lights','laaits','braaitanes'),LEVEL),
    (14,34,('volyoom','volyum','volume','aavaaz','sound','speekar','speaker'),LEVEL),
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def phrase_views(texts):
    surface = [strip_fillers(romanize(text)) for text in texts]
    result = []
    for index,text in enumerate(surface):
        normalized = ' '.join(map(roman_token,text.split()))
        matches = list(PHRASES.finditer(normalized))
        if not matches:
            result.append(text)
            continue
        words = [m[0] for m in re.finditer(r'\S+',text)]
        rng = np.random.default_rng([SEED+73,index])
        selected = range(len(matches)) if len(matches)<=3 else sorted(rng.choice(len(matches),3,replace=False))
        for i in reversed(selected):
            match = matches[i]
            start = normalized[:match.start()].count(' ')
            width = match[0].count(' ')+1
            words[start:start+width] = [TRANSLATION[match[0]]]
        result.append(digitize(' '.join(words)))
    return result


def normalized_words(text):
    return ' '.join(map(roman_token,romanize(text).split()))


def word_pattern(words):
    terms = sorted({normalized_words(word) for word in words},key=lambda w:(-len(w),w))
    return re.compile(r'(?<!\S)(?:'+'|'.join(map(re.escape,terms))+r')(?!\S)')


def counterfactuals(texts,labels):
    rules = {}
    for left,right,objects,(first,second) in OPPOSITES:
        domain = word_pattern(objects)
        a,b = word_pattern(first),word_pattern(second)
        rules[left] = right,domain,a,b,normalized_words(second[0])
        rules[right] = left,domain,b,a,normalized_words(first[0])
    outputs,sources,targets = [],[],[]
    for i,(text,label) in enumerate(zip(texts,labels)):
        if label not in rules:
            continue
        target,domain,action,rival,replacement = rules[label]
        canonical = normalized_words(text)
        hits = list(action.finditer(canonical))
        if not domain.search(canonical) or len(hits)!=1 or rival.search(canonical) or NEGATION.search(canonical):
            continue
        match = hits[0]
        outputs.append(canonical[:match.start()]+replacement+canonical[match.end():])
        sources.append(i)
        targets.append(target)
    return outputs,np.asarray(sources,dtype=int),np.asarray(targets,dtype=int)


def frame_groups(texts, labels):
    views = Views.from_texts(texts)
    words = [text.split() for text in views.roman]
    counts = Counter(word for row in words for word in set(row))
    roots = list(range(len(words)))
    sizes = [1]*len(words)
    def root(i):
        while roots[i]!=i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    def union(i,j,limit):
        i,j = root(i),root(j)
        if i!=j and sizes[i]+sizes[j]<=limit:
            if sizes[i]<sizes[j]:
                i,j = j,i
            roots[j] = i
            sizes[i] += sizes[j]
    known = {}
    for i,key in enumerate(views.phonetic):
        if key:
            union(i,known.setdefault(key,i),len(texts))
    slots = defaultdict(list)
    for i,row in enumerate(words):
        if len(row)<5:
            continue
        signature = tuple('@' if word.isdecimal() or counts[word]<6 else word for word in row)
        if signature.count('@') and len(signature)-signature.count('@')>=4:
            slots[(int(labels[i]),signature)].append(i)
        if len(row)>=7:
            for j,word in enumerate(row):
                if len(word)>=4 and counts[word]<28:
                    slots[(int(labels[i]),(*row[:j],'@',*row[j+1:]))].append(i)
    for family in slots.values():
        if 1<len(family)<=24:
            for i in family[1:]:
                union(i,family[0],64)
    return np.asarray([root(i) for i in range(len(words))])


def splits(texts, labels):
    original = folds(texts,labels)
    groups = frame_groups(texts,labels)
    script_tags = np.array([script(text) for text in texts])
    strata = [f'{y}_{sc=="roman"}' for y,sc in zip(labels,script_tags)]
    challenge = next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=CHALLENGE_SEED).split(
        texts,strata,groups))
    if len(set(labels[challenge[1]]))!=len(set(labels)) or np.intersect1d(groups[challenge[0]],groups[challenge[1]]).size:
        raise ValueError('The locked challenge has missing classes or shared frames')
    return original,challenge,groups,script_tags


def anchor_groups(texts, labels, original, challenge, groups, script_tags, count):
    rng = np.random.default_rng(SEED+618)
    outer = np.empty(len(texts),dtype=np.int8)
    for i,(_,check) in enumerate(original):
        outer[check] = i
    held = np.zeros(len(texts),dtype=bool)
    held[challenge[1]] = True
    plan = defaultdict(list)
    for y in range(FALLBACK):
        seen = set()
        indices = rng.permutation(np.flatnonzero(labels==y))
        eligible = [int(i) for i in indices if 3<=len(texts[i].split())<=22]
        eligible = sorted(eligible,key=lambda i:(held[i],script_tags[i]=='roman'))
        roman = [i for i in eligible if script_tags[i]=='roman']
        devanagari = [i for i in eligible if script_tags[i]!='roman']
        n = 0
        for i in [*roman[:max(1,count//3)],*devanagari,*roman[max(1,count//3):]]:
            if groups[i] not in seen:
                plan[(int(outer[i]),bool(held[i]))].append(i)
                seen.add(groups[i])
                n += 1
                if n==count:
                    break
    shuffled = [rng.permutation(batch).tolist() for batch in plan.values()]
    batches = [batch[i:i+8] for batch in shuffled for i in range(0,len(batch),8)]
    return [batches[i] for i in rng.permutation(len(batches))]


def local_rewrites(session, texts, anchor, num_gpu):
    prompt = ('Rewrite each independent voice-assistant request into one very natural everyday Hindi-English '
              'Hinglish request. Write in Roman script only. Use everyday Hinglish grammar and English content '
              'words where natural. Vary word order and phrasing. Preserve the same specific action, object, '
              'negation, time, numbers, and names. Do not answer the request or change what the user wants. '
              'Keep each rewrite short. Output JSON with a rewrites array, with strings in the exact input order.')
    schema = {'type':'object','properties':{'rewrites':{'type':'array','items':{'type':'string'},
              'minItems':len(anchor),'maxItems':len(anchor)}},'required':['rewrites']}
    data = {'model':MODEL,'messages':[{'role':'system','content':prompt},
            {'role':'user','content':json.dumps([texts[i] for i in anchor],ensure_ascii=False)}],
            'stream':False,'think':False,'format':schema,'keep_alive':'4m',
            'options':{'temperature':.25,'seed':SEED+anchor[0],'num_ctx':2048,'num_predict':len(anchor)*48+50,
                       'num_thread':4,'num_gpu':num_gpu}}
    request = Request(ENDPOINT,data=json.dumps(data).encode('utf-8'),headers={'Content-Type':'application/json'})
    try:
        with session.open(request,timeout=240) as response:
            payload = json.loads(json.load(response)['message']['content'])['rewrites']
    except HTTPError as error:
        raise RuntimeError(f'Local rewrite worker returned HTTP {error.code}') from None
    return payload if isinstance(payload,list) and len(payload)==len(anchor) else []


def valid_rewrite(text, source):
    if not isinstance(text,str) or len(text)>200 or script(text)!='roman' or '\n' in text:
        return False
    left,right = romanize(source),clean(text)
    if not 3<=len(right.split())<=23 or left==right:
        return False
    if bool(NEGATION.search(left))!=bool(NEGATION.search(right)):
        return False
    return set(re.findall(r'\d+',left))<=set(re.findall(r'\d+',right))


def generate(args):
    train = args.data/'train.csv'
    texts,labels = load_training(train)
    original,challenge,groups,script_tags = splits(texts,labels)
    plan = anchor_groups(texts,labels,original,challenge,groups,script_tags,args.per_class)
    with train.open('rb') as file:
        dataset = hashlib.file_digest(file,'sha256').hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    session = build_opener(ProxyHandler({}),NoRedirect())
    with session.open('http://127.0.0.1:11434/api/tags',timeout=10) as response:
        matches = [item['digest'] for item in json.load(response)['models'] if item['name'].casefold()==MODEL.casefold()]
    if len(matches)!=1:
        raise ValueError('The requested local rewrite checkpoint is unavailable or ambiguous')
    digest = matches[0]
    previous = set()
    if args.out.exists():
        with args.out.open(encoding='utf-8') as file:
            for line in file:
                batch = json.loads(line)
                if batch['train_sha256']!=dataset or batch['model']!=MODEL or batch['model_digest']!=digest:
                    raise ValueError('Rewrite artifact model or input changed')
                previous.add(tuple(batch['anchors']))
    plan = [rows for rows in plan if tuple(rows) not in previous]
    if args.batches is not None:
        plan = plan[:args.batches]
    print(f'Local rewrite batches scheduled: {len(plan)}',flush=True)
    accepted = 0
    with args.out.open('a',encoding='utf-8',newline='\n') as file:
        for j,anchor in enumerate(plan,1):
            try:
                rephrases = local_rewrites(session,texts,anchor,args.num_gpu)
            except (json.JSONDecodeError,KeyError,TypeError):
                rephrases = []
            pairs = [[i,text.strip()] for i,text in zip(anchor,rephrases) if valid_rewrite(text,texts[i])]
            file.write(json.dumps({'anchors':anchor,'pairs':pairs,'train_sha256':dataset,'model':MODEL,
                                   'model_digest':digest},ensure_ascii=False)+'\n')
            file.flush()
            accepted += len(pairs)
            if j%12==0 or j==len(plan):
                print(f'{j}/{len(plan)} batches, {accepted} valid rewrites',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path,default=Path('sprint_results/score_push/rewrites.jsonl'))
    parser.add_argument('--per-class',type=int,default=24)
    parser.add_argument('--batches',type=int)
    parser.add_argument('--num-gpu',type=int,default=0)
    generate(parser.parse_args())


if __name__=='__main__':
    main()
