import math
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import onnx

from intent_model import PARAMETER_CAP
from sprint.augment import augment_batch
from sprint.data import SEED, load_data, write_submission
from sprint.export import attach_head, batch_submission_scores
from sprint.muril import json_file
from sprint.runtime import Runtime
from sprint.score_push import WEIGHT


def deploy(args,router,texts,scale,prefix,encoder,tokenizer):
    width = router.features.parameter_count()
    model = attach_head(onnx.load(encoder),router.head.coef_.T,router.head.intercept_,width,
                        router.config['embedding_weight'],True)
    count = sum(math.prod(value.dims) for value in model.graph.initializer)+width
    if count>PARAMETER_CAP:
        json_file(args.output/'deployment.json',{'parameters':count,'passed':False,'reason':'parameter_cap'})
        return
    directory = args.output/'candidate'
    directory.mkdir(exist_ok=False)
    onnx.checker.check_model(model)
    onnx.save(model,directory/'model.int8.onnx')
    del model
    joblib.dump(router.features,directory/'features.joblib',compress=3)
    copyfile(tokenizer,directory/'tokenizer.json')
    json_file(directory/'runtime.json',{'parameters':count,'prefix':prefix,'lexical':'cross-script',
              'scale':scale,'tta_weight':WEIGHT,'cluster_weight':0.,'config_name':args.output.name})
    runtime = Runtime(directory,args.threads)
    rng = np.random.default_rng(SEED+7)
    inputs = [texts[i] for i in rng.choice(len(texts),size=300,replace=False)]
    inputs += augment_batch(inputs[:50],SEED+37,'asr')
    for text in inputs[:12]:
        runtime.predict(text,live=False)
    times = []
    for text in inputs:
        started = perf_counter()
        runtime.predict(text,live=False)
        times.append(1000*(perf_counter()-started))
    p95 = float(np.quantile(times,.95))
    results = {'parameters':count,'p95_ms':p95,'batch1_samples':len(inputs),'passed':p95<50}
    json_file(args.output/'deployment.json',results)
    if not results['passed']:
        return
    _,_,test,ids = load_data(args.data)
    logits = batch_submission_scores(runtime,test)
    write_submission(args.output/'candidate_submission.csv',ids,logits.argmax(axis=1))


def export_frozen(args,router,texts,scale):
    cache = args.cache/'e5-small'
    deploy(args,router,texts,scale,'query: ',cache/'encoder.onnx',cache/'tokenizer.json')


def export_tuned(args,model,tokenizer,router,texts,scale,prefix):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sprint.muril_export import encoder_onnx

    with TemporaryDirectory(dir=args.output,prefix='onnx-') as folder:
        root = Path(folder)
        source = root/'source'
        model.cpu().save_pretrained(source)
        tokenizer.save_pretrained(source)
        encoder_onnx(source,root/'fp32.onnx')
        quantize_dynamic(str(root/'fp32.onnx'),str(root/'int8.onnx'),weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul','Gemm','Gather'],per_channel=True)
        deploy(args,router,texts,scale,prefix,root/'int8.onnx',source/'tokenizer.json')
