import json
import math
from collections import Counter
from shutil import copyfile
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import onnx

from intent_model import PARAMETER_CAP
from sprint.augment import augment_batch
from sprint.data import SEED
from sprint.export import attach_head, benchmark
from sprint.muril import json_file
from sprint.runtime import Runtime
from sprint.train import fit_fold


def export(args, name, result, texts, router, encoder_path, tokenizer_path, scale, training_count=None):
    destination = args.output / f'{name}_int8'
    destination.mkdir(exist_ok=True)
    width = router.features.parameter_count()
    graph = attach_head(onnx.load(encoder_path),router.head.coef_.T,router.head.intercept_,width,
                        router.config['embedding_weight'],True)
    count = sum(math.prod(value.dims) for value in graph.graph.initializer) + width
    largest = max(count,training_count or 0)
    if largest>PARAMETER_CAP:
        raise ValueError(f'{name} exceeds the parameter cap at {largest:,}')
    onnx.checker.check_model(graph)
    onnx.save(graph,destination/'model.int8.onnx')
    del graph
    joblib.dump(router.features,destination/'features.joblib',compress=3)
    copyfile(tokenizer_path,destination/'tokenizer.json')
    json_file(destination/'runtime.json',{'config_name':name,'lexical':'cross-script','prefix':'query: ',
              'tta_weight':result['baseline']['tta_weight'],'cluster_weight':0.,'scale':scale,'parameters':count})
    rng = np.random.default_rng(SEED)
    sample = [texts[i] for i in rng.choice(len(texts),size=300,replace=False)]
    sample += augment_batch(sample[:50],SEED+96,'natural')
    timing = benchmark(Runtime(destination,args.threads),sample)
    result['checks'].update({'int8_p95_le_30_ms':timing['p95_ms']<=30,'parameters_le_300m':largest<=PARAMETER_CAP})
    result['deployment'] = {'parameters':count,'training_parameters':training_count,'latency':timing,
                            'passed':all(result['checks'].values())}
    result['status'] = 'qualified' if result['deployment']['passed'] else 'deployment_failed'
    json_file(args.output/f'{name}.json',result)
    print(json.dumps(result['deployment']),flush=True)


def export_d1(args, inputs, result):
    groups, labels, _, _, _, config, vectors = inputs
    best = Counter((row['selected']['max_iter'],row['selected']['tol']) for row in result['folds'])
    iterations,tol = min(best,key=lambda key:(-best[key],key[0]))
    router = fit_fold({**config,'max_iter':iterations,'tol':tol},groups,labels,vectors,np.arange(len(labels)),SEED+5)
    source = args.cache/'e5-small'
    scale = json.loads((args.reference/'metrics.json').read_text(encoding='utf-8'))['scale']
    export(args,'D1',result,groups['raw'],router,source/'encoder.onnx',source/'tokenizer.json',scale)


def export_partial(args, name, result, texts, model, tokenizer, router, parameters, scale):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sprint.muril_export import encoder_onnx

    with TemporaryDirectory(dir=args.output,prefix=f'{name}_export_') as folder:
        from pathlib import Path
        path = Path(folder)
        source = path/'checkpoint'
        model.cpu().save_pretrained(source)
        tokenizer.save_pretrained(source)
        encoder_onnx(source,path/'fp32.onnx')
        quantize_dynamic(str(path/'fp32.onnx'),str(path/'int8.onnx'),weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul','Gemm','Gather'],per_channel=True)
        export(args,name,result,texts,router,path/'int8.onnx',source/'tokenizer.json',scale,parameters)
