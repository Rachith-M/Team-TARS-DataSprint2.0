"""Rehearse the live demo checklist and time batch-one HTTP requests on localhost."""

import argparse
import json
from http.client import HTTPConnection
from pathlib import Path
from time import perf_counter

import numpy as np

from sprint.data import SEED, load_training
from sprint.defense import WEB
from sprint.encoder import variant_groups
from sprint.train import take


def request(connection, path, body=None, headers=None):
    started = perf_counter()
    headers = headers or {}
    if body is None:
        connection.request('GET', path, headers=headers)
    else:
        payload = json.dumps(body).encode('utf-8')
        connection.request('POST', path, payload, {'Content-Type': 'application/json', **headers})
    response = connection.getresponse()
    content = response.read()
    elapsed = 1000 * (perf_counter() - started)
    return response.status, json.loads(content), elapsed


def timing(values, target):
    return {'queries': len(values), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': max(values),
            'at_or_above_target': sum(value >= target for value in values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--train', type=Path, default=Path('train.csv'))
    parser.add_argument('--cases', type=Path, default=WEB / 'smoke_cases.json')
    args = parser.parse_args()
    if args.repetitions < 1 or args.output.exists():
        parser.error('Use a new output path and a positive repetition count')

    texts, labels = load_training(args.train)
    groups = variant_groups(texts, [])
    chosen = np.random.default_rng(SEED + 7).choice(len(labels), size=200, replace=False)
    queries = take(groups['raw'], chosen) + take(groups['natural'], chosen[:150]) + take(groups['asr'], chosen[:50])
    rng = np.random.default_rng(SEED + 48)
    connection = HTTPConnection('127.0.0.1', args.port, timeout=10)
    try:
        status, model, _ = request(connection, '/api/model')
        if status != 200:
            raise ValueError('The local model is unavailable')
        target = model['latency_limit_ms']
        invalid = {name: request(connection, '/api/predict', {'text': test})[0]
                   for name, test in (('blank', ' '), ('non_string', 55), ('too_long', 'a' * 513),
                                      ('nul', 'hello\x00'), ('invalid_unicode', '\ud800'))}
        access = {'unrecognized_host': request(connection, '/api/model', headers={'Host': 'example.invalid'})[0],
                  'foreign_origin': request(connection, '/api/predict', {'text': queries[0]},
                                            {'Origin': 'https://example.invalid'})[0]}
        smoke = []
        for case in json.loads(args.cases.read_text(encoding='utf-8')):
            code, response, _ = request(connection, '/api/predict', {'text': case['text']})
            if code != 200:
                raise ValueError(f"Demo case request failed: {case['name']}")
            smoke.append({'case': case['name'], 'expected_id': case['intent_id'],
                          'expected_action': case['action'], **response})
        if len(smoke) != 11:
            raise ValueError('Expected all eleven demo checklist cases')

        for query in queries[:24]:
            request(connection, '/api/predict', {'text': query})
        runs = []
        for _ in range(args.repetitions):
            inference, roundtrip = [], []
            for i in rng.permutation(len(queries)):
                code, response, elapsed = request(connection, '/api/predict', {'text': queries[i]})
                if code != 200:
                    raise ValueError(f'Timing request failed at row {i}')
                inference.append(response['inference_ms'])
                roundtrip.append(elapsed)
            runs.append({'cpu_inference': timing(inference, target), 'http_roundtrip': timing(roundtrip, target)})
            print(runs[-1], flush=True)
        long_cpu, long_http = [], []
        long_guarded = 0
        for text in sorted((text for text in texts if len(text) <= model['max_query_chars']), key=len)[-40:]:
            code, response, elapsed = request(connection, '/api/predict', {'text': text})
            if code != 200:
                raise ValueError('A long training query failed')
            long_cpu.append(response['inference_ms'])
            long_http.append(elapsed)
            long_guarded += response['length_guard_applied']
    finally:
        connection.close()
    report = {'model': model, 'demo_checks': smoke, 'invalid_input_status': invalid, 'timings': runs,
              'longest_training_queries': {'cpu_inference': timing(long_cpu, target),
                                           'http_roundtrip': timing(long_http, target),
                                           'length_guard_applied': long_guarded},
              'all_demo_cases_passed': all(case['expected_id'] == case['intent_id'] and
                                           case['expected_action'] == case['action'] for case in smoke),
              'input_validation_passed': all(status == 400 for status in invalid.values()),
              'local_access_enforced': access == {'unrecognized_host': 403, 'foreign_origin': 403},
              'cpu_max_under_target': max([*long_cpu, *(run['cpu_inference']['max_ms'] for run in runs)]) < target,
              'http_max_under_target': max([*long_http, *(run['http_roundtrip']['max_ms'] for run in runs)]) < target}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write('\n')
    print(f"Demo cases {sum(case['expected_id'] == case['intent_id'] for case in smoke)}/{len(smoke)}", flush=True)


if __name__ == '__main__':
    main()
