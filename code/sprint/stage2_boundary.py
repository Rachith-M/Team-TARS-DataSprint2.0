"""Evaluate a live-only intent evidence gate on matched supported and unsupported requests."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import softmax

from intent_model import FALLBACK, Views, script
from sprint.data import SEED, load_training
from sprint.gate_runtime import Stage2Runtime, actions
from sprint.score_data import normalized_words, word_pattern
from sprint.score_push import fingerprint
from sprint.stage2_fixed import supplied


CASES = Path(__file__).with_name('stage2_boundary_cases.json')
INTENTS = {'booking': 51, 'device': 23, 'road': 11}
SCRIPTS = ('roman', 'devanagari', 'mixed')
CONFIDENCE_FLOORS = (.44, .50, .56, .60)

CAB = word_pattern(('cab', 'कैब', 'taxi', 'taiksi', 'टैक्सी', 'uber', 'ubar', 'ऊबर', 'किराये की गाड़ी'))
AIR_TRAVEL = word_pattern(('flight', 'flait', 'फ्लाइट', 'air ticket', 'हवाई टिकट', 'उड़ान', 'udaan',
                           'plane', 'विमान', 'हवाई जहाज़', 'hawai jahaz', 'airline', 'एयरलाइन'))
RAIL = word_pattern(('train', 'tren', 'ट्रेन', 'rail', 'railgaadi', 'रेलगाड़ी', 'रेल'))
RESERVE = word_pattern(('book', 'buk', 'बुक', 'बुकिंग', 'booking', 'reserve', 'रिज़र्व', 'seat', 'सीट',
                        'ticket', 'tikat', 'टिकट'))
SOCKET = word_pattern(('plug', 'plag', 'प्लग', 'socket', 'soket', 'सॉकेट', 'outlet', 'आउटलेट'))
ENABLE = word_pattern(('on', 'ऑन', 'chalu', 'chaalu', 'चालू', 'activate', 'एक्टिवेट', 'enable',
                       'power do', 'पावर दो', 'power laga do', 'पावर लगा दो'))
DISABLE = word_pattern(('off', 'ऑफ', 'band', 'बंद', 'disconnect', 'dont', 'mat', 'मत', 'nahi', 'नहीं'))
DO_NOT_BOOK = word_pattern(('mat', 'मत', 'nahi', 'नहीं', 'cancel', 'रद्द', 'cancelled'))
PHONE = word_pattern(('phone', 'फोन', 'फ़ोन', 'mobile', 'मोबाइल', 'handset', 'हैंडसेट'))
PHONE_SETTING = word_pattern(('wifi', 'wi fi', 'vai fai', 'वाईफाई', 'वाई फ़ाई', 'bluetooth', 'ब्लूटूथ',
                              'hotspot', 'हॉटस्पॉट', 'data', 'डेटा', 'flight mode', 'फ्लाइट मोड',
                              'silent mode', 'साइलेंट मोड', 'network', 'नेटवर्क'))
TRAFFIC = word_pattern(('traffic', 'traiphik', 'ट्रैफिक', 'यातायात', 'yataayat', 'jam', 'जाम',
                        'gaadiyon ki bheed', 'गाड़ियों की भीड़'))
ROAD_SERVICE = word_pattern(('pothole', 'potholes', 'गड्ढे', 'road repair', 'सड़क की मरम्मत',
                             'नगर निगम', 'nagar nigam', 'municipal', 'toll', 'टोल', 'challan', 'चालान'))
SERVICE_ACTION = word_pattern(('report', 'रिपोर्ट', 'file', 'darj', 'दर्ज', 'शिकायत', 'shikayat',
                               'complaint', 'submit', 'जमा', 'jama', 'bhar do', 'भर दो', 'payment',
                               'bhugtan', 'भुगतान'))


def evidence(text, intent):
    words = normalized_words(text)
    if intent in (2, 51):
        if DO_NOT_BOOK.search(words):
            return 0
        cab = CAB.search(words)
        air = AIR_TRAVEL.search(words)
        if air and not (cab or RAIL.search(words)) and RESERVE.search(words):
            return -1
        if intent == 51 and cab and not (air and RESERVE.search(words)):
            return 1
    if intent in (8, 17, 23, 39, 40):
        socket = SOCKET.search(words)
        phone_setting = PHONE.search(words) and PHONE_SETTING.search(words)
        if phone_setting and not socket:
            return -1
        if intent == 23 and socket and ENABLE.search(words) and not (phone_setting or DISABLE.search(words)):
            return 1
    if intent in (11, 12, 41, 46):
        if ROAD_SERVICE.search(words) and SERVICE_ACTION.search(words):
            return -1
        if intent == 11 and TRAFFIC.search(words):
            return 1
    return 0


class BoundaryGate:
    def __init__(self, baseline, floor=None):
        if baseline.method != 'max_softmax' or baseline.policy['mode'] != 'band':
            raise ValueError('Expected the released E5 max-softmax policy with clarification')
        self.baseline = baseline
        self.floor = floor

    def inspect(self, text):
        probabilities = softmax(self.baseline.base.score([text])[0].astype(np.float64)
                                * self.baseline.settings['scale'])
        top = int(probabilities.argmax())
        current = str(actions(1 - probabilities[top], top, self.baseline.policy['chosen'], True))
        if not text.strip():
            current = 'reject'
        return top, probabilities, current, evidence(text, top)

    @staticmethod
    def output(state, floor):
        top, probabilities, current, signal = state
        action = current
        if floor is not None and current == 'accept' and signal < 0:
            action = 'clarify'
        if floor is not None and current == 'clarify' and signal > 0 and probabilities[top] >= floor:
            action = 'accept'
        intent = top if action == 'accept' else FALLBACK
        return {'intent_id': intent, 'confidence': float(probabilities[intent]), 'fallback': intent == FALLBACK,
                'top_model_intent': top, 'action': action}

    def predict(self, text):
        return self.output(self.inspect(text), self.floor)


def matched_cases(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    results = {}
    for phase in ('calibration', 'evaluation'):
        results[phase] = []
        for family, pairs in data[phase].items():
            for index, (supported, unsupported) in enumerate(pairs, 1):
                for texts, target in ((supported, INTENTS[family]), (unsupported, FALLBACK)):
                    if tuple(map(script, texts)) != SCRIPTS:
                        raise ValueError(f'Unexpected scripts in {phase}/{family}/{index}/{target}')
                    results[phase].extend({'text': text, 'target': target, 'script': sc, 'family': family,
                                           'pair': f'{family}_{index}'} for sc, text in zip(SCRIPTS, texts))
    keys = [set(Views.from_texts([row['text'] for row in rows]).phonetic) for rows in results.values()]
    if not keys[0].isdisjoint(keys[1]):
        raise ValueError('Calibration and evaluation share phonetic query families')
    return results


def decisions(gate, cases):
    states = [gate.inspect(row['text']) for row in cases]
    for row, state in zip(cases, states):
        if gate.output(state, None) != gate.baseline.predict(row['text']):
            raise ValueError('Computed baseline differs from the released runtime')
    return states


def counts(cases, outputs):
    result = dict.fromkeys(('supported_correct_accept', 'supported_wrong_accept', 'supported_clarify',
                            'supported_reject', 'unsupported_accept', 'unsupported_clarify',
                            'unsupported_reject'), 0)
    for row, output in zip(cases, outputs):
        action = output['action']
        if row['target'] == FALLBACK:
            key = f'unsupported_{action}'
        elif action == 'accept':
            key = 'supported_correct_accept' if row['target'] == output['intent_id'] else 'supported_wrong_accept'
        else:
            key = f'supported_{action}'
        result[key] += 1
    return result


def fallback_count(summary):
    return summary['supported_clarify'] + summary['supported_reject']


def no_supported_regression(candidate, baseline):
    return (fallback_count(candidate) <= fallback_count(baseline)
            and candidate['supported_correct_accept'] >= baseline['supported_correct_accept']
            and candidate['supported_wrong_accept'] <= baseline['supported_wrong_accept'])


def calibrate(gate, cases):
    states = decisions(gate, cases)
    baseline = counts(cases, [gate.output(state, None) for state in states])
    tried = [{'floor': floor, **counts(cases, [gate.output(state, floor) for state in states])}
             for floor in CONFIDENCE_FLOORS]
    eligible = [trial for trial in tried if no_supported_regression(trial, baseline)
                and trial['unsupported_accept'] <= baseline['unsupported_accept']]
    selected = min(eligible, key=lambda t: (t['unsupported_accept'], -t['supported_correct_accept'],
                                           -t['floor'])) if eligible else None
    return {'chosen_floor': selected['floor'] if selected else None, 'baseline': baseline, 'trials': tried}


def compare(gate, cases):
    states = decisions(gate, cases)
    rows = [{**row, 'baseline': gate.output(state, None), 'candidate': gate.output(state, gate.floor),
             'evidence': state[3]} for row, state in zip(cases, states)]
    return rows


def summary(rows):
    return {name: counts(rows, [row[name] for row in rows]) for name in ('baseline', 'candidate')}


def unsupported_interval(rows):
    rng = np.random.default_rng(SEED + 170)
    grouped = {}
    for row in rows:
        if row['target'] == FALLBACK:
            improvement = int(row['baseline']['action'] == 'accept') - int(row['candidate']['action'] == 'accept')
            grouped[row['pair']] = grouped.get(row['pair'], 0) + improvement
    by_family = [[n for pair, n in grouped.items() if pair.startswith(f'{family}_')] for family in INTENTS]
    estimates = sum(rng.choice(values, size=(5000, len(values)), replace=True).sum(1) for values in by_family)
    return np.quantile(estimates / (sum(map(len, by_family)) * len(SCRIPTS)), [.025, .975]).tolist()


def replay_rows(args):
    queries, categories, _, _, splits, _ = supplied(args.unsupported)
    proxy = [{'text': text, 'target': FALLBACK} for text, category, split in zip(queries, categories, splits)
             if split == 'evaluation' and category != 'navigation_directions']
    directions = [{'text': text, 'target': 41} for text in (
        'mujhe airport tak ka raasta batao', 'मुझे कनॉट प्लेस का रास्ता बताओ',
        'yaar petrol pump tak ka direction bata do', 'give me directions to the mall na jaldi',
    )]
    texts, labels = load_training(args.train)
    rng = np.random.default_rng(SEED + 33)
    supported = [{'text': texts[i], 'target': c} for c in range(FALLBACK)
                 for i in rng.choice(np.flatnonzero(labels == c), 2, replace=False)]
    return proxy, [*supported, *directions]


def benchmark(gate, rows):
    rng = np.random.default_rng(SEED + 77)
    calls = (gate.baseline.predict, gate.predict)
    latencies = [[], []]
    for row in rows[:12]:
        for call in calls:
            call(row['text'])
    for index, row_id in enumerate(rng.permutation(len(rows))):
        text = rows[row_id]['text']
        for offset in range(len(calls)):
            arm = (index + offset) % len(calls)
            start = perf_counter()
            calls[arm](text)
            latencies[arm].append(1000 * (perf_counter() - start))
    return {name: {'queries': len(values), 'p95_ms': float(np.percentile(values, 95)), 'max_ms': max(values)}
            for name, values in zip(('baseline', 'candidate'), latencies)}


def save(path, record):
    with path.open('x', encoding='utf-8') as file:
        json.dump(record, file, indent=2, ensure_ascii=False, default=int)
        file.write('\n')


def run(args):
    baseline = Stage2Runtime.load(args.model, args.gate, args.threads)
    gate = BoundaryGate(baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output / 'calibration.json'
    policy = {'cases_sha256': fingerprint(args.cases), 'code_sha256': fingerprint(Path(__file__)),
              'baseline_policy_sha256': fingerprint(args.gate / 'operating_point.json')}
    if args.phase == 'calibrate':
        cases = matched_cases(args.cases)['calibration']
        result = calibrate(gate, cases)
        save(calibration_path, {**policy, **result})
        print('Calibration', result, flush=True)
        return
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    if any(calibration.get(name) != digest for name, digest in policy.items()):
        raise ValueError('Data, code or baseline changed after the policy was frozen')
    gate.floor = calibration['chosen_floor']
    if args.phase == 'query':
        print(json.dumps(gate.predict(args.text), ensure_ascii=False))
        return
    cases = matched_cases(args.cases)['evaluation']
    rows = compare(gate, cases)
    totals = summary(rows)
    proxy, supported = replay_rows(args)
    regression = {'existing_proxy_replay': summary(compare(gate, proxy)),
                  'training_supported_smoke_replay': summary(compare(gate, supported))}
    latency = benchmark(gate, [*cases, *proxy, *supported])
    old, new = totals['baseline'], totals['candidate']
    proxy_totals = regression['existing_proxy_replay']
    supported_totals = regression['training_supported_smoke_replay']
    checks = {'fewer_unsupported_accepts': new['unsupported_accept'] < old['unsupported_accept'],
              'no_supported_regression': no_supported_regression(new, old),
              'proxy_replay_unsupported_accepts_no_higher': proxy_totals['candidate']['unsupported_accept']
              <= proxy_totals['baseline']['unsupported_accept'],
              'supported_smoke_no_regression': no_supported_regression(supported_totals['candidate'], supported_totals['baseline']),
              'candidate_max_under_50_ms': latency['candidate']['max_ms'] < 50}
    result = {**policy, 'chosen_floor': gate.floor, 'parameters': baseline.settings['parameters'],
              'cpu_threads': args.threads, 'fresh_heldout': totals, 'checks': checks,
              'all_checks_passed': all(checks.values()), 'latency': latency, 'replays': regression,
              'unsupported_accept_reduction_group_interval_95pct': unsupported_interval(rows),
              'by_family': {family: summary([row for row in rows if row['family'] == family]) for family in INTENTS},
              'by_script': {sc: summary([row for row in rows if row['script'] == sc]) for sc in SCRIPTS},
              'heldout_rows': rows}
    save(args.output / 'evaluation.json', result)
    print('Held-out', totals, 'Checks', checks, 'Latency', latency, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('calibrate', 'evaluate', 'query'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--gate', type=Path, default=Path('sprint_results/stage2_provided_gate'))
    parser.add_argument('--unsupported', type=Path, default=Path('unsupported_eval_queries.csv'))
    parser.add_argument('--train', type=Path, default=Path('train.csv'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--text')
    args = parser.parse_args()
    if args.phase == 'query' and args.text is None:
        parser.error('--text is required for query')
    run(args)


if __name__ == '__main__':
    main()
