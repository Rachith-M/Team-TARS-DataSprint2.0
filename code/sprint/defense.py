"""Run the verified Bhasha model as an offline browser or terminal demo."""

import argparse
import json
import math
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock, Timer
from time import perf_counter

from scipy.special import softmax

from intent_model import FALLBACK, PARAMETER_CAP, script
from sprint.augment import tta_variant
from sprint.bhasha_audit import dictionary_view
from sprint.defense_artifacts import ASSETS, MAX_TWO_VIEW_TOKENS, POLICY, SUBMITTED_SHA256, VERSION, verify_release
from sprint.postprocess import average_tta
from sprint.runtime import Runtime
from sprint.score_push import fingerprint


WEB = Path(__file__).with_name('defense_web')
MODEL = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'AnveshanNLP' / 'stage2-bhasha'
MAX_CHARS = 512
MAX_BODY = 8192
TARGET_MS = 50
WARMUP = 'फैन सॉकेट चालू करें'


class DefenseRuntime:
    def __init__(self, model=MODEL, policy=POLICY, threads=8):
        proof = json.loads((model / 'verification.json').read_text(encoding='utf-8'))
        self.policy = json.loads(policy.read_text(encoding='utf-8'))
        if proof['submission_sha256'] != SUBMITTED_SHA256 or self.policy['submission_sha256'] != SUBMITTED_SHA256:
            raise ValueError('The model and fallback policy must match the verified Bhasha submission')
        rule = self.policy['rule']
        scale = self.policy['model_logit_scale']
        if not (math.isfinite(scale) and scale > 0 and 0 <= rule['accept_below'] <= rule['threshold'] <= 1):
            raise ValueError('Invalid calibrated fallback settings')
        if any(fingerprint(model / name) != proof['assets'][name] for name in ASSETS):
            raise ValueError('A rebuilt model asset differs from its verified fingerprint')
        self.base = Runtime(model, threads)
        self.dictionary = json.loads((model / 'dictionary.json').read_text(encoding='utf-8'))
        self.names = json.loads((WEB / 'intent_names.json').read_text(encoding='utf-8'))
        parameters = self.base.settings['parameters']
        entries = sum(map(len, self.dictionary))
        if (parameters != proof['model_parameters'] or entries != proof['dictionary_entries']
                or parameters + entries > PARAMETER_CAP or len(self.names) != FALLBACK + 1):
            raise ValueError('Verified model assets or intent labels are inconsistent')
        self.metadata = {'version': VERSION, 'model': 'Bhasha', 'parameters': parameters, 'dictionary_entries': entries,
                         'parameter_limit': PARAMETER_CAP, 'threads': threads, 'latency_limit_ms': TARGET_MS,
                         'matching_submission_rows': proof['matching_predictions'],
                         'submission_rows': proof['submission_rows'], 'submission_sha256': proof['submission_sha256'],
                         'model_sha256': proof['assets']['model.int8.onnx'], 'policy_sha256': fingerprint(policy),
                         'checkpoint_revision': proof['checkpoint_revision'], 'max_query_chars': MAX_CHARS,
                         'max_two_view_tokens': MAX_TWO_VIEW_TOKENS,
                         'policy': {'accept_below': self.policy['rule']['accept_below'],
                                    'threshold': self.policy['rule']['threshold']}}

    def predict(self, text):
        start = perf_counter()
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Enter a Hindi or Hinglish request.')
        if len(text) > MAX_CHARS:
            raise ValueError(f'Use at most {MAX_CHARS} characters.')
        if '\x00' in text or any(0xD800 <= ord(char) <= 0xDFFF for char in text):
            raise ValueError('Enter valid Unicode text.')
        text = text.strip()
        normalized = tta_variant(text)
        alternate = dictionary_view(text, normalized, self.dictionary)
        views = [text] if alternate == text else [text, alternate]
        encoded = self.base.encoder_inputs(views)
        limited = len(views) == 2 and encoded['input_ids'].shape[1] > MAX_TWO_VIEW_TOKENS
        if limited:
            width = int(encoded['attention_mask'][0].sum())
            encoded = {name: values[:1, :width] for name, values in encoded.items()}
            views = [text]
        scores = self.base.raw(views, encoded=encoded)
        logits = scores[0] if len(scores) == 1 else average_tta(scores[0], scores[1], self.base.settings['tta_weight'])
        top = int(logits.argmax())
        probabilities = softmax(logits.astype('float64') * self.policy['model_logit_scale'])
        confidence = float(probabilities[top])
        risk = 1 - confidence
        rule = self.policy['rule']
        if top == FALLBACK:
            action, reason = 'reject', 'The model predicted Class 55.'
        elif risk >= rule['threshold']:
            action, reason = 'reject', 'I’m not confident I can route that request.'
        elif risk >= rule['accept_below']:
            action, reason = 'clarify', 'Could you rephrase that request?'
        else:
            action, reason = 'accept', 'Routed to a supported intent.'
        intent = top if action == 'accept' else FALLBACK
        result = {'intent_id': intent, 'intent_name': self.names[intent], 'action': action,
                  'fallback': intent == FALLBACK, 'message': reason, 'top_model_intent': top,
                  'top_confidence': round(confidence, 5), 'input_script': script(text),
                  'view': ('Dictionary' if alternate != normalized else 'Normalized') if len(scores) == 2 else 'Original',
                  'views': len(scores), 'length_guard_applied': limited, 'risk': round(risk, 5),
                  'normalized_text': normalized, 'dictionary_text': alternate if alternate != text else None,
                  'alternatives': [{'intent_id': int(i), 'intent_name': self.names[int(i)],
                                    'probability': round(float(probabilities[i]), 5)}
                                   for i in probabilities.argsort()[::-1][:5]],
                  'probabilities': [round(float(p), 4) for p in probabilities]}
        result['inference_ms'] = round(1000 * (perf_counter() - start), 2)
        return result


class LocalServer(ThreadingHTTPServer):
    # Bound sockets separately from the single inference slot so idle clients cannot exhaust threads.
    slots = None

    def __init__(self, address, handler):
        self.slots = BoundedSemaphore(24)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def make_server(runtime, port):
    assets = {'/': ('text/html; charset=utf-8', (WEB / 'index.html').read_bytes()),
              '/styles.css': ('text/css; charset=utf-8', (WEB / 'styles.css').read_bytes()),
              '/app.js': ('text/javascript; charset=utf-8', (WEB / 'app.js').read_bytes()),
              '/favicon.svg': ('image/svg+xml', (WEB / 'favicon.svg').read_bytes())}
    inference_lock = Lock()
    cases = json.loads((WEB / 'smoke_cases.json').read_text(encoding='utf-8'))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = True
        timeout = 15

        def send_content(self, status, content, mime):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'")
            if self.close_connection:
                self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status, payload):
            self.send_content(status, json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                              'application/json; charset=utf-8')

        def local_request(self):
            host = self.headers.get('Host', '').lower()
            origin = self.headers.get('Origin')
            names = ('127.0.0.1', 'localhost')
            hosts = tuple(f'{name}:{self.server.server_port}' for name in names)
            if self.server.server_port == 80:
                hosts += names
            if host in hosts and (origin is None or origin == f'http://{host}'):
                return True
            self.close_connection = True
            self.send_json(403, {'error': 'Use the local demo address printed at startup.'})
            return False

        def do_GET(self):
            if not self.local_request():
                return
            path = self.path.split('?', 1)[0]  # the page accepts ?q= to pre-run a query
            if path == '/api/model':
                self.send_json(200, runtime.metadata)
            elif path == '/api/intents':
                self.send_json(200, runtime.names)
            elif path == '/api/cases':
                self.send_json(200, cases)
            elif path in assets:
                mime, content = assets[path]
                self.send_content(200, content, mime)
            else:
                self.send_json(404, {'error': 'Page not found.'})

        def do_POST(self):
            if not self.local_request():
                return
            if self.path != '/api/predict':
                self.close_connection = True
                self.send_json(404, {'error': 'Route not found.'})
                return
            if self.headers.get_content_type() != 'application/json':
                self.close_connection = True
                self.send_json(415, {'error': 'Send the request as JSON.'})
                return
            try:
                lengths = self.headers.get_all('Content-Length') or []
                if len(lengths) != 1 or self.headers.get('Transfer-Encoding') is not None:
                    raise ValueError
                size = int(lengths[0])
                if not 0 < size <= MAX_BODY:
                    raise ValueError
                data = json.loads(self.rfile.read(size).decode('utf-8'))
                if not isinstance(data, dict):
                    raise ValueError
            except (ValueError, UnicodeError, RecursionError):
                self.close_connection = True
                self.send_json(400, {'error': 'Invalid request body.'})
                return
            if not inference_lock.acquire(blocking=False):
                self.send_json(429, {'error': 'The model is busy. Try the request again.'})
                return
            try:
                result = runtime.predict(data.get('text'))
            except ValueError as error:
                self.send_json(400, {'error': str(error)})
                return
            except Exception as error:
                print(f'Local inference failed ({type(error).__name__})', file=sys.stderr, flush=True)
                self.send_json(500, {'error': 'The local model could not process that request. Try again.'})
                return
            finally:
                inference_lock.release()
            self.send_json(200, result)

        def log_message(self, *_):
            pass

    return LocalServer(('127.0.0.1', port), Handler)


def self_check(runtime):
    cases = json.loads((WEB / 'smoke_cases.json').read_text(encoding='utf-8'))
    outcomes = []
    for case in cases:
        result = runtime.predict(case['text'])
        passed = result['intent_id'] == case['intent_id'] and result['action'] == case['action']
        outcomes.append({'name': case['name'], 'passed': passed, 'intent_id': result['intent_id'],
                         'action': result['action'], 'inference_ms': result['inference_ms']})
    return {'model': runtime.metadata, 'cases': outcomes, 'passed': all(row['passed'] for row in outcomes)}


def terminal(runtime):
    print(f"Bhasha ready · {runtime.metadata['parameters']:,} parameters · type /exit to close")
    while True:
        try:
            query = input('Query › ')
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip() == '/exit':
            break
        if query.strip():
            try:
                print(json.dumps(runtime.predict(query), ensure_ascii=False))
            except ValueError as error:
                print(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--release-root', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--terminal', action='store_true')
    modes.add_argument('--query')
    modes.add_argument('--open', action='store_true')
    modes.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or not 0 <= args.port < 65536:
        parser.error('Use a valid positive thread count and TCP port')
    try:
        verified = verify_release(args.release_root) if args.verify and args.release_root else None
        runtime = DefenseRuntime(args.model, args.policy, args.threads)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f'Stage 2 could not start. {error}\n')
    for _ in range(4):
        runtime.predict(WARMUP)
    if args.verify:
        result = self_check(runtime)
        if verified is not None:
            result['verified_release_files'] = verified
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result['passed']:
            raise SystemExit(1)
    elif args.query is not None:
        print(json.dumps(runtime.predict(args.query), ensure_ascii=False))
    elif args.terminal:
        terminal(runtime)
    else:
        try:
            server = make_server(runtime, args.port)
        except OSError as error:
            parser.exit(2, f'Could not open port {args.port}. Stop the existing server or use --port 0. {error}\n')
        with server:
            url = f'http://127.0.0.1:{server.server_port}/'
            print(f'Bhasha ready at {url} · Ctrl+C to stop', flush=True)
            if args.open:
                Timer(.25, webbrowser.open, args=(url,)).start()
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
