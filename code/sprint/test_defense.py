"""Regression tests for the local Stage 2 HTTP contract."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from threading import Event, Thread

from sprint.defense import make_server


class StubRuntime:
    metadata = {'version': 'test'}

    def __init__(self):
        self.started = Event()
        self.release = Event()

    def predict(self, text):
        if text == 'wait':
            self.started.set()
            if not self.release.wait(4):
                raise RuntimeError('Test request timed out')
        if text == 'failure':
            raise RuntimeError('Private exception detail')
        return {'intent_id': 23, 'action': 'accept'}


class LocalApiTests(unittest.TestCase):
    def setUp(self):
        self.runtime = StubRuntime()
        self.server = make_server(self.runtime, 0)
        self.thread = Thread(target=self.server.serve_forever)
        self.thread.start()

    def tearDown(self):
        self.runtime.release.set()
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def request(self, method, path, data=None, **headers):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request(method, path, data, headers)
            result = connection.getresponse()
            return result.status, dict(result.getheaders()), result.read()
        finally:
            connection.close()

    def predict(self, text):
        return self.request('POST', '/api/predict', json.dumps({'text': text}), **{'Content-Type': 'application/json'})

    def test_local_verification_and_foreign_origin(self):
        code, headers, body = self.request('GET', '/api/model')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)['version'], 'test')
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(self.request('GET', '/api/model', Host='example.invalid')[0], 403)
        self.assertEqual(self.request('POST', '/api/predict', '{}', Origin='https://example.invalid',
                                      **{'Content-Type': 'application/json'})[0], 403)

    def test_malformed_request_recovers(self):
        self.assertEqual(self.request('POST', '/api/predict', '{', **{'Content-Type': 'application/json'})[0], 400)
        self.assertEqual(self.request('POST', '/api/predict', '{}', **{'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.predict('verification')[0], 200)

    def test_busy_inference_returns_fast_and_recovers(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.predict, 'wait')
            try:
                self.assertTrue(self.runtime.started.wait(2))
                self.assertEqual(self.predict('verification')[0], 429)
                self.assertEqual(self.request('GET', '/api/model')[0], 200)
            finally:
                self.runtime.release.set()
            self.assertEqual(first.result(timeout=5)[0], 200)
        self.assertEqual(self.predict('verification')[0], 200)

    def test_runtime_exception_hides_internal_detail(self):
        status, _, body = self.predict('failure')
        self.assertEqual(status, 500)
        self.assertNotIn(b'Private exception detail', body)
        self.assertEqual(self.predict('verification')[0], 200)


if __name__ == '__main__':
    unittest.main()
