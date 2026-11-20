import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

import ai_annotation as ai
from ai_jobs import JobManager
import label_app as server


class AIJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_root, self.old_jobs = server.SOURCE_DIR, server._ai_jobs
        server.SOURCE_DIR, server._ai_jobs = self.root, JobManager(workers=1, limit=8)
        self.jobs = server._ai_jobs
        self.client = server.app.test_client()
        Image.new('RGB', (40, 30)).save(self.root / 'image.jpg')
        server.ai_store().save({'base_url': 'https://fixture.invalid/v1', 'model': 'fixture', 'api_key': 'unit-test-key'})

    def tearDown(self):
        self.jobs.executor.shutdown(wait=True)
        server._ai_stores.pop(self.root / '.ai-config.json', None)
        server.SOURCE_DIR, server._ai_jobs = self.old_root, self.old_jobs
        self.temporary.cleanup()

    def start(self, request_id='request-1'):
        response = self.client.post('/api/ai/jobs', json={'filename': 'image.jpg', 'instruction': '标注', 'request_id': request_id})
        self.assertEqual(response.status_code, 202, response.json)
        return response.json

    def wait(self, identifier):
        for _ in range(300):
            job = self.client.get('/api/ai/jobs/' + identifier).json
            if job['status'] in ('completed', 'failed', 'cancelled'):
                return job
            time.sleep(0.01)
        self.fail('Job failed to terminate')

    def test_async_progress_recovery_and_no_duplicate_request(self):
        release, started = threading.Event(), threading.Event()
        def work(config, key, path, classes, current, instruction, history, progress):
            progress({'phase': 'thinking', 'analysis_chars': 32})
            started.set()
            release.wait(3)
            return {'action': 'replace', 'message': '完成', 'boxes': [], 'elapsed_seconds': 1}
        with patch('label_app.ai_annotation.annotate', side_effect=work) as provider:
            job = self.start()
            self.assertTrue(started.wait(2))
            duplicate = self.start()
            self.assertEqual(duplicate['id'], job['id'])
            self.assertEqual(provider.call_count, 1)
            running = self.client.get('/api/ai/jobs/' + job['id']).json
            self.assertEqual(running['phase'], 'thinking')
            self.assertEqual(running['analysis_chars'], 32)
            self.assertNotIn('unit-test-key', json.dumps(running))
            recovered = self.client.get('/api/ai/jobs').json['jobs']
            self.assertEqual(recovered[0]['id'], job['id'])
            self.assertFalse(list(self.root.glob('**/*.txt')))
            release.set()
            completed = self.wait(job['id'])
            self.assertEqual(completed['status'], 'completed')
            self.assertEqual(completed['result']['filename'], 'image.jpg')
            self.assertTrue(self.client.get('/api/ai/jobs').json['jobs'])
            self.client.post('/api/ai/jobs/' + job['id'] + '/ack', json={})
            self.assertFalse(self.client.get('/api/ai/jobs').json['jobs'])

    def test_cancel_discards_late_output(self):
        started, release = threading.Event(), threading.Event()
        def work(*args):
            started.set()
            release.wait(3)
            return {'action': 'replace', 'message': '迟到结果', 'boxes': []}
        with patch('label_app.ai_annotation.annotate', side_effect=work):
            job = self.start()
            self.assertTrue(started.wait(2))
            self.client.post('/api/ai/jobs/' + job['id'] + '/cancel', json={})
            release.set()
        self.jobs.executor.shutdown(wait=True)
        cancelled = self.client.get('/api/ai/jobs/' + job['id']).json
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertNotIn('result', cancelled)

    def test_worker_failure_and_workspace_isolation(self):
        with patch('label_app.ai_annotation.annotate', side_effect=ai.AIError('API 等待超时')):
            job = self.start()
            failed = self.wait(job['id'])
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('超时', failed['error'])
        self.assertIsNone(self.jobs.get('different-workspace', job['id']))
        self.assertFalse(list(self.root.glob('**/*.txt')))

    def test_stop_also_discards_result_completed_before_cancel_arrives(self):
        with patch('label_app.ai_annotation.annotate', return_value={'action': 'replace', 'message': '完成', 'boxes': []}):
            job = self.start()
            self.assertEqual(self.wait(job['id'])['status'], 'completed')
        cancelled = self.client.post('/api/ai/jobs/' + job['id'] + '/cancel', json={}).json
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertNotIn('result', cancelled)

    def test_token_limit_error_keeps_recovery_settings_without_result_or_key(self):
        error = ai.OutputTokenLimit({'max_tokens': 8192}, analysis_chars=200, received_chars=0)
        with patch('label_app.ai_annotation.annotate', side_effect=error) as provider:
            job = self.start()
            failed = self.wait(job['id'])
        self.assertEqual(failed['error_code'], 'output_token_limit')
        self.assertEqual(failed['error_details']['suggested_max_tokens'], 32768)
        self.assertNotIn('result', failed)
        self.assertNotIn('unit-test-key', json.dumps(failed))
        self.assertEqual(provider.call_count, 1)

    def test_mismatched_request_id_and_active_task_do_not_call_provider(self):
        release, started = threading.Event(), threading.Event()
        def work(*args):
            started.set()
            release.wait(3)
            return {'action': 'none', 'message': '结束', 'boxes': []}
        with patch('label_app.ai_annotation.annotate', side_effect=work) as provider:
            job = self.start()
            self.assertTrue(started.wait(2))
            response = self.client.post('/api/ai/jobs', json={'filename': 'image.jpg', 'instruction': '另一项', 'request_id': job['request_id']})
            self.assertEqual(response.status_code, 400)
            response = self.client.post('/api/ai/jobs', json={'filename': 'image.jpg', 'instruction': '标注', 'request_id': 'new-id'})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(provider.call_count, 1)
            release.set()
            self.wait(job['id'])


if __name__ == '__main__':
    unittest.main()
