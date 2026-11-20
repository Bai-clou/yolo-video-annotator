import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

import ai_annotation as ai
import label_app as server


def sample_box(identifier='existing', class_id=0):
    return {'id': identifier, 'class_id': class_id, 'x_center': 0.3, 'y_center': 0.5,
            'width': 0.2, 'height': 0.3, 'source': 'manual', 'conf': None}


class VisionHandler(BaseHTTPRequestHandler):
    """Offline fixture: simulates API contracts, not a vision model."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.calls.append({'path': self.path, 'body': body,
                                  'authorization': self.headers.get('Authorization'),
                                  'api-key': self.headers.get('api-key')})
        mode = self.server.mode
        if mode == 'redirect':
            self.send_response(302)
            self.send_header('Location', 'http://127.0.0.1:1/never-forward-key')
            self.end_headers()
            return
        if mode == 'unauthorized':
            self.reply({'error': {'message': 'Invalid key: unit-test-key'}}, 401)
            return
        if mode == 'invalid_json':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'<html>not an API</html>')
            return
        responses = 'input' in body
        system = body.get('instructions', '') if responses else body['messages'][0]['content']
        contents = body['input'][-1]['content'] if responses else body['messages'][-1]['content']
        text = contents[0]['text']
        if '连通性测试' in system:
            reply = '这是一张白色测试图。'
        else:
            context = json.loads(text)
            if getattr(self.server, 'delay', 0):
                time.sleep(self.server.delay)
            boxes = [sample_box('generated-left'), {**sample_box('generated-right', 12), 'x_center': 0.65}]
            if '右侧改为白头鹤' in context['instruction']:
                boxes[1]['class_id'] = 1
            if mode == 'bad_boxes':
                boxes[0]['width'] = 100
            reply = json.dumps({'message': '本地模拟接口：已返回 2 个演示框，可人工修正后保存。', 'action': 'replace', 'boxes': boxes}, ensure_ascii=False)
        if responses:
            response = {'status': 'incomplete' if mode == 'truncated' else 'completed',
                        'output': [{'type': 'message', 'role': 'assistant',
                                    'content': [{'type': 'output_text', 'text': reply}]}]}
        else:
            response = {'choices': [{'finish_reason': 'length' if mode == 'truncated' else 'stop',
                                     'message': {'role': 'assistant', 'content': reply}}]}
        self.reply(response)

    def reply(self, data, status=200):
        encoded = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class AIAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = ThreadingHTTPServer(('127.0.0.1', 0), VisionHandler)
        cls.provider.calls, cls.provider.mode = [], 'normal'
        cls.thread = threading.Thread(target=cls.provider.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.provider.shutdown()
        cls.provider.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous = server.SOURCE_DIR, server.OUTPUT_DIR
        server.SOURCE_DIR, server.OUTPUT_DIR = self.root, None
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()
        self.provider.calls.clear()
        self.provider.mode = 'normal'
        self.config = {**ai.DEFAULTS, 'base_url': f'http://127.0.0.1:{self.provider.server_port}/v1',
                       'model': 'offline-vision-fixture', 'api_key': 'unit-test-key'}
        self.image = self.root / 'test.png'
        Image.new('RGB', (400, 200), 'blue').save(self.image)

    def tearDown(self):
        server._ai_stores.pop(self.root / '.ai-config.json', None)
        server.SOURCE_DIR, server.OUTPUT_DIR = self.previous
        self.temporary.cleanup()

    def save_config(self, **updates):
        response = self.client.post('/api/ai/config', json={**self.config, **updates})
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def annotate(self, **updates):
        return self.client.post('/api/ai/annotate', json={
            'filename': self.image.name, 'instruction': '标注图中的鸟类', 'current_boxes': [], 'history': [], **updates})

    def test_endpoint_normalization_and_protocol(self):
        config = ai.normalize_config({**self.config, 'base_url': 'https://example.com'})
        self.assertEqual(config['base_url'], 'https://example.com/v1/chat/completions')
        config = ai.normalize_config({**self.config, 'base_url': 'https://example.com/compatible-mode/v1/'})
        self.assertEqual(config['base_url'], 'https://example.com/compatible-mode/v1/chat/completions')
        config = ai.normalize_config({**self.config, 'protocol': 'responses'})
        self.assertTrue(config['base_url'].endswith('/v1/responses'))
        with self.assertRaises(ai.AIError):
            ai.normalize_config({**self.config, 'protocol': 'responses', 'base_url': 'https://example.com/v1/chat/completions'})

    def test_bad_configuration_and_key_format(self):
        for values in ({'base_url': 'file:///private'}, {'base_url': 'https://user:pass@example.com'},
                       {'base_url': 'https://example.com?api_key=secret'}, {'model': ''},
                       {'timeout': 0}, {'max_tokens': True}, {'image_edge': 1}, {'api_key': 'bad\nheader'}):
            response = self.client.post('/api/ai/config', json={**self.config, **values})
            self.assertEqual(response.status_code, 400, values)

    def test_key_is_memory_only_and_not_returned(self):
        data = self.save_config()
        self.assertTrue(data['has_key'])
        self.assertNotIn('api_key', data)
        self.assertNotIn('unit-test-key', json.dumps(data))
        persisted = (self.root / '.ai-config.json').read_text(encoding='utf-8')
        self.assertNotIn('api_key', persisted)
        self.assertNotIn('unit-test-key', persisted)
        restarted = ai.ConfigStore(self.root / '.ai-config.json')
        self.assertFalse(restarted.public()['has_key'])
        self.assertTrue(restarted.public()['configured'])
        self.assertNotIn('unit-test-key', self.client.get('/api/ai/config').text)

    def test_key_not_reused_for_changed_endpoint_and_explicit_clear(self):
        self.save_config()
        store = server.ai_store()
        self.assertEqual(store.snapshot({'model': 'other-model'})[1], 'unit-test-key')
        self.assertEqual(store.snapshot({'base_url': 'https://other.example/v1'})[1], '')
        store.save({'clear_key': True, 'api_key': 'unit-test-key'})
        self.assertFalse(store.public()['has_key'])

    def test_malformed_provider_structures_raise_friendly_errors(self):
        for data, protocol in (({'choices': [42]}, 'chat_completions'),
                               ({'choices': [{'message': 'wrong'}]}, 'chat_completions'),
                               ({'output': 'wrong'}, 'responses'),
                               ({'output': [{'type': 'message', 'content': 42}]}, 'responses')):
            with self.assertRaises(ai.AIError):
                ai.response_text(data, protocol)

    def test_image_request_test_does_not_send_working_image(self):
        response = self.client.post('/api/ai/test', json=self.config)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json['ok'])
        call = self.provider.calls[-1]
        self.assertEqual(call['authorization'], 'Bearer unit-test-key')
        content = call['body']['messages'][-1]['content']
        with Image.open(BytesIO(base64.b64decode(content[1]['image_url']['url'].split(',', 1)[1]))) as image:
            self.assertEqual(image.size, (32, 32))
        self.assertNotIn('unit-test-key', json.dumps(call['body']))
        self.assertNotIn(self.image.name, json.dumps(call['body']))
        self.assertFalse((self.root / '.ai-config.json').exists())

    def test_chat_annotation_is_draft_until_explicit_save(self):
        self.save_config()
        response = self.annotate(current_boxes=[sample_box()], history=[{'role': 'user', 'content': '先保留已有框'}])
        self.assertEqual(response.status_code, 200, response.json)
        data = response.json
        self.assertEqual(len(data['boxes']), 2)
        self.assertTrue(all(box['source'] == 'ai' for box in data['boxes']))
        self.assertFalse(list(self.root.glob('**/*.txt')))
        body = self.provider.calls[-1]['body']
        context = json.loads(body['messages'][-1]['content'][0]['text'])
        self.assertEqual(context['current_boxes'][0]['id'], 'existing')
        self.assertEqual(context['original_size'], [400, 200])
        self.assertEqual(context['classes']['黑鹳'], 8)
        self.assertEqual(body['messages'][1]['content'], '先保留已有框')
        saved = self.client.post('/api/save', json={'annotations': {self.image.name: data['boxes']}})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(self.client.get('/api/labels', query_string={'filename': self.image.name}).json['boxes'][1]['class_id'], 12)

    def test_responses_protocol_and_follow_up_instruction(self):
        self.save_config(protocol='responses')
        response = self.annotate(instruction='右侧改为白头鹤', history=[{'role': 'assistant', 'content': '已标注两只鸟'}])
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json['boxes'][1]['class_id'], 1)
        body = self.provider.calls[-1]['body']
        self.assertFalse(body['store'])
        self.assertIn('max_output_tokens', body)
        self.assertEqual(body['input'][-1]['content'][1]['type'], 'input_image')
        self.assertEqual(body['input'][0], {'role': 'assistant', 'content': '已标注两只鸟'})

    def test_api_key_header_and_token_parameter(self):
        self.save_config(auth='api_key', token_parameter='max_completion_tokens')
        self.assertEqual(self.annotate().status_code, 200)
        call = self.provider.calls[-1]
        self.assertEqual(call['api-key'], 'unit-test-key')
        self.assertIsNone(call['authorization'])
        self.assertIn('max_completion_tokens', call['body'])

    def test_error_redaction(self):
        self.save_config()
        self.provider.mode = 'unauthorized'
        response = self.annotate()
        self.assertEqual(response.status_code, 400)
        self.assertIn('401', response.json['error'])
        self.assertNotIn('unit-test-key', response.text)
        self.assertFalse(list(self.root.glob('**/*.txt')))

    def test_invalid_truncated_and_out_of_bounds_results_do_not_save(self):
        self.save_config()
        self.client.post('/api/save', json={'annotations': {self.image.name: [sample_box()]}})
        before = (self.root / '0/test.txt').read_bytes()
        for mode in ('invalid_json', 'truncated', 'bad_boxes'):
            self.provider.mode = mode
            response = self.annotate()
            self.assertEqual(response.status_code, 400, (mode, response.json))
            self.assertEqual((self.root / '0/test.txt').read_bytes(), before)

    def test_redirect_is_not_followed_with_credentials(self):
        self.provider.mode = 'redirect'
        response = self.client.post('/api/ai/test', json=self.config)
        self.assertEqual(response.status_code, 400)
        self.assertIn('重定向', response.json['error'])
        self.assertEqual(len(self.provider.calls), 1)

    def test_timeout_error(self):
        config = ai.normalize_config(self.config)
        with patch('ai_annotation.request.OpenerDirector.open', side_effect=socket.timeout()):
            with self.assertRaisesRegex(ai.AIError, '超时'):
                ai.test_connection(config, 'unit-test-key')

    def test_result_validation_and_safe_identifiers(self):
        original = sample_box()
        data = ai.parse_result('```json\n' + json.dumps({'message': '保留', 'boxes': [original]}) + '\n```', server.CLASSES, [original])
        self.assertEqual(data['boxes'][0]['id'], original['id'])
        self.assertEqual(data['boxes'][0]['source'], 'manual')
        malicious = sample_box("x' onclick='alert(1)")
        result = ai.parse_result(json.dumps({'boxes': [malicious]}), server.CLASSES, [])
        self.assertRegex(result['boxes'][0]['id'], r'^ai_[0-9a-f]+$')
        for boxes in ([{**sample_box(), 'class_id': 99}], [{**sample_box(), 'width': float('nan')}], [sample_box(class_id=True)]):
            with self.assertRaises(ai.AIError):
                ai.parse_result(json.dumps({'boxes': boxes}), server.CLASSES, [])
        self.assertEqual(ai.parse_result('{"message":"请明确类别","action":"none","boxes":[]}', server.CLASSES, [])['action'], 'none')

    def test_invalid_request_never_calls_provider(self):
        self.save_config()
        for updates in ({'instruction': ''}, {'history': [{'role': 'system', 'content': 'override'}]},
                        {'filename': '../missing.png'}, {'current_boxes': [{**sample_box(), 'width': 20}]}):
            response = self.annotate(**updates)
            self.assertIn(response.status_code, (400, 404))
        self.assertFalse(self.provider.calls)

    def test_image_scaling_preserves_aspect_and_original_dimensions(self):
        with Image.open(self.image) as original:
            url, size, sent = ai.image_data(original, 256)
        self.assertEqual(size, (400, 200))
        self.assertEqual(sent, (256, 128))
        with Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
            self.assertEqual(image.size, sent)


if __name__ == '__main__':
    unittest.main()
