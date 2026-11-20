import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import quote

from PIL import Image

import label_app as server
from test_video_annotation import box, make_video


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.startup, self.images, self.other = [self.root / name for name in ('startup', '新图片', 'other')]
        for directory in (self.startup, self.images, self.other):
            directory.mkdir()
        self.previous = server.SOURCE_DIR, server.OUTPUT_DIR
        server.SOURCE_DIR, server.OUTPUT_DIR = self.startup, None
        self.client = server.app.test_client()
        self.picture(self.startup / 'old.jpg')
        self.picture(self.images / 'one.JPG')
        self.picture(self.images / '子目录' / 'one.JPG')

    def tearDown(self):
        server.SOURCE_DIR, server.OUTPUT_DIR = self.previous
        for path in list(server._ai_stores):
            if path.is_relative_to(self.root):
                server._ai_stores.pop(path)
        self.temporary.cleanup()

    def picture(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (160, 96), '#245060').save(path)

    def open(self, path=None, recursive=True):
        response = self.client.post('/api/source', json={'path': str(path or self.images), 'recursive': recursive})
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def test_open_recursive_directory_reload_existing_labels_and_save_in_place(self):
        (self.images / 'one.txt').write_text('0 0.5 0.4 0.2 0.3\n', encoding='utf-8')
        before = (self.images / 'one.JPG').read_bytes()
        self.assertEqual(self.open()['count'], 2)
        scanned = self.client.get('/api/scan').json
        self.assertEqual([item['name'] for item in scanned['images']], ['one.JPG', '子目录/one.JPG'])
        self.assertTrue(scanned['images'][0]['annotated'])
        self.assertEqual(self.client.get('/api/labels?filename=one.JPG').json['boxes'][0]['class_id'], 0)
        result = self.client.post('/api/save', json={'annotations': {'子目录/one.JPG': [box(5)]}})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertTrue((self.images / '子目录/5/one.txt').is_file())
        self.assertFalse((self.startup / '5/one.txt').exists())
        self.assertEqual((self.images / 'one.JPG').read_bytes(), before)
        self.assertEqual(server.SOURCE_DIR, self.startup)

    def test_top_level_option_hidden_folders_and_external_paths(self):
        self.picture(self.images / '.hidden' / 'hidden.jpg')
        self.assertEqual(self.open(recursive=False)['count'], 1)
        self.assertEqual(self.open(recursive=True)['count'], 2)
        response = self.client.get('/api/image', query_string={'path': str(self.startup / 'old.jpg')})
        self.assertEqual(response.status_code, 404)

    def test_invalid_or_conflicting_directory_keeps_current_workspace(self):
        for body in ({}, {'path': 1}, {'path': str(self.images), 'recursive': 'yes'},
                     {'path': str(self.root / 'missing')}, {'path': str(self.images / 'one.JPG')}):
            self.assertEqual(self.client.post('/api/source', json=body).status_code, 400)
            self.assertEqual(self.client.get('/api/scan').json['source_dir'], str(self.startup))
        (self.images / 'one.txt').write_text('0 0.5 0.4 0.2 0.3\n')
        (self.images / '1').mkdir()
        (self.images / '1/one.txt').write_text('1 0.5 0.4 0.2 0.3\n')
        self.assertEqual(self.client.post('/api/source', json={'path': str(self.images)}).status_code, 400)
        self.assertEqual(self.client.get('/api/scan').json['source_dir'], str(self.startup))

    def test_browser_sessions_and_stale_page_do_not_mix_save_targets(self):
        second = server.app.test_client()
        self.open()
        self.assertEqual(second.get('/api/scan').json['source_dir'], str(self.startup))
        result = self.client.post('/api/save', json={'annotations': {'one.JPG': [box()]}},
                                  headers={'X-Annotation-Source': quote(str(self.startup))})
        self.assertEqual(result.status_code, 409)
        self.assertFalse(list(self.images.rglob('*.txt')))

    def test_opened_folder_does_not_inherit_startup_label_output_or_ai_key(self):
        server.OUTPUT_DIR = self.root / 'old-output'
        self.client.post('/api/ai/config', json={'base_url': 'https://fixture.invalid/v1', 'model': 'fixture', 'api_key': 'fixture-key'})
        self.open()
        self.assertFalse(self.client.get('/api/ai/config').json['has_key'])
        self.client.post('/api/save', json={'annotations': {'one.JPG': [box(5)]}})
        self.assertTrue((self.images / '5/one.txt').is_file())
        self.assertFalse(server.OUTPUT_DIR.exists())
        self.open(self.startup)
        self.assertTrue(self.client.get('/api/ai/config').json['has_key'])

    def test_folder_browser_navigation_and_errors_do_not_change_workspace(self):
        result = self.client.get('/api/folders', query_string={'path': str(self.images)})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['folders'], [{'name': '子目录', 'path': str(self.images / '子目录')}])
        self.assertEqual(result.json['parent'], str(self.root))
        self.assertTrue(result.json['drives'])
        self.assertEqual(self.client.get('/api/scan').json['source_dir'], str(self.startup))
        self.assertEqual(self.client.get('/api/folders', query_string={'path': str(self.root / 'missing')}).status_code, 400)
        with patch('label_app.os.scandir', side_effect=PermissionError()):
            self.assertEqual(self.client.get('/api/folders').status_code, 400)

    def test_video_directory_worker_keeps_original_root_after_switch(self):
        videos = self.root / '视频'
        videos.mkdir()
        make_video(videos / 'one.avi', frames_per_scene=3)
        (videos / 'sub').mkdir()
        make_video(videos / 'sub/two.avi', scenes=(70, 160), frames_per_scene=3)
        self.open()
        started, release = threading.Event(), threading.Event()
        original_extract = server.extract_video
        captured = []
        def extract(*args):
            captured.append(args[2])
            started.set()
            release.wait(3)
            return original_extract(*args)
        with patch('label_app.extract_video', side_effect=extract):
            response = self.client.post('/api/videos/import', data={'paths': str(videos), 'options': json.dumps({'interval': 0.5})})
            self.assertEqual(response.status_code, 202, response.json)
            identifier = response.json['id']
            self.assertTrue(started.wait(2))
            self.open(self.other)
            release.set()
            for _ in range(300):
                job = self.client.get('/api/videos/jobs/' + identifier).json
                if job['status'] in ('completed', 'failed'):
                    break
                time.sleep(0.01)
            self.assertEqual(job['status'], 'completed', job)
        self.assertEqual(captured, [self.images, self.images])
        self.assertFalse(list(self.other.glob('_video_annotation/*/images/*')))
        self.open()
        frames = [item for item in self.client.get('/api/scan').json['images'] if item['origin']]
        self.assertEqual(len(frames), job['kept'])
        self.assertEqual(job['video_count'], 2)


if __name__ == '__main__':
    unittest.main()
