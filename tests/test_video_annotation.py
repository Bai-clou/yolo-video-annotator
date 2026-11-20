import io
import json
from pathlib import Path
import tempfile
import shutil
import time
import unittest
import zipfile

import cv2
import numpy as np
from PIL import Image
import yaml

import label_app as server
from video_frames import FrameOptions, export_dataset, extract_video


def make_video(path, scenes=(30, 190), frames_per_scene=15, fps=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (160, 96))
    if not writer.isOpened():
        raise RuntimeError('Test video encoder is unavailable')
    for color in scenes:
        for _ in range(frames_per_scene):
            writer.write(np.full((96, 160, 3), color, dtype=np.uint8))
    writer.release()


def box(class_id=0):
    return {'class_id': class_id, 'x_center': 0.5, 'y_center': 0.4, 'width': 0.2, 'height': 0.3}


class VideoAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_source, self.previous_output = server.SOURCE_DIR, server.OUTPUT_DIR
        server.SOURCE_DIR, server.OUTPUT_DIR = self.root, None
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()

    def tearDown(self):
        server.SOURCE_DIR, server.OUTPUT_DIR = self.previous_source, self.previous_output
        self.temporary.cleanup()

    def image(self, name='鸟类测试.JPG'):
        path = self.root / name
        Image.new('RGB', (160, 96), '#204060').save(path)
        return path

    def extraction(self, options):
        source = self.root / 'sample.avi'
        make_video(source)
        records = []
        result = extract_video(source, self.root / '_video_annotation' / 'test', self.root, options,
                               lambda record, stats: records.append(record) if record else None, lambda: False)
        return records, result

    def test_sampling_dedup_and_original_resolution(self):
        records, stats = self.extraction(FrameOptions(interval=1, max_gap=0))
        self.assertEqual((stats['sampled'], stats['kept'], stats['duplicates']), (6, 2, 4))
        self.assertEqual([row['frame_index'] for row in records], [0, 15])
        for row in records:
            with Image.open(self.root / row['image']) as image:
                self.assertEqual(image.size, (160, 96))
        self.assertEqual(records[1]['timestamp_seconds'], 3)

    def test_disable_dedup_and_time_range(self):
        records, stats = self.extraction(FrameOptions(interval=1, dedup=False, start=1, end=4))
        self.assertEqual([row['frame_index'] for row in records], [5, 10, 15])
        self.assertEqual(stats['duplicates'], 0)

    def test_retention_gap_and_frame_limit(self):
        records, stats = self.extraction(FrameOptions(interval=1, max_gap=2, max_frames=3))
        self.assertEqual(len(records), 3)
        self.assertTrue(stats['limited'])
        self.assertEqual([row['frame_index'] for row in records], [0, 10, 15])

    def test_invalid_options_and_unreadable_video(self):
        for values in ({'interval': 0}, {'interval': float('nan')}, {'end': -1}, {'hash_distance': 90}, {'dedup': 'true'}):
            with self.assertRaises(ValueError):
                FrameOptions.parse(values)
        with self.assertRaises(ValueError):
            extract_video(self.root / 'missing.mp4', self.root / 'out', self.root, FrameOptions(), lambda *args: None, lambda: False)

    def test_content_identity_for_renamed_videos_and_cancellation(self):
        original = self.root / 'original.avi'
        duplicate = self.root / 'renamed.avi'
        make_video(original)
        shutil.copyfile(original, duplicate)
        groups = set()
        stats = extract_video(original, self.root / 'first', self.root, FrameOptions(), lambda *args: None, lambda: False, groups)
        other = extract_video(duplicate, self.root / 'second', self.root, FrameOptions(), lambda *args: None, lambda: False, groups)
        self.assertEqual(stats['video_group'], other['video_group'])
        self.assertTrue(other['duplicate_video'])
        self.assertEqual(other['kept'], 0)
        stopped = extract_video(original, self.root / 'stopped', self.root, FrameOptions(), lambda *args: None, lambda: True)
        self.assertEqual(stopped['kept'], 0)
        self.assertFalse((self.root / 'stopped').exists())

    def test_save_reload_change_class_and_clear(self):
        path = self.image()
        response = self.client.post('/api/save', json={'annotations': {path.name: [box(12)]}})
        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.root / '12' / (path.stem + '.txt')).is_file())
        loaded = self.client.get('/api/labels', query_string={'filename': path.name}).json
        self.assertEqual(loaded['boxes'][0]['class_id'], 12)
        self.client.post('/api/save', json={'annotations': {path.name: [box(0)]}})
        self.assertFalse((self.root / '12' / (path.stem + '.txt')).exists())
        self.client.post('/api/save', json={'annotations': {path.name: []}})
        self.assertFalse((self.root / '0' / (path.stem + '.txt')).exists())
        loaded = self.client.get('/api/labels', query_string={'filename': path.name}).json
        self.assertTrue(loaded['annotated'])
        self.assertEqual(loaded['boxes'], [])

    def test_validation_before_writes_and_path_containment(self):
        path = self.image()
        bad = box()
        bad['width'] = 2
        response = self.client.post('/api/save', json={'annotations': {path.name: [bad]}})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(list(self.root.glob('**/*.txt')))
        self.assertEqual(self.client.get('/api/image', query_string={'path': str(Path(__file__).resolve())}).status_code, 404)
        invalid = box(99)
        self.assertEqual(self.client.post('/api/save', json={'annotations': {path.name: [invalid]}}).status_code, 400)

    def test_image_upload_without_model(self):
        data = io.BytesIO()
        Image.new('RGB', (20, 10)).save(data, format='PNG')
        data.seek(0)
        response = self.client.post('/api/upload', data={'images': (data, '../new.png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['images'][0]['name'], 'new.png')
        self.assertIsNone(server._model)

    def test_background_import_save_export_and_history(self):
        video = self.root / 'bird.avi'
        make_video(video)
        response = self.client.post('/api/videos/import', data={'paths': str(video), 'options': json.dumps({'interval': 1, 'max_gap': 0})})
        self.assertEqual(response.status_code, 202)
        job_id = response.json['id']
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = self.client.get('/api/videos/jobs/' + job_id).json
            if job['status'] in ('completed', 'failed', 'cancelled'):
                break
            time.sleep(0.02)
        self.assertEqual(job['status'], 'completed', job)
        # Wait for final persisted metadata before the temporary workspace is removed.
        deadline = time.monotonic() + 3
        saved_job = self.root / '_video_annotation' / job_id / 'job.json'
        while time.monotonic() < deadline:
            if json.loads(saved_job.read_text(encoding='utf-8'))['status'] == 'completed':
                break
            time.sleep(0.01)
        images = self.client.get('/api/scan').json['images']
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]['origin']['source_video'], str(video))
        self.client.post('/api/save', json={'annotations': {images[0]['name']: [box()]}})
        self.assertTrue((self.root / '_video_annotation' / job_id / 'labels').is_dir())
        result = self.client.post('/api/export', json={})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(result.json['count'], 1)  # The untouched frame is excluded.
        with zipfile.ZipFile(result.json['path']) as archive:
            manifest = json.loads(archive.read('source_manifest.jsonl'))
            self.assertEqual(manifest['frame_index'], 0)
            self.assertEqual(len([name for name in archive.namelist() if name.startswith('images/')]), 1)
            self.assertNotIn('data.yaml', archive.namelist())
        split = self.client.post('/api/export', json={'val_fraction': 0.2})
        self.assertEqual(split.status_code, 400)  # Never split one video across partitions.
        self.assertEqual(self.client.get('/api/videos/jobs').json['jobs'][0]['status'], 'completed')
        download = self.client.get(result.json['download_url'])
        self.assertEqual(download.status_code, 200)
        download.close()

    def test_export_split_grouping_and_empty_reviewed_label(self):
        image = self.image()
        rows = []
        for index, group in enumerate(('video:a', 'video:a', 'video:b', 'video:b')):
            rows.append({'name': f'{index}.jpg', 'path': str(image), 'group': group,
                         'box_count': 0, 'label_text': '', 'origin': {'frame_index': index}})
        destination = self.root / 'split.zip'
        export_dataset(rows, server.CLASSES, destination, 0.5)
        with zipfile.ZipFile(destination) as archive:
            config = yaml.safe_load(archive.read('data.yaml'))
            self.assertEqual(config['names'][8], '黑鹳')
            partitions = {}
            for line in archive.read('source_manifest.jsonl').decode().splitlines():
                row = json.loads(line)
                partitions.setdefault(row['group'], set()).add(row['split'])
                self.assertEqual(archive.read(row['annotation']), b'')
                self.assertIn(row['image'], archive.namelist())
            self.assertTrue(all(len(partitions[group]) == 1 for group in partitions))
            self.assertEqual(set.union(*partitions.values()), {'train', 'val'})


if __name__ == '__main__':
    unittest.main()
