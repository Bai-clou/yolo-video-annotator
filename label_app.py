"""Local YOLO image/video annotation server. See VIDEO_ANNOTATION.md."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
import uuid

from flask import Flask, has_request_context, jsonify, render_template, request, send_file, session
from PIL import Image
from urllib.parse import unquote

from video_frames import (FrameOptions, VIDEO_EXTENSIONS, WORK_FOLDER, atomic_json,
                          export_dataset, extract_video, read_manifest)
import ai_annotation
from ai_jobs import JobManager

ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(ROOT / 'yolo26x.pt')
CONFIDENCE_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
SOURCE_DIR = ROOT
OUTPUT_DIR = None
CLASSES = json.loads((ROOT / 'classes.json').read_text(encoding='utf-8-sig'))
app = Flask(__name__)
app.secret_key = secrets.token_bytes(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Strict')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 ** 3
_model = None
_model_lock = threading.Lock()
_save_lock = threading.Lock()
_jobs_lock = threading.Lock()
_jobs = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-frames')
_ai_stores = {}
_ai_stores_lock = threading.Lock()
_ai_jobs = JobManager()


def source_dir():
    return Path(session.get('annotation_source', SOURCE_DIR)).resolve() if has_request_context() else SOURCE_DIR.resolve()


def output_dir():
    # A newly opened folder must not write labels into the startup --output directory.
    return None if has_request_context() and 'annotation_source' in session else OUTPUT_DIR


@app.before_request
def check_workspace():
    expected = request.headers.get('X-Annotation-Source')
    if expected and unquote(expected) != str(source_dir()):
        return jsonify(error='工作目录已在其他页面切换，请先保存原页面草稿或刷新页面'), 409


def ai_store():
    path = source_dir() / '.ai-config.json'
    with _ai_stores_lock:
        if path not in _ai_stores:
            _ai_stores[path] = ai_annotation.ConfigStore(path)
        return _ai_stores[path]


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            # Manual annotation and frame extraction work without Torch/Ultralytics.
            from ultralytics import YOLO
            _model = YOLO(MODEL_PATH)
    return _model


def workspace():
    path = source_dir() / WORK_FOLDER
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_path(filename):
    if not isinstance(filename, str) or not filename:
        return None
    path = Path(filename)
    if not path.is_absolute():
        path = source_dir() / path
    path = path.resolve()
    try:
        path.relative_to(source_dir())
    except ValueError:
        return None
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return path
    return None


def image_name(path):
    return path.resolve().relative_to(source_dir()).as_posix()


def is_video_frame(path):
    relative = path.resolve().relative_to(source_dir())
    return len(relative.parts) == 4 and relative.parts[0] == WORK_FOLDER and relative.parts[2] == 'images'


def label_candidates(path):
    if is_video_frame(path):
        return [path.parent.parent / 'labels' / (path.stem + '.txt')]
    base = output_dir() or path.parent
    direct = base / (path.stem + '.txt')
    # Only numeric category directories are legacy annotation directories.
    nested = sorted(p / (path.stem + '.txt') for p in base.iterdir()
                    if p.is_dir() and p.name.isdecimal()) if base.exists() else []
    return [direct, *nested]


def find_label_path(path):
    matches = [p for p in label_candidates(path) if p.is_file()]
    if len(matches) > 1 and len({p.read_bytes() for p in matches}) > 1:
        raise ValueError(f'{path.name} 存在内容冲突的多份 TXT，请先核对：' + ', '.join(str(p) for p in matches))
    return matches[0] if matches else None


def checked_boxes(boxes):
    if not isinstance(boxes, list):
        raise ValueError('标注框必须是列表')
    result = []
    for box in boxes:
        if not isinstance(box, dict) or type(box.get('class_id')) is not int or box['class_id'] not in CLASSES.values():
            raise ValueError('类别 ID 不在当前类别表中')
        values = [box.get(key) for key in ('x_center', 'y_center', 'width', 'height')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('框坐标必须是有限数值')
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError('YOLO 坐标应在 0 到 1 之间，宽高必须大于 0')
        if x - width / 2 < -1e-6 or y - height / 2 < -1e-6 or x + width / 2 > 1.000001 or y + height / 2 > 1.000001:
            raise ValueError('标注框超出图片边界，请调整后保存')
        result.append({'id': box.get('id', uuid.uuid4().hex[:12]), 'class_id': box['class_id'],
                       'x_center': x, 'y_center': y, 'width': width, 'height': height,
                       'source': box.get('source', 'manual'), 'conf': box.get('conf')})
    return result


def read_boxes(path):
    boxes = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path.name}:{number} 不是 YOLO 检测框的五列格式')
        try:
            boxes.append({'class_id': int(parts[0]), **dict(zip(('x_center', 'y_center', 'width', 'height'), map(float, parts[1:])))})
        except ValueError as error:
            raise ValueError(f'{path.name}:{number} 标签数值无效') from error
    return checked_boxes(boxes)


def box_text(boxes):
    return ''.join(f"{b['class_id']} {b['x_center']:.6f} {b['y_center']:.6f} {b['width']:.6f} {b['height']:.6f}\n" for b in boxes)


def describe_image(path, origin=None):
    label = find_label_path(path)
    return {'name': image_name(path), 'path': str(path.resolve()), 'size': path.stat().st_size,
            'annotated': label is not None, 'origin': origin}


def scan_images(directory):
    if has_request_context() and session.get('annotation_recursive', False):
        candidates = []
        for base, folders, names in os.walk(directory, followlinks=False):
            folders[:] = sorted(name for name in folders if not name.startswith('.')
                                and name not in (WORK_FOLDER, '__pycache__', 'venv', 'env'))
            candidates.extend(Path(base) / name for name in sorted(names) if Path(name).suffix.lower() in IMAGE_EXTENSIONS)
    else:
        candidates = sorted(Path(directory).iterdir())
    images = [describe_image(p) for p in candidates
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and get_image_path(str(p))]
    work = Path(directory) / WORK_FOLDER
    if work.exists():
        for manifest in sorted(work.glob('*/frames.jsonl')):
            for row in read_manifest(manifest):
                path = get_image_path(row['image'])
                if path:
                    images.append(describe_image(path, row))
    return images


def json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError('需要 JSON 对象')
    return data


@app.errorhandler(ValueError)
def value_error(error):
    if isinstance(error, ai_annotation.OutputTokenLimit):
        return jsonify(error=str(error), error_code=error.code, error_details=error.details), 400
    return jsonify(error=str(error)), 400


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='上传超过 20 GB，请用本机视频路径导入，无需复制原视频'), 413


@app.route('/')
def index():
    return render_template('index.html')


@app.get('/api/scan')
def api_scan():
    images = scan_images(source_dir())
    return jsonify(images=images, source_dir=str(source_dir()), count=len(images), classes=CLASSES,
                   recursive=session.get('annotation_recursive', False))


@app.post('/api/source')
def api_open_source():
    data = json_body()
    value, recursive = data.get('path'), data.get('recursive', True)
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or '\0' in value:
        raise ValueError('请选择或填写本机图片目录')
    if type(recursive) is not bool:
        raise ValueError('包含子目录设置应为布尔值')
    try:
        path = Path(value.strip().strip('"')).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('图片目录不存在，请选择已有文件夹')
        with os.scandir(path) as entries:
            next(entries, None)
    except OSError:
        raise ValueError('无法读取该目录，请检查目录路径和访问权限') from None
    # Flask loads a separate session snapshot for each request. Existing workers capture their root.
    previous = dict(session)
    session.update(annotation_source=str(path), annotation_recursive=recursive)
    try:
        images = scan_images(path)
    except (ValueError, OSError) as error:
        session.clear()
        session.update(previous)
        raise ValueError('打开目录失败：' + str(error)) from None
    return jsonify(source_dir=str(path), count=len(images), recursive=recursive)


@app.get('/api/folders')
def api_browse_folders():
    value = request.args.get('path', '')
    if len(value) > 4096 or '\0' in value:
        raise ValueError('目录路径无效')
    try:
        path = Path(value).expanduser().resolve() if value else source_dir()
        if not path.is_dir():
            raise ValueError('目录不存在，请检查路径')
        with os.scandir(path) as entries:
            folders = sorted(({'name': entry.name, 'path': str(Path(entry.path).resolve())}
                              for entry in entries if entry.is_dir() and not entry.name.startswith('.')),
                             key=lambda entry: entry['name'].casefold())
    except OSError:
        raise ValueError('无法浏览该目录，请检查访问权限或填写其他目录路径') from None
    drives = [str(Path(f'{letter}:\\')) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
              if Path(f'{letter}:\\').is_dir()] if os.name == 'nt' else ['/']
    return jsonify(path=str(path), parent=str(path.parent) if path.parent != path else None,
                   folders=folders, drives=drives)


@app.post('/api/upload')
def api_upload():
    results = []
    for upload in request.files.getlist('images'):
        name = Path(upload.filename.replace('\\', '/')).name
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        destination = source_dir() / name
        if destination.exists():
            destination = source_dir() / f'{Path(name).stem}_{uuid.uuid4().hex[:8]}{Path(name).suffix}'
        # Decode before saving; uploads cannot create unreadable image entries.
        with Image.open(upload.stream) as image:
            image.verify()
        upload.stream.seek(0)
        upload.save(destination)
        results.append(describe_image(destination))
    if not results:
        raise ValueError('没有可导入的图片')
    return jsonify(images=results, count=len(results))


@app.get('/api/image')
def api_image():
    path = get_image_path(request.args.get('path', ''))
    return send_file(path) if path else (jsonify(error='图片不存在或不在当前工作目录'), 404)


@app.get('/api/labels')
def api_labels():
    path = get_image_path(request.args.get('filename', ''))
    if path is None:
        return jsonify(error='图片不存在'), 404
    with _save_lock:
        label = find_label_path(path)
        boxes = read_boxes(label) if label else []
    return jsonify(boxes=boxes, annotated=label is not None)


def _run_detection(path, default_class_id, confidence):
    if type(default_class_id) is not int or default_class_id not in CLASSES.values():
        raise ValueError('预设类别 ID 不在类别表中')
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('置信度必须在 0 到 1 之间')
    model = get_model()
    with _model_lock:
        result = model(str(path), conf=confidence, verbose=False)[0]
    height, width = result.orig_shape
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append({'id': uuid.uuid4().hex[:12], 'class_id': default_class_id, 'source': 'auto',
                          'conf': round(float(box.conf[0]), 4), 'x_center': (x1 + x2) / 2 / width,
                          'y_center': (y1 + y2) / 2 / height, 'width': (x2 - x1) / width, 'height': (y2 - y1) / height})
    return boxes


@app.post('/api/detect')
def api_detect():
    data = json_body()
    path = get_image_path(data.get('filename', ''))
    if not path:
        return jsonify(error='图片不存在'), 404
    try:
        boxes = _run_detection(path, data.get('default_class_id', 0), data.get('conf', CONFIDENCE_THRESHOLD))
    except (ImportError, RuntimeError, FileNotFoundError) as error:
        return jsonify(error=f'辅助检测不可用：{error}；仍可手工画框'), 400
    return jsonify(detections=boxes, count=len(boxes))


@app.post('/api/detect-all')
def api_detect_all():
    data = json_body()
    results = {}
    for name in data.get('filenames', []):
        path = get_image_path(name)
        if path:
            results[name] = _run_detection(path, data.get('default_class_id', 0), data.get('conf', CONFIDENCE_THRESHOLD))
    return jsonify(results=results)


@app.get('/api/ai/config')
def api_ai_get_config():
    return jsonify(ai_store().public())


@app.post('/api/ai/config')
def api_ai_save_config():
    return jsonify(ai_store().save(json_body()))


@app.post('/api/ai/test')
def api_ai_test():
    config, key = ai_store().snapshot(json_body())
    return jsonify(ai_annotation.test_connection(config, key))


@app.post('/api/ai/annotate')
def api_ai_annotate():
    data = json_body()
    path = get_image_path(data.get('filename', ''))
    if path is None:
        return jsonify(error='当前图片不存在'), 404
    current = checked_boxes(data.get('current_boxes', []))
    if len(current) > 1000:
        raise ValueError('当前图片超过 1000 个框，请分批处理')
    config, key = ai_store().snapshot()
    result = ai_annotation.annotate(config, key, path, CLASSES, current,
                                    data.get('instruction'), data.get('history', []))
    result['boxes'] = checked_boxes(result['boxes'])
    result['filename'] = image_name(path)
    return jsonify(result)  # Draft only: a separate /api/save call is required to persist labels.


@app.post('/api/ai/jobs')
def api_ai_start_job():
    data = json_body()
    path = get_image_path(data.get('filename', ''))
    if path is None:
        return jsonify(error='当前图片不存在'), 404
    instruction = data.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 12000:
        raise ValueError('请填写 1 到 12000 字的标注需求')
    current = checked_boxes(data.get('current_boxes', []))
    if len(current) > 1000:
        raise ValueError('当前图片超过 1000 个框，请分批处理')
    history = ai_annotation.clean_history(data.get('history', []))
    config, key = ai_store().snapshot()
    classes, filename, root = dict(CLASSES), image_name(path), str(source_dir())
    request_id = data.get('request_id', uuid.uuid4().hex)
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 80:
        raise ValueError('请求编号无效')

    def work(progress):
        result = ai_annotation.annotate(config, key, path, classes, current, instruction, history, progress)
        result['boxes'] = checked_boxes(result['boxes'])
        result['filename'] = filename
        return result

    job = _ai_jobs.create(root, filename, instruction, config['timeout'], request_id, work)
    return jsonify(job), 202


@app.get('/api/ai/jobs')
def api_ai_pending_jobs():
    return jsonify(jobs=_ai_jobs.pending(str(source_dir())))


@app.get('/api/ai/jobs/<identifier>')
def api_ai_job(identifier):
    job = _ai_jobs.get(str(source_dir()), identifier)
    return jsonify(job) if job else (jsonify(error='后台任务不存在或服务已重启；已保存标签不受影响'), 404)


@app.post('/api/ai/jobs/<identifier>/cancel')
def api_ai_cancel_job(identifier):
    job = _ai_jobs.cancel(str(source_dir()), identifier)
    return jsonify(job) if job else (jsonify(error='后台任务不存在'), 404)


@app.post('/api/ai/jobs/<identifier>/ack')
def api_ai_ack_job(identifier):
    acknowledged = _ai_jobs.acknowledge(str(source_dir()), identifier)
    return jsonify(ok=True) if acknowledged else (jsonify(error='后台任务不存在'), 404)


@app.post('/api/save')
def api_save():
    annotations = json_body().get('annotations')
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError('没有待保存的标注')
    prepared = []
    for name, boxes in annotations.items():
        path = get_image_path(name)
        if not path:
            raise ValueError(f'图片不存在：{name}')
        boxes = checked_boxes(boxes)
        if is_video_frame(path):
            target = path.parent.parent / 'labels' / (path.stem + '.txt')
        else:
            counts = {}
            for box in boxes:
                counts[box['class_id']] = counts.get(box['class_id'], 0) + 1
            base = output_dir() or path.parent
            target = base / str(max(counts, key=counts.get)) / (path.stem + '.txt') if counts else base / (path.stem + '.txt')
        prepared.append((name, path, target, box_text(boxes)))
    saved = []
    with _save_lock:
        for name, path, target, text in prepared:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.txt.tmp')
            temporary.write_text(text, encoding='utf-8')
            temporary.replace(target)
            # Changing the dominant class or clearing a label must not leave a stale TXT.
            for old in label_candidates(path):
                if old != target and old.is_file():
                    old.unlink()
            saved.append(str(target))
    return jsonify(saved=saved, count=len(saved))


def public_job(job):
    return {key: value for key, value in job.items() if not key.startswith('_')}


def run_video_job(job_id, directory, sources, options, root):
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(status='running', stage='开始读取视频')
    try:
        known_groups = set()
        with (directory / 'frames.jsonl').open('a', encoding='utf-8') as manifest:
            for number, source in enumerate(sources, 1):
                if job['_cancel'].is_set():
                    break
                def emit(record, stats):
                    if record:
                        manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
                        manifest.flush()
                    with _jobs_lock:
                        job.update(current=stats, video_index=number, stage=f'视频 {number}/{len(sources)}：{source.name}')
                        if record:
                            job['kept'] += 1
                        if time.monotonic() - job.get('_persisted', 0) >= 1:
                            atomic_json(directory / 'job.json', public_job(job))
                            job['_persisted'] = time.monotonic()
                stats = extract_video(source, directory, root, options, emit, job['_cancel'].is_set, known_groups)
                with _jobs_lock:
                    job['videos'].append(stats)
                    job['current'] = stats
            with _jobs_lock:
                job['status'] = 'cancelled' if job['_cancel'].is_set() else 'completed'
                job['stage'] = '已停止，已提取的帧可继续标注' if job['_cancel'].is_set() else '抽帧完成'
                if not job['kept'] and job['status'] == 'completed':
                    job.update(status='failed', error='选定时间范围内没有抽到图片，请检查开始、结束时间')
    except Exception as error:
        with _jobs_lock:
            job.update(status='failed', error=str(error), stage='抽帧未完成，已提取的帧仍然保留')
    finally:
        with _jobs_lock:
            atomic_json(directory / 'job.json', public_job(job))


@app.post('/api/videos/import')
def api_video_import():
    options = FrameOptions.parse(json.loads(request.form.get('options', '{}')))
    sources = []
    for line in request.form.get('paths', '').splitlines():
        path = Path(line.strip().strip('"')).expanduser()
        if not line.strip():
            continue
        if not path.is_absolute():
            path = source_dir() / path
        path = path.resolve()
        if path.is_dir():
            sources.extend(sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS))
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            sources.append(path)
        else:
            raise ValueError(f'视频路径不存在或格式不支持：{path}')
    uploads = request.files.getlist('videos')
    if not sources and not any(upload.filename for upload in uploads):
        raise ValueError('请填写本机视频路径/目录，或选择视频文件')
    job_id = uuid.uuid4().hex
    directory = workspace() / job_id
    directory.mkdir()
    for upload in uploads:
        if not upload.filename:
            continue
        name = Path(upload.filename.replace('\\', '/')).name
        if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f'不支持的视频格式：{name}')
        raw = directory / 'originals'
        raw.mkdir(exist_ok=True)
        path = raw / (uuid.uuid4().hex[:8] + '_' + name)
        upload.save(path)
        sources.append(path.resolve())
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError('所选目录里没有支持的视频')
    job = {'id': job_id, 'status': 'queued', 'stage': '等待抽帧', 'kept': 0, 'videos': [],
           'video_count': len(sources), 'video_index': 0, 'options': asdict(options),
           'sources': [str(p) for p in sources], '_cancel': threading.Event()}
    with _jobs_lock:
        _jobs[job_id] = job
        atomic_json(directory / 'job.json', public_job(job))
    _executor.submit(run_video_job, job_id, directory, sources, options, source_dir())
    return jsonify(id=job_id), 202


@app.get('/api/videos/jobs')
def api_video_jobs():
    jobs = []
    for file in sorted(workspace().glob('*/job.json')):
        value = json.loads(file.read_text(encoding='utf-8'))
        with _jobs_lock:
            active = _jobs.get(value['id'])
            value = public_job(active) if active else value
        if not active and value['status'] in ('queued', 'running'):
            value.update(status='interrupted', stage='上次抽帧已中断，已提取帧可继续标注；需要时重新导入视频')
        jobs.append(value)
    return jsonify(jobs=jobs)


@app.get('/api/videos/jobs/<job_id>')
def api_video_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify(error='本次运行没有该任务，请查看历史抽帧记录'), 404
        return jsonify(public_job(job))


@app.post('/api/videos/jobs/<job_id>/cancel')
def api_cancel_video(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify(error='任务不存在'), 404
        job['_cancel'].set()
    return jsonify(ok=True)


@app.post('/api/export')
def api_export():
    data = json_body()
    fraction = float(data.get('val_fraction', 0))
    rows = []
    with _save_lock:
        for image in scan_images(source_dir()):
            path = get_image_path(image['name'])
            label = find_label_path(path)
            if not label:
                continue
            boxes = read_boxes(label)
            origin = image.get('origin') or {}
            rows.append({'name': image['name'], 'path': str(path), 'label_text': box_text(boxes),
                         'box_count': len(boxes), 'origin': origin,
                         'group': 'video:' + origin['video_group'] if origin else 'image:' + image['name']})
        export_id = uuid.uuid4().hex
        # A synchronous local export streams from disk when downloaded; the browser need not buffer a ZIP.
        result = export_dataset(rows, CLASSES, workspace() / 'exports' / (export_id + '.zip'), fraction)
    result['download_url'] = '/api/export/' + export_id
    return jsonify(result)


@app.get('/api/export/<export_id>')
def api_export_download(export_id):
    if len(export_id) != 32 or any(c not in '0123456789abcdef' for c in export_id):
        return jsonify(error='导出文件不存在'), 404
    path = workspace() / 'exports' / (export_id + '.zip')
    if not path.is_file():
        return jsonify(error='导出文件不存在'), 404
    return send_file(path, as_attachment=True, download_name='birds_yolo_' + export_id[:8] + '.zip')


def main():
    global SOURCE_DIR, OUTPUT_DIR, MODEL_PATH, CLASSES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT, help='图片与抽帧工作目录')
    parser.add_argument('--output', type=Path, help='普通图片标签输出目录；视频帧使用独立 images/labels 目录')
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--classes', type=Path, default=ROOT / 'classes.json')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    SOURCE_DIR = args.source.expanduser().resolve()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = args.output.expanduser().resolve() if args.output else None
    if OUTPUT_DIR:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLASSES = json.loads(args.classes.read_text(encoding='utf-8-sig'))
    if not isinstance(CLASSES, dict) or any(type(v) is not int for v in CLASSES.values()) or sorted(CLASSES.values()) != list(range(len(CLASSES))):
        parser.error('类别表应为类别名称到连续 ID（从 0 开始）的 JSON 对象')
    MODEL_PATH = args.model
    print(f'YOLO 图片与视频标注工具\n工作目录：{SOURCE_DIR}\n访问：http://127.0.0.1:{args.port}\n模型仅在点击辅助检测时加载')
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
