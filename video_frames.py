"""Stream video samples to full-resolution JPEGs and export reviewed YOLO pairs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile

import cv2
import numpy as np


VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mts', '.m2ts', '.webm'}
WORK_FOLDER = '_video_annotation'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@dataclass
class FrameOptions:
    interval: float = 2.0
    dedup: bool = True
    hash_distance: int = 6
    pixel_difference: float = 3.0
    tile_difference: float = 10.0
    max_gap: float = 30.0
    start: float = 0.0
    end: float | None = None
    max_frames: int = 0

    @classmethod
    def parse(cls, values):
        values = values or {}
        if not isinstance(values, dict):
            raise ValueError('抽帧设置应为对象')
        options = cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})
        for key in ('interval', 'pixel_difference', 'tile_difference', 'max_gap', 'start'):
            value = getattr(options, key)
            if isinstance(value, bool):
                raise ValueError(f'{key} 必须是有限数值')
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} 必须是非负有限数值')
            setattr(options, key, value)
        if options.interval < 0.05:
            raise ValueError('抽帧间隔不能小于 0.05 秒')
        for key, maximum in (('hash_distance', 64), ('max_frames', 1000000)):
            value = getattr(options, key)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f'{key} 必须是 0 到 {maximum} 的整数')
        if type(options.dedup) is not bool:
            raise ValueError('dedup 应为布尔值')
        if options.end is not None:
            options.end = float(options.end)
            if not math.isfinite(options.end) or options.end <= options.start:
                raise ValueError('结束时间必须大于开始时间')
        return options


def signature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Fixed-size, aspect-preserving comparison; original pixels are saved separately.
    ratio = min(1.0, 640 / max(gray.shape))
    small = cv2.resize(gray, (max(1, round(gray.shape[1] * ratio)),
                              max(1, round(gray.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
    frequency = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    bits = frequency > np.median(frequency.flatten()[1:])
    bits[0, 0] = False
    return bits, small


def similar(left, right, options):
    if left[1].shape != right[1].shape:
        return False
    if np.count_nonzero(left[0] != right[0]) > options.hash_distance:
        return False
    difference = cv2.absdiff(left[1], right[1]).astype(np.float32)
    if float(difference.mean()) > options.pixel_difference:
        return False
    # A global hash alone can hide changes to a small bird on a large static background.
    for y in range(0, difference.shape[0], 32):
        for x in range(0, difference.shape[1], 32):
            if float(difference[y:y + 32, x:x + 32].mean()) > options.tile_difference:
                return False
    return True


def extract_video(source, directory, source_root, options, emit, cancelled, known_groups=None):
    """Decode sequentially, comparing candidates with the last retained frame only.

    Bounded memory: one reference signature, one decoded frame. No model is needed.
    emit(record, statistics) receives a record only after its image exists on disk.
    """
    source, directory, source_root = Path(source).resolve(), Path(directory), Path(source_root).resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f'无法打开视频：{source.name}，请检查文件或解码器')
    stats = {'source': str(source), 'decoded': 0, 'sampled': 0, 'kept': 0, 'duplicates': 0,
             'timestamp': 0.0, 'limited': False}
    try:
        # Content identity keeps renamed/re-uploaded copies in the same export partition.
        digest = hashlib.sha256()
        stats.update(phase='fingerprint', fingerprint_bytes=0, source_bytes=source.stat().st_size)
        with source.open('rb') as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                if cancelled():
                    return stats
                digest.update(chunk)
                stats['fingerprint_bytes'] += len(chunk)
                emit(None, dict(stats))
        group = digest.hexdigest()
        stats.update(video_group=group, phase='extract')
        if known_groups is not None:
            if group in known_groups:
                stats['duplicate_video'] = True
                return stats
            known_groups.add(group)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'无法读取视频帧率：{source.name}')
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        stats.update(fps=fps, total_frames=frame_count, duration=frame_count / fps)
        images = directory / 'images'
        images.mkdir(parents=True, exist_ok=True)
        reference, previous_time = None, None
        next_time, index = options.start, 0
        while not cancelled():
            if not capture.grab():
                if frame_count and index < frame_count - 2 and not stats['limited']:
                    raise ValueError(f'视频提前解码结束：{source.name}，已读 {index}/{frame_count} 帧；已提取图片仍可标注')
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            # Several codecs return zero or non-monotonic timestamps. Fall back to frame index.
            if not math.isfinite(timestamp) or (index and timestamp <= stats['timestamp']):
                timestamp = index / fps
            stats.update(decoded=index + 1, timestamp=timestamp)
            frame_index = index
            index += 1
            if options.end is not None and timestamp >= options.end:
                break
            if timestamp + 1e-6 < next_time:
                if index % 100 == 0:
                    emit(None, dict(stats))
                continue
            next_time = options.start + (math.floor(max(0, timestamp - options.start) / options.interval) + 1) * options.interval
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise ValueError(f'无法解码视频第 {frame_index} 帧：{source.name}')
            stats['sampled'] += 1
            current = signature(frame) if options.dedup else None
            force = previous_time is None or (options.max_gap > 0 and timestamp - previous_time >= options.max_gap)
            if options.dedup and reference is not None and not force and similar(reference, current, options):
                stats['duplicates'] += 1
                emit(None, dict(stats))
                continue
            name = f'{group[:16]}_f{frame_index:09d}_t{round(timestamp * 1000):012d}.jpg'
            destination = images / name
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError(f'帧编码失败：{source.name} 第 {frame_index} 帧')
            encoded.tofile(str(destination))
            stats['kept'] += 1
            record = {'image': destination.resolve().relative_to(source_root).as_posix(),
                      'source_video': str(source), 'video_group': group, 'frame_index': frame_index,
                      'timestamp_seconds': round(timestamp, 6), 'fps': fps,
                      'width': frame.shape[1], 'height': frame.shape[0]}
            emit(record, dict(stats))
            reference, previous_time = current, timestamp
            if options.max_frames and stats['kept'] >= options.max_frames:
                stats['limited'] = True
                break
        if not stats['decoded'] and not cancelled():
            raise ValueError(f'视频没有可解码帧：{source.name}')
        return stats
    finally:
        capture.release()


def read_manifest(path):
    if not Path(path).is_file():
        return []
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            # A concurrently appended final record may not have been flushed completely yet.
            if line.strip() and line.endswith('\n'):
                rows.append(json.loads(line))
    return rows


def export_dataset(rows, classes, destination, val_fraction=0.0, seed=42):
    """Export only caller-validated, explicitly saved pairs. Never split one video."""
    if not rows:
        raise ValueError('没有已保存的人工标注，请先保存当前图片或视频帧')
    if not math.isfinite(val_fraction) or not 0 <= val_fraction < 1:
        raise ValueError('验证集比例必须在 0（不划分）到 1 之间')
    groups = sorted({row['group'] for row in rows})
    validation = set()
    if val_fraction:
        if len(groups) < 2:
            raise ValueError('当前只有一个来源视频，无法按视频分成训练与验证集。请取消划分导出图片集，或增加独立视频/图片来源。')
        random.Random(seed).shuffle(groups)
        validation = set(groups[:max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {'all': 0} if not val_fraction else {'train': 0, 'val': 0}
    manifest = []
    try:
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for row in rows:
                split = ('val' if row['group'] in validation else 'train') if val_fraction else 'all'
                suffix = f'/{split}' if val_fraction else ''
                image = Path(row['path'])
                stem = hashlib.sha256(row['name'].encode('utf-8')).hexdigest()[:16] + '_' + image.stem
                target = f'images{suffix}/{stem}{image.suffix.lower()}'
                label = f'labels{suffix}/{stem}.txt'
                archive.write(image, target, compress_type=zipfile.ZIP_STORED)
                archive.writestr(label, row['label_text'])
                manifest.append({**row.get('origin', {}), 'original_image': row['name'],
                                 'image': target, 'annotation': label, 'split': split,
                                 'group': row['group'], 'box_count': row['box_count']})
                counts[split] += 1
            ordered = [name for name, _ in sorted(classes.items(), key=lambda item: item[1])]
            archive.writestr('classes.json', json.dumps(classes, ensure_ascii=False, indent=2))
            archive.writestr('classes.txt', '\n'.join(ordered) + '\n')
            archive.writestr('source_manifest.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in manifest))
            if val_fraction:
                # Relative to data.yaml; portable to the existing dataset preparation tools.
                import yaml
                archive.writestr('data.yaml', yaml.safe_dump({'train': 'images/train', 'val': 'images/val',
                                                            'nc': len(ordered), 'names': ordered}, allow_unicode=True, sort_keys=False))
            archive.writestr('README.txt', '原始分辨率图片与 YOLO 五列标签一一对应；空 TXT 表示人工确认无目标。\n'
                             '来源视频、帧号、时间点和分组见 source_manifest.jsonl。\n' +
                             ('已按视频来源分组划分 train/val。\n' if val_fraction else
                              '本包尚未划分训练/验证集，没有 data.yaml；请按来源视频分组后训练，勿随机拆分同段视频帧。\n'))
        return {'count': len(rows), 'splits': counts, 'groups': len(groups), 'path': str(destination)}
    except Exception:
        destination.unlink(missing_ok=True)
        raise
