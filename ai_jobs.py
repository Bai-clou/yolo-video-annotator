"""In-memory annotation jobs: short HTTP polling, no credentials or partial labels."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

from ai_annotation import AICancelled, AIError


class JobManager:
    def __init__(self, workers=2, limit=128):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ai-annotation')
        self.limit = limit
        self.jobs = {}
        self.lock = threading.RLock()

    def _public(self, job):
        now = job.get('_finished') or time.monotonic()
        started = job.get('_started', job['_created'])
        return {**{key: value for key, value in job.items() if not key.startswith('_')},
                'elapsed_seconds': round(now - started, 1)}

    def create(self, root, filename, instruction, timeout, request_id, work):
        with self.lock:
            for job in self.jobs.values():
                if job['_root'] == root and job['request_id'] == request_id:
                    if job['filename'] != filename or job['instruction'] != instruction:
                        raise AIError('请求编号已被另一项任务使用，请重新提交')
                    return self._public(job)
                if job['_root'] == root and job['filename'] == filename and job['status'] in ('queued', 'running'):
                    raise AIError('当前图片已有后台 AI 任务，请刷新页面恢复等待；不会重复向模型提交')
            terminal = sorted((job for job in self.jobs.values() if job['status'] not in ('queued', 'running')),
                              key=lambda item: item['_created'])
            while len(self.jobs) >= self.limit and terminal:
                self.jobs.pop(terminal.pop(0)['id'], None)
            if len(self.jobs) >= self.limit:
                raise AIError('后台 AI 任务较多，请等待已有任务结束')
            job = {'id': uuid.uuid4().hex, 'request_id': request_id, 'filename': filename,
                   'instruction': instruction, 'timeout': timeout, 'status': 'queued', 'phase': 'queued',
                   'received_chars': 0, 'analysis_chars': 0, 'events': 0,
                   '_root': root, '_created': time.monotonic(), '_cancel': threading.Event(), '_ack': False}
            self.jobs[job['id']] = job
            job['_future'] = self.executor.submit(self._run, job, work)
            return self._public(job)

    def _run(self, job, work):
        with self.lock:
            if job['_cancel'].is_set():
                return
            job.update(status='running', phase='preparing', _started=time.monotonic())

        def progress(update):
            with self.lock:
                if job['_cancel'].is_set():
                    raise AICancelled('已停止接收此任务的结果')
                allowed = ('phase', 'received_chars', 'analysis_chars', 'events', 'sent_size', 'original_size')
                job.update({key: value for key, value in update.items() if key in allowed})

        try:
            result = work(progress)
            with self.lock:
                if not job['_cancel'].is_set():
                    job.update(status='completed', phase='completed', result=result)
        except AICancelled:
            with self.lock:
                job.update(status='cancelled', phase='cancelled')
        except Exception as exception:
            with self.lock:
                if not job['_cancel'].is_set():
                    job.update(status='failed', phase='failed',
                               error=str(exception) if isinstance(exception, AIError) else '后台 AI 任务失败，已有标注保持不变')
                    if getattr(exception, 'code', None) == 'output_token_limit':
                        job.update(error_code=exception.code, error_details=exception.details)
        finally:
            with self.lock:
                job['_finished'] = time.monotonic()

    def get(self, root, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            return self._public(job) if job and job['_root'] == root else None

    def pending(self, root):
        with self.lock:
            return [self._public(job) for job in sorted(self.jobs.values(), key=lambda item: item['_created'], reverse=True)
                    if job['_root'] == root and not job['_ack']]

    def cancel(self, root, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if not job or job['_root'] != root:
                return None
            if job['status'] in ('queued', 'running') or (job['status'] == 'completed' and not job['_ack']):
                job['_cancel'].set()
                job['_future'].cancel()
                job.pop('result', None)
                job.update(status='cancelled', phase='cancelled', _finished=time.monotonic())
            return self._public(job)

    def acknowledge(self, root, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if not job or job['_root'] != root:
                return False
            if job['status'] not in ('queued', 'running'):
                job['_ack'] = True
            return True
