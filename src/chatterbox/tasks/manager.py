import threading
import queue
import time
import logging
from typing import Dict, Optional, List
from concurrent.futures import Future

from .models import (
    SpeechJob,
    SpeechJobRequest,
    TaskStatus,
    TaskError,
    generate_job_id,
)
from ..engine.service import SpeechEngine, get_speech_engine, ValidationError, SynthesisError
from ..artifacts.storage import ArtifactManager, get_artifact_manager
from ..artifacts.verifier import AudioVerificationError

logger = logging.getLogger("chatterbox.tasks.manager")


class QueueFullError(Exception):
    """Raised when the task execution queue is at capacity."""
    pass


class TaskManager:
    """
    Manages autonomous execution of speech synthesis tasks.
    Enforces the explicit lifecycle:
    RECEIVED -> VALIDATING -> GENERATING -> VERIFYING -> COMPLETED (or FAILED).
    """

    def __init__(
        self,
        engine: Optional[SpeechEngine] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        max_queue_size: int = 100,
        base_url: str = "",
    ):
        self.engine = engine or get_speech_engine()
        self.artifact_manager = artifact_manager or get_artifact_manager()
        self.max_queue_size = max_queue_size
        self.base_url = base_url

        self._jobs: Dict[str, SpeechJob] = {}
        self._events: Dict[str, threading.Event] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._current_job_id: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Starts the background task worker."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="SpeechTaskWorker")
        self._worker_thread.start()
        logger.info("Speech TaskManager background worker started.")

    def stop(self) -> None:
        """Stops the background task worker."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._queue.put(None)  # Sentinel to wake up worker
            self._worker_thread.join(timeout=2.0)
        logger.info("Speech TaskManager worker stopped.")

    def submit_job(self, request: SpeechJobRequest) -> SpeechJob:
        """
        Enqueues a new speech generation task.
        Sets initial status to RECEIVED.
        """
        if not self._running:
            self.start()

        if self._queue.qsize() >= self.max_queue_size:
            raise QueueFullError("Task queue is at maximum capacity. Please retry shortly.")

        job_id = generate_job_id()
        job = SpeechJob(job_id=job_id, request=request, status=TaskStatus.RECEIVED)
        event = threading.Event()

        with self._lock:
            self._jobs[job_id] = job
            self._events[job_id] = event

        try:
            self._queue.put_nowait(job_id)
            logger.info("Job %s enqueued with status RECEIVED (queue size: %d)", job_id, self._queue.qsize())
        except queue.Full:
            with self._lock:
                del self._jobs[job_id]
                del self._events[job_id]
            raise QueueFullError("Task queue is full.")

        return job

    def wait_for_job(self, job_id: str, timeout: float = 60.0) -> SpeechJob:
        """Blocks until the job completes, fails, or the timeout expires."""
        event = self._events.get(job_id)
        if not event:
            raise KeyError(f"Job {job_id} not found.")

        finished = event.wait(timeout=timeout)
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found.")

        if not finished and job.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            logger.warning("Job %s wait timed out after %.1f seconds", job_id, timeout)
            # We don't mark as FAILED immediately because it may still be processing,
            # but we return the in-flight state.
        return job

    def get_job(self, job_id: str) -> Optional[SpeechJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[SpeechJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False
            job.transition_to(TaskStatus.CANCELLED)
            if job_id in self._events:
                self._events[job_id].set()
            logger.info("Job %s was cancelled by request", job_id)
            return True

    def get_metrics(self) -> Dict[str, any]:
        with self._lock:
            return {
                "queue_depth": self._queue.qsize(),
                "active_job_id": self._current_job_id,
                "total_tracked_jobs": len(self._jobs),
            }

    def _worker_loop(self) -> None:
        """Sequential task consumer loop."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break
                job_id = item
                self._current_job_id = job_id
                try:
                    self._execute_job(job_id)
                finally:
                    self._current_job_id = None
                    self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Unexpected error in worker loop: %s", e, exc_info=True)

    def _execute_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job or job.status == TaskStatus.CANCELLED:
            return

        logger.info("Starting execution of job %s", job_id)
        event = self._events.get(job_id)

        try:
            # 1. VALIDATING
            job.transition_to(TaskStatus.VALIDATING)
            self.engine.validate_request(
                text=job.request.text,
                language=job.request.language,
                exaggeration=job.request.options.exaggeration,
                temperature=job.request.options.temperature,
                cfg_weight=job.request.options.cfg_weight,
                seed=job.request.options.seed,
                repetition_penalty=job.request.options.repetition_penalty,
            )

            # 2. GENERATING
            job.transition_to(TaskStatus.GENERATING)
            sample_rate, waveform = self.engine.synthesize(
                text=job.request.text,
                language=job.request.language,
                voice=job.request.voice,
                exaggeration=job.request.options.exaggeration,
                temperature=job.request.options.temperature,
                cfg_weight=job.request.options.cfg_weight,
                seed=job.request.options.seed,
                repetition_penalty=job.request.options.repetition_penalty,
            )

            # 3. VERIFYING
            job.transition_to(TaskStatus.VERIFYING)
            artifact = self.artifact_manager.save_and_verify(
                job_id=job.job_id,
                audio_np=waveform,
                sample_rate=sample_rate,
                base_url=self.base_url,
            )

            # 4. COMPLETED
            job.artifact = artifact
            job.transition_to(TaskStatus.COMPLETED)
            logger.info("Job %s successfully completed in %.1f ms", job_id, job.duration_ms)

        except ValidationError as e:
            job.error = TaskError(code="VALIDATION_FAILED", message=str(e), category="validation")
            job.transition_to(TaskStatus.FAILED)
            logger.warning("Job %s validation failed: %s", job_id, e)
        except AudioVerificationError as e:
            job.error = TaskError(code="VERIFICATION_FAILED", message=str(e), category="audio_verification")
            job.transition_to(TaskStatus.FAILED)
            logger.error("Job %s audio verification failed: %s", job_id, e)
        except SynthesisError as e:
            job.error = TaskError(code="SYNTHESIS_FAILED", message="Speech synthesis model encountered an error.", category="synthesis")
            job.transition_to(TaskStatus.FAILED)
            logger.error("Job %s synthesis error: %s", job_id, e)
        except Exception as e:
            job.error = TaskError(code="INTERNAL_ERROR", message="An unexpected processing error occurred.", category="system")
            job.transition_to(TaskStatus.FAILED)
            logger.error("Job %s failed with unexpected exception: %s", job_id, e, exc_info=True)
        finally:
            if event:
                event.set()


_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
        _task_manager.start()
    return _task_manager
