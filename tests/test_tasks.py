import time
import pytest
import numpy as np
from unittest.mock import MagicMock

from chatterbox.tasks.models import SpeechJobRequest, SpeechRequestOptions, TaskStatus
from chatterbox.tasks.manager import TaskManager
from chatterbox.artifacts.storage import ArtifactManager


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.is_ready = True
    sr = 24000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    wav = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    engine.synthesize.return_value = (sr, wav)
    return engine


def test_task_lifecycle_completion(tmp_path, mock_engine):
    artifact_mgr = ArtifactManager(base_dir=tmp_path)
    task_mgr = TaskManager(engine=mock_engine, artifact_manager=artifact_mgr, max_queue_size=10)
    task_mgr.start()

    try:
        req = SpeechJobRequest(
            text="اختبار النظام",
            language="ar",
            options=SpeechRequestOptions(wait=True),
        )
        job = task_mgr.submit_job(req)
        assert job.status in (TaskStatus.RECEIVED, TaskStatus.VALIDATING, TaskStatus.GENERATING, TaskStatus.VERIFYING, TaskStatus.COMPLETED)

        completed_job = task_mgr.wait_for_job(job.job_id, timeout=5.0)
        assert completed_job.status == TaskStatus.COMPLETED
        assert completed_job.artifact is not None
        assert completed_job.artifact.verified is True
        assert completed_job.error is None

        # Check observed lifecycle state sequence
        statuses = [s for s, _ in completed_job.transitions]
        assert TaskStatus.RECEIVED in statuses
        assert TaskStatus.VALIDATING in statuses
        assert TaskStatus.GENERATING in statuses
        assert TaskStatus.VERIFYING in statuses
        assert TaskStatus.COMPLETED in statuses
    finally:
        task_mgr.stop()


def test_task_lifecycle_validation_failure(tmp_path):
    engine = MagicMock()
    from chatterbox.engine.service import ValidationError
    engine.validate_request.side_effect = ValidationError("Invalid language")

    artifact_mgr = ArtifactManager(base_dir=tmp_path)
    task_mgr = TaskManager(engine=engine, artifact_manager=artifact_mgr)
    task_mgr.start()

    try:
        req = SpeechJobRequest(
            text="Invalid test",
            language="invalid_lang",
            options=SpeechRequestOptions(wait=True),
        )
        job = task_mgr.submit_job(req)
        failed_job = task_mgr.wait_for_job(job.job_id, timeout=5.0)

        assert failed_job.status == TaskStatus.FAILED
        assert failed_job.error is not None
        assert failed_job.error.code == "VALIDATION_FAILED"
        assert "Invalid language" in failed_job.error.message
    finally:
        task_mgr.stop()
