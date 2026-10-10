from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import time
import uuid


class TaskStatus(str, Enum):
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    GENERATING = "GENERATING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class SpeechRequestOptions:
    exaggeration: float = 0.5
    temperature: float = 0.8
    cfg_weight: float = 0.5
    seed: int = 0
    repetition_penalty: float = 1.2
    wait: bool = False
    timeout: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exaggeration": self.exaggeration,
            "temperature": self.temperature,
            "cfg_weight": self.cfg_weight,
            "seed": self.seed,
            "repetition_penalty": self.repetition_penalty,
            "wait": self.wait,
            "timeout": self.timeout,
        }


@dataclass
class SpeechJobRequest:
    text: str
    language: str = "ar"
    voice: Optional[str] = None
    options: SpeechRequestOptions = field(default_factory=SpeechRequestOptions)


@dataclass
class AudioArtifact:
    job_id: str
    file_path: str
    audio_url: str
    format: str = "wav"
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "file_path": self.file_path,
            "audio_url": self.audio_url,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "file_size_bytes": self.file_size_bytes,
            "verified": self.verified,
        }


@dataclass
class TaskError:
    code: str
    message: str
    category: str = "execution_error"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "timestamp": self.timestamp,
        }


@dataclass
class SpeechJob:
    job_id: str
    request: SpeechJobRequest
    status: TaskStatus = TaskStatus.RECEIVED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    artifact: Optional[AudioArtifact] = None
    error: Optional[TaskError] = None
    transitions: list[tuple[TaskStatus, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.transitions:
            self.transitions.append((self.status, self.created_at))

    def transition_to(self, new_status: TaskStatus) -> None:
        now = time.time()
        self.status = new_status
        self.transitions.append((new_status, now))
        if new_status == TaskStatus.VALIDATING and self.started_at is None:
            self.started_at = now
        elif new_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            self.completed_at = now

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return round((end_time - self.started_at) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "request": {
                "text": self.request.text,
                "language": self.request.language,
                "voice": self.request.voice,
                "options": self.request.options.to_dict(),
            },
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "error": self.error.to_dict() if self.error else None,
            "transitions": [{"status": s.value, "timestamp": t} for s, t in self.transitions],
        }


def generate_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:16]}"
