from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from ..tasks.models import TaskStatus


class SpeechOptionsSchema(BaseModel):
    exaggeration: float = Field(0.5, ge=0.1, le=2.0, description="Speech expressiveness (0.1 to 2.0)")
    temperature: float = Field(0.8, ge=0.05, le=2.0, description="Sampling randomness (0.05 to 2.0)")
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0, description="Classifier-free guidance pace weight (0.0 to 2.0)")
    seed: int = Field(0, ge=0, description="Seed for reproducibility (0 for random)")
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.5, description="Penalize repeated tokens (1.0 to 2.5)")
    wait: bool = Field(False, description="Whether to wait synchronously for completion before returning")
    timeout: float = Field(300.0, ge=1.0, le=1800.0, description="Timeout in seconds when wait=True (up to 1800s)")


class CreateSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize (1 to 5000 characters with auto-chunking)")
    language: str = Field("ar", description="ISO 639-1 language code (e.g. 'ar', 'en', 'fr', 'es')")
    voice: Optional[str] = Field(None, description="Voice preset ID or reference audio path/URL")
    options: SpeechOptionsSchema = Field(default_factory=SpeechOptionsSchema)


class AudioArtifactSchema(BaseModel):
    job_id: str
    file_path: str
    audio_url: str
    format: str = "wav"
    sample_rate: int = 24000
    duration_seconds: float
    file_size_bytes: int
    verified: bool


class TaskErrorSchema(BaseModel):
    code: str
    message: str
    category: str
    timestamp: float


class TaskTransitionSchema(BaseModel):
    status: TaskStatus
    timestamp: float


class SpeechJobResponse(BaseModel):
    job_id: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    request: Dict[str, Any]
    artifact: Optional[AudioArtifactSchema] = None
    error: Optional[TaskErrorSchema] = None
    transitions: List[TaskTransitionSchema] = Field(default_factory=list)


class VoiceProfileSchema(BaseModel):
    voice_id: str
    name: str
    language: str
    gender: str = "neutral"
    description: str = ""
    reference_path: Optional[str] = None
    is_builtin: bool = True


class CreateCustomVoiceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Display name of the voice")
    voice_id: Optional[str] = Field(None, max_length=50, description="Optional custom unique voice identifier")
    language: str = Field("ar", description="Language code (e.g. 'ar', 'en')")
    gender: str = Field("neutral", description="'female', 'male', or 'neutral'")
    description: str = Field("", max_length=300, description="Short bio or description")
    reference_path: Optional[str] = Field(None, description="Direct URL or server path to reference audio")
    reference_base64: Optional[str] = Field(None, description="Optional base64-encoded audio for instant upload/cloning")


class HealthResponse(BaseModel):
    status: str
    ready: bool
    device: str
    model_type: str
    t3_model: str
    sample_rate: int
    supported_languages: List[str]
    queue_depth: int
    active_job_id: Optional[str] = None
    total_tracked_jobs: int
    error: Optional[str] = None
