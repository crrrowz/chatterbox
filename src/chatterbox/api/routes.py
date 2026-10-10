import os
from contextlib import asynccontextmanager
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Header, Depends, status
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    CreateSpeechRequest,
    SpeechJobResponse,
    VoiceProfileSchema,
    CreateCustomVoiceRequest,
    HealthResponse,
)
from ..engine.config import get_config
from ..engine.service import get_speech_engine, SpeechEngine
from ..engine.voices import get_voice_registry, VoiceRegistry, VoiceProfile
from ..tasks.manager import get_task_manager, TaskManager, QueueFullError
from ..tasks.models import SpeechJobRequest, SpeechRequestOptions, TaskStatus
from ..artifacts.storage import get_artifact_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start worker & initialize model in background thread
    task_manager = get_task_manager()
    engine = get_speech_engine()
    
    # Pre-warm model in background
    import threading
    init_thread = threading.Thread(target=engine.initialize, daemon=True, name="ModelPrewarm")
    init_thread.start()
    
    yield
    
    # Shutdown: Stop task manager worker
    task_manager.stop()


app = FastAPI(
    title="Chatterbox Autonomous Speech Engine",
    description="High-performance, reliable speech synthesis engine for AI agents & Orqevyn.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local developer tools and web UI integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """Serves the Chatterbox Speech Studio web dashboard."""
    index_file = WEB_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Chatterbox Speech Studio UI</h1>", status_code=200)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi import Response
    return Response(status_code=204)


def verify_api_key(authorization: Optional[str] = Header(None)):
    config = get_config()
    if not config.enable_auth or not config.api_key:
        return True
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    token = authorization.replace("Bearer ", "").strip()
    if token != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return True


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Inspects service status, model readiness, and task queue metrics."""
    engine = get_speech_engine()
    task_mgr = get_task_manager()
    
    status_dict = engine.get_status()
    metrics = task_mgr.get_metrics()
    
    is_healthy = status_dict["ready"]
    return HealthResponse(
        status="healthy" if is_healthy else "initializing",
        ready=status_dict["ready"],
        device=status_dict["device"],
        model_type=status_dict["model_type"],
        t3_model=status_dict["t3_model"],
        sample_rate=status_dict["sample_rate"],
        supported_languages=status_dict["supported_languages"],
        queue_depth=metrics["queue_depth"],
        active_job_id=metrics["active_job_id"],
        total_tracked_jobs=metrics["total_tracked_jobs"],
        error=status_dict["init_error"],
    )


@app.get("/v1/voices", response_model=List[VoiceProfileSchema], tags=["Voices"])
def list_voices(
    language: Optional[str] = Query(None, description="Filter voices by language code (e.g. 'ar', 'en')"),
    _: bool = Depends(verify_api_key),
):
    """Retrieves available voice presets and reference speaker configurations."""
    registry = get_voice_registry()
    voices = registry.list_voices(language=language)
    return [
        VoiceProfileSchema(
            voice_id=v.voice_id,
            name=v.name,
            language=v.language,
            gender=v.gender,
            description=v.description,
            reference_path=v.reference_path,
            is_builtin=v.is_builtin,
        )
        for v in voices
    ]


@app.post("/v1/voices", response_model=VoiceProfileSchema, status_code=status.HTTP_201_CREATED, tags=["Voices"])
def create_custom_voice(
    req: CreateCustomVoiceRequest,
    _: bool = Depends(verify_api_key),
):
    """Creates, names, and registers a custom voice profile with reference audio."""
    import base64
    import uuid
    import re

    registry = get_voice_registry()

    # Generate or sanitize voice_id
    if req.voice_id and req.voice_id.strip():
        voice_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.voice_id.strip().lower())
    else:
        clean_name = re.sub(r"[^a-zA-Z0-9]", "_", req.name.strip().lower())
        voice_id = f"custom_{clean_name}_{uuid.uuid4().hex[:6]}"

    # Check collision
    if registry.get_voice(voice_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Voice with ID '{voice_id}' already exists."
        )

    ref_path = req.reference_path

    # If base64 audio is provided, decode and convert to standard 16-bit PCM WAV
    if req.reference_base64 and req.reference_base64.strip():
        try:
            audio_bytes = base64.b64decode(req.reference_base64)
            save_path = registry.storage_dir / f"{voice_id}.wav"
            from ..artifacts.converter import convert_to_pcm_wav
            convert_to_pcm_wav(audio_bytes, save_path, sample_rate=24000)
            ref_path = str(save_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode or convert audio data: {e}"
            )

    profile = VoiceProfile(
        voice_id=voice_id,
        name=req.name,
        language=req.language.lower(),
        gender=req.gender,
        description=req.description,
        reference_path=ref_path,
        is_builtin=False,
    )
    registry.register_voice(profile)

    return VoiceProfileSchema(
        voice_id=profile.voice_id,
        name=profile.name,
        language=profile.language,
        gender=profile.gender,
        description=profile.description,
        reference_path=profile.reference_path,
        is_builtin=profile.is_builtin,
    )


@app.delete("/v1/voices/{voice_id}", tags=["Voices"])
def delete_custom_voice(
    voice_id: str,
    _: bool = Depends(verify_api_key),
):
    """Deletes a custom created voice profile."""
    registry = get_voice_registry()
    voice = registry.get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")
    if voice.is_builtin:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete built-in system voice.")

    registry.delete_voice(voice_id)
    return {"status": "deleted", "voice_id": voice_id}


@app.post("/v1/speech", response_model=SpeechJobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Speech"])
def submit_speech(
    request: CreateSpeechRequest,
    _: bool = Depends(verify_api_key),
):
    """
    Submits a speech generation task.
    Supports asynchronous execution (default) or synchronous waiting (`options.wait=True`).
    """
    task_mgr = get_task_manager()

    job_req = SpeechJobRequest(
        text=request.text,
        language=request.language,
        voice=request.voice,
        options=SpeechRequestOptions(
            exaggeration=request.options.exaggeration,
            temperature=request.options.temperature,
            cfg_weight=request.options.cfg_weight,
            seed=request.options.seed,
            repetition_penalty=request.options.repetition_penalty,
            wait=request.options.wait,
            timeout=request.options.timeout,
        ),
    )

    try:
        job = task_mgr.submit_job(job_req)
    except QueueFullError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    if request.options.wait:
        job = task_mgr.wait_for_job(job.job_id, timeout=request.options.timeout)

    return job.to_dict()


@app.get("/v1/speech/jobs/{job_id}", response_model=SpeechJobResponse, tags=["Speech"])
def get_job_status(
    job_id: str,
    _: bool = Depends(verify_api_key),
):
    """Inspects task state, execution lifecycle, verification metrics, or failure diagnostics."""
    task_mgr = get_task_manager()
    job = task_mgr.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speech task '{job_id}' not found.",
        )
    return job.to_dict()


@app.get("/v1/speech/jobs/{job_id}/audio", tags=["Speech"])
def get_job_audio(
    job_id: str,
    _: bool = Depends(verify_api_key),
):
    """Retrieves the verified audio artifact (WAV) for a completed speech task."""
    task_mgr = get_task_manager()
    job = task_mgr.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speech task '{job_id}' not found.",
        )

    if job.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Audio not ready. Current job status is {job.status.value}.",
        )

    artifact_mgr = get_artifact_manager()
    audio_path = artifact_mgr.get_artifact_file(job_id)
    if not audio_path or not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio artifact file missing from storage.",
        )

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=f"{job_id}.wav",
    )


@app.delete("/v1/speech/jobs/{job_id}", tags=["Speech"])
def cancel_job(
    job_id: str,
    _: bool = Depends(verify_api_key),
):
    """Cancels a pending or running speech generation task."""
    task_mgr = get_task_manager()
    success = task_mgr.cancel_job(job_id)
    if not success:
        job = task_mgr.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in state {job.status.value}.",
        )
    return {"status": "cancelled", "job_id": job_id}
