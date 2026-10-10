import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import httpx

logger = logging.getLogger("chatterbox.client")


class SpeechClientError(Exception):
    """Base exception for Chatterbox client errors."""
    pass


class SpeechServiceUnavailableError(SpeechClientError):
    """Raised when the speech service cannot be reached."""
    pass


class SpeechTimeoutError(SpeechClientError):
    """Raised when speech generation times out."""
    pass


class SpeechVerificationError(SpeechClientError):
    """Raised when the downloaded speech artifact is corrupted or invalid."""
    pass


@dataclass
class SpeechResult:
    """Structured result returned to Orqevyn and external agents."""
    job_id: str
    status: str
    audio_path: Optional[str]
    duration_seconds: float
    sample_rate: int
    file_size_bytes: int
    audio_url: str
    verified: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "file_size_bytes": self.file_size_bytes,
            "audio_url": self.audio_url,
            "verified": self.verified,
            "metadata": self.metadata,
        }


class OrqevynSpeechClient:
    """
    Integration client / tool adapter for Orqevyn and AI agents.
    Exposes a stable operation: speak_text(text, voice, language, options).
    
    1. Submits text to the speech service.
    2. Handles timeouts and service errors gracefully.
    3. Tracks asynchronous jobs with polling or synchronous waiting.
    4. Downloads the resulting audio artifact to local disk.
    5. Verifies that the returned artifact is accessible and valid.
    6. Returns a structured result to Orqevyn.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        download_dir: Optional[str | Path] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.download_dir = Path(download_dir).resolve() if download_dir else Path.cwd() / "downloads" / "audio"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.transport = transport

    def _get_client(self, timeout: float) -> httpx.Client:
        kwargs: Dict[str, Any] = {"timeout": timeout}
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return httpx.Client(**kwargs)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_health(self) -> Dict[str, Any]:
        """Inspects speech engine readiness."""
        try:
            with self._get_client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/health")
                res.raise_for_status()
                return res.json()
        except httpx.HTTPError as e:
            raise SpeechServiceUnavailableError(f"Speech service unavailable at {self.base_url}: {e}") from e

    def list_voices(self, language: Optional[str] = None) -> list[Dict[str, Any]]:
        """Queries available voices."""
        try:
            params = {"language": language} if language else {}
            with self._get_client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/v1/voices", headers=self._headers(), params=params)
                res.raise_for_status()
                return res.json()
        except httpx.HTTPError as e:
            raise SpeechClientError(f"Failed to list voices: {e}") from e

    def speak_text(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "ar",
        options: Optional[Dict[str, Any]] = None,
    ) -> SpeechResult:
        """
        Primary adapter function.
        Submits speech request, monitors lifecycle, downloads audio, verifies integrity,
        and returns a structured SpeechResult.
        """
        opts = options.copy() if options else {}
        opts.setdefault("wait", True)
        timeout = opts.get("timeout", self.timeout)

        payload = {
            "text": text,
            "language": language,
            "voice": voice,
            "options": opts,
        }

        # 1. Submit request
        try:
            with self._get_client(timeout=timeout + 10.0) as client:
                res = client.post(
                    f"{self.base_url}/v1/speech",
                    headers=self._headers(),
                    json=payload,
                )
                res.raise_for_status()
                job_data = res.json()
        except httpx.TimeoutException as e:
            raise SpeechTimeoutError(f"Request to speech service timed out after {timeout}s: {e}") from e
        except httpx.HTTPError as e:
            raise SpeechServiceUnavailableError(f"Speech service error during submission: {e}") from e

        job_id = job_data["job_id"]
        status = job_data.get("status")

        # 2. If asynchronous, poll until completion
        if status not in ("COMPLETED", "FAILED", "CANCELLED"):
            job_data = self._poll_job(job_id, timeout=timeout)
            status = job_data.get("status")

        if status != "COMPLETED":
            err = job_data.get("error") or {}
            err_msg = err.get("message", f"Job failed with status {status}")
            raise SpeechClientError(f"Speech synthesis job '{job_id}' did not complete successfully: {err_msg}")

        artifact_info = job_data.get("artifact")
        if not artifact_info:
            raise SpeechClientError(f"Job {job_id} reported COMPLETED but no artifact metadata was returned.")

        # 3. Retrieve resulting audio artifact
        local_audio_path = self.download_dir / f"{job_id}.wav"
        audio_url = f"{self.base_url}/v1/speech/jobs/{job_id}/audio"

        try:
            with self._get_client(timeout=15.0) as client:
                audio_res = client.get(audio_url, headers=self._headers())
                audio_res.raise_for_status()
                with open(local_audio_path, "wb") as f:
                    f.write(audio_res.content)
        except Exception as e:
            raise SpeechClientError(f"Failed to retrieve generated audio for job '{job_id}': {e}") from e

        # 4. Verify downloaded artifact integrity
        if not local_audio_path.is_file() or local_audio_path.stat().st_size < 512:
            raise SpeechVerificationError(f"Downloaded audio artifact '{local_audio_path}' is corrupted or empty.")

        duration = artifact_info.get("duration_seconds", 0.0)
        sample_rate = artifact_info.get("sample_rate", 24000)
        size_bytes = local_audio_path.stat().st_size

        return SpeechResult(
            job_id=job_id,
            status="COMPLETED",
            audio_path=str(local_audio_path),
            duration_seconds=duration,
            sample_rate=sample_rate,
            file_size_bytes=size_bytes,
            audio_url=audio_url,
            verified=True,
            metadata={
                "language": language,
                "voice": voice,
                "duration_ms": job_data.get("duration_ms"),
                "transitions": job_data.get("transitions"),
            },
        )

    def _poll_job(self, job_id: str, timeout: float = 60.0, poll_interval: float = 0.5) -> Dict[str, Any]:
        """Polls job endpoint until final status is reached."""
        start_time = time.time()
        url = f"{self.base_url}/v1/speech/jobs/{job_id}"

        with self._get_client(timeout=5.0) as client:
            while time.time() - start_time < timeout:
                try:
                    res = client.get(url, headers=self._headers())
                    res.raise_for_status()
                    data = res.json()
                    status = data.get("status")
                    if status in ("COMPLETED", "FAILED", "CANCELLED"):
                        return data
                except httpx.HTTPError as e:
                    logger.warning("Polling error for job %s: %s", job_id, e)

                time.sleep(poll_interval)

        raise SpeechTimeoutError(f"Job {job_id} timed out after {timeout} seconds")
