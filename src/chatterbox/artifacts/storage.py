import os
import time
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

from .verifier import AudioVerifier, AudioVerificationError
from ..tasks.models import AudioArtifact

logger = logging.getLogger("chatterbox.artifacts.storage")


class ArtifactManager:
    """Manages audio file storage, serialization, verification, and TTL cleanup."""

    def __init__(self, base_dir: Path | str, ttl_seconds: int = 86400):
        self.base_dir = Path(base_dir).resolve()
        self.ttl_seconds = ttl_seconds
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_artifact_path(self, job_id: str, ext: str = "wav") -> Path:
        # Safe filename sanitization
        safe_id = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
        return self.base_dir / f"{safe_id}.{ext}"

    def save_and_verify(
        self,
        job_id: str,
        audio_np: np.ndarray,
        sample_rate: int,
        base_url: str = "",
    ) -> AudioArtifact:
        """
        Validates the waveform, writes it to a WAV file, verifies the file on disk,
        and constructs an AudioArtifact object.
        """
        # 1. In-memory verification
        AudioVerifier.verify_waveform(audio_np, sample_rate)

        file_path = self.get_artifact_path(job_id, ext="wav")

        # 2. Save WAV file using soundfile
        try:
            # Ensure float32 or int16 range
            if audio_np.dtype != np.float32 and audio_np.dtype != np.int16:
                audio_np = audio_np.astype(np.float32)

            sf.write(str(file_path), audio_np, sample_rate, subtype="PCM_16")
        except Exception as e:
            logger.error("Failed to write audio file to disk: %s", e, exc_info=True)
            raise AudioVerificationError(f"Failed to write audio file: {e}") from e

        # 3. Post-write verification on disk
        duration, sr, file_size = AudioVerifier.verify_file(file_path, expected_sr=sample_rate)

        audio_url = f"{base_url.rstrip('/')}/v1/speech/jobs/{job_id}/audio" if base_url else f"/v1/speech/jobs/{job_id}/audio"

        artifact = AudioArtifact(
            job_id=job_id,
            file_path=str(file_path),
            audio_url=audio_url,
            format="wav",
            sample_rate=sr,
            duration_seconds=duration,
            file_size_bytes=file_size,
            verified=True,
        )
        logger.info(
            "Audio artifact verified and saved: job_id=%s, duration=%.2fs, size=%d bytes",
            job_id, duration, file_size
        )
        return artifact

    def get_artifact_file(self, job_id: str) -> Optional[Path]:
        path = self.get_artifact_path(job_id)
        if path.is_file():
            return path
        return None

    def delete_artifact(self, job_id: str) -> bool:
        path = self.get_artifact_path(job_id)
        if path.exists():
            try:
                path.unlink()
                logger.info("Deleted artifact for job_id: %s", job_id)
                return True
            except Exception as e:
                logger.warning("Failed to delete artifact %s: %s", path, e)
        return False

    def cleanup_expired(self) -> int:
        """Removes artifacts exceeding the configured TTL."""
        now = time.time()
        removed = 0
        try:
            for item in self.base_dir.glob("*.wav"):
                if item.is_file():
                    age = now - item.stat().st_mtime
                    if age > self.ttl_seconds:
                        try:
                            item.unlink()
                            removed += 1
                        except Exception:
                            pass
        except Exception as e:
            logger.error("Error during artifact cleanup: %s", e)
        return removed


_artifact_manager: Optional[ArtifactManager] = None


def get_artifact_manager(base_dir: Optional[Path | str] = None, ttl_seconds: int = 86400) -> ArtifactManager:
    global _artifact_manager
    if _artifact_manager is None:
        from ..engine.config import get_config
        cfg = get_config()
        path = base_dir or cfg.artifact_dir
        _artifact_manager = ArtifactManager(path, ttl_seconds=cfg.artifact_ttl_seconds)
    return _artifact_manager
