import os
import wave
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger("chatterbox.artifacts.verifier")


class AudioVerificationError(Exception):
    """Raised when an audio artifact fails validation."""
    pass


class AudioVerifier:
    """
    Forensic verification for generated audio artifacts.
    Ensures that a task is never marked COMPLETED unless the audio artifact
    actually exists, is readable, has non-zero length, and contains valid audio signal.
    """

    MIN_FILE_SIZE_BYTES = 512  # Minimum valid WAV file size with data chunk
    MIN_DURATION_SECONDS = 0.05  # At least 50 milliseconds

    @classmethod
    def verify_file(cls, file_path: str | Path, expected_sr: Optional[int] = None) -> Tuple[float, int, int]:
        """
        Verifies that an audio file on disk is valid.
        Returns: (duration_seconds, sample_rate, file_size_bytes)
        Raises: AudioVerificationError if invalid
        """
        path = Path(file_path)
        if not path.is_file():
            raise AudioVerificationError(f"Audio artifact does not exist at path: {path}")

        file_size = path.stat().st_size
        if file_size < cls.MIN_FILE_SIZE_BYTES:
            raise AudioVerificationError(
                f"Audio artifact file size too small ({file_size} bytes, min: {cls.MIN_FILE_SIZE_BYTES})"
            )

        try:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sr = wf.getframerate()
                num_frames = wf.getnframes()

                if channels < 1:
                    raise AudioVerificationError("Invalid WAV file: zero channels")
                if sample_width < 1:
                    raise AudioVerificationError("Invalid WAV file: invalid sample width")
                if sr <= 0:
                    raise AudioVerificationError("Invalid WAV file: invalid sample rate")
                if num_frames <= 0:
                    raise AudioVerificationError("Invalid WAV file: empty frame count")

                duration = num_frames / float(sr)
                if duration < cls.MIN_DURATION_SECONDS:
                    raise AudioVerificationError(
                        f"Generated audio duration ({duration:.3f}s) is below minimum threshold ({cls.MIN_DURATION_SECONDS}s)"
                    )

                if expected_sr and sr != expected_sr:
                    logger.warning("Sample rate mismatch: expected %d, got %d", expected_sr, sr)

                return duration, sr, file_size

        except wave.Error as e:
            raise AudioVerificationError(f"Corrupt or unreadable WAV structure: {e}") from e
        except Exception as e:
            raise AudioVerificationError(f"Failed to verify audio artifact: {e}") from e

    @classmethod
    def verify_waveform(cls, audio_np: np.ndarray, sr: int) -> float:
        """
        Verifies raw waveform numpy array before serialization.
        Returns: duration_seconds
        Raises: AudioVerificationError if invalid
        """
        if not isinstance(audio_np, np.ndarray):
            raise AudioVerificationError("Audio waveform must be a numpy ndarray")

        if audio_np.size == 0:
            raise AudioVerificationError("Audio waveform is empty (size 0)")

        if np.isnan(audio_np).any():
            raise AudioVerificationError("Audio waveform contains NaN values")

        if np.isinf(audio_np).any():
            raise AudioVerificationError("Audio waveform contains infinite values")

        # Check for absolute zero signal (degenerate output)
        max_amplitude = np.max(np.abs(audio_np))
        if max_amplitude == 0.0:
            raise AudioVerificationError("Audio waveform is completely silent (all zeros)")

        duration = audio_np.size / float(sr)
        if duration < cls.MIN_DURATION_SECONDS:
            raise AudioVerificationError(
                f"Waveform duration ({duration:.3f}s) is below minimum threshold"
            )

        return duration
