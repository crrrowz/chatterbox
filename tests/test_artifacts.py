import pytest
import numpy as np
from pathlib import Path
from chatterbox.artifacts.verifier import AudioVerifier, AudioVerificationError
from chatterbox.artifacts.storage import ArtifactManager


def test_verify_waveform_valid():
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)  # 1-second 440Hz sine wave
    duration = AudioVerifier.verify_waveform(wav, sr)
    assert abs(duration - 1.0) < 0.01


def test_verify_waveform_all_zeros():
    sr = 24000
    wav = np.zeros(24000, dtype=np.float32)
    with pytest.raises(AudioVerificationError, match="completely silent"):
        AudioVerifier.verify_waveform(wav, sr)


def test_verify_waveform_contains_nan():
    sr = 24000
    wav = np.ones(24000, dtype=np.float32)
    wav[10] = np.nan
    with pytest.raises(AudioVerificationError, match="contains NaN"):
        AudioVerifier.verify_waveform(wav, sr)


def test_artifact_storage_save_and_verify(tmp_path):
    manager = ArtifactManager(base_dir=tmp_path, ttl_seconds=3600)
    sr = 24000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    artifact = manager.save_and_verify("test_job_123", wav, sr, base_url="http://localhost:8000")

    assert artifact.job_id == "test_job_123"
    assert artifact.verified is True
    assert artifact.sample_rate == sr
    assert abs(artifact.duration_seconds - 0.5) < 0.05
    assert artifact.file_size_bytes > 512
    assert Path(artifact.file_path).is_file()

    # Verify file directly
    dur, sample_rate, size = AudioVerifier.verify_file(artifact.file_path, expected_sr=sr)
    assert dur > 0.4
    assert sample_rate == sr
