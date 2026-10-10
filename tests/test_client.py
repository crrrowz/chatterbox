import pytest
import httpx
from chatterbox.client import OrqevynSpeechClient, SpeechResult, SpeechTimeoutError, SpeechVerificationError


def test_client_speak_text_success(tmp_path):
    # Mock responses
    def handler(request: httpx.Request):
        url = str(request.url)
        if "/v1/speech" in url and request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "job_id": "job_abc123",
                    "status": "COMPLETED",
                    "artifact": {
                        "job_id": "job_abc123",
                        "duration_seconds": 1.5,
                        "sample_rate": 24000,
                        "file_size_bytes": 72000,
                        "verified": True,
                    },
                    "duration_ms": 120.0,
                },
            )
        elif "/v1/speech/jobs/job_abc123/audio" in url:
            # Valid WAV header + content bytes (>512 bytes)
            content = b"RIFF" + b"\x00" * 1000
            return httpx.Response(200, content=content, headers={"Content-Type": "audio/wav"})
        elif "/health" in url:
            return httpx.Response(200, json={"status": "healthy", "ready": True})
        return httpx.Response(404)

    mock_transport = httpx.MockTransport(handler)
    client = OrqevynSpeechClient(
        base_url="http://127.0.0.1:8000",
        download_dir=tmp_path,
        transport=mock_transport,
    )

    result = client.speak_text(
        text="مرحبا بكم في أورقيفين",
        voice="ar_female",
        language="ar",
        options={"wait": True},
    )

    assert isinstance(result, SpeechResult)
    assert result.job_id == "job_abc123"
    assert result.status == "COMPLETED"
    assert result.verified is True
    assert result.duration_seconds == 1.5
    assert result.sample_rate == 24000
    assert result.audio_path is not None
