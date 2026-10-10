import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from chatterbox.api.routes import app
from chatterbox.engine.service import get_speech_engine
from chatterbox.tasks.manager import get_task_manager


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = get_speech_engine()
    engine._is_ready = True
    sr = 24000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Mock synthesize to avoid running heavy neural networks in unit tests
    monkeypatch.setattr(engine, "synthesize", lambda *args, **kwargs: (sr, wav))
    monkeypatch.setattr(type(engine), "is_ready", property(lambda self: True))

    with TestClient(app) as test_client:
        yield test_client


def test_ui_endpoints(client):
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Chatterbox Speech Studio" in res_root.text

    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "Chatterbox Speech Studio" in res_ui.text


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "ready" in data
    assert "supported_languages" in data
    assert "ar" in data["supported_languages"]


def test_voices_endpoint(client):
    res = client.get("/v1/voices")
    assert res.status_code == 200
    voices = res.json()
    assert isinstance(voices, list)
    assert any(v["voice_id"] == "ar_female" for v in voices)
    assert any(v["voice_id"] == "zh_female" for v in voices)
    assert any(v["voice_id"] == "ja_female" for v in voices)

    # Filter by language
    res_ar = client.get("/v1/voices?language=ar")
    voices_ar = res_ar.json()
    assert all(v["language"] in ("ar", "multilingual") for v in voices_ar)


def test_create_and_delete_custom_voice(client):
    # 1. Create custom voice
    payload = {
        "name": "My Custom Arabic Voice",
        "voice_id": "my_test_voice",
        "language": "ar",
        "gender": "male",
        "description": "Test voice for unit tests",
        "reference_path": "https://example.com/test.flac",
    }
    create_res = client.post("/v1/voices", json=payload)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["voice_id"] == "my_test_voice"
    assert created_data["name"] == "My Custom Arabic Voice"
    assert created_data["is_builtin"] is False

    # 2. Check it appears in voices list
    list_res = client.get("/v1/voices?language=ar")
    voices = list_res.json()
    assert any(v["voice_id"] == "my_test_voice" for v in voices)

    # 3. Delete custom voice
    del_res = client.delete("/v1/voices/my_test_voice")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # 4. Verify gone
    list_res_after = client.get("/v1/voices?language=ar")
    assert not any(v["voice_id"] == "my_test_voice" for v in list_res_after.json())


def test_submit_speech_sync(client):
    payload = {
        "text": "هذا اختبار تجريبي للصوت",
        "language": "ar",
        "voice": "ar_female",
        "options": {
            "wait": True,
            "timeout": 10.0,
        },
    }
    res = client.post("/v1/speech", json=payload)
    assert res.status_code == 202
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["job_id"] is not None
    assert data["artifact"] is not None
    assert data["artifact"]["verified"] is True

    # Retrieve the audio file
    job_id = data["job_id"]
    audio_res = client.get(f"/v1/speech/jobs/{job_id}/audio")
    assert audio_res.status_code == 200
    assert audio_res.headers["content-type"] == "audio/wav"
    assert len(audio_res.content) > 512
