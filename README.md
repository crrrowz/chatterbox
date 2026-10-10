# Chatterbox Autonomous Speech Engine

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Architecture](https://img.shields.io/badge/architecture-REST%20API%20%7C%20Autonomous%20Tasks-purple)](SPEECH_ENGINE_GUIDE.md)
[![Device](https://img.shields.io/badge/hardware-CUDA%20%7C%20CPU%20%7C%20MPS-green)](https://pytorch.org/)

An enterprise-grade, API-driven autonomous speech synthesis engine wrapping Chatterbox Multilingual neural models. Designed for autonomous AI agents, robotics orchestrators (such as Orqevyn), and high-throughput microservices.

---

## Key Features

- **Autonomous Task Execution Lifecycle**: Full state transitions with observable progression:
  `RECEIVED -> VALIDATING -> GENERATING -> VERIFYING -> COMPLETED` (or `FAILED` / `CANCELLED`).
- **Forensic Audio Artifact Verification**: Guarantees zero corruptions. Tasks are marked `COMPLETED` only after verifying WAV header integrity, PCM bit depth, 24,000 Hz sample rate matching, non-empty frame count, and non-silent signal.
- **Auto-Chunking & Long-Form Synthesis**: Intelligently chunks texts up to **5,000+ characters** across natural sentence boundaries (`.` `!` `?` `,` `;` `\n`) and seamlessly stitches the resulting audio with natural pauses.
- **Voice Lab & Zero-Shot Cloning**: Upload or provide short reference samples (WAV, MP3, M4A, FLAC, OGG) to register, name, and persist custom speaker profiles.
- **Curated 23-Language Voice Catalog**: Built-in reference speakers across Arabic, English, French, Spanish, German, Japanese, Chinese, Turkish, Russian, and 14 other languages.
- **Chatterbox Studio Web Interface**: Modern, responsive, dark-mode Bento Grid studio with real-time waveform spectrum analysis and job history.
- **Universal Integrations**: Native client adapters for Python (`OrqevynSpeechClient`), .NET 8 C# (`ISpeechService`), JavaScript/Node.js, and cURL.

---

## System Architecture

```
                   +-----------------------------------------------+
                   |          External AI Agent / Orchestrator      |
                   +-----------------------------------------------+
                                          |
                                   (REST API Call)
                                          v
+---------------------------------------------------------------------------------+
|                        Chatterbox Speech Service (FastAPI)                       |
|                                                                                 |
|  +--------------------+     +----------------------+     +--------------------+ |
|  |     Endpoints      |     |     Task Manager     |     |  Artifact Manager  | |
|  |                    |     |                      |     |                    | |
|  | GET  /health       | --> | Sequential Queue     | --> | Storage & TTL      | |
|  | GET  /v1/voices    |     | Worker Thread        |     | Forensic Verifier  | |
|  | POST /v1/voices    |     | Transition Tracker   |     | Format Converter   | |
|  | POST /v1/speech    |     +----------------------+     +--------------------+ |
|  | GET  /v1/.../jobs  |                |                                        |
|  | GET  /v1/.../audio |                v                                        |
|  +--------------------+     +----------------------+                            |
|                             |    Speech Engine     |                            |
|                             |                      |                            |
|                             | Thread-Safe Lock     |                            |
|                             | Multilingual Model   |                            |
|                             | Natural Sentence     |                            |
|                             | Chunking & Stitching |                            |
|                             +----------------------+                            |
|                                        |                                        |
|                                        v                                        |
|                             +----------------------+                            |
|                             | PyTorch / CUDA Core  |                            |
|                             +----------------------+                            |
+---------------------------------------------------------------------------------+
```

---

## Installation & Prerequisites

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- NVIDIA GPU with CUDA 12+ recommended (falls back to CPU automatically)
- FFmpeg (optional, recommended for automatic transcoding of non-WAV audio formats)

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/crrrowz/chatterbox.git
cd chatterbox
pip install -e .
```

Verify dependencies:
```bash
python -c "import torch, chatterbox, soundfile; print('Ready! CUDA:', torch.cuda.is_available())"
```

---

## Running the Server

Start the production engine with Uvicorn:
```bash
python -m chatterbox.server
```
Or directly via the CLI:
```bash
uvicorn chatterbox.api.routes:app --host 127.0.0.1 --port 8000
```

Once running, access the web studio at:
**`http://127.0.0.1:8000/`** or **`http://127.0.0.1:8000/ui`**

---

## API Quick Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check service readiness, active compute device (`cuda`/`cpu`), and queue metrics. |
| `GET` | `/v1/voices` | List available voice profiles (filterable via `?language=en`). |
| `POST` | `/v1/voices` | Register, name, and persist a custom speaker reference. |
| `DELETE` | `/v1/voices/{voice_id}` | Remove a custom registered voice. |
| `POST` | `/v1/speech` | Submit text for synthesis (synchronous wait or async background job). |
| `GET` | `/v1/speech/jobs/{job_id}` | Poll task status, timeline transitions, and error diagnostics. |
| `GET` | `/v1/speech/jobs/{job_id}/audio`| Retrieve the verified 24,000 Hz 16-bit PCM WAV audio file. |
| `DELETE` | `/v1/speech/jobs/{job_id}`| Cancel pending or running task. |

---

## Code Integration Examples

### Python (`OrqevynSpeechClient`)
```python
from chatterbox.client import OrqevynSpeechClient

client = OrqevynSpeechClient(base_url="http://127.0.0.1:8000")

result = client.speak_text(
    text="Welcome to the autonomous neural speech engine.",
    language="en",
    voice="en_female",
    options={"exaggeration": 0.5, "temperature": 0.8, "wait": True}
)

print(f"Status: {result.status}")
print(f"Downloaded Audio: {result.audio_path}")
print(f"Duration: {result.duration_seconds}s | Sample Rate: {result.sample_rate}Hz")
```

### cURL / Shell
```bash
curl -X POST "http://127.0.0.1:8000/v1/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world from the Chatterbox speech engine.",
    "language": "en",
    "voice": "en_female",
    "options": {
      "exaggeration": 0.5,
      "wait": true
    }
  }'
```

### C# (.NET 8 Native Adapter)
```csharp
using Orqevyn.Core.Speech;

var httpClient = new HttpClient { BaseAddress = new Uri("http://127.0.0.1:8000") };
ISpeechService speech = new OrqevynSpeechClient(httpClient);

var result = await speech.SpeakTextAsync(
    text: "Welcome to Orqevyn speech engine.",
    language: "en",
    voice: "en_female"
);
Console.WriteLine($"Audio verified at: {result.AudioPath}");
```

---

## Testing

Run the automated test suite:
```bash
pytest tests/ -v
```
All 20 test suites cover:
- API endpoint contracts and input schemas
- Audio waveform and file-level forensic verification
- Sentence chunking and delimiter preservation
- Task state machine and error handling
- Client adapters and timeout behaviors

---

## License

This project is licensed under the [MIT License](LICENSE).
