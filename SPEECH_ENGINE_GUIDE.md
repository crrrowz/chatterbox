# Chatterbox Autonomous Speech Engine — Architecture & Universal Integration Guide

A high-performance, enterprise-grade, REST API-driven speech synthesis engine wrapping Chatterbox Multilingual TTS, designed for autonomous AI agents, Orqevyn, and any external service.

---

## 1. Quick Start & Execution

### Startup
Run the server from the Chatterbox repository root:
```powershell
python -m chatterbox.server
```
Or directly using Uvicorn:
```powershell
uvicorn chatterbox.api.routes:app --host 127.0.0.1 --port 8000
```

### Accessing the Web Dashboard (Chatterbox Studio)
Open your browser to:
- **`http://127.0.0.1:8000/`** or **`http://127.0.0.1:8000/ui`**

Features of the web dashboard:
- **Speech Studio**: Multilingual synthesis with fine-grained expressive controls (exaggeration, temperature, CFG pace weight, repetition penalty, seeds).
- **Voice Lab**: Create, name, and clone custom voices by uploading short audio samples (`WAV`/`FLAC`) or URLs.
- **Voice Library**: Complete reference catalog of built-in speakers across all 23 supported languages.
- **Real-time Lifecycle Stepper**: Forensic status monitoring (`RECEIVED -> VALIDATING -> GENERATING -> VERIFYING -> COMPLETED`).
- **Interactive Audio Visualizer & Player**: Immediate waveform rendering with metadata verification metrics.

---

## 2. API Contract & Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health, hardware acceleration (CUDA/CPU), model readiness, and task queue depth |
| `GET` | `/v1/voices` | List available voices (supports `?language=ar` query parameter) |
| `POST` | `/v1/voices` | Create, name, and persist a new custom voice profile |
| `DELETE`| `/v1/voices/{voice_id}` | Delete a custom voice profile |
| `POST` | `/v1/speech` | Submit a speech generation request (sync or async) |
| `GET` | `/v1/speech/jobs/{job_id}` | Check task status, lifecycle transitions, and error diagnostics |
| `GET` | `/v1/speech/jobs/{job_id}/audio` | Retrieve verified WAV audio artifact |
| `DELETE`| `/v1/speech/jobs/{job_id}` | Cancel pending or in-flight task |

---

## 3. Integration with Any External Project

### A. Python Client (`chatterbox.client.OrqevynSpeechClient`)
```python
from chatterbox.client import OrqevynSpeechClient

client = OrqevynSpeechClient(base_url="http://127.0.0.1:8000")

# 1. Synthesize speech (Arabic female default)
result = client.speak_text(
    text="مرحبا بكم في نظام الذكاء الاصطناعي التكيفي",
    language="ar",
    voice="ar_female",
    options={"exaggeration": 0.5, "wait": True}
)

print(f"Status: {result.status}")
print(f"Verified Audio File: {result.audio_path}")
print(f"Duration: {result.duration_seconds}s | Sample Rate: {result.sample_rate}Hz")
```

### B. JavaScript / TypeScript / Node.js
```javascript
async function synthesizeSpeech(text, voice = "ar_female", language = "ar") {
  const response = await fetch("http://127.0.0.1:8000/v1/speech", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: text,
      language: language,
      voice: voice,
      options: { wait: true, timeout: 60.0 }
    })
  });

  if (!response.ok) throw new Error(await response.text());
  const job = await response.json();
  
  const audioUrl = `http://127.0.0.1:8000${job.artifact.audio_url}`;
  console.log(`Audio synthesized successfully: ${audioUrl}`);
  return audioUrl;
}
```

### C. cURL / Shell
```bash
curl -X POST "http://127.0.0.1:8000/v1/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "هذا اختبار تجريبي للتوليد الصوتي",
    "language": "ar",
    "voice": "ar_female",
    "options": {
      "exaggeration": 0.5,
      "wait": true
    }
  }'
```

### D. C# (.NET 8 - Orqevyn Native Integration)
```csharp
using Orqevyn.Core.Speech;

var httpClient = new HttpClient { BaseAddress = new Uri("http://127.0.0.1:8000") };
ISpeechService speechClient = new OrqevynSpeechClient(httpClient);

var result = await speechClient.SpeakTextAsync(
    text: "مرحبا بكم في أورقيفين",
    voice: "ar_female",
    language: "ar",
    options: new SpeechOptions(Wait: true)
);

Console.WriteLine($"Audio saved: {result.AudioPath}");
```

---

## 4. Voice Lab: Creating and Naming Custom Voices

To clone/register a voice via API:
```bash
curl -X POST "http://127.0.0.1:8000/v1/voices" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "صوت أحمد الإذاعي",
    "voice_id": "ahmed_radio",
    "language": "ar",
    "gender": "male",
    "description": "صوت إذاعي فصيح ورصين",
    "reference_path": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac"
  }'
```
You can also send audio data as a base64-encoded string inside `reference_base64`.

---

## 5. Repository Structure
```
chatterbox/
├── src/
│   └── chatterbox/
│       ├── api/               # FastAPI routing, contracts, schemas
│       ├── artifacts/         # Forensic WAV verification & disk storage
│       ├── engine/            # Model inference wrapper, hardware detector, voice registry
│       ├── tasks/             # Autonomous queue worker & lifecycle manager
│       ├── web/               # Chatterbox Studio Web Interface (Bento Grid, UI/UX Pro Max)
│       ├── client.py          # Universal Python integration client
│       └── server.py          # Production server entry point
├── legacy_examples/           # Archived original Gradio and demo scripts
├── tests/                     # Comprehensive Pytest suite
├── pyproject.toml             # Dependencies and package metadata
└── SPEECH_ENGINE_GUIDE.md     # Architecture documentation
```
