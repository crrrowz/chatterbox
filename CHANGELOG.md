# Changelog

All notable changes to the Chatterbox Autonomous Speech Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-10

### Added
- **Production REST API**: Added FastAPI endpoints for service health (`/health`), voice catalogs (`/v1/voices`), task submission (`/v1/speech`), job state inspection (`/v1/speech/jobs/{job_id}`), and WAV audio retrieval.
- **Autonomous Task Execution**: Implemented explicit task lifecycle state machine (`RECEIVED -> VALIDATING -> GENERATING -> VERIFYING -> COMPLETED`) with bounded sequential queue processing.
- **Forensic Audio Verifier**: Automated validation of generated audio artifacts checking WAV header structure, sample rate matching (24,000 Hz), non-zero frame count, and non-silent signal.
- **Natural Sentence Auto-Chunking & Seamless Stitching**: Engine automatically segments long-form text (up to 5,000+ characters) at sentence punctuation marks (`.` `!` `?` `,` `;` `\n`) and synthesizes/stitches the resulting waveforms with natural pauses.
- **Universal Audio Converter**: Added automatic FFmpeg and Web Audio API transcoding for custom voice samples (supporting MP3, M4A, OGG, FLAC to 16-bit PCM WAV).
- **Voice Lab**: Enabled user-driven creation, naming, cloning, and persistence of custom voice profiles with local JSON registry storage.
- **Multilingual Catalog**: Curated voice configurations covering 23 official languages with pretrained system speakers.
- **Chatterbox Studio Web UI**: High-performance, responsive dark-mode Bento Grid dashboard with live waveform visualizer, generation history, and API docs.
- **Client Integration Adapters**: Implemented Python client (`OrqevynSpeechClient`) and .NET 8 C# integration client (`Orqevyn.Core.Speech`).

### Changed
- Refactored legacy demo scripts into `legacy_examples/` directory to maintain clean root package boundaries.
- Standardized all API responses and web interfaces in English with LTR presentation.

### Fixed
- Fixed audio decode failures when uploading M4A/AAC files from mobile and modern browser recorders by converting to standard PCM WAV upon upload.
- Fixed horizontal layout distortion on long text submissions by enforcing strict CSS Grid bounding and multi-line clamping.
- Fixed premature client timeouts on long audio generations with dynamic timeout scaling.
