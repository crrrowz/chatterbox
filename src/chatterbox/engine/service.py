import os
import threading
import logging
import random
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import torch

from .config import EngineConfig, get_config
from .voices import VoiceRegistry, get_voice_registry
from ..mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

logger = logging.getLogger("chatterbox.engine")


def split_text_chunks(text: str, max_chunk_len: int = 220) -> list[str]:
    """
    Intelligently splits long text into natural sentence-level chunks,
    preserving sentence delimiters (. ! ? ، ؛ : newline) to allow infinite-length
    TTS synthesis without model degeneration or token cutoffs.
    """
    text = text.strip()
    if len(text) <= max_chunk_len:
        return [text]

    import re
    delimiters = r'([.!?،؛:\n]+)'
    parts = re.split(delimiters, text)

    chunks = []
    current = ""

    for i in range(0, len(parts), 2):
        sentence = parts[i].strip()
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        full_segment = (sentence + delim).strip()
        if not full_segment:
            continue

        if len(current) + len(full_segment) + 1 <= max_chunk_len:
            current = (current + " " + full_segment).strip()
        else:
            if current:
                chunks.append(current)
            if len(full_segment) > max_chunk_len:
                words = full_segment.split()
                sub = ""
                for w in words:
                    if len(sub) + len(w) + 1 <= max_chunk_len:
                        sub = (sub + " " + w).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = w
                current = sub
            else:
                current = full_segment

    if current:
        chunks.append(current)
    return chunks if chunks else [text]


class EngineError(Exception):
    """Base exception for speech engine errors."""
    pass


class ModelNotReadyError(EngineError):
    """Raised when inference is attempted before the model is loaded."""
    pass


class ValidationError(EngineError):
    """Raised when input parameters fail domain validation."""
    pass


class SynthesisError(EngineError):
    """Raised when speech synthesis fails during inference."""
    pass


class SpeechEngine:
    """
    Core speech engine wrapper around Chatterbox inference.
    Thread-safe, stateful, loads model once and reuses across requests.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or get_config()
        self.voice_registry: VoiceRegistry = get_voice_registry()
        self._model: Optional[ChatterboxMultilingualTTS] = None
        self._device: str = self._resolve_device(self.config.device)
        self._lock = threading.Lock()
        self._is_ready: bool = False
        self._init_error: Optional[str] = None
        self._conds_cache: Dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_type(self) -> str:
        return self.config.model_type

    def _resolve_device(self, preferred_device: str) -> str:
        if preferred_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def initialize(self) -> None:
        """Loads the model once into memory."""
        with self._lock:
            if self._is_ready:
                return

            logger.info("Initializing SpeechEngine on device: %s", self._device)
            try:
                # 1. Check if explicit local directory is configured
                if self.config.model_dir and Path(self.config.model_dir).exists():
                    logger.info("Loading model from local directory: %s", self.config.model_dir)
                    self._model = ChatterboxMultilingualTTS.from_local(
                        ckpt_dir=self.config.model_dir,
                        device=self._device,
                        t3_model=self.config.t3_model,
                    )
                else:
                    # 2. Check local HuggingFace cache snapshots
                    snapshot_dir = self._find_local_hf_snapshot()
                    if snapshot_dir:
                        logger.info("Found local HuggingFace cache snapshot at: %s", snapshot_dir)
                        self._model = ChatterboxMultilingualTTS.from_local(
                            ckpt_dir=snapshot_dir,
                            device=self._device,
                            t3_model=self.config.t3_model,
                        )
                    else:
                        logger.info("Downloading or loading model via HuggingFace Hub...")
                        self._model = ChatterboxMultilingualTTS.from_pretrained(
                            device=torch.device(self._device),
                            t3_model=self.config.t3_model,
                        )

                self._is_ready = True
                self._init_error = None
                logger.info("SpeechEngine initialized successfully. Sample rate: %d", self._model.sr)
            except Exception as e:
                self._is_ready = False
                self._init_error = str(e)
                logger.error("Failed to initialize SpeechEngine: %s", e, exc_info=True)
                raise EngineError(f"Model initialization failed: {e}") from e

    def _find_local_hf_snapshot(self) -> Optional[str]:
        """Locates cached snapshot for ResembleAI/chatterbox if present."""
        hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub"))
        model_dir = Path(hf_home) / "models--ResembleAI--chatterbox" / "snapshots"
        if model_dir.exists():
            snapshots = [d for d in model_dir.iterdir() if d.is_dir()]
            if snapshots:
                return str(snapshots[0])
        return None

    def validate_request(
        self,
        text: str,
        language: str,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        seed: int = 0,
        repetition_penalty: float = 1.2,
    ) -> None:
        """Performs rigorous domain validation on input parameters."""
        if not text or not text.strip():
            raise ValidationError("Text input must not be empty.")

        if len(text) > self.config.max_text_length:
            raise ValidationError(
                f"Text length ({len(text)}) exceeds maximum allowed limit ({self.config.max_text_length})."
            )

        lang_code = language.lower() if language else "ar"
        if lang_code not in SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
            raise ValidationError(
                f"Unsupported language code '{language}'. Supported language codes are: {supported}"
            )

        if not (0.1 <= exaggeration <= 2.0):
            raise ValidationError(f"Exaggeration must be between 0.1 and 2.0 (got {exaggeration}).")

        if not (0.05 <= temperature <= 2.0):
            raise ValidationError(f"Temperature must be between 0.05 and 2.0 (got {temperature}).")

        if not (0.0 <= cfg_weight <= 2.0):
            raise ValidationError(f"CFG weight must be between 0.0 and 2.0 (got {cfg_weight}).")

        if not (1.0 <= repetition_penalty <= 2.5):
            raise ValidationError(
                f"Repetition penalty must be between 1.0 and 2.5 (got {repetition_penalty})."
            )

    def synthesize(
        self,
        text: str,
        language: str = "ar",
        voice: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        seed: int = 0,
        repetition_penalty: float = 1.2,
    ) -> Tuple[int, np.ndarray]:
        """
        Synthesizes speech from text. Thread-safe execution.
        Returns: (sample_rate, waveform_numpy)
        """
        if not self.is_ready:
            self.initialize()

        self.validate_request(
            text=text,
            language=language,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            seed=seed,
            repetition_penalty=repetition_penalty,
        )

        ref_audio_path = self.voice_registry.resolve_reference_path(voice, language)

        with self._lock:
            try:
                # Set deterministic or random seed
                if seed != 0:
                    torch.manual_seed(seed)
                    if self._device == "cuda":
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                    random.seed(seed)
                    np.random.seed(seed)

                generate_kwargs = {
                    "exaggeration": exaggeration,
                    "temperature": temperature,
                    "cfg_weight": cfg_weight,
                    "repetition_penalty": repetition_penalty,
                }
                if ref_audio_path:
                    generate_kwargs["audio_prompt_path"] = ref_audio_path

                # Split text into natural sentence chunks
                chunks = split_text_chunks(text[:self.config.max_text_length], max_chunk_len=220)
                logger.info("Running TTS synthesis on %d chunk(s) (total chars: %d): lang=%s, voice=%s", len(chunks), len(text), language, voice)

                audio_segments = []
                sr = self._model.sr
                # 150ms natural pause between sentences
                pause_samples = int(sr * 0.15)
                silence_pause = np.zeros(pause_samples, dtype=np.float32)

                for idx, chunk in enumerate(chunks):
                    logger.debug("Synthesizing chunk %d/%d: '%s'", idx + 1, len(chunks), chunk[:40])
                    wav_tensor = self._model.generate(
                        text=chunk,
                        language_id=language.lower(),
                        **generate_kwargs
                    )
                    chunk_audio = wav_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                    audio_segments.append(chunk_audio)
                    if idx < len(chunks) - 1:
                        audio_segments.append(silence_pause)

                # Concatenate all chunks seamlessly
                combined_audio = np.concatenate(audio_segments, axis=0) if len(audio_segments) > 1 else audio_segments[0]
                return sr, combined_audio

            except Exception as e:
                logger.error("Synthesis failed: %s", e, exc_info=True)
                raise SynthesisError(f"Speech synthesis execution failed: {e}") from e

    def get_status(self) -> Dict[str, Any]:
        """Returns engine readiness and operational metrics."""
        return {
            "ready": self.is_ready,
            "device": self._device,
            "model_type": self.config.model_type,
            "t3_model": self.config.t3_model,
            "sample_rate": self._model.sr if self._model else 24000,
            "supported_languages": sorted(list(SUPPORTED_LANGUAGES.keys())),
            "init_error": self._init_error,
        }


# Singleton engine instance
_engine_instance: Optional[SpeechEngine] = None


def get_speech_engine() -> SpeechEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SpeechEngine()
    return _engine_instance
