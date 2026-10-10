import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EngineConfig:
    """Configuration for Chatterbox Speech Engine."""
    host: str = field(default_factory=lambda: os.getenv("CHATTERBOX_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("CHATTERBOX_PORT", "8000")))
    device: str = field(default_factory=lambda: os.getenv("CHATTERBOX_DEVICE", "cuda"))
    model_type: str = field(default_factory=lambda: os.getenv("CHATTERBOX_MODEL_TYPE", "multilingual"))
    t3_model: str = field(default_factory=lambda: os.getenv("CHATTERBOX_T3_MODEL", "v2"))
    model_dir: str | None = field(default_factory=lambda: os.getenv("CHATTERBOX_MODEL_DIR", None))
    artifact_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CHATTERBOX_ARTIFACT_DIR", "artifacts/audio")).resolve()
    )
    artifact_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CHATTERBOX_ARTIFACT_TTL_SECONDS", "86400"))
    )
    max_text_length: int = field(
        default_factory=lambda: int(os.getenv("CHATTERBOX_MAX_TEXT_LEN", "5000"))
    )
    max_queue_size: int = field(
        default_factory=lambda: int(os.getenv("CHATTERBOX_MAX_QUEUE_SIZE", "100"))
    )
    api_key: str | None = field(
        default_factory=lambda: os.getenv("CHATTERBOX_API_KEY", None)
    )
    enable_auth: bool = field(
        default_factory=lambda: os.getenv("CHATTERBOX_ENABLE_AUTH", "false").lower() in ("true", "1", "yes")
    )


# Singleton instance
_config: EngineConfig | None = None


def get_config() -> EngineConfig:
    global _config
    if _config is None:
        _config = EngineConfig()
    return _config
