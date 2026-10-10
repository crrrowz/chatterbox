import os
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from pathlib import Path
import httpx
import logging

from ..mtl_tts import SUPPORTED_LANGUAGES

logger = logging.getLogger("chatterbox.engine.voices")


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    language: str
    gender: str = "neutral"
    description: str = ""
    reference_path: Optional[str] = None
    is_builtin: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


# Built-in reference voice profiles for all 23 supported languages
BUILTIN_VOICES: Dict[str, VoiceProfile] = {
    "default": VoiceProfile(
        voice_id="default",
        name="Default System Voice",
        language="multilingual",
        gender="neutral",
        description="Standard Chatterbox pretrained conditional voice profile",
        reference_path=None,
        is_builtin=True,
    ),
    "ar_female": VoiceProfile(
        voice_id="ar_female",
        name="Arabic Female (Amina)",
        language="ar",
        gender="female",
        description="Fluent, articulate standard Arabic female speaker",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        is_builtin=True,
    ),
    "ar_default": VoiceProfile(
        voice_id="ar_default",
        name="Arabic Standard",
        language="ar",
        gender="female",
        description="Standard reference speaker for Arabic",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        is_builtin=True,
    ),
    "en_female": VoiceProfile(
        voice_id="en_female",
        name="English Female (Sarah)",
        language="en",
        gender="female",
        description="Natural fluent English female speaker",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        is_builtin=True,
    ),
    "en_default": VoiceProfile(
        voice_id="en_default",
        name="English Standard",
        language="en",
        gender="female",
        description="Standard English reference voice",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        is_builtin=True,
    ),
    "fr_female": VoiceProfile(
        voice_id="fr_female",
        name="French Female (Camille)",
        language="fr",
        gender="female",
        description="Voix féminine française naturelle",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        is_builtin=True,
    ),
    "es_female": VoiceProfile(
        voice_id="es_female",
        name="Spanish Female (Sofia)",
        language="es",
        gender="female",
        description="Voz femenina en español neutro y natural",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        is_builtin=True,
    ),
    "de_female": VoiceProfile(
        voice_id="de_female",
        name="German Female (Greta)",
        language="de",
        gender="female",
        description="Natürliche deutsche Frauenstimme",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        is_builtin=True,
    ),
    "da_male": VoiceProfile(
        voice_id="da_male",
        name="Danish Male (Lars)",
        language="da",
        gender="male",
        description="Dansk mandlig stemme",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        is_builtin=True,
    ),
    "el_male": VoiceProfile(
        voice_id="el_male",
        name="Greek Male (Nikos)",
        language="el",
        gender="male",
        description="Ελληνική ανδρική φωνή",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        is_builtin=True,
    ),
    "fi_male": VoiceProfile(
        voice_id="fi_male",
        name="Finnish Male (Matti)",
        language="fi",
        gender="male",
        description="Suomalainen miesääni",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        is_builtin=True,
    ),
    "he_male": VoiceProfile(
        voice_id="he_male",
        name="Hebrew Male (Avi)",
        language="he",
        gender="male",
        description="קול גברי בעברית",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        is_builtin=True,
    ),
    "hi_female": VoiceProfile(
        voice_id="hi_female",
        name="Hindi Female (Priya)",
        language="hi",
        gender="female",
        description="हिंदी महिला आवाज़",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        is_builtin=True,
    ),
    "it_male": VoiceProfile(
        voice_id="it_male",
        name="Italian Male (Marco)",
        language="it",
        gender="male",
        description="Voce maschile italiana naturale",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        is_builtin=True,
    ),
    "ja_female": VoiceProfile(
        voice_id="ja_female",
        name="Japanese Female (Yuki)",
        language="ja",
        gender="female",
        description="日本語の自然な女性の声",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        is_builtin=True,
    ),
    "ko_female": VoiceProfile(
        voice_id="ko_female",
        name="Korean Female (Jiwoo)",
        language="ko",
        gender="female",
        description="자연스러운 한국어 여성 음성",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        is_builtin=True,
    ),
    "ms_female": VoiceProfile(
        voice_id="ms_female",
        name="Malay Female (Siti)",
        language="ms",
        gender="female",
        description="Suara wanita Melayu asli",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        is_builtin=True,
    ),
    "nl_male": VoiceProfile(
        voice_id="nl_male",
        name="Dutch Male (Daan)",
        language="nl",
        gender="male",
        description="Natuurlijke Nederlandse mannenstem",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        is_builtin=True,
    ),
    "no_female": VoiceProfile(
        voice_id="no_female",
        name="Norwegian Female (Ingrid)",
        language="no",
        gender="female",
        description="Norsk kvinnelig stemme",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        is_builtin=True,
    ),
    "pl_male": VoiceProfile(
        voice_id="pl_male",
        name="Polish Male (Piotr)",
        language="pl",
        gender="male",
        description="Naturalny polski głos męski",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        is_builtin=True,
    ),
    "pt_male": VoiceProfile(
        voice_id="pt_male",
        name="Portuguese Male (Lucas)",
        language="pt",
        gender="male",
        description="Voz masculina portuguesa",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        is_builtin=True,
    ),
    "ru_male": VoiceProfile(
        voice_id="ru_male",
        name="Russian Male (Dmitry)",
        language="ru",
        gender="male",
        description="Русский естественный мужской голос",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        is_builtin=True,
    ),
    "sv_female": VoiceProfile(
        voice_id="sv_female",
        name="Swedish Female (Astrid)",
        language="sv",
        gender="female",
        description="Svensk kvinnlig röst",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        is_builtin=True,
    ),
    "sw_male": VoiceProfile(
        voice_id="sw_male",
        name="Swahili Male (Juma)",
        language="sw",
        gender="male",
        description="Sauti ya kiume ya Kiswahili",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        is_builtin=True,
    ),
    "tr_male": VoiceProfile(
        voice_id="tr_male",
        name="Turkish Male (Emre)",
        language="tr",
        gender="male",
        description="Doğal Türkçe erkek sesi",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        is_builtin=True,
    ),
    "zh_female": VoiceProfile(
        voice_id="zh_female",
        name="Chinese Female (Mei)",
        language="zh",
        gender="female",
        description="中文普通话自然女性声音",
        reference_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        is_builtin=True,
    ),
}

# Mapping language codes to default voice presets
LANGUAGE_DEFAULT_VOICE: Dict[str, str] = {
    "ar": "ar_female",
    "da": "da_male",
    "de": "de_female",
    "el": "el_male",
    "en": "en_female",
    "es": "es_female",
    "fi": "fi_male",
    "fr": "fr_female",
    "he": "he_male",
    "hi": "hi_female",
    "it": "it_male",
    "ja": "ja_female",
    "ko": "ko_female",
    "ms": "ms_female",
    "nl": "nl_male",
    "no": "no_female",
    "pl": "pl_male",
    "pt": "pt_male",
    "ru": "ru_male",
    "sv": "sv_female",
    "sw": "sw_male",
    "tr": "tr_male",
    "zh": "zh_female",
}


def download_cached_prompt(url: str) -> str:
    """Downloads remote audio reference prompt and caches it locally on disk."""
    cache_dir = Path.home() / ".cache" / "chatterbox" / "voices"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = url.split(".")[-1].split("?")[0]
    local_path = cache_dir / f"{url_hash}.{ext}"

    if local_path.is_file() and local_path.stat().st_size > 0:
        return str(local_path)

    logger.info("Downloading reference audio from %s to %s...", url, local_path)
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        res = client.get(url)
        res.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(res.content)
    return str(local_path)


class VoiceRegistry:
    def __init__(self, storage_dir: Optional[Path | str] = None):
        self._voices: Dict[str, VoiceProfile] = dict(BUILTIN_VOICES)
        self.storage_dir = Path(storage_dir or (Path.home() / ".cache" / "chatterbox" / "custom_voices")).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.custom_voices_file = self.storage_dir / "voices.json"
        self._load_custom_voices()

    def _load_custom_voices(self) -> None:
        """Loads user-created custom voices from persisted JSON."""
        if self.custom_voices_file.is_file():
            try:
                with open(self.custom_voices_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for v_dict in data:
                        profile = VoiceProfile(**v_dict)
                        self._voices[profile.voice_id] = profile
                logger.info("Loaded %d custom voices from storage", len(data))
            except Exception as e:
                logger.error("Failed to load custom voices: %s", e)

    def _persist_custom_voices(self) -> None:
        """Saves custom voices to persisted JSON."""
        try:
            custom_list = [v.to_dict() for v in self._voices.values() if not v.is_builtin]
            with open(self.custom_voices_file, "w", encoding="utf-8") as f:
                json.dump(custom_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to persist custom voices: %s", e)

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        return self._voices.get(voice_id)

    def list_voices(self, language: Optional[str] = None) -> List[VoiceProfile]:
        voices = list(self._voices.values())
        if language:
            lang_lower = language.lower()
            voices = [v for v in voices if v.language.lower() == lang_lower or v.language == "multilingual"]
        return voices

    def register_voice(self, profile: VoiceProfile) -> None:
        """Registers a new voice profile and persists if custom."""
        self._voices[profile.voice_id] = profile
        if not profile.is_builtin:
            self._persist_custom_voices()
        logger.info("Registered voice: %s (%s)", profile.voice_id, profile.name)

    def delete_voice(self, voice_id: str) -> bool:
        """Deletes a custom voice profile."""
        if voice_id in self._voices:
            if self._voices[voice_id].is_builtin:
                raise ValueError("Cannot delete built-in system voice.")
            del self._voices[voice_id]
            self._persist_custom_voices()
            logger.info("Deleted custom voice: %s", voice_id)
            return True
        return False

    def resolve_reference_path(self, voice_id: Optional[str], language: Optional[str]) -> Optional[str]:
        raw_path: Optional[str] = None
        if voice_id:
            voice = self.get_voice(voice_id)
            if voice:
                raw_path = voice.reference_path
            elif Path(voice_id).is_file() or voice_id.startswith(("http://", "https://")):
                raw_path = voice_id

        if not raw_path and language:
            lang_key = language.lower()
            default_voice_id = LANGUAGE_DEFAULT_VOICE.get(lang_key)
            if default_voice_id and default_voice_id in self._voices:
                raw_path = self._voices[default_voice_id].reference_path

        if raw_path:
            if raw_path.startswith(("http://", "https://")):
                try:
                    return download_cached_prompt(raw_path)
                except Exception as e:
                    logger.warning("Failed to download voice prompt from %s: %s. Using default voice.", raw_path, e)
                    return None
            return raw_path

        return None


_registry: Optional[VoiceRegistry] = None


def get_voice_registry() -> VoiceRegistry:
    global _registry
    if _registry is None:
        _registry = VoiceRegistry()
    return _registry
