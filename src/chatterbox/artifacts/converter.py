import os
import shutil
import subprocess
import logging
import soundfile as sf
from pathlib import Path
from typing import Optional

logger = logging.getLogger("chatterbox.audio_converter")

# Known ffmpeg candidates on Windows
FFMPEG_CANDIDATES = [
    r"C:\Users\hassa\.antigravity-ide\extensions\kilocode.kilo-code-7.8.1-win32-x64\bin\ffmpeg.exe",
    r"C:\Program Files\SteelSeries\GG\apps\moments\ffmpeg.exe",
]


def find_ffmpeg() -> Optional[str]:
    """Locates an available ffmpeg executable."""
    cmd = shutil.which("ffmpeg")
    if cmd:
        return cmd

    for candidate in FFMPEG_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate

    return None


def convert_to_pcm_wav(input_bytes: bytes, output_path: Path, sample_rate: int = 24000) -> bool:
    """
    Takes arbitrary audio bytes (mp4, m4a, ogg, mp3, flac, wav) and converts them
    to standard 16-bit PCM mono WAV at the desired sample rate.
    """
    ffmpeg_exe = find_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary raw file
    temp_raw = output_path.with_suffix(".raw_upload")
    try:
        with open(temp_raw, "wb") as f:
            f.write(input_bytes)

        # 1. Try directly with soundfile first (for valid WAV, FLAC, OGG)
        try:
            data, sr = sf.read(str(temp_raw))
            # Resample or write as mono 16-bit PCM WAV
            if data.ndim > 1:
                data = data.mean(axis=1)  # Convert to mono
            sf.write(str(output_path), data, sr, subtype="PCM_16")
            return True
        except Exception:
            pass  # soundfile doesn't support mp4/m4a without libsndfile mp4 plugin

        # 2. Use ffmpeg if available
        if ffmpeg_exe:
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", str(temp_raw),
                "-vn",                  # discard video
                "-ar", str(sample_rate),
                "-ac", "1",             # mono
                "-c:a", "pcm_s16le",
                str(output_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 512:
                logger.info("Successfully converted audio using ffmpeg: %s", output_path)
                return True
            else:
                logger.error("FFmpeg conversion failed: %s", res.stderr.decode("utf-8", errors="ignore"))

        # Fallback: if all fails, just save raw bytes as-is
        with open(output_path, "wb") as f:
            f.write(input_bytes)
        return False

    finally:
        if temp_raw.exists():
            try:
                temp_raw.unlink()
            except Exception:
                pass
