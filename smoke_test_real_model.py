"""
Real-Model Smoke Test for Chatterbox Multilingual Speech Engine.
Verifies end-to-end synthesis with the actual neural network weights.
"""
import time
import os
from pathlib import Path
import soundfile as sf
import numpy as np

from chatterbox.engine.service import get_speech_engine
from chatterbox.artifacts.verifier import AudioVerifier
from chatterbox.artifacts.storage import get_artifact_manager


def run_smoke_test():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("Starting Chatterbox Real-Model Smoke Test")
    print("=" * 60)

    engine = get_speech_engine()
    artifact_mgr = get_artifact_manager()

    print(f"Device: {engine.device}")
    print(f"Model Type: {engine.model_type}")

    # 1. Initialize model
    start_load = time.time()
    print("Loading Chatterbox Multilingual weights...")
    engine.initialize()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    assert engine.is_ready, "Engine failed to initialize"

    # 2. Test Arabic Speech Generation
    test_text = "مرحبا بكم في أورقيفين، نظام الذكاء الاصطناعي التكيفي."
    print(f"\nGenerating Arabic speech: '{test_text}'")
    start_gen = time.time()
    sr, audio = engine.synthesize(
        text=test_text,
        language="ar",
        voice="ar_female",
        exaggeration=0.5,
        temperature=0.8,
        cfg_weight=0.5,
        seed=42,
    )
    gen_time = time.time() - start_gen
    print(f"Speech synthesized in {gen_time:.2f}s")

    # 3. Verify Artifact
    artifact = artifact_mgr.save_and_verify(
        job_id="smoke_test_arabic_001",
        audio_np=audio,
        sample_rate=sr,
        base_url="http://127.0.0.1:8000",
    )

    print("\n--- Artifact Verification Results ---")
    print(f"Artifact Path:    {artifact.file_path}")
    print(f"Sample Rate:      {artifact.sample_rate} Hz")
    print(f"Duration:         {artifact.duration_seconds:.2f} seconds")
    print(f"File Size:        {artifact.file_size_bytes} bytes")
    print(f"Verified:         {artifact.verified}")
    print(f"RTF (Real-Time Factor): {gen_time / artifact.duration_seconds:.2f}x")

    assert artifact.verified is True
    assert artifact.duration_seconds > 0.5
    assert Path(artifact.file_path).is_file()

    print("\n[SUCCESS] Real-model speech synthesis and verification passed!")


if __name__ == "__main__":
    run_smoke_test()
