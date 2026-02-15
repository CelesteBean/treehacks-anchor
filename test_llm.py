#!/usr/bin/env python3
"""Diagnostic script for LLM warning generation path."""

import os
import sys
import time

sys.path.insert(0, ".")

MODEL_PATH = os.path.expanduser(
    "~/treehacks-anchor/models/qwen-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)


def main() -> int:
    print("=" * 60)
    print("LLM DIAGNOSTIC TEST")
    print("=" * 60)

    # 1) Import llama_cpp
    print("\n[1] Checking llama-cpp-python installation...")
    try:
        from llama_cpp import Llama

        print("    OK llama_cpp imported successfully")
    except ImportError as err:
        print(f"    FAILED: {err}")
        return 1

    # 2) Validate model file
    print("\n[2] Checking model file...")
    if not os.path.exists(MODEL_PATH):
        print(f"    FAILED model not found: {MODEL_PATH}")
        return 1
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"    OK Model found: {MODEL_PATH}")
    print(f"    OK Size: {size_mb:.1f} MB")
    if size_mb < 100:
        print("    WARNING file seems too small and may be corrupted")

    # 3) Load model
    print("\n[3] Loading LLM (may take a few seconds)...")
    start = time.time()
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=128,
            n_threads=6,
            n_gpu_layers=50,
            verbose=False,
        )
    except Exception as err:  # pragma: no cover - diagnostic script
        print(f"    FAILED to load model: {err}")
        return 1
    load_time = time.time() - start
    print(f"    OK Model loaded in {load_time:.2f}s")

    # 4) Generate text
    print("\n[4] Testing generation...")
    prompt = "Warning. This caller is asking you to buy gift cards. You should"
    start = time.time()
    try:
        output = llm(
            prompt,
            max_tokens=25,
            temperature=0.3,
            stop=["\n", ".", "!"],
            echo=False,
        )
    except Exception as err:  # pragma: no cover - diagnostic script
        print(f"    FAILED generation error: {err}")
        return 1
    gen_time = time.time() - start

    text = ""
    if output and "choices" in output and output["choices"]:
        text = output["choices"][0].get("text", "").strip()
    if not text:
        print(f"    FAILED empty output: {output}")
        return 1
    print(f"    OK Generation completed in {gen_time * 1000:.0f}ms")
    print(f'    OK Output: "{text}"')

    # 5) WarningGenerator path
    print("\n[5] Testing WarningGenerator class...")
    try:
        from src.core.warning_generator import WarningGenerator

        print("    OK WarningGenerator imported")
        wg = WarningGenerator()
        print("    OK WarningGenerator initialized")
        warning = wg.generate_warning(
            threat_type="gift_card",
            risk_factors=["gift card purchase"],
            recent_transcript="I will buy the gift cards now",
        )
        print(f'    OK Generated warning: "{warning[:80]}..."')
    except Exception as err:
        print(f"    FAILED warning generator error: {err}")
        return 1

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
