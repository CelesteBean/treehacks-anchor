"""Real-time GPU-accelerated speech recognition for the Anchor pipeline.

Subscribes to base64-encoded int16 audio chunks on ``AUDIO_PORT`` (5555),
accumulates them until ``min_audio_length`` seconds are buffered, runs
`faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ on the Jetson
Orin Nano GPU (CUDA / float16), and publishes transcription results on
``TRANSCRIPT_PORT`` (5556).

Architecture
------------
::

    ┌───────────┐  SUB :5555   ┌──────────────┐  PUB :5556   ┌──────────┐
    │ audio_    │ ───────────→ │ Speech       │ ───────────→ │ stress / │
    │ capture   │  base64 int16│ Recognizer   │  transcript  │ tactic   │
    └───────────┘              │  (Whisper)   │              └──────────┘
                               └──────────────┘

Message published (inside the ``data`` field of the bus envelope)::

    {
        "text":      "concatenated transcript text",
        "segments":  [{"start": 0.0, "end": 1.2, "text": "..."}],
        "language":  "en",
        "timestamp": "<ISO 8601 UTC>"
    }

Usage::

    from src.core.speech_recognition import SpeechRecognizer, ASRConfig
    from src.core.message_bus import MessageBus

    bus = MessageBus()
    asr = SpeechRecognizer(config=ASRConfig(), bus=bus)
    asr.start()   # blocking – call asr.stop() from another thread
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import zmq
from faster_whisper import WhisperModel

from src.core.message_bus import AUDIO_PORT, TRANSCRIPT_PORT, MessageBus

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ASRConfig:
    """Parameters for the speech-recognition stage.

    Attributes
    ----------
    model_size:
        Whisper model size passed to ``faster_whisper.WhisperModel``.
        ``"small"`` is a good default for the Jetson Orin Nano (~500 MB
        VRAM, ~1 s latency per 1 s of audio).
    language:
        BCP-47 language code for transcription.  ``"en"`` skips the
        language-detection head and reduces latency.
    min_audio_length:
        Minimum seconds of audio to buffer before invoking Whisper.
        Shorter values give faster feedback but may reduce accuracy.
    """

    model_size: str = "small"
    language: str = "en"
    min_audio_length: float = 1.0


# ---------------------------------------------------------------------------
# SpeechRecognizer
# ---------------------------------------------------------------------------


class SpeechRecognizer:
    """Subscribes to audio, runs Whisper, publishes transcripts.

    Parameters
    ----------
    config:
        An :class:`ASRConfig` controlling model size, language, and
        buffer threshold.
    bus:
        A :class:`MessageBus` used to subscribe/publish.

    Raises
    ------
    RuntimeError
        If the Whisper model cannot be loaded (e.g. CUDA OOM, missing
        model files).
    """

    def __init__(self, config: ASRConfig, bus: MessageBus) -> None:
        self.config: ASRConfig = config
        self.bus: MessageBus = bus

        # Audio buffer — flat list of float32 samples at 16 kHz.
        self._buffer: list[float] = []
        self._sample_rate: int = 16000  # updated from first received chunk

        self._stop_event: threading.Event = threading.Event()
        self._publisher: zmq.Socket | None = None
        self._subscriber: zmq.Socket | None = None

        self.running: bool = False

        # Load the model eagerly so callers get an immediate error if
        # CUDA / model files are unavailable.
        logger.info(
            "Loading Whisper model: size=%s, device=cuda, compute_type=float16",
            config.model_size,
        )
        self._model: WhisperModel = WhisperModel(
            config.model_size,
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper model loaded successfully")

    # -- Public properties ---------------------------------------------------

    @property
    def buffer_seconds(self) -> float:
        """Duration of audio currently in the buffer (seconds)."""
        return len(self._buffer) / self._sample_rate

    # -- Audio decoding ------------------------------------------------------

    @staticmethod
    def _decode_audio(data: dict[str, Any]) -> np.ndarray:
        """Decode a base64-encoded int16 audio payload to float32.

        The audio_capture module publishes samples as base64-encoded
        little-endian int16 PCM.  This method reverses that encoding and
        normalises to the [-1.0, 1.0] range expected by Whisper.

        Parameters
        ----------
        data:
            The ``data`` dict from an audio bus message.  Must contain a
            ``"samples"`` key with a base64-encoded string.

        Returns
        -------
        np.ndarray
            1-D float32 array normalised to [-1.0, 1.0].
        """
        raw_bytes: bytes = base64.b64decode(data["samples"])
        int16_samples: np.ndarray = np.frombuffer(raw_bytes, dtype=np.int16)
        float32_samples: np.ndarray = int16_samples.astype(np.float32) / 32768.0
        return float32_samples

    # -- Transcription -------------------------------------------------------

    def _transcribe(self, audio: np.ndarray) -> dict[str, Any]:
        """Run Whisper inference on *audio* and return structured results.

        Parameters
        ----------
        audio:
            1-D float32 array at ``self._sample_rate`` Hz.

        Returns
        -------
        dict
            ``{"text": str, "segments": list[dict], "language": str}``
        """
        segments_iter, info = self._model.transcribe(
            audio,
            language=self.config.language,
            beam_size=1,           # greedy — fastest on Jetson
            vad_filter=True,       # skip silence for lower latency
        )

        # Materialise the segment generator.
        segments: list[dict[str, Any]] = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments_iter
        ]
        full_text: str = "".join(seg["text"] for seg in segments).strip()

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language,
        }

    # -- Buffer management ---------------------------------------------------

    def _buffer_ready(self) -> bool:
        """Return ``True`` when the buffer has enough audio to transcribe."""
        return self.buffer_seconds >= self.config.min_audio_length

    def _flush_buffer(self) -> np.ndarray:
        """Return the buffered audio as a NumPy array and clear the buffer."""
        audio = np.array(self._buffer, dtype=np.float32)
        self._buffer.clear()
        return audio

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Subscribe to audio, accumulate, transcribe, publish — blocking.

        Call :meth:`stop` from another thread to exit the loop.
        """
        self._stop_event.clear()

        if self._subscriber is None:
            self._subscriber = self.bus.create_subscriber(
                ports=[AUDIO_PORT], topics=["audio"],
            )
        if self._publisher is None:
            self._publisher = self.bus.create_publisher(TRANSCRIPT_PORT)

        self.running = True
        logger.info(
            "SpeechRecognizer started – subscribing on :%d, publishing on :%d",
            AUDIO_PORT,
            TRANSCRIPT_PORT,
        )

        try:
            self._main_loop()
        finally:
            self.running = False
            logger.info("SpeechRecognizer stopped")

    def stop(self) -> None:
        """Signal the main loop to exit.  Thread-safe and idempotent."""
        self._stop_event.set()
        self.running = False
        logger.info("SpeechRecognizer stop signal sent")

    # -- Main loop -----------------------------------------------------------

    def _main_loop(self) -> None:
        """Core receive → accumulate → transcribe → publish loop."""
        while not self._stop_event.is_set():
            result = self.bus.receive(self._subscriber, timeout_ms=500)
            if result is None:
                continue

            _, envelope = result
            data: dict[str, Any] = envelope["data"]

            # Update sample rate from the source (in case it differs).
            self._sample_rate = int(data.get("sample_rate", self._sample_rate))

            # Decode and accumulate.
            chunk: np.ndarray = self._decode_audio(data)
            self._buffer.extend(chunk.tolist())

            # Transcribe when we have enough audio.
            if self._buffer_ready():
                audio = self._flush_buffer()

                t_start = time.perf_counter()
                transcript = self._transcribe(audio)
                latency_ms = (time.perf_counter() - t_start) * 1000.0

                logger.info(
                    "Transcribed %.2fs audio in %.0fms: %s",
                    len(audio) / self._sample_rate,
                    latency_ms,
                    transcript["text"][:80] if transcript["text"] else "(silence)",
                )

                # Publish.
                transcript["timestamp"] = datetime.now(timezone.utc).isoformat()
                self.bus.publish(self._publisher, topic="transcript", data=transcript)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- 1. Log GPU memory ---------------------------------------------------
    print("\n=== GPU Memory ===")
    try:
        # tegrastats is Jetson-specific; fall back to nvidia-smi.
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if gpu_info.returncode == 0:
            print(f"  {gpu_info.stdout.strip()}")
        else:
            print("  (nvidia-smi not available)")
    except FileNotFoundError:
        print("  (nvidia-smi not found — may be Jetson without nvidia-smi)")

    # -- 2. Load model -------------------------------------------------------
    print("\n=== Loading Whisper Model ===")
    try:
        bus = MessageBus()
        asr = SpeechRecognizer(config=ASRConfig(), bus=bus)
    except Exception as exc:
        print(f"  FAILED to load model: {exc}")
        sys.exit(1)

    # -- 3. Subscribe and transcribe for 30 seconds --------------------------
    RUN_SECONDS: float = 30.0
    print(f"\n=== Listening for audio on port {AUDIO_PORT} for {RUN_SECONDS}s ===")
    print("  (start audio_capture in another terminal to feed audio)\n")

    # Subscribe to our own transcript output for display.
    transcript_sub = bus.create_subscriber(
        ports=[TRANSCRIPT_PORT], topics=["transcript"],
    )
    time.sleep(0.3)

    def _display_transcripts() -> None:
        """Print transcripts as they arrive."""
        deadline = time.monotonic() + RUN_SECONDS + 5.0
        count = 0
        while time.monotonic() < deadline:
            result = bus.receive(transcript_sub, timeout_ms=1000)
            if result is None:
                continue
            _, env = result
            d = env["data"]
            count += 1
            print(
                f"  [{count}] ({d.get('language', '?')}) {d['text']}"
            )
        print(f"\n  Total transcripts: {count}")

    display_thread = threading.Thread(target=_display_transcripts, daemon=True)
    display_thread.start()

    # Timer to stop after RUN_SECONDS.
    def _timer() -> None:
        time.sleep(RUN_SECONDS)
        asr.stop()

    threading.Thread(target=_timer, daemon=True).start()

    try:
        asr.start()
    except KeyboardInterrupt:
        asr.stop()

    display_thread.join(timeout=5)
    transcript_sub.close()
    print("\nSmoke test complete.")
