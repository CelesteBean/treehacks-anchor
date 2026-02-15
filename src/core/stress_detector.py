"""GPU-accelerated vocal-stress detection for the Anchor pipeline.

Subscribes to base64-encoded int16 audio chunks on ``AUDIO_PORT`` (5555),
accumulates them into 2–3 second windows, runs the
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
<https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim>`_
dimensional-emotion model on CUDA, and publishes stress results on
``STRESS_PORT`` (5557).

The model outputs three continuous dimensions:

* **Arousal** – activation / stress level (primary signal).
* **Dominance** – perceived control.
* **Valence** – positivity / negativity.

Architecture
------------
::

    ┌───────────┐  SUB :5555   ┌──────────────┐  PUB :5557   ┌──────────┐
    │ audio_    │ ───────────→ │ Stress       │ ───────────→ │ tactic / │
    │ capture   │  base64 int16│ Detector     │  stress      │ viz      │
    └───────────┘              │  (wav2vec2)  │              └──────────┘
                               └──────────────┘

Message published (inside the ``data`` field of the bus envelope)::

    {
        "timestamp":    "<ISO 8601 UTC>",
        "stress_score": <float 0-1>,   # arousal value
        "emotions": {
            "arousal":   <float 0-1>,
            "valence":   <float 0-1>,
            "dominance": <float 0-1>
        },
        "confidence": <float 0-1>
    }

Usage::

    from src.core.stress_detector import StressDetector, StressConfig
    from src.core.message_bus import MessageBus

    bus = MessageBus()
    detector = StressDetector(config=StressConfig(), bus=bus)
    detector.start()   # blocking – call detector.stop() from another thread
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Jetson cusparselt fix: the pip-installed nvidia-cusparselt package
# ships libcusparseLt.so.0 inside a package directory that the dynamic
# linker doesn't know about.  Preload it with ctypes so that torch's
# subsequent dlopen succeeds.
# ---------------------------------------------------------------------------
try:
    import ctypes
    import nvidia.cusparselt  # type: ignore[import-untyped]

    _cusparselt_so = os.path.join(
        os.path.dirname(nvidia.cusparselt.__path__[0]),
        "cusparselt",
        "lib",
        "libcusparseLt.so.0",
    )
    if os.path.isfile(_cusparselt_so):
        ctypes.cdll.LoadLibrary(_cusparselt_so)
except (ImportError, OSError):
    pass  # cusparselt not installed or not needed

import numpy as np
import torch
import torch.nn as nn
import zmq
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from src.core.message_bus import AUDIO_PORT, STRESS_PORT, MessageBus

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model architecture (matches audeering's published definition)
# ---------------------------------------------------------------------------

_MODEL_NAME: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
"""HuggingFace model identifier for the dimensional emotion model."""


class _RegressionHead(nn.Module):
    """MLP regression head mapping pooled hidden states to emotion dims.

    Architecture mirrors the audeering reference implementation:
    dropout → dense (hidden_size → hidden_size) → tanh → dropout →
    out_proj (hidden_size → num_labels).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 backbone + regression head for dimensional emotion.

    Outputs ``(pooled_hidden_states, logits)`` where logits are
    ``[arousal, dominance, valence]`` in approximately the 0–1 range.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        hidden_states: torch.Tensor = outputs[0]
        # Mean-pool over the time dimension.
        hidden_states = torch.mean(hidden_states, dim=1)
        logits: torch.Tensor = self.classifier(hidden_states)
        return hidden_states, logits


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StressConfig:
    """Parameters for the stress-detection stage.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier.
    min_audio_length:
        Minimum seconds of audio to buffer before running inference.
        2–3 s gives a good trade-off between latency and accuracy.
    device:
        PyTorch device string (``"cuda"`` or ``"cpu"``).
    """

    model_name: str = _MODEL_NAME
    min_audio_length: float = 2.5
    device: str = "cuda"


# ---------------------------------------------------------------------------
# StressDetector
# ---------------------------------------------------------------------------


class StressDetector:
    """Subscribes to audio, runs emotion inference, publishes stress scores.

    Parameters
    ----------
    config:
        A :class:`StressConfig` controlling model, buffer, and device.
    bus:
        A :class:`MessageBus` used to subscribe/publish.

    Raises
    ------
    RuntimeError
        If the emotion model cannot be loaded (e.g. CUDA out of memory,
        missing model files).
    """

    def __init__(self, config: StressConfig, bus: MessageBus) -> None:
        self.config: StressConfig = config
        self.bus: MessageBus = bus

        # Audio buffer — flat list of float32 samples at 16 kHz.
        self._buffer: list[float] = []
        self._sample_rate: int = 16_000

        self._stop_event: threading.Event = threading.Event()
        self._publisher: zmq.Socket | None = None
        self._subscriber: zmq.Socket | None = None

        self.running: bool = False

        # Load model eagerly so callers get an immediate error on failure.
        logger.info(
            "Loading emotion model: %s on %s",
            config.model_name,
            config.device,
        )
        self._processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(
            config.model_name,
        )
        self._model: _EmotionModel = _EmotionModel.from_pretrained(
            config.model_name,
        ).to(config.device)  # type: ignore[arg-type]
        self._model.eval()
        logger.info("Emotion model loaded successfully on %s", config.device)

    # -- Public properties ---------------------------------------------------

    @property
    def buffer_seconds(self) -> float:
        """Duration of audio currently in the buffer (seconds)."""
        return len(self._buffer) / self._sample_rate

    # -- Audio decoding (mirrors speech_recognition.py) ----------------------

    @staticmethod
    def _decode_audio(data: dict[str, Any]) -> np.ndarray:
        """Decode a base64-encoded int16 audio payload to float32.

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
        float32_samples: np.ndarray = int16_samples.astype(np.float32) / 32_768.0
        return float32_samples

    # -- Inference -----------------------------------------------------------

    def _predict_emotions(self, audio: np.ndarray) -> dict[str, Any]:
        """Run emotion inference on *audio* and return structured results.

        Parameters
        ----------
        audio:
            1-D float32 array at ``self._sample_rate`` Hz.

        Returns
        -------
        dict
            ``{"arousal": float, "dominance": float, "valence": float,
              "confidence": float}``
        """
        # Processor normalises the signal; returns a batch (take first item).
        inputs = self._processor(
            audio,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        input_values: torch.Tensor = inputs["input_values"].to(self.config.device)

        with torch.no_grad():
            _, logits = self._model(input_values)

        # logits shape: (1, 3) → [arousal, dominance, valence]
        scores: np.ndarray = logits.detach().cpu().numpy().flatten()

        # Clip to [0, 1] — the model occasionally slightly exceeds bounds.
        scores = np.clip(scores, 0.0, 1.0)

        arousal: float = float(scores[0])
        dominance: float = float(scores[1])
        valence: float = float(scores[2])

        # Confidence heuristic: higher arousal + lower valence → higher
        # confidence that stress is genuine (not just excitement).
        confidence: float = float(np.clip(
            0.5 * arousal + 0.3 * (1.0 - valence) + 0.2 * (1.0 - dominance),
            0.0,
            1.0,
        ))

        return {
            "arousal": arousal,
            "dominance": dominance,
            "valence": valence,
            "confidence": confidence,
        }

    # -- Buffer management ---------------------------------------------------

    def _buffer_ready(self) -> bool:
        """Return ``True`` when the buffer has enough audio to analyse."""
        return self.buffer_seconds >= self.config.min_audio_length

    def _flush_buffer(self) -> np.ndarray:
        """Return the buffered audio as a NumPy array and clear the buffer."""
        audio = np.array(self._buffer, dtype=np.float32)
        self._buffer.clear()
        return audio

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Subscribe to audio, accumulate, analyse, publish — blocking.

        Call :meth:`stop` from another thread to exit the loop.
        """
        self._stop_event.clear()

        if self._subscriber is None:
            self._subscriber = self.bus.create_subscriber(
                ports=[AUDIO_PORT],
                topics=["audio"],
            )
        if self._publisher is None:
            self._publisher = self.bus.create_publisher(STRESS_PORT)

        self.running = True
        logger.info(
            "StressDetector started – subscribing on :%d, publishing on :%d",
            AUDIO_PORT,
            STRESS_PORT,
        )

        try:
            self._main_loop()
        finally:
            self.running = False
            logger.info("StressDetector stopped")

    def stop(self) -> None:
        """Signal the main loop to exit.  Thread-safe and idempotent."""
        self._stop_event.set()
        self.running = False
        logger.info("StressDetector stop signal sent")

    # -- Main loop -----------------------------------------------------------

    def _main_loop(self) -> None:
        """Core receive → accumulate → predict → publish loop."""
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

            # Run inference when we have enough audio.
            if self._buffer_ready():
                audio = self._flush_buffer()
                audio_secs: float = len(audio) / self._sample_rate

                t_start = time.perf_counter()
                emotions = self._predict_emotions(audio)
                latency_ms: float = (time.perf_counter() - t_start) * 1_000.0

                logger.info(
                    "Processed %.1fs audio in %.0fms (GPU) — "
                    "arousal=%.3f valence=%.3f dominance=%.3f",
                    audio_secs,
                    latency_ms,
                    emotions["arousal"],
                    emotions["valence"],
                    emotions["dominance"],
                )

                # Build and publish the stress message.
                stress_msg: dict[str, Any] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stress_score": emotions["arousal"],
                    "emotions": {
                        "arousal": emotions["arousal"],
                        "valence": emotions["valence"],
                        "dominance": emotions["dominance"],
                    },
                    "confidence": emotions["confidence"],
                }
                self.bus.publish(
                    self._publisher,
                    topic="stress",
                    data=stress_msg,
                )


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
        gpu_info = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if gpu_info.returncode == 0:
            print(f"  {gpu_info.stdout.strip()}")
        else:
            print("  (nvidia-smi not available)")
    except FileNotFoundError:
        print("  (nvidia-smi not found — may be Jetson without nvidia-smi)")

    # -- 2. Load model -------------------------------------------------------
    print("\n=== Loading Emotion Model ===")
    try:
        bus = MessageBus()
        detector = StressDetector(config=StressConfig(), bus=bus)
    except Exception as exc:
        print(f"  FAILED to load model: {exc}")
        sys.exit(1)

    # -- 3. Quick inference sanity check with silence ------------------------
    print("\n=== Sanity Check (silence) ===")
    silence = np.zeros(16_000 * 3, dtype=np.float32)  # 3 s of silence
    t0 = time.perf_counter()
    result = detector._predict_emotions(silence)
    dt_ms = (time.perf_counter() - t0) * 1_000.0
    print(f"  Inference on 3s silence: {dt_ms:.0f}ms")
    print(f"  arousal={result['arousal']:.3f}  "
          f"valence={result['valence']:.3f}  "
          f"dominance={result['dominance']:.3f}  "
          f"confidence={result['confidence']:.3f}")

    # -- 4. Subscribe and analyse for 30 seconds -----------------------------
    RUN_SECONDS: float = 3600.0
    print(f"\n=== Listening for audio on port {AUDIO_PORT} for {RUN_SECONDS}s ===")
    print("  (start audio_capture in another terminal to feed audio)\n")

    # Subscribe to our own stress output for display.
    stress_sub = bus.create_subscriber(
        ports=[STRESS_PORT],
        topics=["stress"],
    )
    time.sleep(0.3)

    def _display_stress() -> None:
        """Print stress messages as they arrive."""
        deadline = time.monotonic() + RUN_SECONDS + 5.0
        count = 0
        while time.monotonic() < deadline:
            msg_result = bus.receive(stress_sub, timeout_ms=1000)
            if msg_result is None:
                continue
            _, env = msg_result
            d = env["data"]
            count += 1
            emo = d.get("emotions", {})
            print(
                f"  [{count}] stress={d.get('stress_score', '?'):.3f}  "
                f"a={emo.get('arousal', '?'):.3f}  "
                f"v={emo.get('valence', '?'):.3f}  "
                f"d={emo.get('dominance', '?'):.3f}  "
                f"conf={d.get('confidence', '?'):.3f}"
            )
        print(f"\n  Total stress messages: {count}")

    display_thread = threading.Thread(target=_display_stress, daemon=True)
    display_thread.start()

    # Timer to stop after RUN_SECONDS.
    def _timer() -> None:
        time.sleep(RUN_SECONDS)
        detector.stop()

    threading.Thread(target=_timer, daemon=True).start()

    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()

    display_thread.join(timeout=5)
    stress_sub.close()
    print("\nSmoke test complete.")
