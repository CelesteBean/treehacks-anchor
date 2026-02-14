"""Real-time microphone capture for the Anchor scam-detection pipeline.

Captures audio from a USB microphone (or default input device) via the
``sounddevice`` library and publishes chunks over ZeroMQ for downstream
stages (speech recognition, vocal-stress analysis).

Architecture
------------
::

    ┌────────────┐  float32   ┌───────────┐  base64/JSON  ┌───────────┐
    │ sounddevice│ ─callback→ │  Queue     │ ─pub thread→  │  ZeroMQ   │
    │ InputStream│            │ (int16 b64)│               │ PUB :5555 │
    └────────────┘            └───────────┘               └───────────┘

The sounddevice callback runs on a C-level audio thread that is **not**
compatible with ZeroMQ sockets.  A ``queue.Queue`` bridges the two
threads safely.

Message payload (inside the ``data`` field of the bus envelope)::

    {
        "samples":     "<base64-encoded int16 PCM bytes>",
        "timestamp":   "<ISO 8601 UTC>",
        "sample_rate": 16000
    }

Usage::

    from src.core.audio_capture import AudioCapture, AudioConfig
    from src.core.message_bus import MessageBus

    bus     = MessageBus()
    capture = AudioCapture(config=AudioConfig(), bus=bus)
    capture.start()   # blocking – runs until capture.stop() from another thread
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import sounddevice as sd
import zmq

from src.core.message_bus import AUDIO_PORT, MessageBus

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    """Parameters for the audio capture stage.

    Attributes
    ----------
    sample_rate:
        Samples per second.  16 kHz is the standard for speech models
        (Whisper, wav2vec 2.0).
    channels:
        Number of audio channels.  Mono (1) is sufficient for speech.
    chunk_size:
        Number of samples per callback invocation.  1024 at 16 kHz ≈ 64 ms
        per chunk — a good balance between latency and overhead.
    device_name:
        Optional substring to match against available device names.
        ``None`` selects the system default input device.
    """

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    device_name: str | None = None


# ---------------------------------------------------------------------------
# AudioCapture
# ---------------------------------------------------------------------------

class AudioCapture:
    """Captures audio from a microphone and publishes chunks over ZeroMQ.

    Parameters
    ----------
    config:
        An :class:`AudioConfig` instance controlling sample rate, channels,
        chunk size, and device selection.
    bus:
        A :class:`MessageBus` instance used to create the PUB socket and
        serialise messages.
    """

    # Maximum items buffered between the audio thread and the publish thread.
    _QUEUE_MAXSIZE: int = 256

    def __init__(self, config: AudioConfig, bus: MessageBus) -> None:
        self.config: AudioConfig = config
        self.bus: MessageBus = bus

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=self._QUEUE_MAXSIZE,
        )
        self._stop_event: threading.Event = threading.Event()
        self._publisher: zmq.Socket | None = None

        # Counters for observability.
        self.published_count: int = 0
        self.callback_count: int = 0

        # Public flag consumers can poll.
        self.running: bool = False

    # -- Static helpers ------------------------------------------------------

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """Return a list of available audio input devices.

        Each entry is a dict with at least ``name``, ``max_input_channels``,
        and ``default_samplerate`` keys (matching the ``sounddevice``
        ``query_devices()`` output).

        Returns
        -------
        list[dict[str, Any]]
            Possibly empty list of device-info dictionaries.
        """
        devices = sd.query_devices()
        if isinstance(devices, dict):
            # Single device – wrap in a list for uniform handling.
            return [devices]
        return list(devices)

    # -- Sounddevice callback ------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags | None,
    ) -> None:
        """Called by sounddevice on the audio thread for each chunk.

        Converts float32 samples to int16, base64-encodes them, and puts
        the result on the internal queue.  This method must be fast and
        must **not** touch ZeroMQ sockets.

        Parameters
        ----------
        indata:
            NumPy array of shape ``(chunk_size, channels)`` with float32
            samples in [-1.0, 1.0].
        frames:
            Number of frames (== ``chunk_size``).
        time_info:
            PortAudio time information (unused).
        status:
            PortAudio status flags.  Non-empty status indicates a problem
            (e.g. buffer overflow).
        """
        if status:
            logger.warning("Audio callback status: %s", status)

        self.callback_count += 1

        # float32 → int16 (clamp to avoid wrap-around).
        clamped: np.ndarray = np.clip(indata, -1.0, 1.0)
        int16_samples: np.ndarray = (clamped * 32767).astype(np.int16)

        # Flatten to 1-D before encoding (strips channel dimension).
        raw_bytes: bytes = int16_samples.flatten().tobytes()
        b64_samples: str = base64.b64encode(raw_bytes).decode("ascii")

        payload: dict[str, Any] = {
            "samples": b64_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_rate": self.config.sample_rate,
        }

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Audio queue full – dropping chunk")

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Open the audio stream and publish chunks until :meth:`stop`.

        This method **blocks** the calling thread.  Call :meth:`stop` from
        another thread (or a signal handler) to shut down gracefully.

        Raises
        ------
        sounddevice.PortAudioError
            If the audio device cannot be opened (e.g. no device connected).
        """
        self._stop_event.clear()

        # Resolve device index from name (if given).
        device_index: int | None = self._resolve_device()

        # Bind publisher socket (idempotent guard).
        if self._publisher is None:
            self._publisher = self.bus.create_publisher(AUDIO_PORT)

        self.running = True
        logger.info(
            "Starting audio capture: %d Hz, %d ch, chunk=%d, device=%s",
            self.config.sample_rate,
            self.config.channels,
            self.config.chunk_size,
            device_index if device_index is not None else "default",
        )

        # Open the PortAudio stream.  This may raise PortAudioError.
        stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            dtype="float32",
            device=device_index,
            callback=self._audio_callback,
        )

        try:
            stream.start()
            logger.info("Audio stream opened – publishing on port %d", AUDIO_PORT)
            self._publish_loop()
        except Exception:
            logger.exception("Fatal error in audio capture")
            raise
        finally:
            stream.stop()
            stream.close()
            self.running = False
            logger.info("Audio stream closed")

    def stop(self) -> None:
        """Signal the capture loop to exit.

        Safe to call from any thread, and idempotent (calling ``stop()``
        before ``start()`` is a no-op).
        """
        self._stop_event.set()
        self.running = False
        logger.info("Stop signal sent")

    # -- Internal ------------------------------------------------------------

    def _publish_loop(self) -> None:
        """Drain the queue and publish messages until the stop event is set."""
        logger.debug("_publish_loop started (publisher=%s)", self._publisher)

        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._publisher is not None:
                self.bus.publish(self._publisher, topic="audio", data=payload)
                self.published_count += 1

                if self.published_count % 50 == 1:
                    logger.debug(
                        "_publish_loop: published=%d, queued=%d",
                        self.published_count,
                        self._queue.qsize(),
                    )

        # Drain any remaining items after stop is signalled.
        remaining = 0
        while not self._queue.empty():
            try:
                payload = self._queue.get_nowait()
                if self._publisher is not None:
                    self.bus.publish(self._publisher, topic="audio", data=payload)
                    self.published_count += 1
                    remaining += 1
            except queue.Empty:
                break

        logger.info(
            "_publish_loop exiting: total_published=%d (drained %d after stop), "
            "callbacks=%d",
            self.published_count,
            remaining,
            self.callback_count,
        )

    def _resolve_device(self) -> int | None:
        """Match ``config.device_name`` to a PortAudio device index.

        Returns
        -------
        int | None
            Device index, or ``None`` to use the system default.
        """
        if self.config.device_name is None:
            return None

        devices = self.list_devices()
        for idx, dev in enumerate(devices):
            if self.config.device_name.lower() in dev.get("name", "").lower():
                logger.info(
                    "Matched device %r → index %d", dev["name"], idx,
                )
                return idx

        logger.warning(
            "Device %r not found – falling back to system default",
            self.config.device_name,
        )
        return None


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- 1. List devices -----------------------------------------------------
    print("\n=== Available audio devices ===")
    all_devices = AudioCapture.list_devices()
    if not all_devices:
        print("  (none found)")
    for i, dev in enumerate(all_devices):
        in_ch = dev.get("max_input_channels", 0)
        marker = " <-- input" if in_ch > 0 else ""
        print(f"  [{i}] {dev.get('name', '?')} (in={in_ch}){marker}")

    # -- 2. Capture 5 seconds ------------------------------------------------
    CAPTURE_SECONDS: float = 5.0
    config = AudioConfig()
    bus = MessageBus()
    capture = AudioCapture(config=config, bus=bus)

    # ── FIX: bind the publisher FIRST, then create the subscriber. ──────
    # The original code created the subscriber before the publisher existed,
    # which meant the slow-joiner sleep was wasted (nothing to handshake
    # with).  Now we:
    #   1. Bind the PUB socket eagerly.
    #   2. Inject it into the capture so start() reuses it.
    #   3. Create the SUB socket (connects to the already-bound PUB).
    #   4. Sleep for the ZeroMQ subscription handshake.
    pub = bus.create_publisher(AUDIO_PORT)
    capture._publisher = pub  # start() sees non-None, skips re-bind
    sub = bus.create_subscriber(ports=[AUDIO_PORT], topics=["audio"])
    time.sleep(0.5)  # slow-joiner: SUB ↔ PUB subscription handshake
    logger.info("Publisher bound, subscriber connected, handshake done")

    chunks_received: int = 0

    def _timer_stop() -> None:
        """Stop capture after CAPTURE_SECONDS."""
        time.sleep(CAPTURE_SECONDS)
        capture.stop()

    timer_thread = threading.Thread(target=_timer_stop, daemon=True)
    timer_thread.start()

    # Start capture in a background thread so we can read from the sub.
    capture_thread = threading.Thread(target=capture.start, daemon=True)
    capture_thread.start()

    print(f"\n=== Capturing {CAPTURE_SECONDS}s of audio ===")
    deadline = time.monotonic() + CAPTURE_SECONDS + 2.0  # extra buffer

    # ── FIX: explicit parentheses for correct operator precedence. ──────
    # Old:  (A and B) or C  — exited prematurely when capture.running
    #       flipped to False, even if messages were still in flight.
    # New:  A and (B or C)  — keep looping while within deadline AND
    #       (capture is running OR we haven't received anything yet).
    while time.monotonic() < deadline and (capture.running or chunks_received == 0):
        result = bus.receive(sub, timeout_ms=500)
        if result is None:
            continue
        _, envelope = result
        data = envelope["data"]

        # Decode and compute RMS (Root Mean Square) level.
        raw_bytes = base64.b64decode(data["samples"])
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        chunks_received += 1

        if chunks_received <= 3 or chunks_received % 20 == 0:
            logger.info(
                "Chunk %3d | samples=%5d | RMS=%8.1f | ts=%s",
                chunks_received,
                len(samples),
                rms,
                data["timestamp"],
            )

    capture_thread.join(timeout=3)
    sub.close()

    # -- 3. Verify results ---------------------------------------------------
    published = capture.published_count
    print(f"\nPublished {published} messages, Received {chunks_received} messages")
    print(f"  Callbacks fired : {capture.callback_count}")

    if published == 0:
        print("ERROR: zero messages published — _publish_loop never ran or queue was empty")
        sys.exit(1)

    ratio = chunks_received / published if published > 0 else 0.0
    print(f"  Receive ratio   : {ratio:.1%}")

    if ratio < 0.80:
        print(
            f"FAIL: received only {ratio:.0%} of published "
            f"(threshold: 80%)"
        )
        sys.exit(1)

    print("PASS: audio capture → ZMQ publish → ZMQ subscribe pipeline verified")
