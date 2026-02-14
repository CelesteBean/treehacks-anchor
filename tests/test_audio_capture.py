"""Unit tests for src.core.audio_capture – real-time microphone capture.

Hardware-independent: all sounddevice interactions are mocked so that
tests run on any machine (CI included) without a microphone.

Tests cover:
    - AudioConfig dataclass defaults and overrides
    - AudioCapture construction and socket creation
    - list_devices() static method
    - _audio_callback encodes int16 → base64 correctly
    - Published message structure (samples, timestamp, sample_rate)
    - start / stop lifecycle (running flag, thread join)
    - Graceful handling when no audio device is found
"""

from __future__ import annotations

import base64
import json
import queue
import time
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import zmq

from src.core.audio_capture import AudioCapture, AudioConfig
from src.core.message_bus import AUDIO_PORT, MessageBus


# ---------------------------------------------------------------------------
# AudioConfig dataclass
# ---------------------------------------------------------------------------

class TestAudioConfig:
    """AudioConfig must expose sensible defaults for speech capture."""

    def test_default_sample_rate(self) -> None:
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000

    def test_default_channels(self) -> None:
        cfg = AudioConfig()
        assert cfg.channels == 1

    def test_default_chunk_size(self) -> None:
        cfg = AudioConfig()
        assert cfg.chunk_size == 1024

    def test_default_device_name_is_none(self) -> None:
        cfg = AudioConfig()
        assert cfg.device_name is None

    def test_custom_values(self) -> None:
        cfg = AudioConfig(sample_rate=44100, channels=2, chunk_size=512, device_name="USB Mic")
        assert cfg.sample_rate == 44100
        assert cfg.channels == 2
        assert cfg.chunk_size == 512
        assert cfg.device_name == "USB Mic"


# ---------------------------------------------------------------------------
# AudioCapture construction
# ---------------------------------------------------------------------------

class TestAudioCaptureInit:
    """AudioCapture must wire up config, bus, and publisher."""

    def test_stores_config(self) -> None:
        cfg = AudioConfig()
        bus = MessageBus()
        capture = AudioCapture(config=cfg, bus=bus)
        assert capture.config is cfg

    def test_stores_bus(self) -> None:
        cfg = AudioConfig()
        bus = MessageBus()
        capture = AudioCapture(config=cfg, bus=bus)
        assert capture.bus is bus

    def test_not_running_initially(self) -> None:
        capture = AudioCapture(config=AudioConfig(), bus=MessageBus())
        assert not capture.running


# ---------------------------------------------------------------------------
# list_devices
# ---------------------------------------------------------------------------

class TestListDevices:
    """list_devices() should return structured device information."""

    @patch("src.core.audio_capture.sd")
    def test_returns_list_of_dicts(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = [
            {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 16000.0},
            {"name": "Built-in", "max_input_channels": 2, "default_samplerate": 44100.0},
        ]
        devices = AudioCapture.list_devices()
        assert isinstance(devices, list)
        assert len(devices) == 2
        assert devices[0]["name"] == "USB Mic"

    @patch("src.core.audio_capture.sd")
    def test_empty_when_no_devices(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = []
        devices = AudioCapture.list_devices()
        assert devices == []


# ---------------------------------------------------------------------------
# _audio_callback encoding
# ---------------------------------------------------------------------------

class TestAudioCallback:
    """The sounddevice callback must base64-encode int16 samples and enqueue."""

    @pytest.fixture(autouse=True)
    def _capture(self) -> None:
        self.capture = AudioCapture(config=AudioConfig(), bus=MessageBus())

    def test_callback_enqueues_data(self) -> None:
        """After the callback fires, the internal queue should have one item."""
        # Simulate a 1024-sample mono chunk from sounddevice (float32, [-1, 1]).
        fake_audio = np.random.uniform(-0.5, 0.5, (1024, 1)).astype(np.float32)
        self.capture._audio_callback(fake_audio, frames=1024, time_info=None, status=None)

        assert not self.capture._queue.empty()

    def test_enqueued_data_is_base64_string(self) -> None:
        fake_audio = np.zeros((1024, 1), dtype=np.float32)
        self.capture._audio_callback(fake_audio, frames=1024, time_info=None, status=None)

        item = self.capture._queue.get_nowait()
        # item["samples"] should be a valid base64 string.
        decoded = base64.b64decode(item["samples"])
        assert isinstance(decoded, bytes)

    def test_base64_decodes_to_int16_array(self) -> None:
        """Round-trip: float32 -> int16 bytes -> base64 -> decode -> int16 array."""
        rng = np.random.default_rng(42)
        fake_audio = rng.uniform(-0.8, 0.8, (1024, 1)).astype(np.float32)
        self.capture._audio_callback(fake_audio, frames=1024, time_info=None, status=None)

        item = self.capture._queue.get_nowait()
        raw_bytes = base64.b64decode(item["samples"])
        recovered = np.frombuffer(raw_bytes, dtype=np.int16)
        assert recovered.shape == (1024,)

    def test_enqueued_item_has_timestamp(self) -> None:
        fake_audio = np.zeros((1024, 1), dtype=np.float32)
        self.capture._audio_callback(fake_audio, frames=1024, time_info=None, status=None)

        item = self.capture._queue.get_nowait()
        assert "timestamp" in item
        # Must parse as ISO 8601.
        datetime.fromisoformat(item["timestamp"])

    def test_enqueued_item_has_sample_rate(self) -> None:
        fake_audio = np.zeros((1024, 1), dtype=np.float32)
        self.capture._audio_callback(fake_audio, frames=1024, time_info=None, status=None)

        item = self.capture._queue.get_nowait()
        assert item["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# Publish loop – message format on the wire
# ---------------------------------------------------------------------------

class TestPublishMessageFormat:
    """Messages published to ZeroMQ must match the pipeline envelope spec."""

    PORT = 6500  # Unique port for this test class.

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.bus = MessageBus()
        self.pub = self.bus.create_publisher(port=self.PORT)
        self.sub = self.bus.create_subscriber(ports=[self.PORT], topics=["audio"])
        time.sleep(0.3)  # slow-joiner
        self.capture = AudioCapture(config=AudioConfig(), bus=self.bus)
        # Inject our publisher so the capture doesn't bind its own.
        self.capture._publisher = self.pub
        yield
        self.sub.close()
        self.pub.close()

    def test_published_message_has_samples_field(self) -> None:
        fake_audio = np.zeros((1024, 1), dtype=np.float32)
        self.capture._audio_callback(fake_audio, frames=1024, time_info=None, status=None)
        # Drain queue manually to simulate publish loop.
        item = self.capture._queue.get_nowait()
        self.bus.publish(self.pub, topic="audio", data=item)

        result = self.bus.receive(self.sub, timeout_ms=2000)
        assert result is not None
        _, envelope = result
        assert "samples" in envelope["data"]
        assert "sample_rate" in envelope["data"]
        assert "timestamp" in envelope["data"]


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestStartStop:
    """start() and stop() must manage the capture lifecycle cleanly."""

    @patch("src.core.audio_capture.sd")
    def test_start_sets_running_flag(self, mock_sd: MagicMock) -> None:
        # Mock InputStream as a context-manager-like object.
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture = AudioCapture(config=AudioConfig(), bus=MessageBus())
        # Inject a mock publisher so start() does not bind a real port.
        capture._publisher = MagicMock()

        # Run start in a thread so we can stop it.
        t = threading.Thread(target=capture.start, daemon=True)
        t.start()
        time.sleep(0.3)

        assert capture.running is True

        capture.stop()
        t.join(timeout=3)
        assert capture.running is False

    @patch("src.core.audio_capture.sd")
    def test_stop_is_idempotent(self, mock_sd: MagicMock) -> None:
        capture = AudioCapture(config=AudioConfig(), bus=MessageBus())
        # stop() before start() should not raise.
        capture.stop()
        assert capture.running is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Graceful failure when hardware is missing or disconnects."""

    @patch("src.core.audio_capture.sd")
    def test_start_raises_on_no_device(self, mock_sd: MagicMock) -> None:
        """If sounddevice can't open a stream, start() should raise."""
        import sounddevice  # noqa: F811 – needed for the exception class

        mock_sd.PortAudioError = sounddevice.PortAudioError
        mock_sd.InputStream.side_effect = sounddevice.PortAudioError("No device")

        capture = AudioCapture(config=AudioConfig(), bus=MessageBus())
        # Inject a mock publisher so start() does not bind a real port.
        capture._publisher = MagicMock()
        with pytest.raises(sounddevice.PortAudioError):
            capture.start()
