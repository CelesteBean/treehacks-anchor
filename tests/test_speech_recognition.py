"""Unit tests for src.core.speech_recognition – GPU Whisper transcription.

Hardware-independent: ``faster_whisper.WhisperModel`` is mocked so tests
run without a CUDA GPU.  Audio decoding and buffer accumulation use real
NumPy operations.

Tests cover:
    - ASRConfig dataclass defaults and overrides
    - SpeechRecognizer construction
    - _decode_audio: base64 int16 → float32 normalised
    - _transcribe: mocked WhisperModel returns expected structure
    - Buffer accumulation and threshold logic
    - Published message format on the wire
    - start / stop lifecycle
    - Graceful model-loading error handling
"""

from __future__ import annotations

import base64
import time
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.message_bus import AUDIO_PORT, TRANSCRIPT_PORT, MessageBus
from src.core.speech_recognition import ASRConfig, SpeechRecognizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio_payload(
    samples: np.ndarray | None = None,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Build an audio ``data`` dict matching the audio_capture wire format.

    Parameters
    ----------
    samples:
        1-D int16 array.  If ``None``, generates 1024 samples of silence.
    sample_rate:
        Sample rate to include in the payload.
    """
    if samples is None:
        samples = np.zeros(1024, dtype=np.int16)
    raw = samples.tobytes()
    return {
        "samples": base64.b64encode(raw).decode("ascii"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_rate": sample_rate,
    }


# ---------------------------------------------------------------------------
# ASRConfig dataclass
# ---------------------------------------------------------------------------

class TestASRConfig:
    """ASRConfig must expose sensible defaults for Whisper on Jetson."""

    def test_default_model_size(self) -> None:
        cfg = ASRConfig()
        assert cfg.model_size == "small"

    def test_default_language(self) -> None:
        cfg = ASRConfig()
        assert cfg.language == "en"

    def test_default_min_audio_length(self) -> None:
        cfg = ASRConfig()
        assert cfg.min_audio_length == 1.0

    def test_custom_values(self) -> None:
        cfg = ASRConfig(model_size="tiny", language="es", min_audio_length=2.5)
        assert cfg.model_size == "tiny"
        assert cfg.language == "es"
        assert cfg.min_audio_length == 2.5


# ---------------------------------------------------------------------------
# SpeechRecognizer construction
# ---------------------------------------------------------------------------

class TestSpeechRecognizerInit:
    """SpeechRecognizer stores config and bus; model loads lazily."""

    @patch("src.core.speech_recognition.WhisperModel")
    def test_stores_config(self, mock_model_cls: MagicMock) -> None:
        cfg = ASRConfig()
        sr = SpeechRecognizer(config=cfg, bus=MessageBus())
        assert sr.config is cfg

    @patch("src.core.speech_recognition.WhisperModel")
    def test_stores_bus(self, mock_model_cls: MagicMock) -> None:
        bus = MessageBus()
        sr = SpeechRecognizer(config=ASRConfig(), bus=bus)
        assert sr.bus is bus

    @patch("src.core.speech_recognition.WhisperModel")
    def test_not_running_initially(self, mock_model_cls: MagicMock) -> None:
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        assert not sr.running

    @patch("src.core.speech_recognition.WhisperModel")
    def test_model_loaded_in_init(self, mock_model_cls: MagicMock) -> None:
        """The WhisperModel constructor should be called once during init."""
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        mock_model_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _decode_audio
# ---------------------------------------------------------------------------

class TestDecodeAudio:
    """_decode_audio must convert base64-encoded int16 → float32 in [-1, 1]."""

    @patch("src.core.speech_recognition.WhisperModel")
    def test_output_dtype_is_float32(self, mock_model_cls: MagicMock) -> None:
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        payload = _make_audio_payload()
        result = sr._decode_audio(payload)
        assert result.dtype == np.float32

    @patch("src.core.speech_recognition.WhisperModel")
    def test_output_is_1d(self, mock_model_cls: MagicMock) -> None:
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        payload = _make_audio_payload()
        result = sr._decode_audio(payload)
        assert result.ndim == 1

    @patch("src.core.speech_recognition.WhisperModel")
    def test_length_matches_input(self, mock_model_cls: MagicMock) -> None:
        samples = np.arange(512, dtype=np.int16)
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        result = sr._decode_audio(_make_audio_payload(samples=samples))
        assert len(result) == 512

    @patch("src.core.speech_recognition.WhisperModel")
    def test_normalised_range(self, mock_model_cls: MagicMock) -> None:
        """Full-scale int16 should map to approximately ±1.0."""
        samples = np.array([32767, -32768, 0], dtype=np.int16)
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        result = sr._decode_audio(_make_audio_payload(samples=samples))
        assert result[0] == pytest.approx(1.0, abs=1e-4)
        assert result[1] == pytest.approx(-1.0, abs=1e-4)
        assert result[2] == pytest.approx(0.0, abs=1e-4)

    @patch("src.core.speech_recognition.WhisperModel")
    def test_round_trip_fidelity(self, mock_model_cls: MagicMock) -> None:
        """Encode then decode should recover the original signal."""
        rng = np.random.default_rng(99)
        original_int16 = rng.integers(-20000, 20000, size=2048, dtype=np.int16)
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        decoded = sr._decode_audio(_make_audio_payload(samples=original_int16))
        # Re-quantise back to int16 for comparison.
        recovered = (decoded * 32768.0).astype(np.int16)
        np.testing.assert_array_equal(recovered, original_int16)


# ---------------------------------------------------------------------------
# _transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    """_transcribe wraps WhisperModel.transcribe and normalises output."""

    @patch("src.core.speech_recognition.WhisperModel")
    def test_returns_dict_with_text(self, mock_model_cls: MagicMock) -> None:
        # Build a mock segment (faster-whisper returns NamedTuple-like objects).
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Hello world"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98

        mock_instance = mock_model_cls.return_value
        mock_instance.transcribe.return_value = ([mock_segment], mock_info)

        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        audio = np.zeros(16000, dtype=np.float32)
        result = sr._transcribe(audio)

        assert "text" in result
        assert "Hello world" in result["text"]

    @patch("src.core.speech_recognition.WhisperModel")
    def test_returns_segments_list(self, mock_model_cls: MagicMock) -> None:
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 0.5
        mock_segment.text = " Hi"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_instance = mock_model_cls.return_value
        mock_instance.transcribe.return_value = ([mock_segment], mock_info)

        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        result = sr._transcribe(np.zeros(16000, dtype=np.float32))

        assert "segments" in result
        assert isinstance(result["segments"], list)
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == " Hi"

    @patch("src.core.speech_recognition.WhisperModel")
    def test_returns_language(self, mock_model_cls: MagicMock) -> None:
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_instance = mock_model_cls.return_value
        mock_instance.transcribe.return_value = ([], mock_info)

        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        result = sr._transcribe(np.zeros(16000, dtype=np.float32))

        assert result["language"] == "en"

    @patch("src.core.speech_recognition.WhisperModel")
    def test_empty_segments_returns_empty_text(self, mock_model_cls: MagicMock) -> None:
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_instance = mock_model_cls.return_value
        mock_instance.transcribe.return_value = ([], mock_info)

        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        result = sr._transcribe(np.zeros(16000, dtype=np.float32))

        assert result["text"] == ""
        assert result["segments"] == []


# ---------------------------------------------------------------------------
# Buffer accumulation logic
# ---------------------------------------------------------------------------

class TestBufferAccumulation:
    """Audio chunks must accumulate until min_audio_length is reached."""

    @patch("src.core.speech_recognition.WhisperModel")
    def test_buffer_starts_empty(self, mock_model_cls: MagicMock) -> None:
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        assert len(sr._buffer) == 0

    @patch("src.core.speech_recognition.WhisperModel")
    def test_buffer_length_calculation(self, mock_model_cls: MagicMock) -> None:
        """buffer_seconds property should reflect accumulated samples."""
        sr = SpeechRecognizer(
            config=ASRConfig(min_audio_length=1.0),
            bus=MessageBus(),
        )
        # Append 8000 samples at 16 kHz = 0.5 seconds.
        sr._buffer = list(np.zeros(8000, dtype=np.float32))
        assert sr.buffer_seconds == pytest.approx(0.5, abs=0.01)

    @patch("src.core.speech_recognition.WhisperModel")
    def test_buffer_not_ready_below_threshold(self, mock_model_cls: MagicMock) -> None:
        sr = SpeechRecognizer(
            config=ASRConfig(min_audio_length=1.0),
            bus=MessageBus(),
        )
        sr._buffer = list(np.zeros(8000, dtype=np.float32))  # 0.5 s
        assert not sr._buffer_ready()

    @patch("src.core.speech_recognition.WhisperModel")
    def test_buffer_ready_at_threshold(self, mock_model_cls: MagicMock) -> None:
        sr = SpeechRecognizer(
            config=ASRConfig(min_audio_length=1.0),
            bus=MessageBus(),
        )
        sr._buffer = list(np.zeros(16000, dtype=np.float32))  # 1.0 s
        assert sr._buffer_ready()


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------

class TestStartStop:
    """start() and stop() must manage the recognition loop cleanly."""

    @patch("src.core.speech_recognition.WhisperModel")
    def test_stop_before_start_is_safe(self, mock_model_cls: MagicMock) -> None:
        sr = SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
        sr.stop()  # Must not raise.
        assert not sr.running

    @patch("src.core.speech_recognition.WhisperModel")
    def test_start_sets_running_then_stop_clears(self, mock_model_cls: MagicMock) -> None:
        bus = MessageBus()
        sr = SpeechRecognizer(config=ASRConfig(), bus=bus)

        # Inject mock sockets so start() doesn't bind real ports.
        sr._publisher = MagicMock()
        sr._subscriber = MagicMock()

        # Make receive return None so the loop just spins.
        bus_receive_orig = bus.receive
        bus.receive = MagicMock(return_value=None)

        t = threading.Thread(target=sr.start, daemon=True)
        t.start()
        time.sleep(0.3)
        assert sr.running is True

        sr.stop()
        t.join(timeout=3)
        assert sr.running is False

        # Restore (just in case).
        bus.receive = bus_receive_orig


# ---------------------------------------------------------------------------
# End-to-end on the wire (mocked model, real ZeroMQ)
# ---------------------------------------------------------------------------

class TestEndToEndWire:
    """Full loop: publish audio → SpeechRecognizer → transcript on the wire."""

    AUDIO_TEST_PORT = 6600
    TRANSCRIPT_TEST_PORT = 6601

    @patch("src.core.speech_recognition.TRANSCRIPT_PORT", new=6601)
    @patch("src.core.speech_recognition.AUDIO_PORT", new=6600)
    @patch("src.core.speech_recognition.WhisperModel")
    def test_transcript_published_after_enough_audio(
        self, mock_model_cls: MagicMock
    ) -> None:
        """Feed enough audio for min_audio_length and verify a transcript."""
        # Set up mock model.
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Test transcript"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_instance = mock_model_cls.return_value
        mock_instance.transcribe.return_value = ([mock_segment], mock_info)

        bus = MessageBus()
        cfg = ASRConfig(min_audio_length=0.5)  # low threshold for speed
        sr = SpeechRecognizer(config=cfg, bus=bus)

        # Create audio publisher and transcript subscriber on test ports.
        audio_pub = bus.create_publisher(self.AUDIO_TEST_PORT)
        transcript_sub = bus.create_subscriber(
            ports=[self.TRANSCRIPT_TEST_PORT], topics=["transcript"],
        )
        time.sleep(0.3)  # slow-joiner

        # Start recogniser in background.
        t = threading.Thread(target=sr.start, daemon=True)
        t.start()
        time.sleep(0.3)  # let it subscribe and settle

        # Publish enough audio to cross the 0.5 s threshold.
        # 16000 * 0.5 = 8000 samples.  Each chunk = 1024 → need ~8 chunks.
        for _ in range(10):
            samples = np.zeros(1024, dtype=np.int16)
            bus.publish(audio_pub, topic="audio", data=_make_audio_payload(samples))
            time.sleep(0.02)

        # Wait for a transcript.
        result = bus.receive(transcript_sub, timeout_ms=5000)
        sr.stop()
        t.join(timeout=3)

        transcript_sub.close()
        audio_pub.close()

        assert result is not None
        topic, envelope = result
        assert topic == "transcript"
        assert "text" in envelope["data"]
        assert "Test transcript" in envelope["data"]["text"]
        assert "segments" in envelope["data"]
        assert "language" in envelope["data"]
        assert "timestamp" in envelope["data"]


# ---------------------------------------------------------------------------
# Model loading errors
# ---------------------------------------------------------------------------

class TestModelLoadingError:
    """Graceful failure when the Whisper model cannot be loaded."""

    @patch("src.core.speech_recognition.WhisperModel")
    def test_init_raises_on_model_load_failure(self, mock_model_cls: MagicMock) -> None:
        mock_model_cls.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            SpeechRecognizer(config=ASRConfig(), bus=MessageBus())
