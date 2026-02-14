"""Unit tests for src.viz.judges_window – real-time demo dashboard.

Tests cover:
    - compute_rms: base64-encoded int16 → RMS float
    - create_app: Flask app factory returns valid app + socketio
    - Index route serves HTML with expected elements
    - SocketIO event emission for each topic type
    - Audio chunk throttling (emit every 5th chunk)
    - zmq_listener message routing

ZeroMQ is mocked — no pipeline stages need to be running.
"""

from __future__ import annotations

import base64
import math
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.viz.judges_window import (
    AUDIO_EMIT_INTERVAL,
    compute_rms,
    create_app,
    zmq_listener,
)


# ---------------------------------------------------------------------------
# compute_rms
# ---------------------------------------------------------------------------

class TestComputeRms:
    """compute_rms must decode base64 int16 and return the RMS level."""

    def test_silence_returns_zero(self) -> None:
        samples = np.zeros(1024, dtype=np.int16)
        b64 = base64.b64encode(samples.tobytes()).decode("ascii")
        assert compute_rms(b64) == pytest.approx(0.0, abs=1e-6)

    def test_known_signal(self) -> None:
        """A constant ±16384 (half-scale int16) signal should give RMS ≈ 0.5."""
        samples = np.full(1024, 16384, dtype=np.int16)
        b64 = base64.b64encode(samples.tobytes()).decode("ascii")
        rms = compute_rms(b64)
        # 16384 / 32768 = 0.5
        assert rms == pytest.approx(0.5, abs=0.01)

    def test_full_scale(self) -> None:
        samples = np.full(512, 32767, dtype=np.int16)
        b64 = base64.b64encode(samples.tobytes()).decode("ascii")
        rms = compute_rms(b64)
        assert rms == pytest.approx(1.0, abs=0.01)

    def test_returns_float(self) -> None:
        samples = np.zeros(128, dtype=np.int16)
        b64 = base64.b64encode(samples.tobytes()).decode("ascii")
        assert isinstance(compute_rms(b64), float)


# ---------------------------------------------------------------------------
# Audio emit interval constant
# ---------------------------------------------------------------------------

class TestAudioEmitInterval:
    """The throttle constant must be 5."""

    def test_interval_is_five(self) -> None:
        assert AUDIO_EMIT_INTERVAL == 5


# ---------------------------------------------------------------------------
# create_app factory
# ---------------------------------------------------------------------------

class TestCreateApp:
    """create_app() must return a usable Flask + SocketIO pair."""

    def test_returns_tuple(self) -> None:
        app, sio = create_app()
        assert app is not None
        assert sio is not None

    def test_flask_app_name(self) -> None:
        app, _ = create_app()
        assert app.name == "src.viz.judges_window"


# ---------------------------------------------------------------------------
# Index route
# ---------------------------------------------------------------------------

class TestIndexRoute:
    """GET / must serve the dashboard HTML with key UI elements."""

    @pytest.fixture(autouse=True)
    def _app(self) -> None:
        app, self.sio = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_status_200(self) -> None:
        resp = self.client.get("/")
        assert resp.status_code == 200

    def test_contains_title(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "ANCHOR" in html

    def test_contains_tagline(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "We only hear one side" in html

    def test_contains_transcript_section(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "transcript" in html.lower()

    def test_contains_audio_level_section(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "audio" in html.lower()

    def test_contains_stress_section(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "stress" in html.lower()

    def test_contains_tactic_section(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "tactic" in html.lower()

    def test_contains_socketio_client(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "socket.io" in html.lower()

    def test_contains_dark_theme_bg(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "#0a0a0a" in html

    def test_contains_accent_color(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "#64ffda" in html

    def test_contains_waiting_state(self) -> None:
        resp = self.client.get("/")
        html = resp.data.decode()
        assert "Waiting" in html


# ---------------------------------------------------------------------------
# SocketIO connection (test client)
# ---------------------------------------------------------------------------

class TestSocketIOConnection:
    """Flask-SocketIO test client must be able to connect."""

    def test_connect_succeeds(self) -> None:
        app, sio = create_app()
        app.config["TESTING"] = True
        test_client = sio.test_client(app)
        assert test_client.is_connected()
        test_client.disconnect()


# ---------------------------------------------------------------------------
# zmq_listener message routing
# ---------------------------------------------------------------------------

class TestZmqListenerRouting:
    """zmq_listener must route ZMQ messages to the correct SocketIO events.

    We mock both the MessageBus and SocketIO to verify routing in isolation.
    """

    @staticmethod
    def _make_envelope(topic: str, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": "2026-02-14T12:00:00+00:00",
            "topic": topic,
            "data": data,
        }

    def _run_listener_with_messages(
        self,
        messages: list[tuple[str, dict[str, Any]]],
    ) -> MagicMock:
        """Feed *messages* into zmq_listener and return the mock socketio."""
        mock_sio = MagicMock()
        mock_bus = MagicMock()

        # receive() returns messages one by one, then raises to exit.
        results = [
            (topic, self._make_envelope(topic, data))
            for topic, data in messages
        ]
        # Append a KeyboardInterrupt to break the infinite loop.
        mock_bus.receive.side_effect = [*results, KeyboardInterrupt]

        try:
            zmq_listener(mock_sio, bus=mock_bus)
        except KeyboardInterrupt:
            pass

        return mock_sio

    def test_transcript_emitted(self) -> None:
        mock_sio = self._run_listener_with_messages([
            ("transcript", {"text": "hello", "timestamp": "t"}),
        ])
        mock_sio.emit.assert_any_call("transcript", {
            "text": "hello",
            "timestamp": "t",
        })

    def test_stress_emitted(self) -> None:
        mock_sio = self._run_listener_with_messages([
            ("stress", {"score": 0.72, "indicators": {"jitter": 0.1}}),
        ])
        mock_sio.emit.assert_any_call("stress", {
            "score": 0.72,
            "indicators": {"jitter": 0.1},
            "timestamp": "2026-02-14T12:00:00+00:00",
        })

    def test_tactic_emitted(self) -> None:
        mock_sio = self._run_listener_with_messages([
            ("tactic", {"risk_score": 0.9, "active_tactics": ["urgency"]}),
        ])
        mock_sio.emit.assert_any_call("tactic", {
            "risk_score": 0.9,
            "active_tactics": ["urgency"],
            "timestamp": "2026-02-14T12:00:00+00:00",
        })

    def test_audio_throttled_to_every_5th(self) -> None:
        """Only the 5th, 10th, … audio chunks should trigger an emit."""
        samples = np.zeros(1024, dtype=np.int16)
        b64 = base64.b64encode(samples.tobytes()).decode("ascii")
        audio_data = {"samples": b64, "timestamp": "t", "sample_rate": 16000}

        # Send exactly 5 audio chunks → expect 1 emit.
        mock_sio = self._run_listener_with_messages(
            [("audio", audio_data)] * 5
        )
        audio_calls = [
            c for c in mock_sio.emit.call_args_list
            if c[0][0] == "audio_level"
        ]
        assert len(audio_calls) == 1

    def test_audio_not_emitted_below_interval(self) -> None:
        """Fewer than AUDIO_EMIT_INTERVAL chunks should produce no emit."""
        samples = np.zeros(1024, dtype=np.int16)
        b64 = base64.b64encode(samples.tobytes()).decode("ascii")
        audio_data = {"samples": b64, "timestamp": "t", "sample_rate": 16000}

        mock_sio = self._run_listener_with_messages(
            [("audio", audio_data)] * 4
        )
        audio_calls = [
            c for c in mock_sio.emit.call_args_list
            if c[0][0] == "audio_level"
        ]
        assert len(audio_calls) == 0

    def test_audio_emit_contains_rms(self) -> None:
        """The emitted audio_level event must include an 'rms' key."""
        samples = np.full(1024, 16384, dtype=np.int16)
        b64 = base64.b64encode(samples.tobytes()).decode("ascii")
        audio_data = {"samples": b64, "timestamp": "t", "sample_rate": 16000}

        mock_sio = self._run_listener_with_messages(
            [("audio", audio_data)] * 5
        )
        audio_calls = [
            c for c in mock_sio.emit.call_args_list
            if c[0][0] == "audio_level"
        ]
        assert len(audio_calls) == 1
        payload = audio_calls[0][0][1]
        assert "rms" in payload
        assert payload["rms"] == pytest.approx(0.5, abs=0.01)
