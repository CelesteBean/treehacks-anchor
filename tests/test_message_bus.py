"""Unit tests for src.core.message_bus – ZeroMQ communication spine.

Tests cover:
    - Port constant values
    - MessageBus singleton zmq.Context behavior
    - Publisher / Subscriber socket creation
    - Publish / Receive round-trip with JSON validation
    - Receive timeout returns None
"""

import json
import time
import threading
from datetime import datetime, timezone

import pytest
import zmq

from src.core.message_bus import (
    AUDIO_PORT,
    TRANSCRIPT_PORT,
    STRESS_PORT,
    TACTIC_PORT,
    MessageBus,
)


# ---------------------------------------------------------------------------
# Port constants
# ---------------------------------------------------------------------------

class TestPortConstants:
    """Port constants must match the pipeline spec."""

    def test_audio_port(self) -> None:
        assert AUDIO_PORT == 5555

    def test_transcript_port(self) -> None:
        assert TRANSCRIPT_PORT == 5556

    def test_stress_port(self) -> None:
        assert STRESS_PORT == 5557

    def test_tactic_port(self) -> None:
        assert TACTIC_PORT == 5558


# ---------------------------------------------------------------------------
# MessageBus construction
# ---------------------------------------------------------------------------

class TestMessageBusInit:
    """MessageBus should use a singleton zmq.Context."""

    def test_default_config_path(self) -> None:
        bus = MessageBus()
        assert bus.config_path == "config/pipeline.yaml"

    def test_custom_config_path(self) -> None:
        bus = MessageBus(config_path="custom/path.yaml")
        assert bus.config_path == "custom/path.yaml"

    def test_context_is_zmq_context(self) -> None:
        bus = MessageBus()
        assert isinstance(bus.context, zmq.Context)

    def test_singleton_context_across_instances(self) -> None:
        """Two MessageBus instances must share the same zmq.Context."""
        bus_a = MessageBus()
        bus_b = MessageBus()
        assert bus_a.context is bus_b.context


# ---------------------------------------------------------------------------
# Socket creation
# ---------------------------------------------------------------------------

class TestSocketCreation:
    """Publisher and subscriber sockets must bind/connect correctly."""

    @pytest.fixture(autouse=True)
    def _bus(self) -> None:
        self.bus = MessageBus()

    def test_create_publisher_returns_pub_socket(self) -> None:
        pub = self.bus.create_publisher(port=6100)
        try:
            assert pub.type == zmq.PUB
        finally:
            pub.close()

    def test_create_subscriber_returns_sub_socket(self) -> None:
        # Need a publisher first so the port is bound.
        pub = self.bus.create_publisher(port=6101)
        try:
            sub = self.bus.create_subscriber(ports=[6101])
            try:
                assert sub.type == zmq.SUB
            finally:
                sub.close()
        finally:
            pub.close()

    def test_subscriber_with_topic_filter(self) -> None:
        pub = self.bus.create_publisher(port=6102)
        try:
            sub = self.bus.create_subscriber(ports=[6102], topics=["audio"])
            try:
                assert sub.type == zmq.SUB
            finally:
                sub.close()
        finally:
            pub.close()


# ---------------------------------------------------------------------------
# Publish / Receive round-trip
# ---------------------------------------------------------------------------

class TestPubSubRoundTrip:
    """Messages must survive a publish -> receive round-trip intact."""

    # Use a unique port range to avoid collisions with other test classes.
    PORT = 6200

    @pytest.fixture(autouse=True)
    def _sockets(self) -> None:
        self.bus = MessageBus()
        self.pub = self.bus.create_publisher(port=self.PORT)
        self.sub = self.bus.create_subscriber(ports=[self.PORT])
        # Allow the ZeroMQ "slow joiner" handshake to complete.
        time.sleep(0.3)
        yield
        self.sub.close()
        self.pub.close()

    def test_round_trip_data_integrity(self) -> None:
        payload = {"level": 0.42, "source": "mic0"}
        self.bus.publish(self.pub, topic="audio", data=payload)

        result = self.bus.receive(self.sub, timeout_ms=2000)
        assert result is not None

        topic, message = result
        assert topic == "audio"
        assert message["data"] == payload

    def test_message_has_required_fields(self) -> None:
        self.bus.publish(self.pub, topic="stress", data={"value": 1})

        result = self.bus.receive(self.sub, timeout_ms=2000)
        assert result is not None

        _, message = result
        assert "timestamp" in message
        assert "topic" in message
        assert "data" in message

    def test_timestamp_is_iso8601(self) -> None:
        self.bus.publish(self.pub, topic="audio", data={})

        result = self.bus.receive(self.sub, timeout_ms=2000)
        assert result is not None

        _, message = result
        # datetime.fromisoformat should not raise for a valid ISO 8601 string.
        parsed = datetime.fromisoformat(message["timestamp"])
        assert parsed.tzinfo is not None  # Must include timezone

    def test_topic_in_envelope_matches(self) -> None:
        self.bus.publish(self.pub, topic="tactic", data={"id": 3})

        result = self.bus.receive(self.sub, timeout_ms=2000)
        assert result is not None

        topic, message = result
        assert topic == "tactic"
        assert message["topic"] == "tactic"


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------

class TestReceiveTimeout:
    """receive() must return None when no message arrives within timeout."""

    @pytest.fixture(autouse=True)
    def _sockets(self) -> None:
        self.bus = MessageBus()
        self.pub = self.bus.create_publisher(port=6300)
        self.sub = self.bus.create_subscriber(ports=[self.PORT])
        yield
        self.sub.close()
        self.pub.close()

    PORT = 6300

    def test_returns_none_on_timeout(self) -> None:
        # No messages published — should return None quickly.
        result = self.bus.receive(self.sub, timeout_ms=200)
        assert result is None


# ---------------------------------------------------------------------------
# Multi-message delivery
# ---------------------------------------------------------------------------

class TestMultiMessage:
    """Multiple messages should all be delivered in order."""

    PORT = 6400

    @pytest.fixture(autouse=True)
    def _sockets(self) -> None:
        self.bus = MessageBus()
        self.pub = self.bus.create_publisher(port=self.PORT)
        self.sub = self.bus.create_subscriber(ports=[self.PORT])
        time.sleep(0.3)
        yield
        self.sub.close()
        self.pub.close()

    def test_five_messages_received(self) -> None:
        count = 5
        for i in range(count):
            self.bus.publish(self.pub, topic="audio", data={"seq": i})

        received: list[dict] = []
        for _ in range(count):
            result = self.bus.receive(self.sub, timeout_ms=2000)
            assert result is not None
            _, msg = result
            received.append(msg)

        assert len(received) == count
        for i, msg in enumerate(received):
            assert msg["data"]["seq"] == i
