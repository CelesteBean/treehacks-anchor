"""ZeroMQ message bus — communication spine for the Anchor pipeline.

Each pipeline stage (audio capture, transcription, stress detection,
tactic inference) communicates over ZeroMQ PUB/SUB sockets on localhost.
Messages are JSON-encoded with a standard envelope:

    {"timestamp": "<ISO 8601>", "topic": "<str>", "data": {…}}

The publisher sends a two-frame ZeroMQ message:
    Frame 0 (topic):  UTF-8 topic string used for SUB filtering.
    Frame 1 (body):   JSON-encoded envelope.

Usage:
    bus  = MessageBus()
    pub  = bus.create_publisher(AUDIO_PORT)
    sub  = bus.create_subscriber([AUDIO_PORT])
    bus.publish(pub, "audio", {"level": 0.8})
    result = bus.receive(sub, timeout_ms=500)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import zmq

# ---------------------------------------------------------------------------
# Port constants — one per pipeline stage
# ---------------------------------------------------------------------------

AUDIO_PORT: int = 5555
"""Raw audio chunks from the microphone capture stage."""

TRANSCRIPT_PORT: int = 5556
"""Transcribed text segments from the speech-recognition stage."""

STRESS_PORT: int = 5557
"""Vocal-stress analysis results."""

TACTIC_PORT: int = 5558
"""Detected scam-tactic classifications."""

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class MessageBus:
    """Thin wrapper around ZeroMQ PUB/SUB for inter-stage communication.

    A singleton ``zmq.Context`` is shared across all ``MessageBus`` instances
    within the same process, which is the recommended ZeroMQ pattern (one
    context per process; the context is thread-safe).

    Parameters
    ----------
    config_path:
        Path to an optional YAML pipeline configuration file.  Currently
        stored for future use; the bus works without it.
    """

    # Class-level singleton — one zmq.Context per process.
    _context: zmq.Context | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config_path: str = "config/pipeline.yaml") -> None:
        self.config_path: str = config_path
        self.context: zmq.Context = self._get_context()

    # -- Context management --------------------------------------------------

    @classmethod
    def _get_context(cls) -> zmq.Context:
        """Return the process-wide singleton ``zmq.Context``.

        Thread-safe via a class-level lock.  ``zmq.Context`` itself is
        thread-safe once created, so this lock only guards instantiation.
        """
        if cls._context is None:
            with cls._lock:
                # Double-checked locking.
                if cls._context is None:
                    cls._context = zmq.Context()
                    logger.debug("Created new zmq.Context")
        return cls._context

    # -- Socket factories ----------------------------------------------------

    def create_publisher(self, port: int) -> zmq.Socket:
        """Create and bind a PUB socket on *port* (TCP, localhost).

        Parameters
        ----------
        port:
            TCP port number to ``bind`` to on ``127.0.0.1``.

        Returns
        -------
        zmq.Socket
            A bound ``zmq.PUB`` socket ready for :meth:`publish`.
        """
        socket: zmq.Socket = self.context.socket(zmq.PUB)
        socket.bind(f"tcp://127.0.0.1:{port}")
        logger.info("PUB socket bound on port %d", port)
        return socket

    def create_subscriber(
        self,
        ports: list[int],
        topics: list[str] | None = None,
    ) -> zmq.Socket:
        """Create a SUB socket connected to one or more publisher *ports*.

        Parameters
        ----------
        ports:
            List of TCP ports (on ``127.0.0.1``) to connect to.
        topics:
            Topic strings to subscribe to.  An empty string (``""``)
            subscribes to **all** topics — this is the default.

        Returns
        -------
        zmq.Socket
            A connected ``zmq.SUB`` socket ready for :meth:`receive`.
        """
        if topics is None:
            topics = [""]

        socket: zmq.Socket = self.context.socket(zmq.SUB)
        for port in ports:
            socket.connect(f"tcp://127.0.0.1:{port}")
            logger.debug("SUB socket connected to port %d", port)

        for topic in topics:
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            logger.debug("Subscribed to topic %r", topic)

        return socket

    # -- Publish / Receive ---------------------------------------------------

    def publish(self, socket: zmq.Socket, topic: str, data: dict[str, Any]) -> None:
        """Publish a message on *socket* under *topic*.

        The message is sent as two ZeroMQ frames:

        1. **Topic frame** — UTF-8 encoded topic string (used by SUB filters).
        2. **Body frame** — JSON-encoded envelope containing ``timestamp``,
           ``topic``, and ``data``.

        Parameters
        ----------
        socket:
            A ``zmq.PUB`` socket obtained from :meth:`create_publisher`.
        topic:
            Routing topic (e.g. ``"audio"``, ``"stress"``).
        data:
            Arbitrary JSON-serialisable payload dict.
        """
        envelope: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "data": data,
        }
        payload: str = json.dumps(envelope)
        socket.send_multipart(
            [topic.encode("utf-8"), payload.encode("utf-8")]
        )
        logger.debug("Published [%s]: %s", topic, payload[:120])

    def receive(
        self,
        socket: zmq.Socket,
        timeout_ms: int = 1000,
    ) -> tuple[str, dict[str, Any]] | None:
        """Wait up to *timeout_ms* for a message on *socket*.

        Parameters
        ----------
        socket:
            A ``zmq.SUB`` socket obtained from :meth:`create_subscriber`.
        timeout_ms:
            Maximum time (milliseconds) to wait before returning ``None``.

        Returns
        -------
        tuple[str, dict] | None
            ``(topic, envelope_dict)`` on success, or ``None`` on timeout.
        """
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        events = dict(poller.poll(timeout=timeout_ms))
        if socket not in events:
            return None

        frames: list[bytes] = socket.recv_multipart()
        topic: str = frames[0].decode("utf-8")
        message: dict[str, Any] = json.loads(frames[1].decode("utf-8"))
        return topic, message


# ---------------------------------------------------------------------------
# Standalone integration smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TEST_PORT: int = AUDIO_PORT
    MESSAGE_COUNT: int = 5

    bus = MessageBus()
    pub = bus.create_publisher(TEST_PORT)
    sub = bus.create_subscriber([TEST_PORT])

    # Allow the ZeroMQ slow-joiner handshake to settle.
    time.sleep(0.5)

    received_messages: list[dict[str, Any]] = []
    errors: list[str] = []

    def _subscriber_worker() -> None:
        """Background thread that collects messages from the subscriber."""
        for _ in range(MESSAGE_COUNT):
            result = bus.receive(sub, timeout_ms=3000)
            if result is None:
                errors.append("Timed out waiting for message")
                continue
            _, msg = result
            received_messages.append(msg)

    worker = threading.Thread(target=_subscriber_worker, daemon=True)
    worker.start()

    # Publish test messages.
    for seq in range(MESSAGE_COUNT):
        bus.publish(pub, topic="audio", data={"seq": seq, "test": True})
        logger.info("Sent message seq=%d", seq)
        time.sleep(0.05)  # Small gap to keep things deterministic.

    worker.join(timeout=10)

    # -- Verify results ------------------------------------------------------
    print(f"\nSent:     {MESSAGE_COUNT}")
    print(f"Received: {len(received_messages)}")
    print(f"Errors:   {len(errors)}")

    assert len(errors) == 0, f"Errors encountered: {errors}"
    assert len(received_messages) == MESSAGE_COUNT, (
        f"Expected {MESSAGE_COUNT} messages, got {len(received_messages)}"
    )

    for i, msg in enumerate(received_messages):
        assert msg["data"]["seq"] == i, f"Sequence mismatch at index {i}"
        assert "timestamp" in msg, "Missing timestamp"
        assert msg["topic"] == "audio", f"Wrong topic: {msg['topic']}"
        # Validate ISO 8601 timestamp
        datetime.fromisoformat(msg["timestamp"])
        print(f"  [{i}] OK — seq={msg['data']['seq']}, ts={msg['timestamp']}")

    print("\nAll checks passed.")

    # Cleanup
    sub.close()
    pub.close()
