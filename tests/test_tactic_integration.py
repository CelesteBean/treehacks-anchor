"""Integration test for TacticInferenceService.

Tests the full ZeroMQ pub/sub pipeline:

    1. Start ``TacticInferenceService`` in a background thread.
    2. Publish fake transcript messages (multipart protocol via MessageBus).
    3. Optionally publish a stress score.
    4. Wait for the inference cycle to fire.
    5. Receive the published ``tactics`` message and validate its schema.

Ports are monkey-patched to 16556/16557/16558 so this test can run
alongside the live pipeline without conflicts.

Usage::

    cd ~/treehacks-anchor && source venv/bin/activate
    python -m tests.test_tactic_integration
"""

from __future__ import annotations

import json
import logging
import sys
import time
import threading
from typing import Any

import zmq

# ---------------------------------------------------------------------------
# Monkey-patch port constants BEFORE importing the service so that the
# module-level ``from src.core.message_bus import TRANSCRIPT_PORT, ...``
# bindings inside ``tactic_inference`` point to test-isolated ports.
# ---------------------------------------------------------------------------
TEST_TRANSCRIPT_PORT = 16556
TEST_STRESS_PORT = 16557
TEST_TACTIC_PORT = 16558

import src.core.tactic_inference as _ti_mod  # noqa: E402

_ti_mod.TRANSCRIPT_PORT = TEST_TRANSCRIPT_PORT
_ti_mod.STRESS_PORT = TEST_STRESS_PORT
_ti_mod.TACTIC_PORT = TEST_TACTIC_PORT

from src.core.tactic_inference import TacticInferenceService  # noqa: E402
from src.core.message_bus import MessageBus  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def test_tactic_service() -> bool:
    """Run the full integration test.  Returns ``True`` on success."""

    bus = MessageBus()
    received: dict[str, Any] = {}
    error: str | None = None

    # === STEP 1: Create and start the service in a background thread ====
    # The constructor loads the Qwen model (~60 s on Jetson Orin Nano).
    logger.info("STEP 1: Creating TacticInferenceService (model loading)...")
    service = TacticInferenceService(inference_interval=3.0, bus=bus)
    logger.info("        Model loaded.  Starting service thread...")

    service_thread = threading.Thread(target=service.start, daemon=True)
    service_thread.start()

    # Give the service time to bind its PUB socket and register the poller.
    time.sleep(1)
    logger.info("        Service running.")

    # === STEP 2: Connect the output subscriber BEFORE publishing ========
    logger.info("STEP 2: Subscribing to tactic output on port %d...",
                TEST_TACTIC_PORT)
    sub_output = bus.create_subscriber(
        ports=[TEST_TACTIC_PORT], topics=["tactics"],
    )
    time.sleep(0.5)  # ZeroMQ slow-joiner
    logger.info("        Subscribed.")

    # === STEP 3: Publish fake transcripts (multipart wire protocol) =====
    logger.info("STEP 3: Publishing test transcripts on port %d...",
                TEST_TRANSCRIPT_PORT)

    pub_transcript = bus.create_publisher(TEST_TRANSCRIPT_PORT)
    time.sleep(0.5)  # slow-joiner

    test_messages = [
        "Hello?",
        "The IRS you said?",
        "I owe back taxes?",
        "Pay today or I'll be arrested?",
        "You want gift cards?",
    ]
    for text in test_messages:
        bus.publish(pub_transcript, topic="transcript", data={
            "text": text,
            "is_final": True,
        })
        logger.info("        Sent: %s", text)
        time.sleep(0.2)

    # Also send a stress score.
    pub_stress = bus.create_publisher(TEST_STRESS_PORT)
    time.sleep(0.5)
    bus.publish(pub_stress, topic="stress", data={"stress_score": 0.8})
    logger.info("        Sent stress: 0.8")

    # === STEP 4: Wait for the inference output ==========================
    logger.info("STEP 4: Waiting for inference result (up to 30 s)...")
    result = bus.receive(sub_output, timeout_ms=30_000)

    if result is None:
        error = "Timeout: no tactics message received within 30 s"
        logger.error("        TIMEOUT")
    else:
        topic, envelope = result
        received["topic"] = topic
        received["envelope"] = envelope
        logger.info("        Received message on topic '%s'", topic)

    # === STEP 5: Cleanup ================================================
    service.stop()
    service_thread.join(timeout=5)
    for sock in (sub_output, pub_transcript, pub_stress):
        sock.close()

    # === RESULTS ========================================================
    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    if error:
        print(f"STATUS: FAIL")
        print(f"ERROR:  {error}")
        return False

    envelope = received["envelope"]
    data = envelope.get("data", {})
    tactics = data.get("tactics", {})

    checks: dict[str, bool] = {
        "Has topic 'tactics'": envelope.get("topic") == "tactics",
        "Has timestamp": "timestamp" in envelope,
        "Has data.tactics dict": isinstance(tactics, dict),
        "Has data.risk_level": data.get("risk_level") in ("low", "medium", "high"),
        "Has 5 tactic keys": set(tactics.keys()) == {
            "urgency", "authority", "fear", "isolation", "financial",
        },
        "All scores in [0,1]": all(
            isinstance(v, (int, float)) and 0.0 <= v <= 1.0
            for v in tactics.values()
        ),
        "Has transcript_count": isinstance(data.get("transcript_count"), int),
        "Has inference_time_ms": isinstance(data.get("inference_time_ms"), (int, float)),
        "Risk is medium or high": data.get("risk_level") in ("medium", "high"),
    }

    all_pass = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check_name}")

    print()
    print(f"Tactics: {tactics}")
    print(f"Risk:    {data.get('risk_level')}")
    print(f"Stress:  {data.get('stress_score')}")
    print(f"Count:   {data.get('transcript_count')}")
    print(f"Time:    {data.get('inference_time_ms', 0):.0f} ms")
    print()

    if all_pass:
        print("STATUS: ALL TESTS PASSED")
    else:
        print("STATUS: SOME TESTS FAILED")

    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    success = test_tactic_service()
    sys.exit(0 if success else 1)
