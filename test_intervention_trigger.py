#!/usr/bin/env python3
"""Test script: publish a fake high-risk tactics message to trigger audio intervention.

Bypasses the full pipeline to verify that:
  1. audio_intervention subscribes to TACTIC_PORT (5558)
  2. Message envelope format matches content_analyzer output
  3. USB speaker plays a warning when risk_level is high/medium

Usage:
  1. STOP the full pipeline (port 5558 must be free):
     pkill -f 'src.core\|src.viz' || true
  2. Start ONLY audio_intervention (in a separate terminal):
     python -m src.core.audio_intervention
  3. Run this script:
     python test_intervention_trigger.py

Expected: USB speaker plays "Warning. This call shows signs of a scam..."
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import zmq

from src.core.message_bus import TACTIC_PORT, MessageBus


# Message format must match content_analyzer publish (topic "tactics")
FAKE_HIGH_RISK_PAYLOAD = {
    "tactics": {
        "urgency": 0.9,
        "authority": 0.85,
        "fear": 0.8,
        "isolation": 0.7,
        "financial": 0.95,
    },
    "risk_level": "high",
    "risk_score": 0.85,
    "risk_factors": [
        "Tier 1: 'read you the numbers on the back'",
        "Tier 2: gift card scenario (similarity 0.72)",
    ],
    "transcript": "Yes I will buy the gift cards right now and read you the codes",
    "word_count": 12,
    "inference_time_ms": 45.0,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}


def main() -> int:
    print(f"Publishing fake HIGH-RISK message to port {TACTIC_PORT} (topic: tactics)")
    print(f"Payload: risk_level={FAKE_HIGH_RISK_PAYLOAD['risk_level']}, risk_score={FAKE_HIGH_RISK_PAYLOAD['risk_score']:.2f}")
    print()

    bus = MessageBus()
    try:
        pub = bus.create_publisher(TACTIC_PORT)
    except zmq.ZMQError as e:
        if "Address already in use" in str(e):
            print("ERROR: Port 5558 in use. Stop the pipeline first:")
            print("  pkill -f 'src.core|src.viz'")
            print("Then start ONLY audio_intervention, then re-run this script.")
            return 1
        raise

    # Allow subscriber (audio_intervention) to connect; ZMQ slow-joiner
    time.sleep(1.0)

    bus.publish(pub, "tactics", FAKE_HIGH_RISK_PAYLOAD)
    print("Published. If audio_intervention is running, speaker should play warning within ~3 seconds.")
    print()

    pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
