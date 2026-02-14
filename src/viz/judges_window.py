"""Real-time dashboard for the Anchor phone-scam detection demo.

Serves a dark-themed web UI on port 8080 that displays live data from
every pipeline stage — audio levels, transcription, vocal-stress scores,
and scam-tactic detection — streamed over WebSockets via Flask-SocketIO.

Architecture
------------
::

    ZMQ ports 5555-5558       SocketIO            Browser
    ┌──────────────────┐     ┌──────────┐      ┌──────────────┐
    │ audio / transcript│ ──→│ Flask +  │ ═WS═→│ Dark-themed  │
    │ stress / tactic   │    │ SocketIO │      │  dashboard   │
    └──────────────────┘     └──────────┘      └──────────────┘

A background daemon thread subscribes to all four ZMQ publisher ports,
decodes each message, and emits the appropriate SocketIO event.  The
browser reconnects automatically on disconnect.

Usage::

    python -m src.viz.judges_window          # from project root
    python src/viz/judges_window.py          # direct execution
"""

from __future__ import annotations

# -- Path fixup for direct execution ----------------------------------------
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# -- Standard / third-party imports -----------------------------------------
import base64
import logging
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO

from src.core.message_bus import (
    AUDIO_PORT,
    STRESS_PORT,
    TACTIC_PORT,
    TRANSCRIPT_PORT,
    MessageBus,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DASHBOARD_HOST: str = "0.0.0.0"  # noqa: S104 – intentional for LAN demo
DASHBOARD_PORT: int = 8080

AUDIO_EMIT_INTERVAL: int = 5
"""Emit an ``audio_level`` SocketIO event every *N*-th audio chunk.

At 16 kHz / 1024-sample chunks ≈ 15.6 chunks/s, emitting every 5th
yields ~3 updates/s — smooth without flooding the browser.
"""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_rms(b64_samples: str) -> float:
    """Decode base64-encoded int16 PCM and return the RMS level in [0, 1].

    Parameters
    ----------
    b64_samples:
        Base64-encoded bytes of little-endian int16 samples (matching
        the wire format produced by ``audio_capture``).

    Returns
    -------
    float
        Root Mean Square of the normalised signal.  0.0 = silence,
        1.0 = full-scale.
    """
    raw: bytes = base64.b64decode(b64_samples)
    samples: np.ndarray = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    rms: float = float(np.sqrt(np.mean(samples ** 2)))
    return rms


# ---------------------------------------------------------------------------
# HTML template (inline)
# ---------------------------------------------------------------------------

DASHBOARD_HTML: str = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>.anchor — Judge's Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  /* ── Reset & Base ─────────────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: #1E272E;
    color: #FFFFFF;
    font-size: 16px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Palette ──────────────────────────────────────────────────── */
  :root {
    --bg:      #1E272E;
    --surface: rgba(255, 255, 255, 0.05);
    --border:  rgba(178, 190, 195, 0.20);
    --text:    #FFFFFF;
    --dim:     #B2BEC3;
    --accent:  #5BBFB3;
    --danger:  #E17055;
    --success: #00B894;
    --warn:    #FDCB6E;
  }

  /* ── Pulsing status-dot animation ───────────────────────────── */
  @keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(91, 191, 179, 0.5); }
    50%      { box-shadow: 0 0 0 6px rgba(91, 191, 179, 0); }
  }

  /* ── Header ───────────────────────────────────────────────────── */
  header {
    text-align: center;
    padding: 24px 16px 12px;
    border-bottom: 1px solid var(--border);
  }
  header h1 {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--accent);
    margin-bottom: 4px;
  }
  header p {
    font-size: 0.875rem;
    color: var(--dim);
    font-weight: 400;
  }
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--danger);
    margin-right: 8px;
    vertical-align: middle;
    transition: background 0.3s;
  }
  .status-dot.connected {
    background: var(--accent);
    animation: pulse-dot 2s ease-in-out infinite;
  }

  /* ── Grid layout ──────────────────────────────────────────────── */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr auto;
    gap: 12px;
    padding: 12px;
    height: calc(100vh - 100px);
  }

  /* ── Cards ────────────────────────────────────────────────────── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .card h2 {
    font-size: 14px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--dim);
    margin-bottom: 10px;
  }

  /* ── Transcript panel ─────────────────────────────────────────── */
  #transcript-card { grid-column: 1; grid-row: 1; }
  #transcript-lines {
    flex: 1;
    overflow-y: auto;
    font-size: 1rem;
    line-height: 1.6;
    color: var(--dim);
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }
  #transcript-lines .line { padding: 2px 0; color: var(--text); }
  #transcript-lines .line .ts {
    color: var(--dim);
    font-size: 0.75rem;
    margin-right: 8px;
  }
  .waiting-msg {
    color: var(--dim);
    font-style: italic;
  }

  /* ── Audio level panel ────────────────────────────────────────── */
  #audio-card { grid-column: 2; grid-row: 1; }
  .meter-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 12px;
  }
  .meter-bar-bg {
    width: 100%;
    height: 28px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 14px;
    overflow: hidden;
    position: relative;
  }
  .meter-bar-fill {
    height: 100%;
    width: 0%;
    background: var(--accent);
    border-radius: 14px;
    transition: width 0.15s ease-out;
  }
  .rms-value {
    font-size: 2.5rem;
    font-weight: 600;
    color: var(--accent);
  }
  .rms-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .chunk-counter {
    font-size: 0.75rem;
    color: var(--dim);
  }

  /* ── Bottom cards ─────────────────────────────────────────────── */
  #stress-card { grid-column: 1; grid-row: 2; min-height: 140px; }
  #tactic-card { grid-column: 2; grid-row: 2; min-height: 140px; }

  .score-big {
    font-size: 2.2rem;
    font-weight: 600;
    transition: color 0.3s;
  }
  .score-low    { color: var(--accent); }
  .score-medium { color: var(--warn); }
  .score-high   { color: var(--danger); }

  .indicators {
    font-size: 0.875rem;
    color: var(--dim);
    margin-top: 6px;
    line-height: 1.5;
  }

  .tactic-list {
    list-style: none;
    margin-top: 8px;
  }
  .tactic-list li {
    padding: 6px 10px;
    margin: 4px 0;
    background: rgba(225, 112, 85, 0.08);
    border-left: 3px solid var(--danger);
    border-radius: 0 4px 4px 0;
    font-size: 0.875rem;
    color: var(--text);
  }

  /* ── Alert glow for high-risk tactic card ──────────────────────── */
  #tactic-card.alert-active {
    border-color: var(--danger);
    box-shadow: 0 0 12px rgba(225, 112, 85, 0.25);
  }
</style>
</head>
<body>

<header>
  <h1><span class="status-dot" id="status-dot"></span>.anchor</h1>
  <p>We only hear one side. Watch what we know.</p>
</header>

<div class="grid">
  <!-- Transcript -->
  <div class="card" id="transcript-card">
    <h2>Live Transcript</h2>
    <div id="transcript-lines">
      <div class="waiting-msg" id="transcript-waiting">Waiting for transcript data&hellip;</div>
    </div>
  </div>

  <!-- Audio Level -->
  <div class="card" id="audio-card">
    <h2>Audio Level</h2>
    <div class="meter-container">
      <div class="rms-value" id="rms-value">&mdash;</div>
      <div class="rms-label">RMS Level</div>
      <div class="meter-bar-bg">
        <div class="meter-bar-fill" id="meter-fill"></div>
      </div>
      <div class="chunk-counter" id="chunk-counter">Waiting for audio data&hellip;</div>
    </div>
  </div>

  <!-- Stress -->
  <div class="card" id="stress-card">
    <h2>Vocal Stress</h2>
    <div class="score-big score-low" id="stress-score">&mdash;</div>
    <div class="indicators" id="stress-indicators">Waiting for stress data&hellip;</div>
  </div>

  <!-- Tactic -->
  <div class="card" id="tactic-card">
    <h2>Scam Tactic Detection</h2>
    <div class="score-big score-low" id="risk-score">&mdash;</div>
    <ul class="tactic-list" id="tactic-list">
      <li class="waiting-msg">Waiting for tactic data&hellip;</li>
    </ul>
  </div>
</div>

<!-- Socket.IO client -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script>
(function() {
  "use strict";

  /* ── Connect with auto-reconnect ─────────────────────────────── */
  const socket = io({
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
  });

  const dot = document.getElementById("status-dot");
  socket.on("connect",    () => { dot.classList.add("connected"); });
  socket.on("disconnect", () => { dot.classList.remove("connected"); });

  /* ── Transcript ──────────────────────────────────────────────── */
  const MAX_LINES = 10;
  const transcriptEl = document.getElementById("transcript-lines");
  const transcriptWait = document.getElementById("transcript-waiting");

  socket.on("transcript", (data) => {
    if (transcriptWait) transcriptWait.remove();
    const line = document.createElement("div");
    line.className = "line";
    const ts = data.timestamp ? data.timestamp.split("T")[1].substring(0, 8) : "";
    line.innerHTML = '<span class="ts">' + ts + '</span>' + escapeHtml(data.text);
    transcriptEl.appendChild(line);

    /* Keep only last MAX_LINES */
    while (transcriptEl.children.length > MAX_LINES) {
      transcriptEl.removeChild(transcriptEl.firstChild);
    }
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  });

  /* ── Audio Level ─────────────────────────────────────────────── */
  const rmsEl    = document.getElementById("rms-value");
  const meterEl  = document.getElementById("meter-fill");
  const counterEl = document.getElementById("chunk-counter");
  let audioChunks = 0;

  socket.on("audio_level", (data) => {
    audioChunks++;
    const rms = data.rms || 0;

    /* Scale RMS to a visible percentage.
       Typical values: 0.01 = quiet, 0.1 = normal speech, 0.3 = loud.
       Multiply by 500 so the bar moves meaningfully, cap at 100%. */
    const pct = Math.min(100, rms * 500);

    rmsEl.textContent = pct.toFixed(1) + "%";
    meterEl.style.width = pct.toFixed(1) + "%";
    counterEl.textContent = "Chunks: " + audioChunks;

    /* Color the value based on scaled level */
    rmsEl.style.color = pct > 80 ? "var(--danger)" :
                        pct > 40 ? "var(--warn)" : "var(--accent)";
  });

  /* ── Stress ──────────────────────────────────────────────────── */
  const stressScoreEl = document.getElementById("stress-score");
  const stressIndEl   = document.getElementById("stress-indicators");

  socket.on("stress", (data) => {
    const score = data.score != null ? data.score : 0;
    stressScoreEl.textContent = score.toFixed(2);
    stressScoreEl.className = "score-big " +
      (score > 0.7 ? "score-high" : score > 0.4 ? "score-medium" : "score-low");

    const parts = [];
    if (data.arousal != null)    parts.push("arousal: " + data.arousal.toFixed(2));
    if (data.valence != null)    parts.push("valence: " + data.valence.toFixed(2));
    if (data.dominance != null)  parts.push("dominance: " + data.dominance.toFixed(2));
    if (data.confidence != null) parts.push("confidence: " + data.confidence.toFixed(2));
    stressIndEl.textContent = parts.length > 0 ? parts.join("  \u00b7  ") : "No indicators";
  });

  /* ── Tactic ──────────────────────────────────────────────────── */
  const riskScoreEl = document.getElementById("risk-score");
  const tacticListEl = document.getElementById("tactic-list");

  socket.on("tactic", (data) => {
    const risk = data.risk_score != null ? data.risk_score : 0;
    riskScoreEl.textContent = risk.toFixed(2);
    riskScoreEl.className = "score-big " +
      (risk > 0.7 ? "score-high" : risk > 0.4 ? "score-medium" : "score-low");

    tacticListEl.innerHTML = "";
    if (data.active_tactics && data.active_tactics.length > 0) {
      data.active_tactics.forEach((t) => {
        const li = document.createElement("li");
        li.textContent = typeof t === "string" ? t : JSON.stringify(t);
        tacticListEl.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "No active tactics detected";
      li.style.color = "var(--accent)";
      li.style.borderLeftColor = "var(--accent)";
      li.style.background = "rgba(91, 191, 179, 0.06)";
      tacticListEl.appendChild(li);
    }

    /* Toggle coral alert glow on tactic card when risk is high */
    const tacticCard = document.getElementById("tactic-card");
    if (risk > 0.7) {
      tacticCard.classList.add("alert-active");
    } else {
      tacticCard.classList.remove("alert-active");
    }
  });

  /* ── Utility ─────────────────────────────────────────────────── */
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(text || ""));
    return div.innerHTML;
  }
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> tuple[Flask, SocketIO]:
    """Create and configure the Flask + SocketIO application.

    Returns
    -------
    tuple[Flask, SocketIO]
        The Flask app and its SocketIO wrapper.  Neither is started —
        call ``socketio.run(app, ...)`` to serve.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "anchor-demo"  # non-secret; local demo only

    socketio = SocketIO(
        app,
        cors_allowed_origins="*",  # LAN demo — no auth
        async_mode="threading",
    )

    @app.route("/")
    def index() -> str:
        return render_template_string(DASHBOARD_HTML)

    return app, socketio


# ---------------------------------------------------------------------------
# ZeroMQ → SocketIO bridge (background thread)
# ---------------------------------------------------------------------------


def zmq_listener(socketio: SocketIO, bus: MessageBus | None = None) -> None:
    """Subscribe to all pipeline ZMQ ports and emit SocketIO events.

    Designed to run as a daemon thread.  Loops forever until the process
    exits.

    Parameters
    ----------
    socketio:
        The SocketIO instance to emit events on.
    bus:
        Optional ``MessageBus`` override (useful for testing).
    """
    if bus is None:
        bus = MessageBus()

    all_ports = [AUDIO_PORT, TRANSCRIPT_PORT, STRESS_PORT, TACTIC_PORT]
    sub = bus.create_subscriber(ports=all_ports)

    # ── FIX: slow-joiner sleep ──────────────────────────────────────
    # The subscriber needs time to complete the subscription handshake
    # with any publishers that are already bound.  Without this sleep,
    # the PUB side silently drops messages because it hasn't received
    # the SUB's subscription filter yet.
    time.sleep(0.5)

    logger.info(
        "ZMQ listener ready — subscribed to ports %s (slow-joiner handshake done)",
        all_ports,
    )

    audio_chunk_count: int = 0
    total_received: int = 0
    total_emitted: int = 0

    while True:
        result = bus.receive(sub, timeout_ms=200)
        if result is None:
            continue

        topic, envelope = result
        data: dict[str, Any] = envelope.get("data", {})
        timestamp: str = envelope.get(
            "timestamp", datetime.now(timezone.utc).isoformat(),
        )

        total_received += 1

        # Log first message from each topic, then periodically.
        if total_received == 1 or total_received % 100 == 0:
            logger.info(
                "ZMQ received: topic=%s total_received=%d total_emitted=%d",
                topic,
                total_received,
                total_emitted,
            )

        try:
            if topic == "audio":
                audio_chunk_count += 1
                if audio_chunk_count % AUDIO_EMIT_INTERVAL != 0:
                    continue
                b64_samples: str = data.get("samples", "")
                if b64_samples:
                    rms = compute_rms(b64_samples)
                    payload = {
                        "rms": round(rms, 4),
                        "timestamp": timestamp,
                    }
                    socketio.emit("audio_level", payload)
                    total_emitted += 1
                    logger.info(
                        "Emitted audio_level: rms=%.4f (chunk #%d, emitted #%d)",
                        rms,
                        audio_chunk_count,
                        total_emitted,
                    )

            elif topic == "transcript":
                text = data.get("text", "")
                socketio.emit("transcript", {
                    "text": text,
                    "timestamp": data.get("timestamp", timestamp),
                })
                total_emitted += 1
                logger.info(
                    "Emitted transcript: %s (emitted #%d)",
                    text[:60] if text else "(empty)",
                    total_emitted,
                )

            elif topic == "stress":
                # stress_detector publishes: stress_score, emotions{}, confidence
                stress_score: float = float(data.get("stress_score", 0.0))
                emotions: dict[str, Any] = data.get("emotions", {})
                if not isinstance(emotions, dict):
                    logger.warning(
                        "Malformed stress message: 'emotions' is %s, expected dict",
                        type(emotions).__name__,
                    )
                    emotions = {}

                arousal: float = float(emotions.get("arousal", stress_score))
                valence: float = float(emotions.get("valence", 0.0))
                dominance: float = float(emotions.get("dominance", 0.0))
                confidence: float = float(data.get("confidence", 0.0))

                socketio.emit("stress", {
                    "score": stress_score,
                    "arousal": arousal,
                    "valence": valence,
                    "dominance": dominance,
                    "confidence": confidence,
                    "timestamp": timestamp,
                })
                total_emitted += 1
                logger.info("Emitted stress: score=%.2f", stress_score)

            elif topic == "tactic":
                socketio.emit("tactic", {
                    "risk_score": data.get("risk_score", 0.0),
                    "active_tactics": data.get("active_tactics", []),
                    "timestamp": timestamp,
                })
                total_emitted += 1
                logger.info("Emitted tactic: risk=%.2f", data.get("risk_score", 0.0))

            else:
                logger.debug("Unknown topic: %s", topic)

        except Exception:
            logger.exception("Error processing %s message", topic)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app, socketio = create_app()

    # Start the ZMQ listener in a background daemon thread.
    listener_thread = threading.Thread(
        target=zmq_listener,
        args=(socketio,),
        daemon=True,
        name="zmq-listener",
    )
    listener_thread.start()
    logger.info("ZMQ listener thread launched (tid=%s)", listener_thread.ident)

    # Give the listener thread time to create the subscriber and complete
    # the slow-joiner handshake before we start the HTTP server.
    time.sleep(1.0)
    if listener_thread.is_alive():
        logger.info("ZMQ listener thread confirmed alive")
    else:
        logger.error("ZMQ listener thread died during startup!")

    # Print access info.
    print("\n" + "=" * 56)
    print("  ANCHOR — Real-Time Scam Detection Dashboard")
    print("=" * 56)
    print(f"  Local:   http://127.0.0.1:{DASHBOARD_PORT}")
    print(f"  Network: http://0.0.0.0:{DASHBOARD_PORT}")
    print()
    print("  Open in a browser on any device on the same network.")
    print("  Pipeline stages will stream data as they start.")
    print("=" * 56 + "\n")

    socketio.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
        allow_unsafe_werkzeug=True,  # required for Flask 3.x dev server
    )
