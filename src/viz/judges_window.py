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
    display: flex;
    flex-direction: column;
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

  /* ── Component status panel ───────────────────────────────────── */
  .status-panel {
    display: flex;
    justify-content: center;
    gap: 20px;
    padding: 12px 16px;
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .status-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: var(--dim);
  }
  .status-item.connected { color: var(--accent); }
  .status-dot-sm {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--dim);
    opacity: 0.5;
  }
  .status-item.connected .status-dot-sm {
    background: var(--accent);
    opacity: 1;
  }

  /* ── Ready / Timer banner ─────────────────────────────────────── */
  .ready-banner {
    text-align: center;
    padding: 14px;
    font-size: 1.1rem;
    font-weight: 500;
    transition: all 0.3s;
  }
  .ready-banner.starting {
    color: var(--dim);
    background: rgba(253, 203, 110, 0.08);
  }
  .ready-banner.ready {
    color: var(--success);
    background: rgba(0, 184, 148, 0.25);
    font-size: 1.4rem;
    font-weight: 600;
    animation: pulse-ready 2s ease-in-out infinite;
  }
  .ready-banner.listening {
    color: var(--success);
    background: rgba(91, 191, 179, 0.2);
    animation: pulse-ready 2s ease-in-out infinite;
  }
  .ready-banner.analyzing {
    color: #74b9ff;
    background: rgba(116, 185, 255, 0.15);
  }
  .ready-banner.complete {
    color: var(--success);
    background: rgba(0, 184, 148, 0.12);
  }
  @keyframes pulse-ready {
    0%, 100% { opacity: 1; }
    50%     { opacity: 0.85; }
  }
  .timer-bar {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
  }
  .timer-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.5s linear;
  }

  /* ── Grid layout for the 4 main cards ──────────────────────────── */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 16px;
    min-width: 0;
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
  #transcript-card {
    grid-column: 1;
    grid-row: 1;
    min-height: 200px;
  }
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
  #audio-card {
    grid-column: 2;
    grid-row: 1;
    min-height: 200px;
  }
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
  #stress-card {
    grid-column: 1;
    grid-row: 2;
    min-height: 200px;
  }
  #tactic-card {
    grid-column: 2;
    grid-row: 2;
    min-height: 200px;
  }

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

  /* ── Risk badge ─────────────────────────────────────────────── */
  .risk-badge {
    font-size: 1.5rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 14px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 10px;
    transition: background 0.3s, color 0.3s;
  }
  .risk-badge.risk-low {
    color: var(--success);
    background: rgba(0, 184, 148, 0.12);
  }
  .risk-badge.risk-medium {
    color: var(--warn);
    background: rgba(253, 203, 110, 0.12);
  }
  .risk-badge.risk-high {
    color: var(--danger);
    background: rgba(225, 112, 85, 0.15);
  }

  /* ── Tactic bars ────────────────────────────────────────────── */
  .tactic-bars {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 6px;
  }
  .tactic-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .tactic-bar-label {
    width: 80px;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: capitalize;
    color: var(--dim);
    text-align: right;
  }
  .tactic-bar-bg {
    flex: 1;
    height: 18px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 9px;
    overflow: hidden;
  }
  .tactic-bar-fill {
    height: 100%;
    border-radius: 9px;
    transition: width 0.5s ease-out, background 0.3s;
    min-width: 0;
  }
  .tactic-bar-pct {
    width: 42px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--dim);
    text-align: left;
  }
  .tactic-meta {
    font-size: 0.7rem;
    color: var(--dim);
    margin-top: 8px;
    text-align: right;
  }

  /* ── Alert glow for high-risk tactic card ──────────────────────── */
  #tactic-card.alert-active {
    border-color: var(--danger);
    box-shadow: 0 0 12px rgba(225, 112, 85, 0.25);
  }

  /* ── Main layout: fixed grid, script panel 300px left ───────────── */
  .dashboard-container {
    display: grid;
    grid-template-columns: 300px 1fr;
    grid-template-rows: 1fr;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    gap: 0;
    padding: 0;
  }
  .script-panel {
    grid-row: 1 / -1;
    width: 300px;
    min-width: 300px;
    max-width: 300px;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.15);
    overflow: hidden;
  }
  .dashboard-panel {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    overflow: auto;
    padding: 16px;
  }

  /* ── Scenario dropdown: fixed width, all 6 options visible ──────── */
  .script-selector {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }
  .script-selector label {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--dim);
    margin-bottom: 6px;
  }
  .scenario-select {
    width: 100%;
    padding: 12px;
    font-size: 16px;
    background: #2a2a2a;
    color: #fff;
    border: 1px solid #444;
    border-radius: 4px;
    cursor: pointer;
  }
  .scenario-select:focus {
    outline: none;
    border-color: var(--accent);
  }
  .scenario-select option {
    background: #2a2a2a;
    color: #fff;
    padding: 8px;
  }
  .scenario-select optgroup {
    font-weight: 600;
    color: var(--dim);
  }
  .scenario-type-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .scenario-type-badge.scam {
    background: rgba(225, 112, 85, 0.2);
    color: var(--danger);
  }
  .scenario-type-badge.benign {
    background: rgba(0, 184, 148, 0.2);
    color: var(--success);
  }

  /* ── Teleprompter: fixed height, scrollable script lines ───────── */
  .teleprompter {
    flex: 1;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    cursor: pointer;
  }
  .script-lines {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }
  .teleprompter-placeholder {
    color: var(--dim);
    font-style: italic;
    font-size: 0.95rem;
    text-align: center;
    padding: 16px;
  }
  .script-line {
    padding: 12px;
    margin: 8px 0;
    border-radius: 4px;
    font-size: 18px;
    line-height: 1.4;
    transition: background 0.2s;
    color: var(--dim);
  }
  .script-line.current {
    background: #3a5a3a;
    border-left: 4px solid #4ade80;
    font-weight: bold;
    color: var(--text);
  }
  .script-line.completed {
    opacity: 0.5;
  }
  .script-progress {
    padding: 10px 16px;
    font-size: 0.75rem;
    color: var(--dim);
    border-top: 1px solid var(--border);
  }
</style>
</head>
<body>

<header>
  <h1><span class="status-dot" id="status-dot"></span>.anchor</h1>
  <p>We only hear one side. Watch what we know.</p>
</header>

<div class="status-panel" id="status-panel">
  <span class="status-item" id="status-audio"><span class="status-dot-sm"></span>Audio Capture</span>
  <span class="status-item" id="status-speech"><span class="status-dot-sm"></span>Speech Recognition</span>
  <span class="status-item" id="status-stress"><span class="status-dot-sm"></span>Stress Detector</span>
  <span class="status-item" id="status-tactic"><span class="status-dot-sm"></span>Tactic Inference</span>
</div>

<div class="ready-banner starting" id="ready-banner">
  <span id="ready-text">⏳ Starting up&hellip;</span>
  <div class="timer-bar" id="timer-bar" style="display:none;">
    <div class="timer-bar-fill" id="timer-fill"></div>
  </div>
</div>

<div class="main-content dashboard-container">
  <aside class="script-panel">
    <div class="script-selector">
      <div class="scenario-type-badge" id="scenario-type-badge" style="display:none;"></div>
      <label for="script-select">Scenario</label>
      <select id="script-select" class="scenario-select">
        <option value="">— Select scenario —</option>
        <optgroup label="Scam scenarios">
          <option value="authority">[SCAM] Authority/Government</option>
          <option value="bank">[SCAM] Bank Security</option>
          <option value="grandchild">[SCAM] Grandchild Emergency</option>
          <option value="tech">[SCAM] Tech Support</option>
          <option value="payment">[SCAM] Payment Demand</option>
          <option value="romance">[SCAM] Romance Scam</option>
        </optgroup>
        <optgroup label="Benign scenarios">
          <option value="doctors">[BENIGN] Doctor&rsquo;s Office</option>
          <option value="friend">[BENIGN] Friend Calling</option>
          <option value="pharmacy">[BENIGN] Pharmacy Refill</option>
          <option value="family">[BENIGN] Family Check-in</option>
        </optgroup>
      </select>
    </div>
    <div class="teleprompter" id="teleprompter">
      <div class="teleprompter-placeholder" id="teleprompter-placeholder">
        Select a scenario to see the elder&rsquo;s lines
      </div>
      <div id="script-lines" class="script-lines" style="display:none;"></div>
    </div>
    <div class="script-progress" id="script-progress" style="display:none;">
      Line <span id="current-line-num">0</span> of <span id="total-lines">0</span>
    </div>
  </aside>

  <div class="dashboard-panel">
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
    <div class="risk-badge risk-low" id="risk-badge">&mdash;</div>
    <div class="tactic-bars" id="tactic-bars">
      <div class="waiting-msg" id="tactic-waiting">Waiting for tactic data&hellip;</div>
    </div>
    <div class="tactic-meta" id="tactic-meta"></div>
  </div>
</div>
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

  /* ── Script selector & teleprompter ────────────────────────────── */
  const SCRIPTS = {
    authority: [
      "Oh my, this sounds serious.",
      "Yes, I understand. I don't want any trouble.",
      "My social security number? Well, if you need it...",
      "I'm just a little confused about what I did wrong.",
      "Please don't send anyone to my house.",
      "Okay, okay, let me get my information.",
      "How much did you say I need to pay to fix this?",
    ],
    bank: [
      "Suspicious activity? Oh no, what happened?",
      "Yes, I have my card right here.",
      "You need me to verify my account number?",
      "I want to make sure my money is safe.",
      "A verification code just came to my phone.",
      "You want me to read it to you? Okay, it says...",
      "Should I stay on the line while you fix it?",
    ],
    grandchild: [
      "Tommy? Is that you? You sound different.",
      "Oh my god, are you hurt? What happened?",
      "You're in jail? How did this happen?",
      "Of course I'll help you, sweetheart.",
      "I won't tell your parents, I promise.",
      "How much do you need for the bail?",
      "I can go to the store and get that for you.",
    ],
    tech: [
      "A virus? Oh dear, I don't know how that happened.",
      "Yes, I can see the screen you're talking about.",
      "You need remote access? Is that safe?",
      "I'm typing in what you told me.",
      "It's asking for my password now.",
      "Three hundred dollars to fix it? That seems like a lot.",
      "Let me get my credit card.",
    ],
    payment: [
      "A warrant for my arrest? There must be a mistake.",
      "I've never been in trouble with the law.",
      "If I pay now, this goes away?",
      "Gift cards? That's an unusual way to pay.",
      "Which store should I go to?",
      "I'm writing down the amounts you need.",
      "I'll go right now. Please don't hang up.",
    ],
    romance: [
      "I feel so lucky we found each other.",
      "Of course I trust you, darling.",
      "You're stuck overseas? That's terrible.",
      "I wish I could be there to help you.",
      "How much do you need to get home?",
      "Western Union? I can figure out how to do that.",
      "I'd do anything for you, you know that.",
    ],
    /* Benign scenarios for false-positive testing */
    doctors: [
      "Hello? Yes, this is she.",
      "Oh yes, I remember. Tuesday at two o'clock.",
      "Do I need to bring my insurance card?",
      "And I shouldn't eat anything before the appointment?",
      "Okay, I'll be there fifteen minutes early.",
      "Thank you for calling to remind me.",
    ],
    friend: [
      "Margaret! It's so good to hear from you.",
      "I've been meaning to call you too.",
      "How are the grandchildren doing?",
      "That sounds wonderful. I'd love to see the photos.",
      "Yes, let's have lunch this week.",
      "Thursday works perfectly for me. See you then!",
    ],
    pharmacy: [
      "Yes, this is the right number.",
      "My blood pressure medication, yes.",
      "It's ready for pickup? Great.",
      "I can come by this afternoon.",
      "Do I need to bring anything?",
      "Thank you, I appreciate the reminder.",
    ],
    family: [
      "Hi sweetheart! How are you?",
      "Work is going well? That's good to hear.",
      "Yes, I'm feeling much better this week.",
      "I made that soup recipe you sent me.",
      "Sunday dinner sounds lovely.",
      "I love you too. Talk soon.",
    ],
  };

  const SCAM_KEYS = ["authority", "bank", "grandchild", "tech", "payment", "romance"];
  const BENIGN_KEYS = ["doctors", "friend", "pharmacy", "family"];

  const scriptSelect = document.getElementById("script-select");
  const scriptLinesEl = document.getElementById("script-lines");
  const teleprompterPlaceholder = document.getElementById("teleprompter-placeholder");
  const scriptProgress = document.getElementById("script-progress");
  const currentLineNumEl = document.getElementById("current-line-num");
  const totalLinesEl = document.getElementById("total-lines");
  const scenarioTypeBadge = document.getElementById("scenario-type-badge");

  let currentScript = null;
  let currentLineIndex = 0;

  function renderScript() {
    if (!currentScript || !SCRIPTS[currentScript]) {
      teleprompterPlaceholder.style.display = "block";
      scriptLinesEl.style.display = "none";
      scriptProgress.style.display = "none";
      if (scenarioTypeBadge) scenarioTypeBadge.style.display = "none";
      return;
    }
    if (scenarioTypeBadge) {
      scenarioTypeBadge.style.display = "block";
      scenarioTypeBadge.textContent = BENIGN_KEYS.indexOf(currentScript) >= 0 ? "BENIGN" : "SCAM";
      scenarioTypeBadge.className = "scenario-type-badge " + (BENIGN_KEYS.indexOf(currentScript) >= 0 ? "benign" : "scam");
    }
    const lines = SCRIPTS[currentScript];
    teleprompterPlaceholder.style.display = "none";
    scriptLinesEl.style.display = "block";
    scriptProgress.style.display = "block";
    scriptLinesEl.innerHTML = "";
    totalLinesEl.textContent = lines.length;
    lines.forEach(function(line, i) {
      const div = document.createElement("div");
      var cls = "script-line";
      if (i === currentLineIndex) cls += " current";
      else if (i < currentLineIndex) cls += " completed";
      div.className = cls;
      div.textContent = line;
      div.id = i === currentLineIndex ? "script-line-current" : "";
      scriptLinesEl.appendChild(div);
    });
    var cur = document.getElementById("script-line-current");
    if (cur) cur.scrollIntoView({ behavior: "smooth", block: "center" });
    currentLineNumEl.textContent = currentLineIndex + 1;
  }

  function advanceLine() {
    if (!currentScript || !SCRIPTS[currentScript]) return;
    const lines = SCRIPTS[currentScript];
    if (currentLineIndex < lines.length - 1) {
      currentLineIndex++;
      renderScript();
    }
  }

  function resetScript() {
    currentLineIndex = 0;
    renderScript();
  }

  function selectScript(key) {
    if (SCRIPTS[key]) {
      currentScript = key;
      currentLineIndex = 0;
      scriptSelect.value = key;
      renderScript();
    }
  }

  scriptSelect.addEventListener("change", function() {
    const val = this.value;
    if (val) {
      selectScript(val);
    } else {
      currentScript = null;
      currentLineIndex = 0;
      renderScript();
    }
  });

  document.addEventListener("keydown", function(e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) return;
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      advanceLine();
    } else if (e.key === "r" || e.key === "R") {
      e.preventDefault();
      resetScript();
    } else if (e.key >= "1" && e.key <= "9") {
      var keys = ["authority", "bank", "grandchild", "tech", "payment", "romance", "doctors", "friend", "pharmacy"];
      selectScript(keys[parseInt(e.key, 10) - 1]);
    } else if (e.key === "0") {
      selectScript("family");
    }
  });

  document.getElementById("teleprompter").addEventListener("click", function() {
    advanceLine();
  });

  /* ── Component status & ready banner ──────────────────────────── */
  const components = {
    audio: document.getElementById("status-audio"),
    speech: document.getElementById("status-speech"),
    stress: document.getElementById("status-stress"),
    tactic: document.getElementById("status-tactic"),
  };
  const readyBanner = document.getElementById("ready-banner");
  const readyText = document.getElementById("ready-text");
  const timerBar = document.getElementById("timer-bar");
  const timerFill = document.getElementById("timer-fill");

  function markConnected(name) {
    const el = components[name];
    if (el && !el.classList.contains("connected")) {
      el.classList.add("connected");
      updateReadyBanner();
    }
  }

  function updateReadyBanner() {
    const all = components.audio.classList.contains("connected") &&
               components.speech.classList.contains("connected") &&
               components.stress.classList.contains("connected") &&
               components.tactic.classList.contains("connected");
    if (all && !readyBanner.classList.contains("listening") &&
        !readyBanner.classList.contains("analyzing") &&
        !readyBanner.classList.contains("complete")) {
      readyBanner.classList.remove("starting");
      readyBanner.classList.add("ready");
      readyText.textContent = "🛡️ READY TO PROTECT";
    } else if (!all && !readyBanner.classList.contains("listening") &&
               !readyBanner.classList.contains("analyzing") &&
               !readyBanner.classList.contains("complete")) {
      readyBanner.classList.remove("ready");
      readyBanner.classList.add("starting");
      readyText.textContent = "⏳ Starting up…";
    }
  }

  /* ── Speaking timer (30s when transcripts start) ───────────────── */
  let timerInterval = null;
  const SPEAKING_DURATION = 30;

  function startSpeakingTimer() {
    if (readyBanner.classList.contains("complete")) return;
    if (timerInterval) return;  /* Already running */
    readyBanner.classList.remove("ready", "starting", "analyzing");
    readyBanner.classList.add("listening");
    timerBar.style.display = "block";
    let elapsed = 0;
    function tick() {
      elapsed++;
      const pct = Math.min(100, (elapsed / SPEAKING_DURATION) * 100);
      readyText.textContent = "🎤 LISTENING… " + formatTime(elapsed) + " / " + formatTime(SPEAKING_DURATION);
      timerFill.style.width = pct + "%";
      if (elapsed >= SPEAKING_DURATION) {
        clearInterval(timerInterval);
        timerInterval = null;
        readyBanner.classList.remove("listening");
        readyBanner.classList.add("complete");
        readyText.textContent = "✓ Complete — Processing…";
        timerBar.style.display = "none";
      }
    }
    tick();
    timerInterval = setInterval(tick, 1000);
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  /* ── Transcript ──────────────────────────────────────────────── */
  const MAX_LINES = 10;
  const transcriptEl = document.getElementById("transcript-lines");
  const transcriptWait = document.getElementById("transcript-waiting");

  socket.on("transcript", (data) => {
    const text = (data.text || "").trim();
    if (!text || text === "(silence)") return;  /* Filter empty/whitespace/silence */
    markConnected("speech");
    startSpeakingTimer();
    if (transcriptWait) transcriptWait.remove();
    const line = document.createElement("div");
    line.className = "line";
    const ts = data.timestamp ? data.timestamp.split("T")[1].substring(0, 8) : "";
    line.innerHTML = '<span class="ts">' + ts + '</span>' + escapeHtml(text);
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
    markConnected("audio");
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
    markConnected("stress");
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
  const riskBadgeEl  = document.getElementById("risk-badge");
  const tacticBarsEl = document.getElementById("tactic-bars");
  const tacticMetaEl = document.getElementById("tactic-meta");
  const tacticCard   = document.getElementById("tactic-card");

  function tacticBarColor(v) {
    return v > 0.7 ? "var(--danger)" : v > 0.4 ? "var(--warn)" : "var(--accent)";
  }

  socket.on("tactics", (data) => {
    markConnected("tactic");
    if (readyBanner.classList.contains("listening") && !readyBanner.classList.contains("complete")) {
      readyBanner.classList.remove("listening");
      readyBanner.classList.add("analyzing");
      readyText.textContent = "🔍 ANALYZING…";
      setTimeout(function() {
        if (readyBanner.classList.contains("analyzing") && !readyBanner.classList.contains("complete")) {
          readyBanner.classList.remove("analyzing");
          readyBanner.classList.add("listening");
          readyText.textContent = "🎤 LISTENING…";
        }
      }, 2000);
    }
    /* ── Risk badge ─────────────────────────────────────────── */
    const risk = (data.risk_level || "low").toLowerCase();
    riskBadgeEl.textContent = risk.toUpperCase();
    riskBadgeEl.className   = "risk-badge risk-" + risk;

    /* ── Tactic bars ────────────────────────────────────────── */
    const tactics = data.tactics || {};
    tacticBarsEl.innerHTML = "";
    var keys = ["urgency", "authority", "fear", "isolation", "financial"];
    keys.forEach(function(key) {
      var val = tactics[key] != null ? tactics[key] : 0;
      var pct = (val * 100).toFixed(0);

      var row   = document.createElement("div");
      row.className = "tactic-bar-row";

      var label = document.createElement("span");
      label.className = "tactic-bar-label";
      label.textContent = key;

      var bg    = document.createElement("div");
      bg.className = "tactic-bar-bg";
      var fill  = document.createElement("div");
      fill.className = "tactic-bar-fill";
      fill.style.width = pct + "%";
      fill.style.background = tacticBarColor(val);
      bg.appendChild(fill);

      var pctEl = document.createElement("span");
      pctEl.className = "tactic-bar-pct";
      pctEl.textContent = pct + "%";

      row.appendChild(label);
      row.appendChild(bg);
      row.appendChild(pctEl);
      tacticBarsEl.appendChild(row);
    });

    /* ── Metadata (word count, inference time, timestamp) ──── */
    var parts = [];
    if (data.word_count != null)         parts.push(data.word_count + " words");
    else if (data.transcript_count != null) parts.push(data.transcript_count + " transcripts");
    if (data.inference_time_ms != null)  parts.push((data.inference_time_ms / 1000).toFixed(1) + "s inference");
    if (data.timestamp) {
      var t = data.timestamp;
      var tsShort = t.indexOf("T") >= 0 ? t.split("T")[1].substring(0, 8) : t;
      parts.push("analyzed " + tsShort);
    }
    tacticMetaEl.textContent = parts.length > 0 ? parts.join(" \u00b7 ") : "";

    /* ── Alert glow ─────────────────────────────────────────── */
    if (risk === "high") {
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
                text = (data.get("text", "") or "").strip()
                # Skip empty, whitespace-only, or silence placeholders
                if not text or text == "(silence)":
                    logger.debug("Filtering transcript: empty or (silence)")
                    continue
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

            elif topic == "tactics":
                tactics_dict: dict[str, Any] = data.get("tactics", {})
                risk_level: str = data.get("risk_level", "low")
                socketio.emit("tactics", {
                    "tactics": tactics_dict,
                    "risk_level": risk_level,
                    "transcript_count": data.get("transcript_count"),
                    "word_count": data.get("word_count"),
                    "inference_time_ms": data.get("inference_time_ms", 0),
                    "timestamp": timestamp,
                })
                total_emitted += 1
                logger.info(
                    "Emitted tactics: risk=%s tactics=%s",
                    risk_level,
                    tactics_dict,
                )

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
