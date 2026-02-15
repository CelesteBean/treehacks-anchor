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
    SYSTEM_PORT,
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

  /* ── Speech patterns ─────────────────────────────────────────── */
  .speech-patterns-grid {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 0.9rem;
  }
  .pattern-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .pattern-label {
    width: 90px;
    color: var(--dim);
  }
  .pattern-tag {
    font-size: 0.75rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(255,255,255,0.08);
    color: var(--dim);
  }
  .pattern-tag.elevated { background: rgba(253,203,110,0.2); color: var(--warn); }
  .recent-matches {
    margin-top: 10px;
    font-size: 0.75rem;
    color: var(--dim);
  }
  .recent-label { font-weight: 500; }

  /* ── Detection trigger & risk factors ─────────────────────────── */
  .detection-trigger {
    display: none;
    padding: 10px 12px;
    margin-bottom: 10px;
    background: rgba(225,112,85,0.15);
    border: 1px solid rgba(225,112,85,0.4);
    border-radius: 6px;
    font-size: 0.85rem;
  }
  .detection-trigger.visible { display: block; }
  .detection-trigger .trigger-phrase { font-weight: 600; color: var(--danger); }
  .detection-trigger .trigger-meta { font-size: 0.75rem; color: var(--dim); margin-top: 4px; }
  .risk-factors-panel {
    display: none;
    margin-bottom: 10px;
    padding: 8px 12px;
    background: rgba(0,0,0,0.2);
    border-radius: 6px;
    font-size: 0.8rem;
    max-height: 80px;
    overflow-y: auto;
  }
  .risk-factors-panel.visible { display: block; }
  .risk-factors-panel ul { margin: 0; padding-left: 18px; }

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
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    gap: 0;
    padding: 0;
  }
  #tab-pipeline.dashboard-grid {
    display: grid;
    grid-template-columns: 300px 1fr;
    grid-template-rows: 1fr;
    flex: 1;
    min-width: 0;
    overflow: hidden;
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

  /* ── Tab navigation ───────────────────────────────────────────── */
  .tab-bar {
    display: flex;
    gap: 0;
    padding: 0 16px;
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid var(--border);
  }
  .tab-btn {
    padding: 14px 24px;
    font-size: 0.9rem;
    font-weight: 500;
    background: none;
    border: none;
    color: var(--dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.2s, border-color 0.2s;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active {
    color: #76B900;
    border-bottom-color: #76B900;
  }
  .tab-panel { display: none; }
  .tab-panel.active { flex: 1; flex-direction: column; min-height: 0; min-width: 0; overflow: hidden; }
  .tab-panel.active#tab-pipeline { display: grid; }
  .tab-panel.active#tab-system { display: flex; }

  /* ── System Performance tab ───────────────────────────────────── */
  #tab-system { overflow-y: auto; overflow-x: hidden; }
  .perf-header { font-size: 1.2rem; color: #76B900; margin-bottom: 16px; padding: 0 16px; flex-shrink: 0; }
  .metric-cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; padding: 0 16px 16px; flex-shrink: 0; }
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    min-width: 0;
  }
  .metric-card .label { font-size: 0.7rem; text-transform: uppercase; color: var(--dim); margin-bottom: 4px; }
  .metric-card .value { font-size: 1.6rem; font-weight: 600; color: var(--accent); }
  .metric-card .sub { font-size: 0.75rem; color: var(--dim); margin-top: 2px; }
  .metric-card .value.warn { color: var(--warn); }
  .metric-card .value.danger { color: var(--danger); }
  .perf-charts { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; padding: 0 16px; min-height: 180px; max-height: 380px; flex-shrink: 1; }
  .perf-chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0; overflow: hidden; }
  .perf-chart-card h3 { font-size: 0.8rem; color: var(--dim); margin-bottom: 8px; flex-shrink: 0; }
  .perf-chart-wrap { height: 140px; min-height: 140px; max-height: 140px; overflow: hidden; position: relative; }
  .perf-chart { width: 100% !important; height: 140px !important; max-height: 140px; }
  .process-bars { padding: 16px; flex-shrink: 0; }
  .process-bars h3 { font-size: 0.8rem; color: var(--dim); margin-bottom: 10px; }
  .process-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .process-row .name { width: 140px; font-size: 0.85rem; }
  .process-row .bar-bg { flex: 1; min-width: 0; height: 16px; background: rgba(255,255,255,0.06); border-radius: 8px; overflow: hidden; }
  .process-row .bar-fill { height: 100%; border-radius: 8px; transition: width 0.3s; }
  .process-row .mb { width: 70px; font-size: 0.8rem; color: var(--dim); text-align: right; }
  .latency-panel { padding: 16px; }
  .latency-panel h3 { font-size: 0.8rem; color: var(--dim); margin-bottom: 10px; }
  .latency-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 0.9rem; }
  .latency-row .stage { color: var(--dim); }
  .latency-row .ms { font-weight: 600; color: var(--accent); }
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

<nav class="tab-bar">
  <button class="tab-btn active" data-tab="pipeline">Pipeline</button>
  <button class="tab-btn" data-tab="system">System Performance</button>
</nav>

<div class="main-content dashboard-container">
  <div id="tab-pipeline" class="tab-panel active dashboard-grid">
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
    <h2>Speech Patterns</h2>
    <div class="speech-patterns-grid" id="speech-patterns">
      <div class="pattern-row"><span class="pattern-label">Speech Rate:</span><span id="wpm-value">&mdash;</span> <span id="wpm-label" class="pattern-tag"></span></div>
      <div class="pattern-row"><span class="pattern-label">Hesitations:</span><span id="hesitation-value">&mdash;</span> <span id="hesitation-tag" class="pattern-tag"></span></div>
      <div class="pattern-row"><span class="pattern-label">Questions:</span><span id="question-value">&mdash;</span></div>
      <div class="pattern-row"><span class="pattern-label">Uncertainty:</span><span id="uncertainty-value">&mdash;</span> phrase(s)</div>
    </div>
    <div class="recent-matches" id="recent-matches">
      <span class="recent-label">Recent:</span> <span id="recent-list">—</span>
    </div>
  </div>

  <!-- Tactic -->
  <div class="card" id="tactic-card">
    <h2>Scam Tactic Detection</h2>
    <div class="detection-trigger" id="detection-trigger"></div>
    <div class="risk-factors-panel" id="risk-factors-panel"></div>
    <div class="risk-badge risk-low" id="risk-badge">&mdash;</div>
    <div class="tactic-bars" id="tactic-bars">
      <div class="waiting-msg" id="tactic-waiting">Waiting for tactic data&hellip;</div>
    </div>
    <div class="tactic-meta" id="tactic-meta"></div>
  </div>
</div>
  </div>
  </div><!-- /tab-pipeline -->

  <div id="tab-system" class="tab-panel" style="display:none;flex:1;flex-direction:column;min-height:0;overflow:auto;">
    <h2 class="perf-header">Jetson Orin Nano &mdash; Edge Performance</h2>
    <div class="metric-cards" id="perf-cards">
      <div class="metric-card"><div class="label">CPU</div><div class="value" id="perf-cpu">&mdash;</div><div class="sub" id="perf-cpu-temp">&mdash;</div></div>
      <div class="metric-card"><div class="label">RAM</div><div class="value" id="perf-ram">&mdash;</div><div class="sub" id="perf-ram-detail">&mdash;</div></div>
      <div class="metric-card"><div class="label">GPU</div><div class="value" id="perf-gpu">&mdash;</div><div class="sub" id="perf-gpu-temp">&mdash;</div></div>
      <div class="metric-card"><div class="label">Power</div><div class="value" id="perf-power">&mdash;</div><div class="sub" id="perf-power-mode">&mdash;</div></div>
    </div>
    <div class="perf-charts">
      <div class="perf-chart-card"><h3>CPU Usage (60s)</h3><div class="perf-chart-wrap"><canvas class="perf-chart" id="chart-cpu"></canvas></div></div>
      <div class="perf-chart-card"><h3>Memory (60s)</h3><div class="perf-chart-wrap"><canvas class="perf-chart" id="chart-mem"></canvas></div></div>
      <div class="perf-chart-card"><h3>Power (60s)</h3><div class="perf-chart-wrap"><canvas class="perf-chart" id="chart-power"></canvas></div></div>
      <div class="perf-chart-card"><h3>Inference Latency (60s)</h3><div class="perf-chart-wrap"><canvas class="perf-chart" id="chart-latency"></canvas></div></div>
    </div>
    <div class="process-bars">
      <h3>Process Memory Usage</h3>
      <div id="process-bars-container"><div class="waiting-msg">Waiting for system metrics&hellip;</div></div>
      <div class="latency-panel" style="margin-top:12px;">
        <h3>Pipeline Latency (avg last 10)</h3>
        <div class="latency-row"><span class="stage">Audio &rarr; Whisper</span><span class="ms" id="lat-whisper">&mdash;</span></div>
        <div class="latency-row"><span class="stage">Whisper &rarr; Analyzer</span><span class="ms" id="lat-analyzer">&mdash;</span></div>
        <div class="latency-row"><span class="stage">Analyzer &rarr; TTS</span><span class="ms" id="lat-tts">&mdash;</span></div>
        <div class="latency-row"><span class="stage">End-to-end</span><span class="ms" id="lat-e2e">&mdash;</span></div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
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

  /* ── Tab switching ─────────────────────────────────────────────── */
  document.querySelectorAll(".tab-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      const tab = this.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(function(b) { b.classList.remove("active"); });
      document.querySelectorAll(".tab-panel").forEach(function(p) {
        p.style.display = "none";
        p.classList.remove("active");
      });
      this.classList.add("active");
      var panel = document.getElementById("tab-" + tab);
      if (panel) {
        panel.style.display = tab === "pipeline" ? "grid" : "flex";
        panel.classList.add("active");
      }
    });
  });

  /* ── System metrics: charts (60-point rolling) ──────────────────── */
  const HISTORY = 60;
  const cpuHistory = []; const memHistory = []; const powerHistory = []; const latencyHistory = [];
  function ensureChart(id, label, color, unit) {
    const el = document.getElementById(id);
    if (!el || el.chart) return el.chart;
    const ctx = el.getContext("2d");
    el.chart = new Chart(ctx, {
      type: "line",
      data: { labels: [], datasets: [{ label: label, data: [], borderColor: color, backgroundColor: color + "20", fill: true, tension: 0.3 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { x: { display: false }, y: { min: 0, suggestedMax: 100 } } }
    });
    return el.chart;
  }
  function updateChart(chart, history, label, unit) {
    if (!chart) return;
    chart.data.labels = history.map(function(_, i) { return i; });
    chart.data.datasets[0].data = history;
    chart.data.datasets[0].label = label;
    chart.update("none");
  }
  socket.on("system_metrics", function(data) {
    var cpu = (data.cpu || {}).percent;
    var mem = (data.memory || {}).percent;
    var pw = (data.power || {}).current_w;
    var lat = (data.pipeline || {}).e2e_ms;
    if (typeof cpu === "number") { cpuHistory.push(cpu); if (cpuHistory.length > HISTORY) cpuHistory.shift(); }
    if (typeof mem === "number") { memHistory.push(mem); if (memHistory.length > HISTORY) memHistory.shift(); }
    if (typeof pw === "number") { powerHistory.push(pw); if (powerHistory.length > HISTORY) powerHistory.shift(); }
    if (typeof lat === "number") { latencyHistory.push(lat); if (latencyHistory.length > HISTORY) latencyHistory.shift(); }
    document.getElementById("perf-cpu").textContent = cpu != null ? cpu.toFixed(1) + "%" : "—";
    document.getElementById("perf-cpu").className = "value" + (cpu > 80 ? " danger" : cpu > 60 ? " warn" : "");
    document.getElementById("perf-cpu-temp").textContent = (data.cpu || {}).temp_c != null ? (data.cpu.temp_c + "°C") : "—";
    var usedMb = (data.memory || {}).used_mb; var totalMb = (data.memory || {}).total_mb;
    document.getElementById("perf-ram").textContent = usedMb != null ? (usedMb / 1024).toFixed(2) + " GB" : "—";
    document.getElementById("perf-ram-detail").textContent = totalMb != null ? "/ " + (totalMb / 1024).toFixed(2) + " GB (" + (mem || 0).toFixed(0) + "%)" : "—";
    var gpu = (data.gpu || {}).percent;
    document.getElementById("perf-gpu").textContent = gpu != null ? gpu + "%" : "—";
    document.getElementById("perf-gpu-temp").textContent = (data.gpu || {}).temp_c != null ? (data.gpu.temp_c + "°C") : "—";
    document.getElementById("perf-power").textContent = pw != null ? pw.toFixed(2) + " W" : "—";
    document.getElementById("perf-power-mode").textContent = (data.power || {}).mode || "—";
    var pp = (data.memory || {}).per_process_mb || {};
    var total = (data.memory || {}).total_mb || 1;
    var container = document.getElementById("process-bars-container");
    if (typeof pp === "object" && Object.keys(pp).length) {
      var names = { audio_capture: "audio_capture", speech_recognition: "speech_recognition (Whisper)", content_analyzer: "content_analyzer", audio_intervention: "audio_intervention (Piper)", judges_window: "judges_window" };
      var sumMb = 0;
      var html = "";
      for (var k in names) {
        var mb = pp[k] || 0;
        sumMb += mb;
        var pct = total > 0 ? (mb / total) * 100 : 0;
        html += "<div class=\"process-row\"><span class=\"name\">" + names[k] + "</span><div class=\"bar-bg\"><div class=\"bar-fill\" style=\"width:" + pct + "%;background:#76B900\"></div></div><span class=\"mb\">" + mb.toFixed(0) + " MB</span></div>";
      }
      html += "<div class=\"latency-row\" style=\"margin-top:8px;\"><span class=\"stage\">Total Pipeline</span><span class=\"ms\">" + sumMb.toFixed(0) + " MB / " + (total / 1024).toFixed(2) + " GB</span></div>";
      container.innerHTML = html;
    }
    var pl = data.pipeline || {};
    document.getElementById("lat-whisper").textContent = pl.whisper_ms != null ? pl.whisper_ms + "ms" : "—";
    document.getElementById("lat-analyzer").textContent = pl.analyzer_ms != null ? pl.analyzer_ms + "ms" : "—";
    document.getElementById("lat-tts").textContent = pl.tts_ms != null ? pl.tts_ms + "ms" : "—";
    document.getElementById("lat-e2e").textContent = pl.e2e_ms != null ? "~" + pl.e2e_ms + "ms" : "—";
    if (typeof Chart !== "undefined") {
      updateChart(ensureChart("chart-cpu", "CPU %", "#76B900"), cpuHistory, "CPU %");
      updateChart(ensureChart("chart-mem", "RAM %", "#5BBFB3"), memHistory, "RAM %");
      updateChart(ensureChart("chart-power", "Power W", "#FDCB6E"), powerHistory, "Power W");
      updateChart(ensureChart("chart-latency", "E2E ms", "#74b9ff"), latencyHistory, "Latency ms");
    }
  });

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

  /* ── Stress (Speech Patterns) ──────────────────────────────────── */
  const wpmEl = document.getElementById("wpm-value");
  const wpmLabelEl = document.getElementById("wpm-label");
  const hesitationEl = document.getElementById("hesitation-value");
  const hesitationTagEl = document.getElementById("hesitation-tag");
  const questionEl = document.getElementById("question-value");
  const uncertaintyEl = document.getElementById("uncertainty-value");
  const recentListEl = document.getElementById("recent-list");

  socket.on("stress", (data) => {
    markConnected("stress");
    const sp = data.speech_patterns || {};
    wpmEl.textContent = sp.wpm != null ? sp.wpm + " WPM" : "—";
    wpmLabelEl.textContent = sp.wpm_label || "";
    wpmLabelEl.className = "pattern-tag " + (sp.wpm_label === "fast" || sp.wpm_label === "slow" ? "elevated" : "");
    hesitationEl.textContent = sp.hesitations != null ? sp.hesitations : "—";
    hesitationTagEl.textContent = sp.hesitation_label || "";
    hesitationTagEl.className = "pattern-tag " + (sp.hesitation_label === "elevated" ? "elevated" : "");
    questionEl.textContent = sp.questions != null ? sp.questions : "—";
    uncertaintyEl.textContent = sp.uncertainty != null ? sp.uncertainty : "—";
    const recent = sp.recent || [];
    recentListEl.textContent = recent.length > 0 ? recent.join(", ") : "—";
  });

  /* ── Tactic ──────────────────────────────────────────────────── */
  const riskBadgeEl  = document.getElementById("risk-badge");
  const tacticBarsEl = document.getElementById("tactic-bars");
  const tacticMetaEl = document.getElementById("tactic-meta");
  const tacticCard   = document.getElementById("tactic-card");
  const detectionTriggerEl = document.getElementById("detection-trigger");
  const riskFactorsPanelEl = document.getElementById("risk-factors-panel");

  function tacticBarColor(v) {
    var pct = v * 100;
    return pct >= 60 ? "var(--danger)" : pct >= 30 ? "var(--warn)" : "var(--success)";
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
    /* ── Detection trigger ──────────────────────────────────── */
    var trigger = data.detection_trigger || {};
    if (trigger.phrase) {
      detectionTriggerEl.className = "detection-trigger visible";
      detectionTriggerEl.innerHTML = "⚠️ DETECTED: <span class=\"trigger-phrase\">\"" + escapeHtml(trigger.phrase) + "\"</span>" +
        "<div class=\"trigger-meta\">Match Type: " + escapeHtml(trigger.match_type || "") + " · Category: " + escapeHtml(trigger.category || "") + "</div>";
    } else {
      detectionTriggerEl.className = "detection-trigger";
      detectionTriggerEl.innerHTML = "";
    }
    /* ── Risk factors ────────────────────────────────────────── */
    var factors = data.risk_factors || [];
    if (factors.length > 0) {
      riskFactorsPanelEl.className = "risk-factors-panel visible";
      riskFactorsPanelEl.innerHTML = "<strong>Risk Factors:</strong><ul>" + factors.map(function(f) { return "<li>" + escapeHtml(f) + "</li>"; }).join("") + "</ul>";
    } else {
      riskFactorsPanelEl.className = "risk-factors-panel";
      riskFactorsPanelEl.innerHTML = "";
    }
    /* ── Risk badge ─────────────────────────────────────────── */
    const risk = (data.risk_level || "low").toLowerCase();
    riskBadgeEl.textContent = risk.toUpperCase();
    riskBadgeEl.className   = "risk-badge risk-" + risk;

    /* ── Tactic bars ────────────────────────────────────────── */
    const tactics = data.tactics || {};
    const tacticLabels = data.tactic_labels || {};
    tacticBarsEl.innerHTML = "";
    var keys = ["urgency", "authority", "fear", "isolation", "financial"];
    keys.forEach(function(key) {
      var val = tactics[key] != null ? tactics[key] : 0;
      var pct = (val * 100).toFixed(0);
      var barLabel = tacticLabels[key];

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
      if (barLabel && val >= 0.3) {
        var reasonEl = document.createElement("span");
        reasonEl.className = "tactic-bar-reason";
        reasonEl.textContent = " ← " + barLabel;
        reasonEl.style.fontSize = "0.7rem";
        reasonEl.style.color = "var(--dim)";
        row.appendChild(reasonEl);
      }
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

    all_ports = [AUDIO_PORT, TRANSCRIPT_PORT, STRESS_PORT, TACTIC_PORT, SYSTEM_PORT]
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
                # content_analyzer publishes: stress_score, speech_patterns{}, emotions{}, confidence
                stress_score: float = float(data.get("stress_score", 0.0))
                speech_patterns: dict[str, Any] = data.get("speech_patterns", {})
                if not isinstance(speech_patterns, dict):
                    speech_patterns = {}
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
                    "speech_patterns": speech_patterns,
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
                tactic_labels: dict[str, str] = data.get("tactic_labels", {}) or {}
                detection_trigger: dict[str, str] = data.get("detection_trigger", {}) or {}
                risk_factors: list[str] = data.get("risk_factors", []) or []
                socketio.emit("tactics", {
                    "tactics": tactics_dict,
                    "tactic_labels": tactic_labels,
                    "detection_trigger": detection_trigger,
                    "risk_factors": risk_factors,
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

            elif topic == "system":
                data = envelope.get("data", {})
                socketio.emit("system_metrics", data)
                total_emitted += 1

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
