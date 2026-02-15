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
    padding: 12px 16px 8px;
    border-bottom: 1px solid var(--border);
  }
  .header-main-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }
  header h1 {
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--accent);
    margin-bottom: 0;
  }
  header p {
    font-size: 0.875rem;
    color: var(--dim);
    font-weight: 400;
    margin-top: 4px;
    text-align: left;
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

  /* ── Demo View Badge ────────────────────────────────────────────── */
  .demo-view-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff6b6b 0%, #e17055 100%);
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 6px 16px;
    border-radius: 20px;
    margin-top: 8px;
    box-shadow: 0 2px 8px rgba(225, 112, 85, 0.4);
    animation: pulse-badge 2s ease-in-out infinite;
  }
  @keyframes pulse-badge {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.85; transform: scale(1.02); }
  }

  /* ── Component status panel ───────────────────────────────────── */
  .status-panel {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0;
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
    display: inline-flex;
    align-items: center;
    text-align: left;
    padding: 4px 10px;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s;
    border-radius: 999px;
  }
  .ready-banner.starting {
    color: var(--dim);
    background: rgba(253, 203, 110, 0.08);
  }
  .ready-banner.ready {
    color: var(--success);
    background: rgba(0, 184, 148, 0.25);
    font-size: 0.95rem;
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
    width: 100px;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    margin-top: 0;
    margin-left: 8px;
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
    grid-template-rows: auto auto;
    gap: 16px;
    min-width: 0;
    align-content: start;
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
    min-height: 0;
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
    grid-row: 2;
    min-height: 0;
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
    min-height: 0;
  }
  #tactic-card {
    grid-column: 2;
    grid-row: 1;
    min-height: 0;
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
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
    overflow: hidden;
  }
  .pipeline-main-row {
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
  .script-panel {
    width: 300px;
    min-width: 300px;
    max-width: 300px;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.15);
    overflow: hidden;
    flex-shrink: 0;
  }
  .dashboard-panel {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    overflow: auto;
    padding: 16px;
  }

  /* ── Scenario dropdown: fixed width, dark theme ──────── */
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

  /* Toggle buttons for SCAM/BENIGN */
  .scenario-toggle {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }
  .scenario-toggle button {
    flex: 1;
    padding: 10px 16px;
    border: 2px solid #444;
    border-radius: 6px;
    background-color: #1e1e1e;
    color: #888;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: all 0.2s ease;
  }
  .scenario-toggle button:hover:not(.active) {
    background-color: #2a2a2a;
    color: #ccc;
    border-color: #555;
  }
  .scenario-toggle button.active {
    background-color: #2a2a2a;
  }
  .scenario-toggle button.scam.active {
    border-color: var(--danger);
    color: var(--danger);
    box-shadow: 0 0 8px rgba(225, 112, 85, 0.3);
  }
  .scenario-toggle button.benign.active {
    border-color: var(--success);
    color: var(--success);
    box-shadow: 0 0 8px rgba(0, 184, 148, 0.3);
  }

  /* Scenario dropdown - dark theme, forced styling for Jetson */
  .scenario-select,
  #script-select,
  select.scenario-select {
    width: 100%;
    padding: 12px 36px 12px 12px;
    font-size: 15px;
    background-color: #1e1e1e !important;
    color: #e0e0e0 !important;
    border: 1px solid #444;
    border-radius: 6px;
    cursor: pointer;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 12px center;
  }
  .scenario-select:hover,
  #script-select:hover {
    border-color: #666;
    background-color: #252525 !important;
  }
  .scenario-select:focus,
  #script-select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(91, 191, 179, 0.2);
  }
  .scenario-select option,
  #script-select option {
    background-color: #1e1e1e !important;
    color: #e0e0e0 !important;
    padding: 10px;
  }
  .scenario-select optgroup,
  #script-select optgroup {
    font-weight: 600;
    color: var(--dim);
    background-color: #1a1a1a !important;
  }

  .scenario-type-badge {
    display: inline-block;
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

  /* Script lines styling with scam/benign distinction */
  .script-line.scam-line {
    border-left: 4px solid var(--danger);
  }
  .script-line.scam-line.current {
    background: #3a2a2a;
    border-left-color: #ff6b6b;
  }
  .script-line.benign-line {
    border-left: 4px solid var(--success);
  }
  .script-line.benign-line.current {
    background: #2a3a2a;
    border-left-color: #4ade80;
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
    padding: 8px 10px;
    margin: 6px 0;
    border-radius: 4px;
    font-size: 13px;
    line-height: 1.35;
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
  #tab-system { overflow-y: auto; overflow-x: hidden; padding-bottom: 16px; }
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
  .perf-bottom-grid {
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 16px;
    padding: 16px;
    align-items: start;
  }
  .process-bars,
  .latency-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    flex-shrink: 0;
  }
  .process-bars h3,
  .latency-panel h3 {
    font-size: 0.85rem;
    color: var(--text);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .process-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .process-row .name { width: 150px; font-size: 0.85rem; color: var(--text); }
  .process-row .bar-bg { flex: 1; min-width: 0; height: 16px; background: rgba(255,255,255,0.1); border-radius: 8px; overflow: hidden; }
  .process-row .bar-fill { height: 100%; border-radius: 8px; transition: width 0.3s; }
  .process-row .mb { width: 80px; font-size: 0.82rem; color: var(--dim); text-align: right; }
  .latency-row {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    padding: 8px 0;
    font-size: 0.92rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }
  .latency-row:last-child { border-bottom: none; }
  .latency-row .stage { color: var(--text); }
  .latency-row .ms { font-weight: 600; color: var(--accent); }
  @media (max-width: 1100px) {
    .metric-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .perf-bottom-grid { grid-template-columns: 1fr; }
  }

  /* ── Hero / About Page Styles ────────────────────────────────────── */
  :root {
    --anchor-teal: #00D9C0;
  }

  .hero-page {
    padding: 40px 24px;
    max-width: 1200px;
    margin: 0 auto;
    text-align: center;
    overflow-y: auto;
    flex: 1;
  }

  /* Logo */
  .hero-header {
    margin-bottom: 32px;
  }

  .hero-logo {
    font-size: 56px;
    font-weight: 300;
    margin-bottom: 12px;
    letter-spacing: 6px;
  }

  .logo-dot {
    color: var(--anchor-teal);
    animation: logoPulse 2s ease-in-out infinite;
    display: inline-block;
  }

  .logo-text {
    color: #ffffff;
  }

  @keyframes logoPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.1); }
  }

  .hero-tagline {
    font-size: 20px;
    color: #888;
    font-style: italic;
    font-weight: 300;
  }

  /* Hero top row: Problem + Demo side by side */
  .hero-top-row {
    display: flex;
    gap: 24px;
    justify-content: center;
    align-items: stretch;
    margin-bottom: 32px;
    flex-wrap: wrap;
  }

  @media (max-width: 800px) {
    .hero-top-row {
      flex-direction: column;
      align-items: center;
    }
  }

  /* Problem banner - the hook */
  .problem-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid var(--anchor-teal);
    border-radius: 16px;
    padding: 28px 24px;
    box-shadow: 0 0 30px rgba(0, 217, 192, 0.15);
    flex: 1;
    max-width: 400px;
    min-width: 280px;
    margin-bottom: 0;
  }

  .stat-large {
    margin-bottom: 16px;
  }

  .stat-number {
    font-size: 80px;
    font-weight: 700;
    color: var(--anchor-teal);
    display: block;
    line-height: 1;
    text-shadow: 0 0 40px rgba(0, 217, 192, 0.3);
  }

  .stat-label {
    font-size: 24px;
    color: #ccc;
    margin-top: 8px;
    display: block;
  }

  .stat-detail {
    color: #888;
    font-size: 16px;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
  }

  .stat-detail .divider {
    color: #444;
  }

  /* Value prop cards */
  .value-props {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-bottom: 48px;
    flex-wrap: wrap;
  }

  .prop-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid #333;
    border-radius: 16px;
    padding: 28px 20px;
    width: 220px;
    transition: all 0.3s ease;
  }

  .prop-card:hover {
    border-color: var(--anchor-teal);
    transform: translateY(-6px);
    box-shadow: 0 8px 24px rgba(0, 217, 192, 0.15);
  }

  .prop-icon {
    font-size: 44px;
    margin-bottom: 14px;
  }

  .prop-title {
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 10px;
  }

  .prop-desc {
    font-size: 13px;
    color: #888;
    line-height: 1.5;
  }

  /* Section titles */
  .hero-section {
    margin-bottom: 48px;
  }

  .section-title {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 4px;
    color: var(--anchor-teal);
    margin-bottom: 28px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
  }

  .section-title::before,
  .section-title::after {
    content: '';
    width: 80px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #333, transparent);
  }

  /* Pipeline visual */
  .pipeline-visual {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }

  .pipeline-step {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 18px 14px;
    min-width: 90px;
    transition: all 0.3s;
  }

  .pipeline-step:hover {
    border-color: var(--anchor-teal);
  }

  .pipeline-step.highlight {
    border-color: var(--anchor-teal);
    background: rgba(0, 217, 192, 0.1);
    box-shadow: 0 0 20px rgba(0, 217, 192, 0.2);
  }

  .step-icon {
    font-size: 28px;
    margin-bottom: 10px;
  }

  .step-label {
    font-size: 12px;
    color: #aaa;
    line-height: 1.4;
  }

  .pipeline-arrow {
    color: var(--anchor-teal);
    font-size: 24px;
    font-weight: 300;
  }

  .pipeline-note {
    font-size: 14px;
    color: #666;
  }

  .pipeline-note strong {
    color: #76B900;
  }

  /* Tech grid */
  .tech-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }

  @media (max-width: 900px) {
    .tech-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  .tech-item {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 24px 16px;
    border: 1px solid #2a2a2a;
  }

  .tech-stat {
    display: block;
    font-size: 36px;
    font-weight: 700;
    color: var(--anchor-teal);
    margin-bottom: 6px;
  }

  .tech-desc {
    font-size: 13px;
    color: #888;
    line-height: 1.4;
  }

  .tech-list {
    text-align: left;
    max-width: 560px;
    margin: 0 auto;
  }

  .tech-bullet {
    color: #ccc;
    font-size: 15px;
    margin: 12px 0;
    padding-left: 10px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  .tech-bullet .check {
    color: var(--anchor-teal);
    font-weight: bold;
    flex-shrink: 0;
  }

  /* Demo CTA */
  .demo-cta {
    background: linear-gradient(135deg, rgba(0, 217, 192, 0.12) 0%, rgba(0, 217, 192, 0.04) 100%);
    border: 2px solid var(--anchor-teal);
    border-radius: 20px;
    padding: 28px 24px;
    position: relative;
    overflow: hidden;
    flex: 1;
    max-width: 400px;
    min-width: 280px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .cta-pulse {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0, 217, 192, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: ctaPulse 3s ease-in-out infinite;
    pointer-events: none;
  }

  @keyframes ctaPulse {
    0%, 100% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.6; }
    50% { transform: translate(-50%, -50%) scale(1.3); opacity: 0; }
  }

  .cta-content {
    position: relative;
    z-index: 1;
  }

  .cta-icon {
    font-size: 56px;
    margin-bottom: 16px;
    animation: ctaBounce 2s ease-in-out infinite;
    display: inline-block;
  }

  @keyframes ctaBounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-12px); }
  }

  .demo-cta h3 {
    font-size: 28px;
    color: #fff;
    margin-bottom: 10px;
    font-weight: 600;
  }

  .demo-cta p {
    font-size: 18px;
    color: var(--anchor-teal);
    font-style: italic;
  }

  /* Footer */
  .hero-footer {
    margin-top: 48px;
    padding-top: 28px;
    border-top: 1px solid #333;
    color: #666;
    font-size: 15px;
  }

  .hero-footer p {
    margin: 6px 0;
  }

  .hero-footer strong {
    color: var(--anchor-teal);
    font-weight: 500;
  }

  /* Entrance animations */
  .hero-page .hero-top-row,
  .hero-page .value-props,
  .hero-page .hero-footer {
    animation: heroFadeInUp 0.7s ease-out forwards;
    opacity: 0;
  }

  .hero-page .hero-top-row { animation-delay: 0.1s; }
  .hero-page .value-props { animation-delay: 0.25s; }
  .hero-page .hero-footer { animation-delay: 0.4s; }

  @keyframes heroFadeInUp {
    from {
      opacity: 0;
      transform: translateY(24px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* About tab panel */
  .tab-panel.active#tab-about {
    display: flex;
    flex-direction: column;
  }

  /* ── Key Facts Banner ────────────────────────────────────────────── */
  .key-facts-banner {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    background: #0d0d0d;
    border: 2px solid var(--anchor-teal);
    border-radius: 12px;
    padding: 10px 12px;
    position: fixed;
    left: 16px;
    right: 230px;
    bottom: 16px;
    margin: 0;
    z-index: 900;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
  }

  .fact-item {
    flex: 1;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 10px;
    border-radius: 8px;
  }

  .fact-item.local { background: rgba(0, 217, 192, 0.12); }
  .fact-item.privacy { background: rgba(74, 158, 255, 0.12); }
  .fact-item.dignity { background: rgba(255, 107, 107, 0.12); }

  .fact-icon {
    font-size: 28px;
    line-height: 1;
    flex-shrink: 0;
  }

  .fact-content {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .fact-content strong {
    color: #fff;
    font-size: 14px;
    font-weight: 600;
  }

  .fact-content span {
    color: #999;
    font-size: 12px;
    line-height: 1.4;
  }

  .fact-content em {
    color: var(--anchor-teal);
    font-style: normal;
    font-weight: 600;
  }

  @media (max-width: 900px) {
    .key-facts-banner {
      flex-direction: column;
      right: 16px;
      bottom: 72px;
    }
  }

  /* ── Judge Mode Badge (floating) ─────────────────────────────────── */
  .judge-mode-badge {
    position: fixed;
    top: auto;
    bottom: 16px;
    right: 16px;
    background: rgba(255, 107, 107, 0.95);
    color: #fff;
    padding: 10px 16px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    z-index: 1000;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    animation: badgePulse 3s ease-in-out infinite;
  }

  @keyframes badgePulse {
    0%, 100% { box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4); }
    50% { box-shadow: 0 4px 24px rgba(255, 107, 107, 0.5); }
  }

  .badge-icon {
    font-size: 18px;
  }

  .badge-text {
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1px;
  }

  .badge-note {
    font-size: 10px;
    opacity: 0.9;
    margin-left: 8px;
    padding-left: 10px;
    border-left: 1px solid rgba(255, 255, 255, 0.4);
  }

  /* ── Elder Only Badge (transcript header) ───────────────────────── */
  .transcript-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .transcript-header h2 {
    margin-bottom: 0;
  }

  .elder-only-badge {
    background: rgba(74, 158, 255, 0.2);
    color: #4a9eff;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 12px;
    font-weight: 500;
    white-space: nowrap;
  }

  /* ── Demo Clarification Note ─────────────────────────────────────── */
  .demo-clarification {
    text-align: center;
    font-size: 13px;
    color: #888;
    margin-top: -24px;
    margin-bottom: 32px;
    padding: 0 20px;
  }

  .demo-clarification strong {
    color: #aaa;
  }

  /* ── Architecture Page Styles ────────────────────────────────────── */
  .tab-panel.active#tab-architecture {
    display: flex;
    flex-direction: column;
  }

  .arch-page {
    padding: 32px;
    max-width: 1100px;
    margin: 0 auto;
    overflow-y: auto;
    flex: 1;
  }

  .arch-title {
    font-size: 28px;
    color: #fff;
    margin-bottom: 8px;
    text-align: center;
  }

  .arch-subtitle {
    color: var(--anchor-teal);
    text-align: center;
    margin-bottom: 40px;
    font-size: 16px;
  }

  .arch-section {
    margin-bottom: 48px;
  }

  .arch-section h2 {
    font-size: 18px;
    color: var(--anchor-teal);
    border-bottom: 1px solid #333;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }

  /* Why Edge Grid */
  .why-edge-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }

  @media (max-width: 900px) {
    .why-edge-grid { grid-template-columns: repeat(2, 1fr); }
  }

  .why-card {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.3s;
  }

  .why-card:hover {
    border-color: var(--anchor-teal);
  }

  .why-icon {
    font-size: 32px;
    margin-bottom: 12px;
  }

  .why-card h3 {
    font-size: 14px;
    color: #fff;
    margin-bottom: 8px;
  }

  .why-card p {
    font-size: 12px;
    color: #888;
    line-height: 1.4;
  }

  /* Hardware Section */
  .hardware-spec {
    display: flex;
    gap: 32px;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .jetson-diagram {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 24px;
    position: relative;
    min-width: 280px;
  }

  .jetson-box {
    background: #76b900;
    color: #000;
    padding: 16px;
    border-radius: 4px;
    text-align: center;
  }

  .jetson-label {
    font-weight: 700;
    font-size: 14px;
  }

  .jetson-specs {
    font-size: 11px;
    margin-top: 8px;
    opacity: 0.8;
  }

  .peripheral {
    background: #2a2a2a;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    margin-top: 12px;
    color: #aaa;
  }

  .arch-spec-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    flex: 1;
    min-width: 300px;
  }

  .arch-spec-table th, .arch-spec-table td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #333;
  }

  .arch-spec-table th {
    color: var(--anchor-teal);
    font-weight: 600;
  }

  .arch-spec-table td {
    color: #ccc;
  }

  /* Pipeline Diagram */
  .arch-pipeline-diagram {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 24px;
  }

  .pipeline-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .pipeline-row.second-row {
    margin-top: 20px;
    flex-direction: row-reverse;
  }

  .pipe-box {
    background: #252525;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 16px;
    min-width: 140px;
    text-align: center;
  }

  .pipe-box.model { border-color: var(--anchor-teal); }
  .pipe-box.input { border-color: #4a9eff; }
  .pipe-box.output { border-color: #ff6b6b; }
  .pipe-box.viz { border-color: #888; border-style: dashed; }

  .pipe-icon { font-size: 24px; margin-bottom: 8px; }
  .pipe-name { font-weight: 600; color: #fff; font-size: 12px; }
  .pipe-tech { font-size: 10px; color: #888; margin-top: 4px; }
  .pipe-mem { font-size: 10px; color: var(--anchor-teal); margin-top: 4px; }
  .pipe-latency { font-size: 10px; color: #ffcc00; margin-top: 2px; }
  .pipe-note { font-size: 9px; color: #666; font-style: italic; margin-top: 4px; }

  .pipe-arrow {
    color: var(--anchor-teal);
    font-size: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .arrow-label {
    font-size: 9px;
    color: #666;
    text-align: center;
  }

  .pipeline-summary {
    display: flex;
    justify-content: space-around;
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid #333;
    flex-wrap: wrap;
    gap: 16px;
  }

  .summary-stat {
    text-align: center;
  }

  .stat-val {
    display: block;
    font-size: 24px;
    font-weight: 700;
    color: var(--anchor-teal);
  }

  .stat-desc {
    font-size: 11px;
    color: #888;
  }

  /* Detection Tiers */
  .detection-tiers {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  @media (max-width: 800px) {
    .detection-tiers { grid-template-columns: 1fr; }
  }

  .tier-box {
    background: #1a1a1a;
    border-radius: 8px;
    padding: 20px;
  }

  .tier-box.tier1 { border-left: 4px solid #ff6b6b; }
  .tier-box.tier2 { border-left: 4px solid var(--anchor-teal); }

  .tier-box h3 {
    color: #fff;
    font-size: 14px;
    margin-bottom: 12px;
  }

  .tier-desc {
    font-size: 12px;
    color: #aaa;
    line-height: 1.5;
  }

  .tier-desc p { margin: 8px 0; }
  .tier-desc ul { margin: 8px 0 8px 20px; }
  .tier-desc li { margin: 4px 0; color: #888; }
  .tier-desc strong { color: #ccc; }

  /* ZMQ Diagram */
  .zmq-intro {
    color: #aaa;
    font-size: 13px;
    margin-bottom: 16px;
  }

  .zmq-ports {
    background: #1a1a1a;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .port-item {
    display: flex;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #2a2a2a;
    flex-wrap: wrap;
    gap: 8px;
  }

  .port-item:last-child { border-bottom: none; }

  .port-num {
    background: var(--anchor-teal);
    color: #000;
    padding: 4px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
    min-width: 50px;
    text-align: center;
  }

  .port-desc {
    font-size: 12px;
    color: #aaa;
  }

  .zmq-benefits {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
  }

  .benefit {
    font-size: 12px;
    color: #888;
  }

  /* Models Table */
  .models-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  .models-table th, .models-table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #333;
  }

  .models-table th {
    background: #1a1a1a;
    color: var(--anchor-teal);
  }

  .models-table td {
    color: #aaa;
  }

  /* Proof Section */
  .proof-section {
    background: linear-gradient(135deg, rgba(0, 217, 192, 0.1) 0%, rgba(0, 217, 192, 0.02) 100%);
    border: 1px solid var(--anchor-teal);
    border-radius: 12px;
    padding: 24px;
  }

  .proof-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 16px;
  }

  @media (max-width: 800px) {
    .proof-grid { grid-template-columns: repeat(2, 1fr); }
  }

  .proof-item {
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    padding: 16px;
  }

  .proof-item h4 {
    color: #fff;
    font-size: 12px;
    margin-bottom: 8px;
  }

  .proof-item code {
    display: block;
    background: #000;
    color: var(--anchor-teal);
    padding: 8px;
    border-radius: 4px;
    font-size: 11px;
    margin-bottom: 8px;
    font-family: monospace;
  }

  .proof-item p {
    font-size: 11px;
    color: #888;
  }

  .proof-cta {
    text-align: center;
    color: var(--anchor-teal);
    font-weight: 600;
  }

  /* Architecture page entrance animations */
  .arch-page .arch-section {
    animation: heroFadeInUp 0.7s ease-out forwards;
    opacity: 0;
  }

  /* Architecture Sub-Navigation */
  .arch-subnav {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 24px;
    flex-wrap: wrap;
    padding: 12px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
  }
  .arch-subnav-btn {
    padding: 10px 20px;
    border: 1px solid #444;
    border-radius: 6px;
    background: #1a1a1a;
    color: #888;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  .arch-subnav-btn:hover {
    background: #252525;
    color: #ccc;
    border-color: #555;
  }
  .arch-subnav-btn.active {
    background: var(--anchor-teal);
    color: #000;
    border-color: var(--anchor-teal);
    font-weight: 600;
  }
  .arch-panel {
    animation: heroFadeInUp 0.5s ease-out forwards;
  }
</style>
</head>
<body>

<!-- Judge Mode Badge - visible on all pages -->
<div class="judge-mode-badge">
  <span class="badge-icon">&#128065;</span>
  <span class="badge-text">DEMO VIEW</span>
  <span class="badge-note">Elder sees nothing&mdash;only hears voice alerts</span>
</div>

<header>
  <div class="header-main-row">
    <h1><span class="status-dot" id="status-dot"></span>ANCHOR</h1>
    <div class="status-panel" id="status-panel">
      <span class="status-item" id="status-audio"><span class="status-dot-sm"></span>Audio Capture</span>
      <span class="status-item" id="status-speech"><span class="status-dot-sm"></span>Speech Recognition</span>
      <span class="status-item" id="status-stress"><span class="status-dot-sm"></span>Stress Detector</span>
      <span class="status-item" id="status-tactic"><span class="status-dot-sm"></span>Tactic Inference</span>
      <div class="ready-banner starting" id="ready-banner">
        <span id="ready-text">Starting up&hellip;</span>
        <div class="timer-bar" id="timer-bar" style="display:none;">
          <div class="timer-bar-fill" id="timer-fill"></div>
        </div>
      </div>
    </div>
  </div>
  <p>We only hear one side. Watch what we know.</p>
</header>

<nav class="tab-bar">
  <button class="tab-btn active" data-tab="about">About</button>
      <button class="tab-btn" data-tab="pipeline">Demo View</button>
  <button class="tab-btn" data-tab="system">System Performance</button>
  <button class="tab-btn" data-tab="architecture">Architecture</button>
</nav>

<div class="main-content dashboard-container">
  <div id="tab-pipeline" class="tab-panel dashboard-grid" style="display:none;">
  
  <!-- Key Facts Banner - Critical points for judges -->
  <div class="key-facts-banner">
    <div class="fact-item local">
      <div class="fact-icon">&#127968;</div>
      <div class="fact-content">
        <strong>100% Local Processing</strong>
        <span>All AI runs on this device. No cloud. No internet required. <em>Ask us to prove it!</em></span>
      </div>
    </div>
    <div class="fact-item privacy">
      <div class="fact-icon">&#128066;</div>
      <div class="fact-content">
        <strong>One-Sided Listening</strong>
        <span>We only hear the elder&rsquo;s voice. The scammer&rsquo;s audio is never captured or analyzed.</span>
      </div>
    </div>
    <div class="fact-item dignity">
      <div class="fact-icon">&#128263;</div>
      <div class="fact-content">
        <strong>Elder Doesn&rsquo;t See This</strong>
        <span>This dashboard is for judges only. The elder simply hears a calm voice intervention.</span>
      </div>
    </div>
  </div>

  <div class="pipeline-main-row">
  <aside class="script-panel">
    <div class="script-selector">
      <label>Scenario Type</label>
      <!-- Toggle buttons for SCAM/BENIGN -->
      <div class="scenario-toggle">
        <button class="scam active" id="toggle-scam" onclick="setScenarioType('scam')">SCAM</button>
        <button class="benign" id="toggle-benign" onclick="setScenarioType('benign')">BENIGN</button>
      </div>
      <div class="scenario-type-badge" id="scenario-type-badge" style="display:none;"></div>
      <label for="script-select">Select Scenario</label>
      <select id="script-select" class="scenario-select">
        <!-- Options populated by JavaScript based on toggle -->
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
    <div class="transcript-header">
      <h2>Live Transcript</h2>
      <span class="elder-only-badge">&#128066; Elder&rsquo;s voice only&mdash;caller not captured</span>
    </div>
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
  </div><!-- /pipeline-main-row -->
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
    <div class="perf-bottom-grid">
      <div class="process-bars">
        <h3>Process Memory Usage</h3>
        <div id="process-bars-container"><div class="waiting-msg">Waiting for system metrics&hellip;</div></div>
      </div>
      <div class="latency-panel">
        <h3>Pipeline Latency (avg last 10)</h3>
        <div class="latency-row"><span class="stage">Audio &rarr; Whisper</span><span class="ms" id="lat-whisper">&mdash;</span></div>
        <div class="latency-row"><span class="stage">Whisper &rarr; Analyzer</span><span class="ms" id="lat-analyzer">&mdash;</span></div>
        <div class="latency-row"><span class="stage">LLM Warning Gen</span><span class="ms" id="lat-llm">&mdash;</span></div>
        <div class="latency-row"><span class="stage">Analyzer &rarr; TTS</span><span class="ms" id="lat-tts">&mdash;</span></div>
        <div class="latency-row"><span class="stage">End-to-end</span><span class="ms" id="lat-e2e">&mdash;</span></div>
      </div>
    </div>
  </div>

  <!-- Architecture Deep Dive Tab -->
  <div id="tab-architecture" class="tab-panel" style="display:none;">
    <div class="arch-page">

      <h1 class="arch-title">Edge Architecture Deep Dive</h1>
      <p class="arch-subtitle">Everything runs on-device. No cloud. No internet required.</p>

      <!-- Architecture Sub-Navigation -->
      <div class="arch-subnav">
        <button class="arch-subnav-btn active" data-arch-section="pipeline">Pipeline</button>
        <button class="arch-subnav-btn" data-arch-section="proof">Proof</button>
        <button class="arch-subnav-btn" data-arch-section="why-edge">Why Edge</button>
        <button class="arch-subnav-btn" data-arch-section="hardware">Hardware</button>
        <button class="arch-subnav-btn" data-arch-section="detection">Detection</button>
        <button class="arch-subnav-btn" data-arch-section="models">Models</button>
      </div>

      <!-- Pipeline Architecture (FIRST) -->
      <div class="arch-section arch-panel" id="arch-pipeline">
        <h2>Pipeline Architecture</h2>
        <div class="arch-pipeline-diagram">
          <div class="pipeline-row">
            <div class="pipe-box input">
              <div class="pipe-icon">&#127908;</div>
              <div class="pipe-name">audio_capture</div>
              <div class="pipe-tech">PyAudio + Resampling</div>
              <div class="pipe-mem">~80 MB</div>
            </div>
            <div class="pipe-arrow">
              <span class="arrow-label">ZMQ<br>PCM audio</span>
              &rarr;
            </div>
            <div class="pipe-box model">
              <div class="pipe-icon">&#128172;</div>
              <div class="pipe-name">speech_recognition</div>
              <div class="pipe-tech">Whisper small.en</div>
              <div class="pipe-mem">~915 MB</div>
              <div class="pipe-latency">~700ms</div>
            </div>
            <div class="pipe-arrow">
              <span class="arrow-label">ZMQ<br>transcript</span>
              &rarr;
            </div>
            <div class="pipe-box model">
              <div class="pipe-icon">&#129504;</div>
              <div class="pipe-name">content_analyzer</div>
              <div class="pipe-tech">all-MiniLM-L6-v2</div>
              <div class="pipe-mem">~450 MB</div>
              <div class="pipe-latency">~120ms</div>
            </div>
          </div>
          <div class="pipeline-row second-row">
            <div class="pipe-box output">
              <div class="pipe-icon">&#128266;</div>
              <div class="pipe-name">audio_intervention</div>
              <div class="pipe-tech">Piper TTS + Qwen 0.5B</div>
              <div class="pipe-mem">~650 MB</div>
              <div class="pipe-latency">~500ms LLM + ~300ms TTS</div>
            </div>
            <div class="pipe-arrow reverse">
              &larr;
              <span class="arrow-label">ZMQ<br>risk + tactics</span>
            </div>
            <div class="pipe-box viz">
              <div class="pipe-icon">&#128202;</div>
              <div class="pipe-name">judges_window</div>
              <div class="pipe-tech">Flask + Chart.js</div>
              <div class="pipe-mem">~60 MB</div>
              <div class="pipe-note">(Demo only&mdash;elder doesn&rsquo;t see this)</div>
            </div>
          </div>
        </div>
        <div class="pipeline-summary">
          <div class="summary-stat">
            <span class="stat-val">~2.1 GB</span>
            <span class="stat-desc">Total RAM Usage</span>
          </div>
          <div class="summary-stat">
            <span class="stat-val">&lt;2 sec</span>
            <span class="stat-desc">End-to-End Latency</span>
          </div>
          <div class="summary-stat">
            <span class="stat-val">5 processes</span>
            <span class="stat-desc">Isolated via ZeroMQ</span>
          </div>
          <div class="summary-stat">
            <span class="stat-val">~6W</span>
            <span class="stat-desc">Power Draw</span>
          </div>
        </div>
        <!-- ZMQ Ports inline -->
        <div class="zmq-ports" style="margin-top: 20px;">
          <div class="port-item">
            <span class="port-num">:5555</span>
            <span class="port-desc">audio_capture &rarr; speech_recognition (PCM audio)</span>
          </div>
          <div class="port-item">
            <span class="port-num">:5556</span>
            <span class="port-desc">speech_recognition &rarr; content_analyzer (transcripts)</span>
          </div>
          <div class="port-item">
            <span class="port-num">:5557</span>
            <span class="port-desc">content_analyzer &rarr; audio_intervention (risk + tactics)</span>
          </div>
          <div class="port-item">
            <span class="port-num">:5558</span>
            <span class="port-desc">content_analyzer &rarr; judges_window (visualization)</span>
          </div>
        </div>
      </div>

      <!-- Proof Points (SECOND) -->
      <div class="arch-section arch-panel proof-section" id="arch-proof" style="display:none;">
        <h2>&#128269; Want Proof?</h2>
        <div class="proof-grid">
          <div class="proof-item">
            <h4>1. Network Isolation</h4>
            <code>ip addr show</code>
            <p>We can disconnect WiFi and the system keeps working.</p>
          </div>
          <div class="proof-item">
            <h4>2. Process List</h4>
            <code>ps aux | grep python</code>
            <p>See all 5 pipeline processes running locally.</p>
          </div>
          <div class="proof-item">
            <h4>3. GPU Usage</h4>
            <code>tegrastats</code>
            <p>Watch real-time GPU load during inference.</p>
          </div>
          <div class="proof-item">
            <h4>4. No External Calls</h4>
            <code>netstat -tuln</code>
            <p>Only localhost ports in use. No external connections.</p>
          </div>
        </div>
        <p class="proof-cta">&#128070; Ask us to run any of these commands live!</p>
      </div>

      <!-- Why Edge Section -->
      <div class="arch-section arch-panel" id="arch-why-edge" style="display:none;">
        <h2>Why Edge Computing?</h2>
        <div class="why-edge-grid">
          <div class="why-card">
            <div class="why-icon">&#128274;</div>
            <h3>Privacy by Design</h3>
            <p>Audio never leaves the device. No cloud storage. No transcripts shared.</p>
          </div>
          <div class="why-card">
            <div class="why-icon">&#128225;</div>
            <h3>No Internet Required</h3>
            <p>54% of low-income seniors lack broadband. Anchor works offline.</p>
          </div>
          <div class="why-card">
            <div class="why-icon">&#9889;</div>
            <h3>Real-Time Response</h3>
            <p>Cloud adds 200-500ms latency. Edge enables sub-2-second intervention.</p>
          </div>
          <div class="why-card">
            <div class="why-icon">&#128176;</div>
            <h3>No Subscription</h3>
            <p>No cloud costs. One-time hardware purchase. Sustainable protection.</p>
          </div>
        </div>
      </div>

      <!-- Hardware Section -->
      <div class="arch-section arch-panel" id="arch-hardware" style="display:none;">
        <h2>Hardware Platform</h2>
        <div class="hardware-spec">
          <div class="jetson-diagram">
            <div class="jetson-box">
              <span class="jetson-label">NVIDIA Jetson Orin Nano</span>
              <div class="jetson-specs">
                <div>6-core ARM Cortex-A78AE</div>
                <div>1024-core Ampere GPU</div>
                <div>8GB LPDDR5 RAM</div>
                <div>40 TOPS AI Performance</div>
              </div>
            </div>
            <div class="peripheral">&#127908; USB Microphone</div>
            <div class="peripheral">&#128266; Speaker Output</div>
          </div>
          <table class="arch-spec-table">
            <tr><th>Component</th><th>Specification</th></tr>
            <tr><td>Platform</td><td>NVIDIA Jetson Orin Nano 8GB</td></tr>
            <tr><td>CPU</td><td>6-core ARM @ 1.5 GHz</td></tr>
            <tr><td>GPU</td><td>1024 CUDA cores (Ampere)</td></tr>
            <tr><td>RAM</td><td>8GB LPDDR5 (shared)</td></tr>
            <tr><td>Power</td><td>5-15W (~6W active)</td></tr>
            <tr><td>OS</td><td>Ubuntu 22.04 (JetPack 6.0)</td></tr>
          </table>
        </div>
      </div>

      <!-- Detection System -->
      <div class="arch-section arch-panel" id="arch-detection" style="display:none;">
        <h2>Two-Tier Detection System</h2>
        <div class="detection-tiers">
          <div class="tier-box tier1">
            <h3>Tier 1: Exact Phrase Match</h3>
            <div class="tier-desc">
              <p><strong>Speed:</strong> &lt;1ms (regex/string matching)</p>
              <p><strong>Trigger:</strong> Known scam phrases from FBI/FTC databases</p>
              <p><strong>Examples:</strong></p>
              <ul>
                <li>&ldquo;read you the gift card numbers&rdquo;</li>
                <li>&ldquo;don&rsquo;t tell anyone about this&rdquo;</li>
                <li>&ldquo;social security number is suspended&rdquo;</li>
              </ul>
              <p><strong>Result:</strong> Immediate HIGH risk (score = 0.7+)</p>
            </div>
          </div>
          <div class="tier-box tier2">
            <h3>Tier 2: Semantic Similarity</h3>
            <div class="tier-desc">
              <p><strong>Speed:</strong> ~120ms (embedding comparison)</p>
              <p><strong>Model:</strong> all-MiniLM-L6-v2 (384-dim embeddings)</p>
              <p><strong>Method:</strong> Cosine similarity to 50+ scam scenario descriptions</p>
              <p><strong>Thresholds:</strong></p>
              <ul>
                <li>&gt;0.65 &rarr; HIGH risk</li>
                <li>&gt;0.40 &rarr; MEDIUM risk</li>
                <li>&lt;0.30 &rarr; LOW risk</li>
              </ul>
              <p><strong>Benign Override:</strong> Birthday/family context &rarr; force LOW</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Models Used -->
      <div class="arch-section arch-panel" id="arch-models" style="display:none;">
        <h2>AI Models</h2>
        <table class="models-table">
          <tr>
            <th>Model</th>
            <th>Purpose</th>
            <th>Size</th>
            <th>Inference</th>
            <th>Hardware</th>
          </tr>
          <tr>
            <td><strong>Whisper small.en</strong></td>
            <td>Speech-to-text</td>
            <td>~244 MB</td>
            <td>~700ms / 2.6s audio</td>
            <td>GPU (CUDA)</td>
          </tr>
          <tr>
            <td><strong>all-MiniLM-L6-v2</strong></td>
            <td>Semantic embeddings</td>
            <td>~90 MB</td>
            <td>~50ms / sentence</td>
            <td>CPU</td>
          </tr>
          <tr>
            <td><strong>Qwen2.5-0.5B-Instruct</strong></td>
            <td>Context-aware warnings</td>
            <td>~468 MB (Q4)</td>
            <td>~500ms / 25 tokens</td>
            <td>GPU (CUDA)</td>
          </tr>
          <tr>
            <td><strong>Piper TTS</strong></td>
            <td>Voice synthesis</td>
            <td>~60 MB</td>
            <td>~300ms / sentence</td>
            <td>CPU (ONNX)</td>
          </tr>
        </table>
      </div>

    </div>
  </div>

  <!-- About / Hero Tab -->
  <div id="tab-about" class="tab-panel active">
    <div class="hero-page">

      <!-- Top row: Problem + Demo CTA side by side -->
      <div class="hero-top-row">
        <!-- Problem statement - the hook -->
        <div class="problem-banner">
          <div class="stat-large">
            <span class="stat-number">$5B</span>
            <span class="stat-label">lost to elder fraud annually</span>
          </div>
          <div class="stat-detail">
            <span>67% involve phone scams</span>
            <span class="divider">&bull;</span>
            <span>Only 1 in 24 cases reported</span>
          </div>
        </div>

      </div>

      <!-- Value prop cards - Key differentiators -->
      <div class="value-props">
        <div class="prop-card">
          <div class="prop-icon">&#127968;</div>
          <div class="prop-title">100% On-Device</div>
          <div class="prop-desc">All AI runs locally. No cloud. No data leaves the home.</div>
        </div>
        <div class="prop-card">
          <div class="prop-icon">&#128066;</div>
          <div class="prop-title">One-Sided Only</div>
          <div class="prop-desc">We only hear your loved one. Caller is never recorded.</div>
        </div>
        <div class="prop-card">
          <div class="prop-icon">&#128266;</div>
          <div class="prop-title">Audio, Not Screens</div>
          <div class="prop-desc">Elder hears a gentle voice&mdash;no app, no tech to learn.</div>
        </div>
      </div>

      <!-- Footer -->
      <div class="hero-footer">
        <p>Built in 36 hours by <strong>Celeste Bean</strong> &amp; <strong>Harrison Lee</strong></p>
        <p>Stanford TreeHacks 2026</p>
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
        if (tab === "pipeline") {
          panel.style.display = "grid";
        } else {
          panel.style.display = "flex";
        }
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
      var names = { audio_capture: "audio_capture", speech_recognition: "speech_recognition (Whisper)", content_analyzer: "content_analyzer (Embeddings)", audio_intervention: "audio_intervention (Piper + LLM)", judges_window: "judges_window" };
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
    document.getElementById("lat-llm").textContent = pl.llm_ms != null ? pl.llm_ms + "ms" : "—";
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
  /* Scenario data organized by type */
  const SCENARIOS = {
    scam: [
      {
        id: "gift_card",
        name: "[SCAM] Gift Card Purchase",
        lines: [
          "Hello? Yes, this is she speaking.",
          "The IRS? Oh my goodness, what's wrong?",
          "I owe back taxes? I had no idea.",
          "Gift cards? That seems unusual, but okay.",
          "Yes, I can go to Walgreens right now.",
          "How many gift cards do you need?",
          "Five hundred dollars in Google Play cards, got it.",
          "Should I scratch off the back and read you the numbers?"
        ]
      },
      {
        id: "irs",
        name: "[SCAM] IRS Impersonation",
        lines: [
          "Hello? Yes, I can hear you.",
          "The IRS? Oh my, what did I do wrong?",
          "I didn't know I owed back taxes.",
          "A warrant for my arrest? Please, I don't want any trouble.",
          "How much do I owe exactly?",
          "You need my social security number to verify my identity?",
          "Okay, let me get my card. It's in my purse.",
          "My social security number is 4-8-3..."
        ]
      },
      {
        id: "tech_support",
        name: "[SCAM] Tech Support",
        lines: [
          "Hello? Microsoft support?",
          "My computer has a virus? Oh no.",
          "Yes, I have been getting strange pop-ups lately.",
          "You can fix it remotely? That would be wonderful.",
          "TeamViewer? Let me write that down.",
          "Okay, I'm downloading it now.",
          "The code on my screen is 4-5-7-8-9-2.",
          "You need my bank login to process the refund?"
        ]
      },
      {
        id: "grandchild",
        name: "[SCAM] Grandchild Emergency",
        lines: [
          "Hello? Who is this?",
          "Tommy? Is that you? You sound different, sweetie.",
          "Oh no, you were in a car accident?",
          "You're in jail? What happened?",
          "Of course I'll help you, don't worry.",
          "How much do you need for bail?",
          "Three thousand dollars? That's a lot, but okay.",
          "Don't worry, I won't tell your parents about this."
        ]
      },
      {
        id: "romance",
        name: "[SCAM] Romance Scam",
        lines: [
          "Oh, it's so good to hear your voice!",
          "I feel so lucky we found each other online.",
          "Of course I trust you, darling.",
          "You're stuck overseas? That's terrible.",
          "I wish I could be there to help you.",
          "How much do you need to get home?",
          "Two thousand dollars? I can figure that out.",
          "Western Union? I'll go there today, don't worry."
        ]
      },
      {
        id: "ssn_suspension",
        name: "[SCAM] SSN Suspension",
        lines: [
          "Hello? Social Security Administration?",
          "My social security number is suspended?",
          "Oh my God, I don't understand what happened.",
          "I don't want to be arrested, please help me.",
          "What do I need to do to fix this?",
          "Pay a fine? How much is it?",
          "Gift cards? That's how I pay the government?",
          "Okay, I'll go get them right now."
        ]
      },
      {
        id: "utility",
        name: "[SCAM] Utility Shutoff",
        lines: [
          "Hello? The electric company?",
          "You're going to shut off my power today?",
          "But I thought I paid my bill already.",
          "I must have missed a payment? I'm so sorry.",
          "I have to pay right now or you'll cut it off?",
          "I don't have my checkbook with me.",
          "A Bitcoin ATM? I've never used one of those.",
          "There's one at the gas station? Okay, I'll go now."
        ]
      },
      {
        id: "bank_fraud",
        name: "[SCAM] Bank Fraud Alert",
        lines: [
          "Hello? Yes, this is my phone number.",
          "You're calling from my bank's fraud department?",
          "Someone's been using my account fraudulently?",
          "Oh no, how much did they take?",
          "Yes, I'll verify my information with you.",
          "My account number? Let me get my checkbook.",
          "You need me to transfer money to a safe account?",
          "How do I do a wire transfer?"
        ]
      },
      {
        id: "lottery",
        name: "[SCAM] Lottery Winner",
        lines: [
          "Hello? I won something?",
          "A million dollars? Are you serious?",
          "I don't remember entering, but that's wonderful!",
          "What do I need to do to claim my prize?",
          "A processing fee? How much is it?",
          "Two thousand dollars? That seems like a lot.",
          "I have to pay with gift cards?",
          "Okay, if that's how it works. Where do I get them?"
        ]
      },
      {
        id: "medicare",
        name: "[SCAM] Medicare Scam",
        lines: [
          "Hello? Medicare services?",
          "Yes, I'm on Medicare, that's correct.",
          "A new card? I didn't know I needed a new one.",
          "Oh, the old ones are expiring? I see.",
          "My Medicare number? Let me get my card.",
          "It's 1-E-G-4... let me find my glasses.",
          "You need my bank account for the deposit?",
          "For the refund? Okay, let me get my checkbook."
        ]
      }
    ],
    benign: [
      {
        id: "birthday_gift",
        name: "[BENIGN] Birthday Gift Shopping",
        lines: [
          "Hi honey, I'm at Target right now.",
          "I'm buying a gift card for Tommy's birthday.",
          "He loves video games, so I'm getting PlayStation.",
          "Fifty dollars should be a nice gift, right?",
          "They have such cute birthday cards here too.",
          "I'll wrap it up nice when I get home.",
          "The party is Saturday at two o'clock.",
          "Should I bring my famous chocolate cake?"
        ]
      },
      {
        id: "family_call",
        name: "[BENIGN] Catching Up With Family",
        lines: [
          "Hi sweetie! I'm so glad you called.",
          "How are the kids doing in school?",
          "That's wonderful! Tommy got an A? I'm so proud.",
          "And how's work going for you?",
          "Let's have dinner together this Sunday.",
          "I'll make your favorite pot roast.",
          "Bring the kids, I haven't seen them in weeks.",
          "I love you too, dear. See you Sunday!"
        ]
      },
      {
        id: "doctor",
        name: "[BENIGN] Doctor Appointment",
        lines: [
          "Hello? Yes, this is Margaret speaking.",
          "Oh, Dr. Johnson's office! Thank you for calling.",
          "Yes, I have my appointment tomorrow at 2pm.",
          "My prescription is ready at the pharmacy?",
          "That's wonderful, I was running low.",
          "I'll pick it up on my way home from the checkup.",
          "Do I need to fast before the blood work?",
          "Okay, no food after midnight. Got it!"
        ]
      },
      {
        id: "bank_legit",
        name: "[BENIGN] Legitimate Bank Call",
        lines: [
          "Yes, I actually called the bank earlier today.",
          "I had a question about a charge on my statement.",
          "I'm looking at my statement right now.",
          "The charge on February 3rd, that was me.",
          "I bought groceries at Safeway that day.",
          "Yes, everything else looks correct too.",
          "Thank you for following up on my call.",
          "I appreciate your help. Have a nice day!"
        ]
      },
      {
        id: "food_order",
        name: "[BENIGN] Ordering Food",
        lines: [
          "Hi, I'd like to place an order for delivery.",
          "A large pepperoni pizza, please.",
          "And an order of garlic bread.",
          "Yes, delivery to 123 Oak Street.",
          "The apartment number is 4B.",
          "How long will that take?",
          "Thirty minutes? Perfect.",
          "I'll pay with my credit card when it arrives."
        ]
      },
      {
        id: "trip_planning",
        name: "[BENIGN] Planning a Trip",
        lines: [
          "Hi dear, I'm so excited about our trip!",
          "We're thinking about visiting Florida next month.",
          "The grandkids really want to go to Disney World.",
          "I found a nice hotel near the park.",
          "It has a pool, the kids will love it.",
          "I'll book everything online tonight.",
          "Should we fly or drive down there?",
          "Let me check the flight prices first."
        ]
      },
      {
        id: "church",
        name: "[BENIGN] Church Event",
        lines: [
          "Hello, Pastor Williams! How are you?",
          "Yes, I'm planning to come to the potluck Saturday.",
          "It starts at noon at the fellowship hall, right?",
          "I'm bringing my famous apple pie.",
          "The one with the crumb topping everyone loves.",
          "Should I pick up Helen on the way?",
          "She doesn't drive anymore, poor thing.",
          "See you Saturday! God bless."
        ]
      },
      {
        id: "home_repair",
        name: "[BENIGN] Home Maintenance",
        lines: [
          "Hello? Yes, this is the Johnson residence.",
          "Oh, the plumber! Thanks for calling back.",
          "The kitchen sink has been leaking for days.",
          "Tomorrow morning works perfectly for me.",
          "You quoted two hundred dollars, right?",
          "That sounds reasonable to me.",
          "I'll have coffee ready when you get here.",
          "Thank you so much. See you at nine!"
        ]
      },
      {
        id: "neighbor",
        name: "[BENIGN] Talking to Neighbor",
        lines: [
          "Good morning, Helen! Beautiful weather today.",
          "How's your garden coming along?",
          "My tomatoes are finally turning red!",
          "It's been such a good summer for vegetables.",
          "Would you like some from my garden?",
          "I have more zucchini than I know what to do with.",
          "Come by this afternoon, I'll have a bag ready.",
          "We should have tea together soon!"
        ]
      },
      {
        id: "tv_chat",
        name: "[BENIGN] TV Discussion",
        lines: [
          "Hi Susan! Did you watch that show last night?",
          "The mystery on Netflix, I forget the name.",
          "It was so good, I couldn't stop watching!",
          "I stayed up way past my bedtime.",
          "You haven't seen it yet? Oh, you have to!",
          "No spoilers, I promise.",
          "Come over Saturday and we'll watch together.",
          "I'll make popcorn. See you then!"
        ]
      }
    ]
  };

  /* Build flat SCRIPTS object for backward compatibility */
  const SCRIPTS = {};
  const SCAM_KEYS = [];
  const BENIGN_KEYS = [];

  SCENARIOS.scam.forEach(function(s) {
    SCRIPTS[s.id] = s.lines;
    SCAM_KEYS.push(s.id);
  });
  SCENARIOS.benign.forEach(function(s) {
    SCRIPTS[s.id] = s.lines;
    BENIGN_KEYS.push(s.id);
  });

  let currentScenarioType = "scam";

  const scriptSelect = document.getElementById("script-select");
  const scriptLinesEl = document.getElementById("script-lines");
  const teleprompterPlaceholder = document.getElementById("teleprompter-placeholder");
  const scriptProgress = document.getElementById("script-progress");
  const currentLineNumEl = document.getElementById("current-line-num");
  const totalLinesEl = document.getElementById("total-lines");
  const scenarioTypeBadge = document.getElementById("scenario-type-badge");
  const toggleScamBtn = document.getElementById("toggle-scam");
  const toggleBenignBtn = document.getElementById("toggle-benign");

  let currentScript = null;
  let currentLineIndex = 0;

  /* ── Toggle between SCAM and BENIGN scenarios ─────────────────── */
  window.setScenarioType = function(type) {
    currentScenarioType = type;

    /* Update toggle button styling */
    if (toggleScamBtn && toggleBenignBtn) {
      toggleScamBtn.classList.remove("active");
      toggleBenignBtn.classList.remove("active");
      if (type === "scam") {
        toggleScamBtn.classList.add("active");
      } else {
        toggleBenignBtn.classList.add("active");
      }
    }

    /* Populate dropdown with scenarios of this type */
    scriptSelect.innerHTML = "";
    var defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "— Select " + type + " scenario —";
    scriptSelect.appendChild(defaultOption);

    SCENARIOS[type].forEach(function(scenario) {
      var option = document.createElement("option");
      option.value = scenario.id;
      option.textContent = scenario.name;
      scriptSelect.appendChild(option);
    });

    /* Reset current script */
    currentScript = null;
    currentLineIndex = 0;
    renderScript();
  };

  function renderScript() {
    if (!currentScript || !SCRIPTS[currentScript]) {
      teleprompterPlaceholder.style.display = "block";
      scriptLinesEl.style.display = "none";
      scriptProgress.style.display = "none";
      if (scenarioTypeBadge) scenarioTypeBadge.style.display = "none";
      return;
    }
    var isBenign = BENIGN_KEYS.indexOf(currentScript) >= 0;
    var lineTypeClass = isBenign ? "benign-line" : "scam-line";

    if (scenarioTypeBadge) {
      scenarioTypeBadge.style.display = "inline-block";
      scenarioTypeBadge.textContent = isBenign ? "BENIGN" : "SCAM";
      scenarioTypeBadge.className = "scenario-type-badge " + (isBenign ? "benign" : "scam");
    }
    const lines = SCRIPTS[currentScript];
    teleprompterPlaceholder.style.display = "none";
    scriptLinesEl.style.display = "block";
    scriptProgress.style.display = "block";
    scriptLinesEl.innerHTML = "";
    totalLinesEl.textContent = lines.length;
    lines.forEach(function(line, i) {
      const div = document.createElement("div");
      var cls = "script-line " + lineTypeClass;
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
    } else if (e.key === "s" || e.key === "S") {
      e.preventDefault();
      setScenarioType("scam");
    } else if (e.key === "b" || e.key === "B") {
      e.preventDefault();
      setScenarioType("benign");
    } else if (e.key >= "1" && e.key <= "9") {
      var scenarios = SCENARIOS[currentScenarioType];
      var idx = parseInt(e.key, 10) - 1;
      if (idx < scenarios.length) {
        selectScript(scenarios[idx].id);
      }
    } else if (e.key === "0") {
      var scenarios = SCENARIOS[currentScenarioType];
      if (scenarios.length >= 10) {
        selectScript(scenarios[9].id);
      }
    }
  });

  document.getElementById("teleprompter").addEventListener("click", function() {
    advanceLine();
  });

  /* ── Initialize with SCAM scenarios on page load ──────────────── */
  setScenarioType("scam");

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

  /* ── Architecture sub-tab switching ─────────────────────────────── */
  document.querySelectorAll(".arch-subnav-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var section = this.dataset.archSection;
      /* Update button states */
      document.querySelectorAll(".arch-subnav-btn").forEach(function(b) {
        b.classList.remove("active");
      });
      this.classList.add("active");
      /* Show/hide panels */
      document.querySelectorAll(".arch-panel").forEach(function(panel) {
        panel.style.display = "none";
      });
      var targetPanel = document.getElementById("arch-" + section);
      if (targetPanel) {
        targetPanel.style.display = "block";
      }
    });
  });
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
