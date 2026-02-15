#!/usr/bin/env python3
"""Parse Anchor pipeline logs and produce a unified timeline summary.

Reads /tmp/analyzer.log, /tmp/speech.log, /tmp/intervention.log and:
- Shows time range
- Counts transcripts processed, detections by level
- Lists interventions triggered
- Shows recent transcript detail
- Highlights potential issues (gaps, etc.)

Usage: python analyze_logs.py [--analyzer PATH] [--speech PATH] [--intervention PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

LOG_PATHS = {
    "analyzer": "/tmp/analyzer.log",
    "speech": "/tmp/speech.log",
    "intervention": "/tmp/intervention.log",
}

# Regex patterns for log lines (timestamp format: 2026-02-14 20:25:33,123)
TS_PAT = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)"
# Example: 2026-02-14 20:25:33,123 or 2026-02-14 20:25:33
TS_FMT = "%Y-%m-%d %H:%M:%S"
TS_FMT_MS = "%Y-%m-%d %H:%M:%S,%f"


def parse_ts(s: str) -> datetime | None:
    """Parse timestamp from string."""
    s = s.strip()
    try:
        if "," in s:
            return datetime.strptime(s[:26], TS_FMT_MS)
        return datetime.strptime(s[:19], TS_FMT)
    except ValueError:
        return None


def parse_analyzer_log(path: Path) -> list[dict]:
    """Extract analyzer events: Published risk=... and DETECTION lines."""
    events = []
    # [ANALYZER] Published risk=low (0.00) 17 words in 45ms
    # [ANALYZER] [PUBLISH] transcript='Yes I will buy...' port=5558
    # [ANALYZER] DETECTION HIGH (0.70): 'Yes I will buy...' tier1=['buy a gift card'] tier2=0.41 benign=False
    pub_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?).*\[ANALYZER\] Published risk=(\w+) \((\d+\.?\d*)\) (\d+) words"
    )
    pub_debug_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?).*\[ANALYZER\] \[PUBLISH\] transcript=(.+?) port=\d+"
    )
    det_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?).*\[ANALYZER\] DETECTION (\w+) \((\d+\.?\d*)\): (.+)$"
    )
    for line in path.read_text(errors="replace").splitlines():
        m = pub_re.search(line)
        if m:
            ts = parse_ts(m.group(1))
            events.append({
                "ts": ts, "type": "published",
                "risk_level": m.group(2).lower(), "risk_score": float(m.group(3)),
                "words": int(m.group(4)), "raw": line[:120], "transcript": "",
            })
            continue
        m = pub_debug_re.search(line)
        if m:
            transcript = m.group(2).strip("'\"")[:80]
            if events and events[-1]["type"] == "published":
                events[-1]["transcript"] = transcript
            continue
        m = det_re.search(line)
        if m:
            ts = parse_ts(m.group(1))
            transcript = m.group(4).split(" tier1=")[0].strip("'\"")[:80]
            events.append({
                "ts": ts, "type": "detection",
                "risk_level": m.group(2).lower(), "risk_score": float(m.group(3)),
                "transcript": transcript, "raw": line[:120],
            })
    # Attach last_transcript to most recent pub event if we parsed a PUBLISH after
    return events


def parse_speech_log(path: Path) -> list[dict]:
    """Extract speech events: Transcribed ... audio in ...ms: ..."""
    events = []
    # [SPEECH] Transcribed 2.50s audio in 1234ms: Yes I will buy a gift card
    re_speech = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?).*\[SPEECH\] Transcribed ([\d.]+)s audio in ([\d.]+)ms: (.+)$"
    )
    for line in path.read_text(errors="replace").splitlines():
        m = re_speech.search(line)
        if m:
            ts = parse_ts(m.group(1))
            text = (m.group(4) or "").strip()
            if not text or text == "(silence)":
                continue
            events.append({
                "ts": ts, "type": "transcribed",
                "duration_s": float(m.group(2)), "latency_ms": float(m.group(3)),
                "text": text[:80], "raw": line[:120],
            })
    return events


def parse_intervention_log(path: Path) -> list[dict]:
    """Extract intervention events: RECV, DECIDE, INTERVENTION."""
    events = []
    # [INTERVENTION] [RECV] msg #1 risk=high score=0.70 transcript='...'
    # [INTERVENTION] INTERVENTION [gift_card]: Warning. Someone is asking...
    recv_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?).*\[INTERVENTION\] \[RECV\] msg #(\d+) risk=(\w+) score=([\d.]+)"
    )
    interv_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?).*\[INTERVENTION\] INTERVENTION \[(\w+)\]: (.+)$"
    )
    for line in path.read_text(errors="replace").splitlines():
        m = recv_re.search(line)
        if m:
            ts = parse_ts(m.group(1))
            transcript = ""
            if " transcript=" in line:
                idx = line.find(" transcript=")
                transcript = line[idx + 12:idx + 90].strip("'\"")
            events.append({
                "ts": ts, "type": "recv",
                "msg_num": int(m.group(2)), "risk_level": m.group(3).lower(),
                "risk_score": float(m.group(4)), "transcript": transcript,
            })
            continue
        m = interv_re.search(line)
        if m:
            ts = parse_ts(m.group(1))
            events.append({
                "ts": ts, "type": "intervention",
                "scam_type": m.group(2), "warning": m.group(3)[:60],
            })
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Anchor pipeline logs")
    parser.add_argument("--analyzer", default=LOG_PATHS["analyzer"])
    parser.add_argument("--speech", default=LOG_PATHS["speech"])
    parser.add_argument("--intervention", default=LOG_PATHS["intervention"])
    args = parser.parse_args()

    analyzer_path = Path(args.analyzer)
    speech_path = Path(args.speech)
    intervention_path = Path(args.intervention)

    analyzer_events = parse_analyzer_log(analyzer_path) if analyzer_path.exists() else []
    speech_events = parse_speech_log(speech_path) if speech_path.exists() else []
    intervention_events = parse_intervention_log(intervention_path) if intervention_path.exists() else []

    # Time range
    all_ts = [
        e["ts"] for e in analyzer_events + speech_events + intervention_events
        if e.get("ts")
    ]
    if not all_ts:
        print("No events found in logs. Ensure pipeline has run with --debug.")
        return 1

    t_min = min(all_ts)
    t_max = max(all_ts)
    duration_s = (t_max - t_min).total_seconds()

    print("=" * 60)
    print("Pipeline Log Analysis")
    print("=" * 60)
    print(f"\nTime range: {t_min.strftime('%H:%M:%S')} - {t_max.strftime('%H:%M:%S')} ({duration_s:.0f}s)")

    # Transcripts
    pub_events = [e for e in analyzer_events if e.get("type") == "published"]
    high = sum(1 for e in pub_events if e.get("risk_level") == "high")
    med = sum(1 for e in pub_events if e.get("risk_level") == "medium")
    low = sum(1 for e in pub_events if e.get("risk_level") == "low")

    print(f"\nTranscripts processed: {len(pub_events)}")
    print(f"  - high risk:   {high}")
    print(f"  - medium risk: {med}")
    print(f"  - low risk:    {low}")

    # Interventions
    interv_played = [e for e in intervention_events if e.get("type") == "intervention"]
    print(f"\nInterventions: {len(interv_played)}")
    for e in interv_played[:5]:
        ts = e.get("ts", "")
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        print(f"  {ts_str} [{e.get('scam_type', '?')}] \"{e.get('warning', '')[:50]}...\"")

    # Potential issues
    issues = []
    if speech_events and not pub_events:
        issues.append("Speech transcribed but analyzer published nothing (check min_words / interval)")
    if pub_events and not intervention_events:
        issues.append("Analyzer published but intervention received nothing (check ZMQ port 5558)")

    print(f"\nPotential issues:")
    if not issues:
        print("  - No issues detected")
    else:
        for i in issues:
            print(f"  - {i}")

    # Recent transcript detail
    print(f"\nRecent transcript detail (last 10):")
    for e in sorted(pub_events, key=lambda x: x.get("ts") or datetime.min)[-10:]:
        ts = e.get("ts", "")
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        level = {"low": "LOW", "medium": "MED", "high": "HIGH"}.get((e.get("risk_level") or "low").lower(), "???")
        score = e.get("risk_score", 0)
        words = e.get("words", 0)
        transcript = (e.get("transcript") or "").strip() or "(no transcript)"
        t_preview = transcript[:55] + "…" if len(transcript) > 55 else transcript
        print(f"  {ts_str} {level:3} ({score:.2f}) {words:2}w  \"{t_preview}\"")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
