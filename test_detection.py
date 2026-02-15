#!/usr/bin/env python3
"""Comprehensive test for scam detection in content_analyzer.

Tests exact phrases against expected risk levels. Exit code 0 if ≥80% pass.

Usage: python test_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.content_analyzer import ContentAnalyzer

# (phrase, expected_risk_level) - expected is one of "high", "medium", "low"
TEST_PHRASES = [
    # Should detect as HIGH risk (gift card scam)
    ("Yes I will buy a gift card and send it to you", "high"),
    ("Yes I will buy the gift cards right now", "high"),
    ("I am going to the store to get gift cards for you", "medium"),
    ("Let me read you the numbers on the back of the card", "high"),
    # Should detect as MEDIUM risk (suspicious)
    ("You need remote access to my computer?", "medium"),
    ("I'll download that software you mentioned", "medium"),
    # Should be LOW risk (benign)
    ("Let me get my credit card for the purchase", "low"),
    ("I'm buying a gift card for my grandson's birthday", "low"),
    ("The weather is nice today", "low"),
]

# Accept "high" when "medium" is expected (conservative OK) for some phrases
ACCEPT_HIGH_WHEN_MEDIUM = {"I'll download that software you mentioned"}


def main() -> int:
    analyzer = ContentAnalyzer(embedding_model="all-MiniLM-L6-v2")

    passed = 0
    failed = []

    print("=" * 75)
    print("SCAM DETECTION TEST")
    print("=" * 75)

    for phrase, expected in TEST_PHRASES:
        result = analyzer.analyze(phrase)
        actual = result["risk_level"].lower()
        risk_score = result["risk_score"]

        # Check pass: actual matches expected, or actual=high when expected=medium (OK)
        if actual == expected:
            status = "PASS"
            passed += 1
        elif expected == "medium" and actual == "high" and phrase in ACCEPT_HIGH_WHEN_MEDIUM:
            status = "PASS (high acceptable for medium)"
            passed += 1
        else:
            status = "FAIL"
            failed.append((phrase, expected, actual))

        print(f"\n{status}: {phrase!r}")
        print(f"  Expected: {expected} | Got: {actual} (score={risk_score:.2f})")
        if result["risk_factors"]:
            for rf in result["risk_factors"][:2]:
                print(f"    {rf}")

    total = len(TEST_PHRASES)
    pct = 100 * passed / total if total else 0
    print("\n" + "=" * 75)
    print(f"Result: {passed}/{total} passed ({pct:.0f}%)")
    if failed:
        print("\nFailed:")
        for phrase, exp, act in failed:
            print(f"  {phrase!r}: expected {exp}, got {act}")
    print("=" * 75)

    return 0 if pct >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
