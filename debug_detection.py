#!/usr/bin/env python3
"""Diagnose scam detection and intervention trigger regressions."""

import inspect
import sys

sys.path.insert(0, ".")

from src.core.content_analyzer import (  # noqa: E402
    BENIGN_PATTERNS,
    SCAM_SCENARIOS,
    TIER1_PHRASES,
    ContentAnalyzer,
)
from src.core.audio_intervention import AudioIntervention  # noqa: E402


MUST_BE_HIGH = [
    "Yes I'll buy the gift cards and read you the numbers",
    "Let me give you my social security number",
    "I'll download TeamViewer and give you remote access",
    "Don't worry I won't tell my parents about the bail money",
    "I'll go to Western Union right now to wire you money",
    "My Medicare number is 1EG4",
    "I'll pay the processing fee with gift cards",
    "You need me to transfer money to a safe account",
]

SHOULD_BE_MEDIUM = [
    "The IRS is calling me?",
    "You're going to shut off my power?",
    "I won a million dollars?",
]

MUST_BE_LOW = [
    "I'm buying a gift card for my grandson's birthday",
    "The church potluck is Saturday",
    "My doctor appointment is tomorrow",
]


def print_result_header(title: str) -> None:
    print(f"\n### {title} ###\n")


def main() -> int:
    analyzer = ContentAnalyzer()

    print("=" * 70)
    print("SCAM DETECTION DIAGNOSTIC")
    print("=" * 70)

    print_result_header("MUST BE HIGH (intervention should trigger)")
    must_high_pass = 0
    for phrase in MUST_BE_HIGH:
        result = analyzer.analyze(phrase)
        level = result["risk_level"]
        score = result["risk_score"]
        status = "OK" if level == "high" else "FAIL"
        if level == "high":
            must_high_pass += 1
        print(f"{status:5} [{level.upper():6}] (score={score:.2f}) {phrase[:55]}")
        if level != "high":
            print("      Risk factors:")
            for rf in result.get("risk_factors", [])[:4]:
                print(f"        - {rf}")
            print(f"      benign_override={result.get('benign_override', False)}")
            print(f"      benign_match={result.get('benign_match', 'none')}")

    print_result_header("SHOULD BE MEDIUM")
    for phrase in SHOULD_BE_MEDIUM:
        result = analyzer.analyze(phrase)
        level = result["risk_level"]
        score = result["risk_score"]
        status = "OK" if level in {"medium", "high"} else "CHECK"
        print(f"{status:5} [{level.upper():6}] (score={score:.2f}) {phrase[:55]}")

    print_result_header("MUST BE LOW (no false positives)")
    must_low_pass = 0
    for phrase in MUST_BE_LOW:
        result = analyzer.analyze(phrase)
        level = result["risk_level"]
        score = result["risk_score"]
        status = "OK" if level == "low" else "FALSE+"
        if level == "low":
            must_low_pass += 1
        print(f"{status:6} [{level.upper():6}] (score={score:.2f}) {phrase[:55]}")

    print("\n" + "=" * 70)
    print("CONFIGURATION CHECK")
    print("=" * 70)
    print(f"\nTier 1 phrases count: {len(TIER1_PHRASES)}")
    print(f"Scam scenarios count: {len(SCAM_SCENARIOS)}")
    print(f"Benign patterns count: {len(BENIGN_PATTERNS)}")

    analyze_source = inspect.getsource(analyzer.analyze)
    if "semantic_score > 0.65" in analyze_source:
        print("Tier 2 high threshold: semantic_score > 0.65")
    if "semantic_score > 0.40" in analyze_source:
        print("Tier 2 medium threshold: semantic_score > 0.40")
    if "risk_score = max(risk_score, 0.7)" in analyze_source:
        print("Tier 1 risk score assignment: 0.7")

    should_source = inspect.getsource(AudioIntervention.should_intervene)
    medium_high_trigger = '("medium", "high")' in should_source
    print(
        "Intervention trigger accepts medium/high:"
        f" {'yes' if medium_high_trigger else 'no'}"
    )

    print("\n" + "=" * 70)
    print("PASS COUNTS")
    print("=" * 70)
    print(f"MUST_BE_HIGH passing: {must_high_pass}/{len(MUST_BE_HIGH)}")
    print(f"MUST_BE_LOW passing: {must_low_pass}/{len(MUST_BE_LOW)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
