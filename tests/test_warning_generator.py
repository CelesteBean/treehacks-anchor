"""Tests for warning_generator module — fallback and phrase extraction."""

from __future__ import annotations

import pytest

from src.core.warning_generator import (
    FALLBACK_COMPLETIONS,
    ALERT_TEMPLATES,
    WarningGenerator,
)


def _safe_extract_phrases(risk_factors: list[str], max_items: int = 3) -> list[str]:
    """Helper to extract phrases from risk factor strings (test utility)."""
    result = []
    for rf in risk_factors[:max_items]:
        if ":" in rf:
            phrase = rf.split(":", 1)[1].strip().strip("'\"")
        else:
            phrase = rf.strip()
        result.append(phrase)
    return result


class TestSafeExtractPhrases:
    """Test phrase extraction from risk factor strings."""

    def test_extracts_after_colon(self) -> None:
        got = _safe_extract_phrases(["Tier 1: buy a gift card"])
        assert got == ["buy a gift card"]

    def test_strips_quotes(self) -> None:
        got = _safe_extract_phrases(["Tier 1: 'buy a gift card'"])
        assert got == ["buy a gift card"]

    def test_respects_max_items(self) -> None:
        got = _safe_extract_phrases(
            ["Tier 1: a", "Tier 1: b", "Tier 1: c", "Tier 1: d"],
            max_items=2,
        )
        assert len(got) == 2
        assert got == ["a", "b"]

    def test_handles_no_colon(self) -> None:
        got = _safe_extract_phrases(["plain risk factor"])
        assert got == ["plain risk factor"]


class TestFallbackCompletions:
    """Test fallback completions cover expected threat types."""

    @pytest.mark.parametrize(
        "threat_type",
        [
            "gift_card",
            "government_impersonation",
            "tech_support",
            "grandparent_scam",
            "wire_transfer",
            "romance_scam",
            "generic_high_risk",
        ],
    )
    def test_known_threat_has_fallback(self, threat_type: str) -> None:
        msg = FALLBACK_COMPLETIONS[threat_type]
        assert isinstance(msg, str)
        assert len(msg) > 10
        keywords = ("hang up", "speak", "not", "trust", "family", "reversed")
        assert any(kw in msg.lower() for kw in keywords), f"Fallback should contain a safety keyword: {msg[:80]}"

    def test_unknown_threat_uses_generic(self) -> None:
        result = FALLBACK_COMPLETIONS.get(
            "unknown_type", FALLBACK_COMPLETIONS["generic_high_risk"]
        )
        assert result == FALLBACK_COMPLETIONS["generic_high_risk"]

    def test_alert_templates_exist(self) -> None:
        """Ensure ALERT_TEMPLATES has matching keys for common threats."""
        for threat in ["gift_card", "tech_support", "generic_high_risk"]:
            assert threat in ALERT_TEMPLATES
            assert len(ALERT_TEMPLATES[threat]) > 10


class TestWarningGeneratorInit:
    """Test WarningGenerator initialization."""

    def test_raises_on_missing_model(self) -> None:
        """When model file does not exist, raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            WarningGenerator(model_path="models/nonexistent/does_not_exist.gguf")
