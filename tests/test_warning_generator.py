"""Tests for warning_generator module — fallback and phrase extraction."""

from __future__ import annotations

import pytest

from src.core.warning_generator import (
    FALLBACK_TEMPLATES,
    _safe_extract_phrases,
    WarningGenerator,
)


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


class TestFallbackTemplates:
    """Test fallback templates cover expected threat types."""

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
        msg = FALLBACK_TEMPLATES[threat_type]
        assert isinstance(msg, str)
        assert len(msg) > 10
        keywords = ("scam", "warning", "stop", "cautious", "risky", "hang up")
        assert any(kw in msg.lower() for kw in keywords), f"Fallback should contain a safety keyword: {msg[:80]}"

    def test_unknown_threat_uses_generic(self) -> None:
        result = FALLBACK_TEMPLATES.get(
            "unknown_type", FALLBACK_TEMPLATES["generic_high_risk"]
        )
        assert result == FALLBACK_TEMPLATES["generic_high_risk"]


class TestWarningGeneratorInit:
    """Test WarningGenerator initialization."""

    def test_raises_on_missing_model(self) -> None:
        """When model file does not exist, raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            WarningGenerator(model_path="models/nonexistent/does_not_exist.gguf")
