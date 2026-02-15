"""Lightweight LLM warning generator.

Generates context-aware scam warnings in <500ms using Qwen2.5-0.5B-Instruct.
Falls back to static templates if the LLM is unavailable or fails.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback templates (same semantics as audio_intervention.INTERVENTION_TEMPLATES)
FALLBACK_TEMPLATES = {
    "gift_card": (
        "Warning. This caller is asking for gift card payments. "
        "Legitimate organizations never request gift cards. Please hang up."
    ),
    "government_impersonation": (
        "Stop. Government agencies never call demanding immediate payment. "
        "This is likely a scam. Please hang up."
    ),
    "tech_support": (
        "Warning. This caller wants access to your computer. "
        "Legitimate tech companies don't cold call. Please hang up."
    ),
    "grandparent_scam": (
        "Stop. Please verify this is really your family member "
        "by calling them directly on a number you trust."
    ),
    "wire_transfer": (
        "Stop. Someone is asking you to wire money. "
        "These payments cannot be reversed. Please hang up."
    ),
    "romance_scam": (
        "Please be cautious. Sending money to someone you have not met in person "
        "is very risky."
    ),
    "generic_high_risk": (
        "Warning. This call shows signs of a scam. "
        "Do not share personal information or send money."
    ),
}

DEFAULT_MODEL_PATH = "models/qwen-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"


def _safe_extract_phrases(risk_factors: list[str], max_items: int = 3) -> list[str]:
    """Extract human-readable phrases from risk factor strings."""
    phrases: list[str] = []
    for rf in risk_factors[:max_items]:
        if ": " in rf:
            part = rf.split(": ", 1)[1]
            # Remove surrounding quotes if present
            if part.startswith("'") and part.endswith("'"):
                part = part[1:-1]
            phrases.append(part.strip())
        else:
            phrases.append(rf.strip())
    return phrases


class WarningGenerator:
    """Generates context-aware scam warnings via on-device LLM."""

    def __init__(self, model_path: str | None = None) -> None:
        """Load the GGUF model. Raises if the model file is missing or load fails."""
        resolved = self._resolve_model_path(model_path or DEFAULT_MODEL_PATH)
        if not resolved.exists():
            raise FileNotFoundError(f"LLM model not found: {resolved}")

        logger.info("Loading LLM from %s", resolved)
        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(resolved),
                n_ctx=512,
                n_threads=4,
                n_gpu_layers=20,
                verbose=False,
                chat_format="chatml",
            )
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for LLM warnings. "
                "Install with: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python"
            ) from e

        logger.info("LLM loaded successfully")

    def _resolve_model_path(self, path: str) -> Path:
        """Resolve model path relative to project root."""
        p = Path(path)
        if not p.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            p = project_root / path
        return p

    def generate_warning(
        self,
        threat_type: str,
        risk_factors: list[str],
        recent_transcript: str,
        max_tokens: int = 60,
    ) -> str:
        """Generate a contextual warning message.

        Args:
            threat_type: e.g., "gift_card", "government_impersonation", "tech_support"
            risk_factors: e.g., ["Tier 1: buy a gift card", "Tier 2: similarity 0.56"]
            recent_transcript: Last 50-100 words of conversation

        Returns:
            Warning text suitable for TTS (2-3 sentences, <15 seconds spoken)
        """
        key_phrases = _safe_extract_phrases(risk_factors)
        transcript_snippet = recent_transcript[-300:] if recent_transcript else "(no transcript)"

        system_prompt = (
            "You are a protective AI assistant. Generate a brief, calm warning "
            "for an elderly person who may be on a scam call. Keep it under 30 words."
        )
        user_prompt = (
            f"Threat type: {threat_type}\n"
            f"Suspicious phrases: {', '.join(key_phrases) if key_phrases else 'general scam indicators'}\n"
            f"Recent conversation: \"{transcript_snippet}\"\n\n"
            "Generate a 2-sentence warning that: "
            "1) Alerts them to the specific danger without panic, "
            "2) Gives one concrete action (e.g., hang up, verify). "
            "Output only the warning text, nothing else."
        )

        start = time.perf_counter()
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                stop=["\n\n", "Warning:", "Threat", "<|"],
            )
        except Exception as e:
            logger.warning("LLM generation failed: %s", e)
            return self._fallback_warning(threat_type)

        # Chat completion returns choices[0]["message"]["content"]
        choices = response.get("choices", [])
        if not choices:
            return self._fallback_warning(threat_type)

        raw = choices[0].get("message", {}).get("content", "")
        warning = (raw or "").strip()
        if not warning or len(warning) > 200:
            warning = self._fallback_warning(threat_type)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("[LLM] Generated warning in %.0fms: %s...", elapsed_ms, warning[:50])
        return warning

    def _fallback_warning(self, threat_type: str) -> str:
        """Static fallback when LLM output is invalid or generation fails."""
        return FALLBACK_TEMPLATES.get(
            threat_type, FALLBACK_TEMPLATES["generic_high_risk"]
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    gen = WarningGenerator()
    warning = gen.generate_warning(
        threat_type="gift_card",
        risk_factors=["Tier 1: buy a gift card", "Tier 2: 0.56 similarity to gift card scam"],
        recent_transcript="Yes I will buy the gift cards right now. How many do you need?",
    )
    print(f"Warning: {warning}")
