"""
Lightweight LLM warning generator using template completion.
Provides consistent alert prefix, LLM completes with contextual advice.
~700ms per warning on Jetson Orin Nano.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Alert prefixes by threat type - we control framing, LLM completes
ALERT_TEMPLATES = {
    "gift_card": "Warning. This caller is asking you to buy gift cards as payment.",
    "government_impersonation": "Stop. This caller claims to be from the government demanding payment.",
    "tech_support": "Warning. This caller wants remote access to your computer.",
    "grandparent_scam": "Stop. This caller claims to be a family member in trouble.",
    "financial": "Warning. This caller is pressuring you to send money.",
    "isolation": "Warning. This caller is asking you to keep secrets from your family.",
    "urgency": "Warning. This caller is creating false urgency to pressure you.",
    "authority": "Stop. This caller is impersonating an authority figure.",
    "wire_transfer": "Stop. This caller is asking you to wire money or send cryptocurrency.",
    "romance_scam": "Warning. This caller may be attempting a romance scam.",
    "generic_high_risk": "Warning. This call shows signs of a scam.",
}

# Static fallback completions if LLM fails
FALLBACK_COMPLETIONS = {
    "gift_card": "You should hang up. No legitimate company asks for gift card payments.",
    "government_impersonation": "You should hang up. The real government never demands immediate phone payments.",
    "tech_support": "You should hang up. Do not let strangers access your computer.",
    "grandparent_scam": "You should hang up and call your family member directly to verify.",
    "financial": "You should hang up and speak with a trusted family member first.",
    "isolation": "You should tell a family member about this call right away.",
    "urgency": "You should hang up. Take time to think before making any decisions.",
    "authority": "You should hang up and contact the agency directly using a number you trust.",
    "wire_transfer": "You should hang up. Wire transfers and cryptocurrency cannot be reversed.",
    "romance_scam": "You should not send money to someone you have not met in person.",
    "generic_high_risk": "You should hang up and speak with someone you trust before taking any action.",
}

DEFAULT_MODEL_PATH = "models/qwen-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"


class WarningGenerator:
    """Generates context-aware scam warnings via on-device LLM using template completion."""

    def __init__(self, model_path: str | None = None) -> None:
        """Load the GGUF model. Raises if the model file is missing or load fails."""
        resolved = self._resolve_model_path(model_path or DEFAULT_MODEL_PATH)
        if not resolved.exists():
            raise FileNotFoundError(f"LLM model not found: {resolved}")

        logger.info("Loading LLM from %s", resolved)
        start = time.time()
        
        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(resolved),
                n_ctx=128,        # Small context for speed
                n_threads=6,      # Use more CPU threads
                n_gpu_layers=50,  # Full GPU offload
                verbose=False,
            )
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for LLM warnings. "
                "Install with: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python"
            ) from e

        self.load_time = time.time() - start
        logger.info("LLM loaded in %.1fs", self.load_time)

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
        detected_phrases: list[str] | None = None,
        risk_factors: list[str] | None = None,
        recent_transcript: str | None = None,
    ) -> str:
        """
        Generate contextual warning using template completion.
        
        Args:
            threat_type: e.g., "gift_card", "government_impersonation"
            detected_phrases: Optional list of detected scam phrases (alias for risk_factors)
            risk_factors: Optional list of risk factors (kept for API compatibility)
            recent_transcript: Optional recent transcript (unused in template approach but kept for API compatibility)
            
        Returns:
            Complete warning text suitable for TTS (~2-3 sentences)
        """
        # Get template prefix for this threat type
        template = ALERT_TEMPLATES.get(threat_type, ALERT_TEMPLATES["generic_high_risk"])
        
        # Build completion prompt - we provide the prefix, LLM completes with protective advice
        # The key is framing: "You should hang up" or "You should not" guides toward protective actions
        prompt = f"{template} You should hang up and"
        
        start = time.perf_counter()
        
        try:
            response = self._llm(
                prompt,
                max_tokens=20,      # Short completion
                temperature=0.2,    # More consistent output
                stop=["\n", ".", "!", "?", "\"", "'"],  # Stop at sentence end or quotes
            )
            
            completion = response["choices"][0]["text"].strip()
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Ensure completion ends properly
            if completion and not completion.endswith((".", "!", "?")):
                completion += "."
            
            # Combine template + completion
            if completion and len(completion) > 3:
                warning = f"{template} You should hang up and {completion}"
            else:
                # LLM returned empty/bad output, use fallback
                fallback = FALLBACK_COMPLETIONS.get(
                    threat_type, FALLBACK_COMPLETIONS["generic_high_risk"]
                )
                warning = f"{template} {fallback}"
            
            logger.info("[LLM] %.0fms: %s...", elapsed_ms, warning[:60])
            return warning
            
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return self.get_fallback_warning(threat_type)

    def get_fallback_warning(self, threat_type: str) -> str:
        """Get static fallback warning without LLM."""
        template = ALERT_TEMPLATES.get(threat_type, ALERT_TEMPLATES["generic_high_risk"])
        fallback = FALLBACK_COMPLETIONS.get(
            threat_type, FALLBACK_COMPLETIONS["generic_high_risk"]
        )
        return f"{template} {fallback}"

    # Alias for backward compatibility
    def _fallback_warning(self, threat_type: str) -> str:
        """Static fallback when LLM output is invalid or generation fails."""
        return self.get_fallback_warning(threat_type)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    gen = WarningGenerator()
    
    test_threats = [
        "gift_card",
        "government_impersonation",
        "tech_support",
        "grandparent_scam",
        "financial",
    ]
    
    print("\n=== Testing Warning Generator ===\n")
    for threat in test_threats:
        start = time.perf_counter()
        warning = gen.generate_warning(threat)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{threat} ({elapsed:.0f}ms):")
        print(f"  \"{warning}\"")
        print()
