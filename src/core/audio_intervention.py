"""Real-time audio warnings when scam detected.

Subscribes to content_analyzer output on TACTIC_PORT (5558), synthesizes dynamic
warnings with Piper TTS, plays through configured ALSA device (e.g. USB speaker).

Requires: piper-tts, aplay (ALSA)
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import logging
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Optional

if TYPE_CHECKING:
    from src.core.warning_generator import WarningGenerator

import zmq
from piper import PiperVoice

from src.core.message_bus import TACTIC_PORT, MessageBus

try:
    from src.core.warning_generator import WarningGenerator
    _WARNING_GENERATOR_AVAILABLE = True
except ImportError:
    _WARNING_GENERATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AUDIO_DEVICE = "plughw:3,0"
DEFAULT_MODEL_PATH = "models/piper/en_US-lessac-medium.onnx"
COOLDOWN_SECONDS = 30

# ---------------------------------------------------------------------------
# Intervention templates and scenario detection
# ---------------------------------------------------------------------------

INTERVENTION_TEMPLATES = {
    "gift_card": (
        "Warning. Someone is asking you to buy {payment_method}. "
        "Real companies never request gift card payments. Please hang up."
    ),
    "government_impersonation": (
        "Stop. This caller claims to be from {authority}. "
        "The real {authority} never demands immediate payment by phone. Please hang up."
    ),
    "grandparent_scam": (
        "Before sending money for an emergency, call your family member "
        "directly on their real phone number to verify."
    ),
    "tech_support": (
        "Warning. Someone is asking you to download software or give remote access. "
        "This is a common scam. Do not proceed."
    ),
    "wire_transfer": (
        "Stop. Someone is asking you to wire money or use {payment_method}. "
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

SCENARIO_KEYWORDS = {
    "gift_card": [
        "gift card",
        "cards at",
        "cvs",
        "target",
        "walmart",
        "itunes",
        "google play",
        "read the code",
        "scratch off",
        "redemption",
        "numbers on the back",
    ],
    "government_impersonation": [
        "irs",
        "social security",
        "ssa",
        "warrant",
        "arrest",
        "police",
        "sheriff",
        "lawsuit",
        "legal action",
        "suspended",
        "back taxes",
    ],
    "grandparent_scam": [
        "grandson",
        "granddaughter",
        "grandchild",
        "bail",
        "accident",
        "hospital",
        "lawyer fee",
        "don't tell mom",
        "don't tell dad",
        "emergency",
    ],
    "tech_support": [
        "remote access",
        "download",
        "teamviewer",
        "anydesk",
        "virus",
        "infected",
        "microsoft",
        "tech support",
        "refund",
        "remote software",
    ],
    "wire_transfer": [
        "wire transfer",
        "western union",
        "moneygram",
        "bitcoin",
        "cryptocurrency",
        "zelle",
        "cash deposit",
        "atm",
        "crypto",
    ],
    "romance_scam": [
        "send money",
        "wire me",
        "help me financially",
        "stuck overseas",
        "need funds",
        "love you",
    ],
}

ENTITY_PATTERNS = {
    "payment_method": [
        "gift cards",
        "wire transfer",
        "Bitcoin",
        "cryptocurrency",
        "Zelle",
        "Western Union",
        "MoneyGram",
        "cash",
    ],
    "authority": [
        "IRS",
        "Social Security Administration",
        "police",
        "sheriff",
        "federal agent",
        "government",
    ],
}


# ---------------------------------------------------------------------------
# AudioIntervention
# ---------------------------------------------------------------------------


class AudioIntervention:
    """Plays TTS warnings when high-risk scams are detected."""

    def __init__(
        self,
        model_path: str,
        audio_device: str = DEFAULT_AUDIO_DEVICE,
        cooldown: int = COOLDOWN_SECONDS,
        use_llm: bool = True,
        llm_model_path: str | None = None,
    ) -> None:
        self.audio_device = audio_device
        self.cooldown_seconds = cooldown
        self.last_intervention_time: float = 0.0
        self._warning_gen: Optional["WarningGenerator"] = None
        self._use_llm = False

        if use_llm and _WARNING_GENERATOR_AVAILABLE:
            try:
                self._warning_gen = WarningGenerator(model_path=llm_model_path)
                self._use_llm = True
                logger.info("LLM warning generator loaded")
            except (FileNotFoundError, ImportError, OSError) as e:
                logger.warning("LLM unavailable, using templates: %s", e)
        elif use_llm and not _WARNING_GENERATOR_AVAILABLE:
            logger.warning("llama-cpp-python not installed, using templates")

        model_file = Path(model_path)
        if not model_file.is_absolute():
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parents[2]
            model_file = project_root / model_path
        if not model_file.exists():
            raise FileNotFoundError(f"Piper model not found: {model_file}")

        logger.info("Loading Piper voice from %s", model_file)
        self.voice = PiperVoice.load(str(model_file))

        self.fallback_path = Path("/tmp/anchor_fallback_warning.wav")
        self._generate_fallback()
        logger.info("AudioIntervention ready (device=%s)", audio_device)

    def _generate_fallback(self) -> None:
        text = INTERVENTION_TEMPLATES["generic_high_risk"]
        self._synthesize_to_file(text, str(self.fallback_path))

    def _synthesize_to_file(self, text: str, path: str) -> None:
        with wave.open(path, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

    def _play_audio(self, path: str) -> None:
        try:
            subprocess.run(
                ["aplay", "-D", self.audio_device, path],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.error("Audio playback timed out")
        except subprocess.CalledProcessError as e:
            logger.error("Audio playback failed: %s", e.stderr)
        except FileNotFoundError:
            logger.error("aplay not found — install alsa-utils")

    def detect_scam_type(self, analysis: dict[str, Any]) -> str:
        risk_factors = analysis.get("risk_factors", [])
        transcript = (analysis.get("transcript", "") or "").lower()
        combined_text = transcript + " " + " ".join(risk_factors).lower()

        best_match = "generic_high_risk"
        best_score = 0

        for scam_type, keywords in SCENARIO_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in combined_text)
            if score > best_score:
                best_score = score
                best_match = scam_type

        return best_match

    def extract_entities(self, analysis: dict[str, Any]) -> dict[str, str]:
        transcript = (analysis.get("transcript", "") or "").lower()
        risk_factors = " ".join(analysis.get("risk_factors", [])).lower()
        combined = transcript + " " + risk_factors

        entities: dict[str, str] = {
            "payment_method": "gift cards",
            "authority": "government",  # Avoid "the the government" in templates
        }

        for payment in ENTITY_PATTERNS["payment_method"]:
            if payment.lower() in combined:
                entities["payment_method"] = payment
                break

        for authority in ENTITY_PATTERNS["authority"]:
            if authority.lower() in combined:
                entities["authority"] = authority
                break

        return entities

    def should_intervene(self, analysis: dict[str, Any]) -> bool:
        risk_level = (analysis.get("risk_level") or "low").lower()
        # Trigger on "medium" OR "high" (previously only "high")
        if risk_level not in ("medium", "high"):
            return False

        now = time.time()
        if now - self.last_intervention_time < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (now - self.last_intervention_time)
            logger.debug("Cooldown active, %.0fs remaining", remaining)
            return False

        return True

    def _generate_warning(
        self,
        threat_type: str,
        risk_factors: list[str],
        transcript: str,
        entities: dict[str, str],
    ) -> str:
        """Generate warning text: try LLM first, fallback to template."""
        if self._use_llm and self._warning_gen:
            try:
                return self._warning_gen.generate_warning(
                    threat_type=threat_type,
                    risk_factors=risk_factors,
                    recent_transcript=transcript or "",
                )
            except Exception as e:
                logger.error("LLM generation failed: %s", e)

        template = INTERVENTION_TEMPLATES.get(
            threat_type, INTERVENTION_TEMPLATES["generic_high_risk"]
        )
        try:
            return template.format(**entities)
        except KeyError:
            return INTERVENTION_TEMPLATES["generic_high_risk"]

    def intervene(self, analysis: dict[str, Any]) -> None:
        if not self.should_intervene(analysis):
            return

        scam_type = self.detect_scam_type(analysis)
        entities = self.extract_entities(analysis)
        risk_factors = analysis.get("risk_factors", [])
        transcript = analysis.get("transcript") or ""

        warning_text = self._generate_warning(
            threat_type=scam_type,
            risk_factors=risk_factors,
            transcript=transcript,
            entities=entities,
        )

        logger.info(
            "[INTERVENTION] INTERVENTION [%s]: %s...",
            scam_type,
            warning_text[:60],
        )
        logger.debug(
            "[INTERVENTION] [TTS] generating: %r",
            (warning_text[:80] + "…") if len(warning_text) > 80 else warning_text,
        )

        try:
            output_path = "/tmp/anchor_intervention.wav"
            self._synthesize_to_file(warning_text, output_path)
            logger.debug("[INTERVENTION] [PLAY] playing on %s", self.audio_device)
            self._play_audio(output_path)
            self.last_intervention_time = time.time()
        except Exception as e:
            logger.error("TTS failed, using fallback: %s", e)
            self._play_audio(str(self.fallback_path))
            self.last_intervention_time = time.time()


# ---------------------------------------------------------------------------
# MessageBus service
# ---------------------------------------------------------------------------


class AudioInterventionService:
    """Subscribe to tactics, play warnings on high risk — blocking."""

    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        audio_device: str = DEFAULT_AUDIO_DEVICE,
        cooldown: int = COOLDOWN_SECONDS,
        use_llm: bool = True,
        llm_model_path: str | None = None,
    ) -> None:
        self.bus = bus or MessageBus()
        self._intervention = AudioIntervention(
            model_path,
            audio_device,
            cooldown,
            use_llm=use_llm,
            llm_model_path=llm_model_path,
        )
        self._subscriber: Optional[zmq.Socket] = None
        self._stop = threading.Event()
        self.running = False

    def start(self) -> None:
        """Subscribe and process — blocking."""
        self._stop.clear()
        self._subscriber = self.bus.create_subscriber(
            ports=[TACTIC_PORT],
            topics=["tactics"],
        )
        self.running = True
        logger.info(
            "AudioInterventionService started — SUB tactics :%d",
            TACTIC_PORT,
        )

        try:
            self._main_loop()
        finally:
            self.running = False
            self._cleanup()
            logger.info("AudioInterventionService stopped")

    def stop(self) -> None:
        self._stop.set()
        self.running = False

    def _cleanup(self) -> None:
        if self._subscriber:
            self._subscriber.close()
            self._subscriber = None

    def _main_loop(self) -> None:
        msg_count = 0
        while not self._stop.is_set():
            result = self.bus.receive(self._subscriber, timeout_ms=500)
            if result is None:
                continue

            topic, envelope = result
            data = envelope.get("data", {})
            if not isinstance(data, dict):
                logger.warning("Received non-dict data for topic=%s: %s", topic, type(data))
                continue

            msg_count += 1
            risk_level = (data.get("risk_level") or "low").lower()
            risk_score = data.get("risk_score", 0.0)
            transcript_preview = ((data.get("transcript") or "")[:80] + "…") if len(data.get("transcript") or "") > 80 else (data.get("transcript") or "")

            logger.info(
                "[INTERVENTION] [RECV] msg #%d risk=%s score=%.2f transcript=%r",
                msg_count, risk_level, risk_score, transcript_preview,
            )

            will_intervene = self._intervention.should_intervene(data)
            if not will_intervene:
                reason = "risk_level not medium/high"
                if risk_level in ("medium", "high"):
                    reason = "cooldown active"
                logger.info(
                    "[INTERVENTION] [DECIDE] will_intervene=False reason=%s",
                    reason,
                )
            else:
                scam_type = self._intervention.detect_scam_type(data)
                logger.info(
                    "[INTERVENTION] [DECIDE] will_intervene=True scam_type=%s cooldown_active=False",
                    scam_type,
                )

            self._intervention.intervene(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Audio intervention — TTS warnings on scam detection",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Piper model path (default: models/piper/en_US-lessac-medium.onnx)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_AUDIO_DEVICE,
        help="ALSA audio device (default: plughw:3,0)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=COOLDOWN_SECONDS,
        help="Seconds between interventions (default: 30)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM, use static templates only",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Path to GGUF model for LLM warnings (default: models/qwen-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    service = AudioInterventionService(
        model_path=args.model,
        audio_device=args.device,
        cooldown=args.cooldown,
        use_llm=not args.no_llm,
        llm_model_path=args.llm_model,
    )
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()
