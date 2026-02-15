"""Two-tier scam detection — replaces stress_detector + tactic_inference.

Tier 1: Instant pattern alerts — unambiguous phrases (substring match).
Tier 2: Semantic similarity — sentence-transformers vs 30–40 scam scenario descriptions.

Subscribes to TRANSCRIPT_PORT (5556), publishes to STRESS_PORT (5557) and TACTIC_PORT (5558).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import numpy as np
import zmq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.message_bus import (
    STRESS_PORT,
    TACTIC_PORT,
    TRANSCRIPT_PORT,
    MessageBus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier 1: Unambiguous scam compliance phrases (zero legitimate use)
# ---------------------------------------------------------------------------

TIER1_PHRASES = [
    # Reading gift card numbers/codes to caller
    "read you the numbers on the back",
    "read you the numbers",
    "read the numbers on the back",
    "read you the code",
    "read the code to you",
    "call you back with the codes",
    "call back with the codes",
    "give you the numbers on the back",
    "scratch off the code",
    "scratch off the numbers",
    # Not telling family
    "won't tell my family",
    "wont tell my family",
    "will not tell my family",
    "won't tell anyone",
    "wont tell anyone",
    "don't tell my family",
    "dont tell my family",
    "keep this between us",
    "promise not to tell",
    # Giving SSN
    "my social security number is",
    "give you my social security",
    "my ssn is",
    # Wiring/paying to avoid arrest
    "wire the money to avoid",
    "pay to clear the warrant",
    "pay the fine to avoid arrest",
    "how much to avoid arrest",
    "going to the bitcoin atm",
    "at the bitcoin atm",
    # Remote access
    "download that software for you",
    "installing the program you sent",
    "give you remote access",
    "downloading the software now",
    "teamviewer",
    "anydesk",
    "logmein",
]

# ---------------------------------------------------------------------------
# Tier 2: Scam scenario descriptions (for semantic similarity)
# ---------------------------------------------------------------------------

SCAM_SCENARIOS: list[tuple[str, str]] = [
    # (description, category for tactics)
    ("The elderly person agrees to purchase gift cards and read the redemption codes to the caller", "financial"),
    ("The victim is buying iTunes or Google Play cards and will read the codes over the phone", "financial"),
    ("Someone is instructing the elder to buy gift cards and scratch off the codes", "financial"),
    ("The person is at a store purchasing gift cards for someone on the phone", "financial"),
    ("The victim is being threatened with arrest if they don't pay immediately", "fear"),
    ("The caller claims there is a warrant and the person must pay to avoid jail", "authority"),
    ("Someone claiming to be from the IRS is demanding immediate payment", "authority"),
    ("The person is being told they will be arrested unless they wire money", "fear"),
    ("A grandchild or family member is urgently asking for bail money", "urgency"),
    ("Someone claiming to be a grandchild in jail needs money for a lawyer", "financial"),
    ("The victim is being asked to send money to help a family member in trouble", "financial"),
    ("The caller is requesting remote access to the victim's computer", "isolation"),
    ("Someone is guiding the victim to download software to fix their computer", "isolation"),
    ("The victim is being told their computer has a virus and needs remote access", "authority"),
    ("The person is being directed to install TeamViewer or AnyDesk", "isolation"),
    ("The victim is being told to keep this call secret from family members", "isolation"),
    ("The caller insists the victim must not tell anyone about this", "isolation"),
    ("The person is promising not to tell their family about the call", "isolation"),
    ("The victim is being directed to withdraw cash and deposit it at a cryptocurrency ATM", "financial"),
    ("Someone is instructing the elder to buy Bitcoin and send it", "financial"),
    ("The person is going to a Bitcoin ATM to send money", "financial"),
    ("The victim is being told to wire money through Western Union", "financial"),
    ("Someone is directing the elder to transfer money or send a wire", "financial"),
    ("The victim is providing their social security number to the caller", "authority"),
    ("The person is giving their bank account number to someone on the phone", "financial"),
    ("The elder is reading a verification code from their phone to the caller", "financial"),
    ("The victim is providing sensitive personal information to stop a supposed fraud", "authority"),
    ("Someone the victim met online is asking for money to get home", "financial"),
    ("A romantic interest online needs money for an emergency", "financial"),
    ("The person is sending money to someone they care about online", "financial"),
    ("The victim is being told their bank account has fraud and must verify", "authority"),
    ("Someone claims to be from the bank and needs account verification", "authority"),
    ("The elder is being directed to move money to protect it from fraud", "financial"),
    ("The victim won a prize but must pay fees to claim it", "financial"),
    ("Someone is telling the person they need to pay taxes on winnings", "authority"),
    ("The victim is picking up gift cards at a store and will provide the codes to the caller", "financial"),
    ("The person is getting cards from the store and will call back with the redemption codes", "financial"),
]

# ---------------------------------------------------------------------------
# Benign context patterns (reduce false positives)
# ---------------------------------------------------------------------------

BENIGN_PATTERNS = [
    r"\b(doctor|physician|hospital|pharmacy|prescription|appointment|checkup)\b",
    r"\b(tax preparer|accountant|financial advisor|my banker)\b",
    r"\b(birthday|nephew|niece|grandchild.*visit|family dinner)\b",
    r"\b(gift card.*birthday|gift.*nephew|gift.*niece)\b",
    r"\b(just checking in|how are you|good to hear|catch up)\b",
    r"\b(lunch plans|dinner plans|coffee|visiting)\b",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProsodicsResult:
    speech_rate: float = 0.0
    pause_ratio: float = 0.0
    hesitation_count: int = 0
    question_indicators: int = 0
    confusion_score: float = 0.0


@dataclass
class SentimentResult:
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0
    compound: float = 0.0


# ---------------------------------------------------------------------------
# ContentAnalyzer
# ---------------------------------------------------------------------------


class ContentAnalyzer:
    """Two-tier scam detection: instant phrases + semantic similarity."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        logger.info("Initializing ContentAnalyzer...")
        self.vader = SentimentIntensityAnalyzer()
        logger.info("Loading sentence transformer: %s", embedding_model)
        self.embedder = SentenceTransformer(embedding_model, device="cpu")

        # Pre-compute scenario embeddings at startup
        logger.info("Pre-computing scenario embeddings...")
        self.scenario_descriptions = [s[0] for s in SCAM_SCENARIOS]
        self.scenario_categories = [s[1] for s in SCAM_SCENARIOS]
        self.scenario_embeddings = self.embedder.encode(self.scenario_descriptions)

        self.benign_patterns = [re.compile(p, re.IGNORECASE) for p in BENIGN_PATTERNS]
        self.call_start_time: Optional[float] = None
        self.risk_history: deque[float] = deque(maxlen=20)
        logger.info("ContentAnalyzer ready (%d Tier 1 phrases, %d scenarios)",
                    len(TIER1_PHRASES), len(SCAM_SCENARIOS))

    def _check_tier1(self, transcript: str) -> list[str]:
        """Tier 1: Unambiguous phrase matches (substring)."""
        transcript_lower = transcript.lower()
        matches = []
        for phrase in TIER1_PHRASES:
            if phrase in transcript_lower:
                matches.append(phrase)
        return matches

    def _check_tier2(self, transcript: str) -> Tuple[float, str, str]:
        """Tier 2: Semantic similarity to scam scenarios. Returns (score, scenario, category)."""
        words = transcript.split()
        if len(words) < 3:
            return 0.0, "", ""

        encoding = self.embedder.encode([transcript])
        similarities = cosine_similarity(encoding, self.scenario_embeddings)[0]
        max_idx = int(np.argmax(similarities))
        score = float(similarities[max_idx])
        scenario = self.scenario_descriptions[max_idx]
        category = self.scenario_categories[max_idx]
        return score, scenario, category

    def _check_benign_context(self, transcript: str) -> bool:
        """Strong benign context that could explain suspicious words."""
        for pattern in self.benign_patterns:
            if pattern.search(transcript):
                return True
        return False

    def _analyze_prosodics(self, transcript: str, duration_hint: float = 2.5) -> ProsodicsResult:
        words = transcript.split()
        word_count = len(words)
        speech_rate = word_count / max(duration_hint, 0.1)
        hesitation_markers = ["um", "uh", "er", "ah", "hmm", "well", "like"]
        hesitation_count = sum(
            1 for w in words if w.lower().strip(".,!?") in hesitation_markers
        )
        question_words = ["what", "why", "how", "when", "where", "who", "huh"]
        question_indicators = transcript.count("?") + sum(
            1 for w in words if w.lower().strip(".,!?") in question_words
        )
        pause_indicators = transcript.count("...") + transcript.count("..")
        pause_ratio = min(pause_indicators / max(word_count, 1), 1.0)
        confusion_score = min(
            1.0,
            hesitation_count * 0.15 + question_indicators * 0.1 + pause_ratio * 0.3,
        )
        return ProsodicsResult(
            speech_rate=speech_rate,
            pause_ratio=pause_ratio,
            hesitation_count=hesitation_count,
            question_indicators=question_indicators,
            confusion_score=confusion_score,
        )

    def _analyze_sentiment(self, transcript: str) -> SentimentResult:
        scores = self.vader.polarity_scores(transcript)
        return SentimentResult(
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            compound=scores["compound"],
        )

    def _infer_tactics(
        self,
        tier1_matches: list[str],
        semantic_score: float,
        matched_category: str,
        sentiment: SentimentResult,
    ) -> dict[str, float]:
        """Infer tactics from Tier 1 matches and matched scenario category."""
        tactics = {k: 0.1 for k in ("urgency", "authority", "fear", "isolation", "financial")}

        if tier1_matches or semantic_score > 0.45:
            if matched_category == "authority":
                tactics["authority"] = 0.85
            if matched_category == "fear":
                tactics["fear"] = 0.85
            if matched_category == "urgency":
                tactics["urgency"] = 0.85
            if matched_category == "isolation":
                tactics["isolation"] = 0.85
            if matched_category == "financial":
                tactics["financial"] = 0.85

        # Tier 1 phrase hints
        for m in tier1_matches:
            m_l = m.lower()
            if "arrest" in m_l or "warrant" in m_l or "jail" in m_l:
                tactics["fear"] = max(tactics["fear"], 0.8)
                tactics["authority"] = max(tactics["authority"], 0.7)
            if "don't tell" in m_l or "won't tell" in m_l or "secret" in m_l:
                tactics["isolation"] = max(tactics["isolation"], 0.85)
            if "social security" in m_l or "ssn" in m_l:
                tactics["authority"] = max(tactics["authority"], 0.75)
            if "gift card" in m_l or "bitcoin" in m_l or "wire" in m_l:
                tactics["financial"] = max(tactics["financial"], 0.85)
            if "remote access" in m_l or "download" in m_l or "teamviewer" in m_l:
                tactics["isolation"] = max(tactics["isolation"], 0.8)

        if sentiment.negative > 0.3:
            tactics["fear"] = max(tactics["fear"], 0.6)

        return tactics

    def analyze(self, transcript: str, duration_hint: float = 2.5) -> dict[str, Any]:
        """Run two-tier analysis. Returns dict compatible with dashboard."""
        start = time.perf_counter()

        prosodics = self._analyze_prosodics(transcript, duration_hint)
        sentiment = self._analyze_sentiment(transcript)

        tier1_matches = self._check_tier1(transcript)
        semantic_score, matched_scenario, matched_category = self._check_tier2(transcript)
        is_benign = self._check_benign_context(transcript)

        risk_factors: list[str] = []
        risk_score = 0.0

        # Tier 1: Any match -> HIGH
        if tier1_matches:
            risk_score = max(risk_score, 0.7)
            for m in tier1_matches[:2]:
                risk_factors.append(f"Tier 1: '{m}'")

        # Tier 2: Semantic thresholds
        if semantic_score > 0.65:
            risk_score = max(risk_score, 0.6)
            risk_factors.append(
                f"Tier 2: {matched_scenario[:60]}... (similarity {semantic_score:.2f})"
            )
        elif semantic_score > 0.45:
            risk_score = max(risk_score, 0.35)
            risk_factors.append(
                f"Tier 2: {matched_scenario[:50]}... (similarity {semantic_score:.2f})"
            )

        # Benign context reduction
        if is_benign and not tier1_matches and semantic_score < 0.5:
            risk_score *= 0.5
            risk_factors.append("(Reduced: benign context)")

        # Final level from score
        if semantic_score < 0.3 and not tier1_matches:
            risk_level = "low"
        elif risk_score >= 0.6 or tier1_matches:
            risk_level = "high"
        elif risk_score >= 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"

        risk_score = min(risk_score, 1.0)
        confidence = min(risk_score + 0.1, 1.0) if risk_factors else 0.1

        tactics = self._infer_tactics(
            tier1_matches, semantic_score, matched_category, sentiment
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("Analysis %.1fms: %s (%.2f)", elapsed_ms, risk_level, risk_score)

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "tactics": tactics,
            "prosodics": asdict(prosodics),
            "sentiment": asdict(sentiment),
            "stress_score": prosodics.confusion_score,
            "inference_time_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# ZeroMQ service
# ---------------------------------------------------------------------------


class ContentAnalyzerService:
    """MessageBus service: subscribe to transcripts, publish stress + tactics."""

    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        min_words: int = 8,
        analysis_interval: float = 5.0,
    ) -> None:
        self.bus = bus or MessageBus()
        self.min_words = min_words
        self.analysis_interval = analysis_interval
        self._analyzer = ContentAnalyzer()
        self._accumulated: list[str] = []
        self._last_analysis = 0.0
        self._stop = threading.Event()
        self._subscriber: Optional[zmq.Socket] = None
        self._stress_pub: Optional[zmq.Socket] = None
        self._tactic_pub: Optional[zmq.Socket] = None
        self.running = False

    def start(self) -> None:
        """Subscribe, accumulate, analyze, publish — blocking."""
        self._stop.clear()
        self._subscriber = self.bus.create_subscriber(
            ports=[TRANSCRIPT_PORT],
            topics=["transcript"],
        )
        self._stress_pub = self.bus.create_publisher(STRESS_PORT)
        self._tactic_pub = self.bus.create_publisher(TACTIC_PORT)
        self.running = True
        logger.info(
            "ContentAnalyzerService started — SUB :%d, PUB stress:%d tactics:%d",
            TRANSCRIPT_PORT, STRESS_PORT, TACTIC_PORT,
        )
        try:
            self._main_loop()
        finally:
            self.running = False
            self._cleanup()
            logger.info("ContentAnalyzerService stopped")

    def stop(self) -> None:
        self._stop.set()
        self.running = False

    def _cleanup(self) -> None:
        for sock in (self._subscriber, self._stress_pub, self._tactic_pub):
            if sock:
                sock.close()

    def _main_loop(self) -> None:
        while not self._stop.is_set():
            result = self.bus.receive(self._subscriber, timeout_ms=200)
            if result is None:
                self._maybe_analyze()
                continue
            topic, envelope = result
            data = envelope.get("data", {})
            text = (data.get("text", "") or "").strip()
            if not text or text == "(silence)":
                continue
            if data.get("is_final", True):
                self._accumulated.append(text)
                logger.debug("Buffered transcript (%d): %s", len(self._accumulated), text[:50])
            self._maybe_analyze()

    def _maybe_analyze(self) -> None:
        now = time.time()
        if now - self._last_analysis < self.analysis_interval:
            return
        combined = " ".join(self._accumulated).strip()
        words = combined.split()
        if len(words) < self.min_words:
            return
        self._last_analysis = now
        result = self._analyzer.analyze(combined)
        self._accumulated.clear()
        ts = datetime.now(timezone.utc).isoformat()
        stress_data = {
            "stress_score": result["stress_score"],
            "emotions": {
                "arousal": result["stress_score"],
                "valence": max(0, (result["sentiment"]["compound"] + 1) / 2),
                "dominance": 1.0 - result["stress_score"],
            },
            "confidence": result["confidence"],
            "timestamp": ts,
        }
        self.bus.publish(self._stress_pub, "stress", stress_data)
        tactic_data = {
            "tactics": result["tactics"],
            "risk_level": result["risk_level"],
            "risk_factors": result["risk_factors"],
            "transcript": combined,
            "word_count": len(words),
            "inference_time_ms": result["inference_time_ms"],
            "timestamp": ts,
        }
        self.bus.publish(self._tactic_pub, "tactics", tactic_data)
        logger.info(
            "Published risk=%s (%.2f) %d words in %.0fms",
            result["risk_level"], result["risk_score"], len(words), result["inference_time_ms"],
        )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Content Analyzer — two-tier scam detection")
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    service = ContentAnalyzerService(min_words=args.min_words, analysis_interval=args.interval)
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()
