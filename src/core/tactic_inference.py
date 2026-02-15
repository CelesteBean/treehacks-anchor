"""Tactic inference module.

Phase 1: Core LLM inference — loads Qwen2.5-0.5B-Instruct on CUDA and
analyzes transcripts for manipulation tactics.

Phase 2: ZeroMQ service wrapper — subscribes to ``transcript`` and
``stress`` topics, runs periodic inference, and publishes ``tactics``
assessments on the message bus.

Scam Tactic Detection — How It Works
------------------------------------

* **Model**: ``Qwen/Qwen2.5-0.5B-Instruct`` — a 0.5B parameter instruction-tuned
  LLM. Runs in float16 on CUDA for Jetson Orin Nano compatibility.

* **Prompt**: A fixed prompt asks the model to score five manipulation tactics
  (0.0–1.0) from the elder's side of a phone conversation:
  - urgency (time pressure)
  - authority (official claims: IRS, police, government)
  - fear (threats: arrest, lawsuit, jail)
  - isolation (secrecy: don't tell, keep secret)
  - financial (money requests: gift cards, wire transfer)

* **Tactic extraction**: The model returns JSON like
  ``{"urgency": 0.8, "authority": 0.9, ...}``. We parse the first ``{...}``
  block; malformed output falls back to all zeros.

* **Risk level** (simple heuristic):
  - **high**: max_tactic > 0.7 AND stress_score > 0.6
  - **medium**: max_tactic > 0.5 OR stress_score > 0.7
  - **low**: otherwise

* **Known limitations**:
  - Small model; can miss nuanced or novel phrasings.
  - Greedy decoding (deterministic); no sampling diversity.
  - Relies on transcript quality; ASR errors affect detection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any, Optional
import json
import time
import logging

import torch
import torch.nn.functional as F
import zmq
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.message_bus import (
    STRESS_PORT,
    TACTIC_PORT,
    TRANSCRIPT_PORT,
    MessageBus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jetson torch 2.5.0a0 compatibility patch
# ---------------------------------------------------------------------------
# The Jetson Orin Nano ships torch 2.5.0a0 which lacks the ``enable_gqa``
# keyword in ``scaled_dot_product_attention``.  Newer ``transformers``
# (>=4.44) passes ``enable_gqa=True`` for models that use Grouped-Query
# Attention (GQA) like Qwen2.5.  We patch SDPA to manually expand K/V
# heads when the flag is requested but unsupported by the runtime.
_orig_sdpa = F.scaled_dot_product_attention


def _patched_sdpa(query, key, value, *args, enable_gqa=False, **kwargs):
    """SDPA wrapper that emulates ``enable_gqa`` on older torch builds."""
    if enable_gqa and query.size(1) != key.size(1):
        n_rep = query.size(1) // key.size(1)
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    return _orig_sdpa(query, key, value, *args, **kwargs)


# Only apply the patch if the native SDPA lacks enable_gqa support.
try:
    _q = torch.empty(1, 1, 1, 1)
    _orig_sdpa(_q, _q, _q, enable_gqa=False)
except TypeError:
    F.scaled_dot_product_attention = _patched_sdpa
    logger.debug("Applied SDPA enable_gqa compatibility patch for Jetson torch")


@dataclass
class TacticConfig:
    """Configuration for the tactic inference engine.

    Attributes:
        model_name: HuggingFace model identifier (must be cached for offline use).
        device: Compute device — "cuda" for GPU, "cpu" for fallback.
        max_new_tokens: Maximum tokens the model may generate per call.
        tactic_threshold: Minimum score to consider a tactic "active" (reserved for Phase 2).
    """

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"
    max_new_tokens: int = 80
    tactic_threshold: float = 0.5


# Canonical list of manipulation tactics we detect.
TACTIC_KEYS = ("urgency", "authority", "fear", "isolation", "financial")

# Inference gating and context limits.
MIN_WORDS_FOR_INFERENCE = 15  # Don't analyze until we have enough context
MAX_TRANSCRIPT_AGE_SECONDS = 60  # Rolling window - ignore older transcripts
MAX_CONTEXT_WORDS = 200  # Limit context size to control latency


class TacticInference:
    """Phase 1: Core inference only. ZeroMQ added in Phase 2."""

    TACTIC_PROMPT = (
        "Analyze this phone conversation transcript for scam tactics "
        "(elder's statements only).\n\n"
        "Score 0.0 (absent) to 1.0 (clearly present):\n"
        "- urgency: time pressure ('pay today','right now')\n"
        "- authority: official claims ('IRS','police','government')\n"
        "- fear: threats ('arrested','lawsuit','jail')\n"
        "- isolation: secrecy ('don't tell','keep secret')\n"
        "- financial: money requests ('gift cards','wire money')\n\n"
        "EXAMPLE:\n"
        "- You're from Social Security?\n"
        "- I need to transfer money now?\n"
        '{{"urgency": 0.8, "authority": 0.9, "fear": 0.6, '
        '"isolation": 0.0, "financial": 0.8}}\n\n'
        "ANALYZE:\n{transcripts}\n"
        "JSON only:"
    )

    def __init__(self, config: Optional[TacticConfig] = None) -> None:
        self.config = config or TacticConfig()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load Qwen model on CUDA with float16 precision."""
        cfg = self.config
        logger.info("Loading %s on %s", cfg.model_name, cfg.device)

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
        ).to(cfg.device)
        self._model.eval()

        vram_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("Model loaded. VRAM: %.2f GB", vram_gb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, transcripts: list[str], stress_score: float = 0.5) -> dict:
        """Analyze transcripts for manipulation tactics.

        Args:
            transcripts: Recent transcript strings from the elder's side.
            stress_score: Current vocal-stress level in [0, 1].

        Returns:
            Dict with keys: tactics, risk_level, stress_score, inference_time_ms.
        """
        start = time.perf_counter()

        # Build the prompt from transcript(s). Single concatenated context
        # or multiple fragments both supported.
        transcript_text = "\n".join(f"- {t}" for t in transcripts[-5:])
        prompt = self.TACTIC_PROMPT.format(transcripts=transcript_text)

        # Tokenize using the chat template expected by instruct models.
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.config.device)

        # Run inference (greedy decoding — deterministic).
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens.
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        logger.debug("Raw LLM response: %s", response)
        tactics = self._parse_tactics(response)

        # Risk heuristic: max_tactic + stress_score.
        # high: max_tactic > 0.7 AND stress > 0.6
        # medium: max_tactic > 0.5 OR stress > 0.7
        # low: otherwise
        max_tactic = max(tactics.values()) if tactics else 0.0
        if max_tactic > 0.7 and stress_score > 0.6:
            risk_level = "high"
        elif max_tactic > 0.5 or stress_score > 0.7:
            risk_level = "medium"
        else:
            risk_level = "low"

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "tactics": tactics,
            "risk_level": risk_level,
            "stress_score": stress_score,
            "inference_time_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_tactics(self, response: str) -> dict:
        """Extract tactic scores from the LLM JSON response.

        Falls back to all-zeros if the model output is malformed.
        """
        defaults = {key: 0.0 for key in TACTIC_KEYS}
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                for key in TACTIC_KEYS:
                    if key in parsed:
                        defaults[key] = max(0.0, min(1.0, float(parsed[key])))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse tactics from model output: %s", exc)
        return defaults


# ======================================================================
# Phase 2: ZeroMQ service wrapper
# ======================================================================


class TacticInferenceService:
    """ZeroMQ service wrapper around :class:`TacticInference`.

    Subscribes to ``transcript`` (port 5556) and ``stress`` (port 5557),
    runs LLM inference every *inference_interval* seconds when new
    transcripts have arrived, and publishes ``tactics`` (port 5558).

    Parameters
    ----------
    config:
        Model configuration forwarded to :class:`TacticInference`.
    bus:
        A :class:`MessageBus` used to create sockets.
    inference_interval:
        Minimum seconds between consecutive inference runs.
    """

    def __init__(
        self,
        config: Optional[TacticConfig] = None,
        bus: Optional[MessageBus] = None,
        inference_interval: float = 10.0,
    ) -> None:
        self.config = config or TacticConfig()
        self.bus = bus or MessageBus()
        self.inference_interval = inference_interval

        # Shared state protected by ``_lock``.
        self._transcripts: deque[tuple[float, str]] = deque(maxlen=50)
        self._current_stress: float = 0.5
        self._lock = Lock()

        # Lifecycle
        self._stop_event = Event()
        self.running: bool = False

        # Sockets (created lazily in start()).
        self._sub_transcript: zmq.Socket | None = None
        self._sub_stress: zmq.Socket | None = None
        self._publisher: zmq.Socket | None = None

        # Core inference engine (loads the model immediately).
        self._engine = TacticInference(self.config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe, infer, and publish — **blocking**.

        Call :meth:`stop` from another thread (or signal handler) to exit.
        """
        self._stop_event.clear()

        self._sub_transcript = self.bus.create_subscriber(
            ports=[TRANSCRIPT_PORT], topics=["transcript"],
        )
        self._sub_stress = self.bus.create_subscriber(
            ports=[STRESS_PORT], topics=["stress"],
        )
        self._publisher = self.bus.create_publisher(TACTIC_PORT)

        self.running = True
        logger.info(
            "TacticInferenceService started — "
            "SUB transcript:%d, SUB stress:%d, PUB tactics:%d, interval=%ds",
            TRANSCRIPT_PORT,
            STRESS_PORT,
            TACTIC_PORT,
            self.inference_interval,
        )

        inference_thread = Thread(
            target=self._inference_loop, daemon=True, name="tactic-inference",
        )
        inference_thread.start()

        try:
            self._receive_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            self.running = False
            self._stop_event.set()
            inference_thread.join(timeout=3)
            self._cleanup()
            logger.info("TacticInferenceService stopped")

    def stop(self) -> None:
        """Signal the main loop to exit.  Thread-safe and idempotent."""
        self._stop_event.set()
        self.running = False
        logger.info("TacticInferenceService stop signal sent")

    # ------------------------------------------------------------------
    # Receive loop (main thread)
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        """Poll both subscriber sockets and buffer incoming data."""
        poller = zmq.Poller()
        poller.register(self._sub_transcript, zmq.POLLIN)
        poller.register(self._sub_stress, zmq.POLLIN)

        while not self._stop_event.is_set():
            events = dict(poller.poll(timeout=500))

            if self._sub_transcript in events:
                logger.debug("Received message on transcript socket")
                self._handle_transcript()

            if self._sub_stress in events:
                logger.debug("Received message on stress socket")
                self._handle_stress()

    def _handle_transcript(self) -> None:
        """Decode a transcript multipart frame and buffer the text."""
        try:
            frames: list[bytes] = self._sub_transcript.recv_multipart(zmq.NOBLOCK)
            raw_body: str = frames[1].decode("utf-8")
            logger.debug("Transcript raw message: %s", raw_body[:500] + ("..." if len(raw_body) > 500 else ""))
            envelope: dict[str, Any] = json.loads(raw_body)
            data = envelope.get("data", {})
            text = data.get("text", "").strip()
            if text and data.get("is_final", False):
                with self._lock:
                    self._transcripts.append((time.time(), text))
                logger.info("Transcript buffered (%d total): %s",
                            len(self._transcripts), text[:60])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error handling transcript message: %s", exc)

    def _handle_stress(self) -> None:
        """Decode a stress multipart frame and update current score."""
        try:
            frames: list[bytes] = self._sub_stress.recv_multipart(zmq.NOBLOCK)
            raw_body: str = frames[1].decode("utf-8")
            logger.debug("Stress raw message: %s", raw_body[:300] + ("..." if len(raw_body) > 300 else ""))
            envelope: dict[str, Any] = json.loads(raw_body)
            data = envelope.get("data", {})
            score = data.get("stress_score", data.get("score", 0.5))
            with self._lock:
                self._current_stress = max(0.0, min(1.0, float(score)))
            logger.debug("Stress updated: %.2f", self._current_stress)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error handling stress message: %s", exc)

    def _build_conversation_context(self) -> tuple[str, int]:
        """Build concatenated conversation from recent transcripts.

        Returns (context_string, word_count).
        Filters out transcripts older than MAX_TRANSCRIPT_AGE_SECONDS.
        """
        now = time.time()
        cutoff = now - MAX_TRANSCRIPT_AGE_SECONDS

        # Filter recent transcripts
        recent = [(ts, text) for ts, text in self._transcripts if ts > cutoff]

        # Concatenate into conversation
        conversation = " ".join(text for _, text in recent)

        # Trim to max words if needed
        words = conversation.split()
        if len(words) > MAX_CONTEXT_WORDS:
            words = words[-MAX_CONTEXT_WORDS:]  # Keep most recent
            conversation = " ".join(words)

        return conversation, len(words)

    # ------------------------------------------------------------------
    # Inference loop (background thread)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        """Periodically run LLM inference and publish results."""
        last_context = ""

        while not self._stop_event.wait(timeout=self.inference_interval):
            with self._lock:
                context, word_count = self._build_conversation_context()
                stress = self._current_stress

            # Skip if not enough context or no change
            if word_count < MIN_WORDS_FOR_INFERENCE:
                logger.debug(
                    "Skipping inference: only %d words (need %d)",
                    word_count, MIN_WORDS_FOR_INFERENCE,
                )
                continue

            if context == last_context:
                continue  # No new content

            last_context = context

            logger.info(
                "Running inference on %d words (stress=%.2f): %s...",
                word_count, stress, context[:80],
            )

            result = self._engine.analyze([context], stress_score=stress)

            tactic_data: dict[str, Any] = {
                "tactics": result["tactics"],
                "risk_level": result["risk_level"],
                "stress_score": stress,
                "word_count": word_count,
                "inference_time_ms": result["inference_time_ms"],
            }
            self.bus.publish(self._publisher, topic="tactics", data=tactic_data)
            logger.info(
                "Published tactics — risk=%s time=%.0fms tactics=%s",
                result["risk_level"],
                result["inference_time_ms"],
                result["tactics"],
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Close ZeroMQ sockets (context is managed by MessageBus)."""
        for sock in (self._sub_transcript, self._sub_stress, self._publisher):
            if sock is not None:
                sock.close()


# ------------------------------------------------------------------
# Standalone smoke test (Phase 1 exit criteria)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tactic Inference — Phase 1 test or Phase 2 ZeroMQ service",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run Phase 1 standalone inference tests (no ZeroMQ)",
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Seconds between inference runs in service mode (default: 10)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging (shows raw LLM output)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.test:
        # ============================================================
        # Phase 1: standalone smoke test
        # ============================================================
        print("=== Phase 1: Standalone Inference Test ===\n")

        print("Loading model...")
        ti = TacticInference()
        vram_used = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM: {vram_used:.2f} GB\n")

        # Warmup — first CUDA inference is slower due to JIT compilation.
        print("Warmup...")
        _ = ti.analyze(["hello"], stress_score=0.3)

        print("=== Test 1: Scam Scenario ===")
        scam_result = ti.analyze(
            [
                "Yes hello?",
                "The IRS you said?",
                "I have to pay today or I'll be arrested?",
                "Gift cards? You want me to buy gift cards?",
            ],
            stress_score=0.75,
        )
        print(f"Risk: {scam_result['risk_level']}")
        print(f"Tactics: {scam_result['tactics']}")
        print(f"Time: {scam_result['inference_time_ms']:.0f}ms\n")

        print("=== Test 2: Benign Scenario ===")
        benign_result = ti.analyze(
            ["Hi sweetie", "Yes dinner at six sounds lovely", "See you then"],
            stress_score=0.25,
        )
        print(f"Risk: {benign_result['risk_level']}")
        print(f"Tactics: {benign_result['tactics']}")
        print(f"Time: {benign_result['inference_time_ms']:.0f}ms\n")

        scam_pass = scam_result["risk_level"] in ("medium", "high")
        benign_pass = benign_result["risk_level"] == "low"
        print(f"{'PASS' if scam_pass else 'FAIL'}  Scam detection: {scam_result['risk_level']}")
        print(f"{'PASS' if benign_pass else 'FAIL'}  Benign detection: {benign_result['risk_level']}")
        print(f"NOTE  VRAM: {vram_used:.2f} GB")

    else:
        # ============================================================
        # Phase 2: ZeroMQ service mode
        # ============================================================
        print("=== Phase 2: ZeroMQ Service Mode ===")
        service = TacticInferenceService(inference_interval=args.interval)
        service.start()
