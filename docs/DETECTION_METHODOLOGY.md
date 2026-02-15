# Anchor Detection Methodology

This document summarizes how the Anchor phone-scam detection system analyzes transcripts to infer scam risk.

## Overview

The detection pipeline uses a **lightweight multi-signal architecture** (`src/core/content_analyzer.py`) that replaces the previous wav2vec2 + Qwen setup. No GPU is required for detection.

**Memory:** ~200MB total (vs 2GB before)  
**Latency:** <500ms (vs 13s before)

Four signal classes are combined:

1. **Prosodic features** — speech rate, pauses, hesitations (from transcript text)
2. **VADER sentiment** — rule-based, ~1ms
3. **Semantic similarity** — sentence-transformers (80MB) + scam phrase database
4. **Temporal tracking** — escalation, call duration, repetition

---

## 1. Prosodic Features

**Source:** Inferred from transcript text (no audio).

| Signal             | Derivation                                      |
| ------------------ | ----------------------------------------------- |
| Speech rate        | Words per second (from duration hint)           |
| Hesitation count   | "um", "uh", "er", "well", "like"                |
| Question indicators| "?", "what", "why", "how", etc.                 |
| Pause ratio        | Ellipsis, repeated punctuation                   |
| Confusion score    | Composite (hesitation + questions + pauses)     |

The **confusion_score** is published as the stress proxy for dashboard compatibility.

---

## 2. VADER Sentiment

**Library:** `vaderSentiment` (rule-based, no ML)

VADER returns positive, negative, neutral, and compound (-1 to +1) scores. Highly negative compound (≤ -0.5) contributes to risk. Fast (~1ms).

---

## 3. Semantic Similarity

**Model:** `all-MiniLM-L6-v2` (sentence-transformers, 80MB, CPU)

- Pre-computed embeddings for ~20 scam scenario phrases (IRS, bail, virus, etc.).
- Transcript embedding compared via cosine similarity.
- Threshold 0.45; ≥0.6 adds significant risk.

---

## 4. Keyword & Phrase Matching

**High-confidence phrases** (instant high risk): e.g. "I'll buy the gift cards", "my social security number is", "I won't tell anyone".

**Regex categories:**
- `gift_card_payment`, `wire_transfer` — critical
- `government_threat`, `remote_access`, `urgency_pressure`, `personal_info_request` — concerning

**Benign context reduction:** Patterns like "doctor", "pharmacy", "friend", "appointment" reduce risk when score < 0.5.

---

## 5. Temporal Tracking

- **Escalation:** Risk increasing over last 3 analyses (Δ ≥ 0.2).
- **Duration:** Extended calls (>5 min) add slight risk.
- **Repetition:** Tracked in risk history (deque).

---

## 6. Risk Level Heuristic

| Level   | Condition                                                                 |
| ------- | ------------------------------------------------------------------------- |
| **high**| risk_score ≥ 0.6, or (high-confidence match AND risk_score ≥ 0.4)        |
| **medium** | risk_score ≥ 0.3, or ≥2 regex categories matched                      |
| **low** | otherwise                                                                  |

Contributing factors (additive): high-confidence phrases (+0.6), regex categories (+0.15–0.25 each), semantic similarity (+0.15–0.3), sentiment (+0.1), confusion (+0.1), escalation (+0.15).

---

## 7. Pipeline Flow

```
Transcript → [speech_recognition] → text
     ↓
[content_analyzer]
  ├─ Prosodics (text-based)
  ├─ VADER sentiment
  ├─ Phrase + regex matching
  ├─ Semantic similarity
  └─ Temporal tracking
     ↓
  ├─ PUB :5557 (stress)  → dashboard stress panel
  └─ PUB :5558 (tactics) → dashboard tactic panel
```

Content analyzer subscribes to TRANSCRIPT_PORT (5556), accumulates until ≥8 words, runs analysis every 5 s, publishes to both STRESS_PORT and TACTIC_PORT for dashboard compatibility.

---

## Legacy Modules (Deprecated)

- `stress_detector.py` — wav2vec2 arousal-based (replaced by content_analyzer prosodics + sentiment)
- `tactic_inference.py` — Qwen LLM (replaced by content_analyzer phrase/semantic/regex)
