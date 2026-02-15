# Anchor Detection Methodology

This document summarizes how the Anchor phone-scam detection system analyzes audio and transcripts to infer scam risk.

## Overview

The pipeline combines two signals:

1. **Vocal stress** — acoustic model on raw audio
2. **Scam tactics** — language model on transcript text

Both feed into a final risk level (low / medium / high) displayed on the dashboard.

---

## 1. Vocal Stress Detection

**Module:** `src/core/stress_detector.py`  
**Model:** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`

### What It Does

A Wav2Vec2-based model fine-tuned on the MSP-Podcast corpus predicts three dimensional emotion values from 2–3 second audio windows:

| Output   | Meaning                                      |
| -------- | -------------------------------------------- |
| Arousal  | Activation / stress level (0–1)              |
| Valence  | Positivity vs negativity (0–1)                |
| Dominance| Perceived control (0–1)                       |

### Stress Score Derivation

The published **stress_score** is the **arousal** value. High arousal indicates heightened emotional activation, often associated with scam-induced anxiety.

A confidence heuristic (internal) combines arousal, valence, and dominance:
```
confidence = 0.5 * arousal + 0.3 * (1 - valence) + 0.2 * (1 - dominance)
```
Higher values suggest genuine stress rather than excitement.

### Limitations

- Trained on English; may not generalize to other languages.
- Short windows (2–3 s) can be noisy.
- Cannot distinguish stress from excitement without transcript context (e.g. joyful reunion vs scam fear).

---

## 2. Scam Tactic Detection

**Module:** `src/core/tactic_inference.py`  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`

### What It Does

An instruction-tuned 0.5B parameter LLM analyzes recent transcripts for five manipulation tactics:

| Tactic    | Examples                                          |
| --------- | -------------------------------------------------- |
| urgency   | "pay today", "right now"                           |
| authority | "IRS", "police", "government"                      |
| fear      | "arrested", "lawsuit", "jail"                      |
| isolation | "don't tell", "keep secret"                        |
| financial | "gift cards", "wire money"                         |

The model returns JSON scores from 0.0 (absent) to 1.0 (clearly present).

### Prompt

A fixed prompt instructs the model to score tactics from the elder's side of the conversation. Example phrases and a JSON format are provided. Parsing extracts the first `{...}` block; malformed output falls back to all zeros.

### Risk Level Heuristic

| Level   | Condition                                      |
| ------- | ---------------------------------------------- |
| **high**| max_tactic > 0.7 **and** stress_score > 0.6     |
| **medium** | max_tactic > 0.5 **or** stress_score > 0.7  |
| **low** | otherwise                                       |

### Limitations

- Small model; may miss nuanced or novel phrasings.
- Greedy decoding; deterministic, no sampling.
- Depends on transcript quality; ASR errors affect detection.

---

## 3. Pipeline Flow

```
Audio → [stress_detector] → stress_score → [tactic_inference] → risk_level
         (arousal)                           (max_tactic + stress)
         ↑
Transcript → [speech_recognition] → text ────────────────────→
```

Tactic inference runs periodically (default 10 s) when enough transcript words (≥15) are available. It uses the last 5 transcript chunks and the current stress score.
