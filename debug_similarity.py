#!/usr/bin/env python3
"""Debug script: print semantic similarity scores for test phrases.

Shows the actual cosine similarity between each test phrase and each scenario,
and identifies the best-matching scenario. Use this to diagnose why
gift card phrases return risk=low.

Usage: python debug_similarity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.content_analyzer import SCAM_SCENARIOS, ContentAnalyzer

# Test phrases from user report
TEST_PHRASES = [
    "Yes I will buy a gift card and send it to you",
    "Yes I will buy the gift cards right now",
    "I am going to the store to get gift cards for you",
    "Let me read you the numbers on the back of the card",
    "You need remote access to my computer?",
    "I'll download that software you mentioned",
    "Let me get my credit card for the purchase",
    "I'm buying a gift card for my grandson's birthday",
    "The weather is nice today",
]


def main() -> int:
    print("=" * 70)
    print("DEBUG: Semantic Similarity Scores (all-MiniLM-L6-v2)")
    print("=" * 70)

    analyzer = ContentAnalyzer(embedding_model="all-MiniLM-L6-v2")
    scenario_descriptions = [s[0] for s in SCAM_SCENARIOS]
    scenario_categories = [s[1] for s in SCAM_SCENARIOS]

    # Verify embeddings are not zeros/NaN
    sample = analyzer.scenario_embeddings[0]
    print(f"\nScenario embedding shape: {analyzer.scenario_embeddings.shape}")
    print(f"Sample embedding norms: min={np.min(np.linalg.norm(analyzer.scenario_embeddings, axis=1)):.4f}, "
          f"max={np.max(np.linalg.norm(analyzer.scenario_embeddings, axis=1)):.4f}")
    if np.any(np.isnan(analyzer.scenario_embeddings)):
        print("ERROR: NaN in scenario embeddings!")
        return 1

    for phrase in TEST_PHRASES:
        print(f"\n{'─' * 70}")
        print(f"PHRASE: {phrase!r}")
        print("─" * 70)

        encoding = analyzer.embedder.encode([phrase])
        similarities = cosine_similarity(encoding, analyzer.scenario_embeddings)[0]

        # Top 5 matches
        top_indices = np.argsort(similarities)[::-1][:5]
        for rank, idx in enumerate(top_indices, 1):
            score = float(similarities[idx])
            scenario = scenario_descriptions[idx]
            category = scenario_categories[idx]
            print(f"  #{rank} score={score:.3f} [{category}] {scenario[:65]}...")

        best_score = float(np.max(similarities))
        best_idx = int(np.argmax(similarities))
        best_scenario = scenario_descriptions[best_idx]

        # Run full analyze to get risk_level
        result = analyzer.analyze(phrase)
        print(f"\n  → Best: {best_score:.3f} vs {best_scenario[:50]}...")
        print(f"  → analyze() result: risk_level={result['risk_level']} risk_score={result['risk_score']:.2f}")
        if result["risk_factors"]:
            for rf in result["risk_factors"][:3]:
                print(f"     {rf}")

    print("\n" + "=" * 70)
    print("Thresholds: >0.65=high, >0.40=medium, <0.3+no_tier1=low (benign overrides)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
