#!/usr/bin/env python3
"""
Attention-Guided RAG Demo

Demonstrates how attention fingerprints can improve RAG quality by:
1. Identifying which query tokens need retrieval
2. Scoring chunks by attention coherence
3. Re-ranking retrieved documents

Usage:
    # With live server (requires --attention-fingerprint-mode):
    python examples/attention_explorer/attention_rag_demo.py --server http://localhost:30000

    # With synthetic data (no server needed):
    python examples/attention_explorer/attention_rag_demo.py --synthetic

Author: SGLang Team
"""

import argparse
import sys
from typing import List, Optional

import numpy as np

# Add sglang to path
sys.path.insert(0, "python")

from sglang.srt.mem_cache.attention_guided_rag import (
    AttentionGuidedRetriever,
    QueryAnalysis,
    analyze_retrieval_need,
)


def create_synthetic_fingerprints(
    n_tokens: int, zone_type: str = "mixed"
) -> List[np.ndarray]:
    """Create synthetic fingerprints for testing."""
    fps = []
    for i in range(n_tokens):
        if zone_type == "semantic_bridge":
            # Mid-range focused, moderate entropy
            fp = np.array([0.15, 0.45, 0.40, 0.5] + list(np.random.rand(16) * 0.1))
        elif zone_type == "syntax_floor":
            # Local focused, low entropy
            fp = np.array([0.70, 0.20, 0.10, 0.2] + list(np.random.rand(16) * 0.1))
        elif zone_type == "long_range":
            # Long-range focused
            fp = np.array([0.10, 0.30, 0.60, 0.4] + list(np.random.rand(16) * 0.1))
        else:  # mixed
            if i % 3 == 0:
                fp = np.array([0.15, 0.45, 0.40, 0.6 + np.random.rand() * 0.3] + list(np.random.rand(16) * 0.1))
            elif i % 3 == 1:
                fp = np.array([0.70, 0.20, 0.10, 0.2] + list(np.random.rand(16) * 0.1))
            else:
                fp = np.array([0.10, 0.30, 0.60, 0.4] + list(np.random.rand(16) * 0.1))
        fps.append(fp)
    return fps


def demo_query_analysis():
    """Demonstrate query analysis for retrieval need."""
    print("\n" + "=" * 60)
    print("DEMO 1: Query Analysis")
    print("=" * 60)

    retriever = AttentionGuidedRetriever()

    # Test queries with different retrieval needs
    test_cases = [
        {
            "query": "What is the capital of France?",
            "tokens": ["What", "is", "the", "capital", "of", "France", "?"],
            "zones": ["semantic_bridge", "syntax_floor", "syntax_floor",
                     "semantic_bridge", "syntax_floor", "long_range", "syntax_floor"],
            "expected": "high",
        },
        {
            "query": "Hello, how are you today?",
            "tokens": ["Hello", ",", "how", "are", "you", "today", "?"],
            "zones": ["syntax_floor"] * 7,
            "expected": "low",
        },
        {
            "query": "Explain quantum entanglement and its applications in computing",
            "tokens": ["Explain", "quantum", "entanglement", "and", "its",
                      "applications", "in", "computing"],
            "zones": ["semantic_bridge", "long_range", "long_range", "syntax_floor",
                     "syntax_floor", "semantic_bridge", "syntax_floor", "long_range"],
            "expected": "high",
        },
    ]

    for case in test_cases:
        fps = create_synthetic_fingerprints(len(case["tokens"]), "mixed")

        # Override zones based on test case
        for i, zone in enumerate(case["zones"]):
            if zone == "semantic_bridge":
                fps[i] = np.array([0.15, 0.45, 0.40, 0.7] + list(np.random.rand(16) * 0.1))
            elif zone == "long_range":
                fps[i] = np.array([0.10, 0.30, 0.60, 0.5] + list(np.random.rand(16) * 0.1))
            else:
                fps[i] = np.array([0.70, 0.20, 0.10, 0.2] + list(np.random.rand(16) * 0.1))

        analysis = retriever.analyze_query(
            query=case["query"],
            fingerprints=fps,
            zones=case["zones"],
            tokens=case["tokens"],
        )

        print(f"\nQuery: \"{case['query']}\"")
        print(f"  Expected need: {case['expected']}")
        print(f"  Actual score:  {analysis.retrieval_need_score:.2f}")
        print(f"  Anchor tokens: {len(analysis.anchor_tokens)}")

        if analysis.anchor_tokens:
            anchors = [(t, f"{s:.2f}") for _, t, s in analysis.anchor_tokens[:3]]
            print(f"  Top anchors:   {anchors}")


def demo_chunk_scoring():
    """Demonstrate chunk scoring with fingerprint similarity."""
    print("\n" + "=" * 60)
    print("DEMO 2: Chunk Scoring")
    print("=" * 60)

    retriever = AttentionGuidedRetriever()

    query = "What are the health benefits of green tea?"
    query_tokens = ["What", "are", "the", "health", "benefits", "of", "green", "tea", "?"]

    # Query fingerprints - "health", "benefits", "green", "tea" are anchors
    query_fps = []
    for i, token in enumerate(query_tokens):
        if token in ["health", "benefits", "green", "tea"]:
            # Semantic bridge with high entropy (searching for info)
            fp = np.array([0.15, 0.45, 0.40, 0.75] + list(np.random.rand(16) * 0.1))
        else:
            # Syntax floor (filler words)
            fp = np.array([0.70, 0.20, 0.10, 0.2] + list(np.random.rand(16) * 0.1))
        query_fps.append(fp)

    # Chunks with different relevance
    chunks = [
        "Green tea contains catechins and antioxidants that boost metabolism and support heart health.",
        "Coffee is a popular beverage made from roasted coffee beans.",
        "Drinking tea has been associated with reduced risk of cardiovascular disease.",
        "The weather today is sunny with temperatures around 75 degrees.",
    ]

    # Create chunk fingerprints - relevant chunks have matching patterns
    chunk_fps = []
    for i, chunk in enumerate(chunks):
        if "tea" in chunk.lower() or "health" in chunk.lower():
            # Good retrieval target - focused attention, low entropy
            fps = [np.array([0.20, 0.50, 0.30, 0.25] + list(np.random.rand(16) * 0.1))
                   for _ in range(5)]
        else:
            # Poor target - diffuse attention, high entropy
            fps = [np.array([0.35, 0.30, 0.25, 0.80] + list(np.random.rand(16) * 0.1))
                   for _ in range(5)]
        chunk_fps.append(fps)

    # Score chunks
    analysis = retriever.analyze_query(query, query_fps, tokens=query_tokens)
    scores = retriever.score_chunks(
        query=query,
        chunks=chunks,
        query_analysis=analysis,
        chunk_fingerprints=chunk_fps,
    )

    print(f"\nQuery: \"{query}\"")
    print(f"Retrieval need: {analysis.retrieval_need_score:.2f}")
    print(f"\nRanked chunks:")

    for rank, score in enumerate(scores, 1):
        print(f"\n  {rank}. Score: {score.final_score:.3f}")
        print(f"     FP sim: {score.fingerprint_similarity:.3f}, Target quality: {score.target_quality:.3f}")
        print(f"     Text: {score.chunk[:60]}...")


def demo_retrieval_decision():
    """Demonstrate should_retrieve decision making."""
    print("\n" + "=" * 60)
    print("DEMO 3: Retrieval Decision")
    print("=" * 60)

    retriever = AttentionGuidedRetriever(min_retrieval_need=0.4)

    queries = [
        ("What is the GDP of Japan in 2023?", "high"),  # Needs specific facts
        ("Hello!", "low"),  # Simple greeting
        ("Summarize the key findings of the research paper", "high"),  # Needs context
        ("2 + 2 equals what?", "low"),  # Self-contained
        ("Who won the Nobel Prize in Physics last year?", "high"),  # Needs retrieval
    ]

    for query, expected in queries:
        # Create synthetic fingerprints based on expected need
        if expected == "high":
            zones = ["semantic_bridge", "long_range"] * 5
        else:
            zones = ["syntax_floor"] * 10

        fps = create_synthetic_fingerprints(len(zones),
                                           "semantic_bridge" if expected == "high" else "syntax_floor")

        analysis = retriever.analyze_query(query, fps, zones)
        should, score, reason = retriever.should_retrieve(query)

        # Override with our synthetic analysis
        should = analysis.retrieval_need_score >= 0.4

        status = "RETRIEVE" if should else "SKIP"
        match = "OK" if (should and expected == "high") or (not should and expected == "low") else "MISMATCH"

        print(f"\n  [{status}] \"{query[:40]}...\"")
        print(f"         Score: {analysis.retrieval_need_score:.2f}, Expected: {expected}, {match}")


def demo_live_server(base_url: str):
    """Demo with live server."""
    print("\n" + "=" * 60)
    print("DEMO 4: Live Server Integration")
    print("=" * 60)

    try:
        import requests

        # Check server health
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"Server not ready at {base_url}")
            return

        print(f"Connected to server at {base_url}")

        retriever = AttentionGuidedRetriever(base_url=base_url)

        # Test query analysis with live fingerprints
        query = "What are the applications of machine learning in healthcare?"
        print(f"\nAnalyzing: \"{query}\"")

        analysis = retriever.analyze_query(query)

        print(f"  Tokens analyzed: {len(analysis.fingerprints)}")
        print(f"  Retrieval need: {analysis.retrieval_need_score:.2f}")
        print(f"  Anchor tokens: {len(analysis.anchor_tokens)}")

        if analysis.retrieval_phrases:
            print(f"  Key phrases: {analysis.retrieval_phrases[:3]}")

    except Exception as e:
        print(f"Live demo failed: {e}")
        print("Run with --synthetic for offline demo")


def main():
    parser = argparse.ArgumentParser(description="Attention-Guided RAG Demo")
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="SGLang server URL for live demo",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run with synthetic data (no server needed)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ATTENTION-GUIDED RAG DEMO")
    print("=" * 60)
    print("\nThis demo shows how attention fingerprints improve RAG by:")
    print("  1. Identifying tokens that need retrieval (semantic_bridge)")
    print("  2. Scoring chunks by attention coherence")
    print("  3. Making smart retrieval decisions")

    # Run demos
    demo_query_analysis()
    demo_chunk_scoring()
    demo_retrieval_decision()

    if args.server:
        demo_live_server(args.server)
    elif not args.synthetic:
        print("\n" + "-" * 60)
        print("TIP: Run with --server URL for live demo")
        print("     Run with --synthetic for offline demo only")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
