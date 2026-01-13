#!/usr/bin/env python3
"""
Token Importance Scoring Demo

Demonstrates how to use attention-based token importance scoring for
smart context truncation.

Usage:
    # First start a server with attention fingerprint mode:
    python -m sglang.launch_server \
        --model Qwen/Qwen3-1.7B \
        --attention-fingerprint-mode \
        --return-attention-tokens \
        --port 30000

    # Then run this demo:
    python examples/attention_explorer/token_importance_demo.py

Author: SGLang Team
"""

import argparse
import sys
from typing import List, Optional

import numpy as np

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

# Add sglang to path for imports
sys.path.insert(0, "python")

from sglang.srt.mem_cache.token_importance import (
    ImportanceResult,
    SmartTruncator,
    TokenImportanceScorer,
)


def generate_with_attention(
    base_url: str,
    prompt: str,
    max_tokens: int = 100,
) -> dict:
    """Generate text with attention token capture."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "return_attention_tokens": True,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    return response.json()


def extract_fingerprints(data: dict) -> Optional[np.ndarray]:
    """Extract fingerprints from API response."""
    attention_tokens = data.get("choices", [{}])[0].get("attention_tokens", [])

    if not attention_tokens:
        print("No attention tokens in response")
        return None

    # Check if fingerprint mode
    if not attention_tokens[0].get("fingerprint"):
        print("Response not in fingerprint mode")
        return None

    fingerprints = [item["fingerprint"] for item in attention_tokens]
    return np.array(fingerprints)


def demo_importance_scoring(fingerprints: np.ndarray) -> ImportanceResult:
    """Demonstrate token importance scoring."""
    print("\n" + "=" * 60)
    print("TOKEN IMPORTANCE SCORING DEMO")
    print("=" * 60)

    scorer = TokenImportanceScorer(
        zone_weight=0.4,
        entropy_weight=0.3,
        long_range_weight=0.2,
        position_weight=0.1,
    )

    result = scorer.score_from_fingerprints(fingerprints, keep_ratio=0.5)

    print(f"\nScored {len(result.scores)} tokens")
    print(f"Mean importance: {result.mean_score:.3f}")

    # Show top 10 most important tokens
    print("\n📊 TOP 10 MOST IMPORTANT TOKENS:")
    print("-" * 40)
    sorted_scores = sorted(result.scores, key=lambda s: s.score, reverse=True)
    for i, score in enumerate(sorted_scores[:10]):
        print(
            f"  {i+1:2d}. Token {score.index:3d}: {score.score:.3f} " f"({score.zone})"
        )
        print(f"      Components: {score.components}")

    # Show bottom 5 (candidates for truncation)
    print("\n🗑️  BOTTOM 5 TOKENS (truncation candidates):")
    print("-" * 40)
    for i, score in enumerate(sorted_scores[-5:]):
        print(f"  {i+1}. Token {score.index:3d}: {score.score:.3f} " f"({score.zone})")

    # Zone distribution
    zones = {}
    for s in result.scores:
        zones[s.zone] = zones.get(s.zone, 0) + 1

    print("\n🔮 ZONE DISTRIBUTION:")
    print("-" * 40)
    for zone, count in sorted(zones.items(), key=lambda x: -x[1]):
        pct = count / len(result.scores) * 100
        bar = "█" * int(pct / 5)
        print(f"  {zone:18s}: {count:3d} ({pct:5.1f}%) {bar}")

    return result


def demo_truncation(fingerprints: np.ndarray, tokens: List[str]) -> None:
    """Demonstrate smart truncation."""
    print("\n" + "=" * 60)
    print("SMART TRUNCATION DEMO")
    print("=" * 60)

    # Simulate needing to truncate to 50% of original length
    original_length = len(tokens)
    target_length = original_length // 2

    print(f"\nOriginal: {original_length} tokens")
    print(f"Target:   {target_length} tokens")

    truncator = SmartTruncator(
        max_context=target_length,
        preserve_start_ratio=0.2,
        preserve_end_ratio=0.1,
    )

    truncated_tokens, _, kept_indices = truncator.truncate(tokens, fingerprints)

    print(f"\nKept {len(truncated_tokens)} tokens")
    print(f"Reduction: {(1 - len(truncated_tokens)/original_length)*100:.1f}%")

    # Show which indices were kept
    print(f"\nKept indices (first 20): {kept_indices[:20]}...")

    # Show kept vs dropped distribution
    dropped = set(range(original_length)) - set(kept_indices)
    print(
        f"\nDropped {len(dropped)} tokens from indices: {sorted(list(dropped))[:10]}..."
    )


def demo_synthetic() -> None:
    """Demo with synthetic data when no server is available."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA DEMO (no server needed)")
    print("=" * 60)

    # Create synthetic fingerprints for 100 tokens
    n_tokens = 100
    np.random.seed(42)

    # Simulate different zones
    fingerprints = []
    zones = []
    for i in range(n_tokens):
        # Position-based zone assignment
        if i < 10:  # Start tokens - important instructions
            local_mass = 0.3
            mid_mass = 0.5
            long_mass = 0.2
            entropy = 1.5
            zones.append("semantic_bridge")
        elif i < 30:  # Early context - mixed
            local_mass = 0.5
            mid_mass = 0.3
            long_mass = 0.2
            entropy = 2.5
            zones.append("structure_ripple")
        elif i < 70:  # Middle - mostly syntax
            local_mass = 0.7
            mid_mass = 0.2
            long_mass = 0.1
            entropy = 3.0
            zones.append("syntax_floor")
        else:  # Recent context - important
            local_mass = 0.4
            mid_mass = 0.3
            long_mass = 0.3
            entropy = 2.0
            zones.append("long_range")

        # Add some noise
        fp = [
            local_mass + np.random.normal(0, 0.1),
            mid_mass + np.random.normal(0, 0.1),
            long_mass + np.random.normal(0, 0.1),
            entropy + np.random.normal(0, 0.2),
        ]
        # Add histogram bins (8 bins)
        fp.extend(np.random.dirichlet(np.ones(8)).tolist())
        fingerprints.append(fp)

    fingerprints = np.array(fingerprints)

    # Score tokens
    scorer = TokenImportanceScorer()
    result = scorer.score_from_fingerprints(fingerprints, zones=zones)

    print(f"\nGenerated {n_tokens} synthetic tokens with fingerprints")
    print(f"Mean importance: {result.mean_score:.3f}")

    # Zone distribution
    print("\n📊 Zone distribution in synthetic data:")
    from collections import Counter

    zone_counts = Counter(zones)
    for zone, count in zone_counts.most_common():
        print(f"  {zone}: {count}")

    # Show importance distribution by zone
    print("\n📈 Average importance by zone:")
    zone_scores = {}
    for s, z in zip(result.scores, zones):
        if z not in zone_scores:
            zone_scores[z] = []
        zone_scores[z].append(s.score)

    for zone, scores in sorted(zone_scores.items(), key=lambda x: -np.mean(x[1])):
        avg = np.mean(scores)
        print(f"  {zone:18s}: {avg:.3f}")

    # Demo truncation
    tokens = [f"token_{i}" for i in range(n_tokens)]
    demo_truncation(fingerprints, tokens)


def main():
    parser = argparse.ArgumentParser(description="Token Importance Scoring Demo")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data (no server needed)"
    )
    parser.add_argument(
        "--prompt",
        default="Explain how attention mechanisms work in transformers and why they are important for language understanding.",
        help="Prompt to analyze",
    )
    args = parser.parse_args()

    if args.synthetic:
        demo_synthetic()
        return

    base_url = f"http://{args.host}:{args.port}"

    # Check server connection
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        if resp.status_code != 200:
            print(f"Server not available at {base_url}")
            print("Running synthetic demo instead...")
            demo_synthetic()
            return
    except Exception as e:
        print(f"Could not connect to server: {e}")
        print("Running synthetic demo instead...")
        demo_synthetic()
        return

    print(f"Connected to server at {base_url}")

    # Generate with attention capture
    print(f"\nGenerating response for: '{args.prompt[:50]}...'")
    data = generate_with_attention(base_url, args.prompt, max_tokens=50)

    # Extract fingerprints
    fingerprints = extract_fingerprints(data)

    if fingerprints is None:
        print(
            "\nNo fingerprints available. Make sure server has --attention-fingerprint-mode"
        )
        print("Running synthetic demo instead...")
        demo_synthetic()
        return

    # Demo importance scoring
    result = demo_importance_scoring(fingerprints)

    # Demo truncation
    # Create dummy token list matching fingerprint count
    tokens = [f"tok_{i}" for i in range(len(fingerprints))]
    demo_truncation(fingerprints, tokens)

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
