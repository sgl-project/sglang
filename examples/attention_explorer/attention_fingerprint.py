#!/usr/bin/env python3
"""
Attention Fingerprinting for Self-Discovery Loop

Computes compact "cognitive fingerprints" from sparse attention data
for UMAP/HDBSCAN manifold discovery without storing full matrices.

Three axes:
- Hubness (X): Concentration of attention on "anchor" tokens (SpaceSaving approx)
- Consensus (Y): Cross-layer agreement on important tokens
- Spectral (Z): FFT of offset histogram (local syntax vs long-range reasoning)

Usage:
    python attention_fingerprint.py --input attention_trace.json
    python attention_fingerprint.py --server http://localhost:8000 --prompt "Hello"
"""

import argparse
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import sys

try:
    import scipy.fft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AttentionFingerprint:
    """Compute compact fingerprints from attention token traces."""

    def __init__(self, num_offset_bins: int = 24, hub_capacity: int = 64):
        """
        Args:
            num_offset_bins: Number of log2 bins for offset histogram (covers 2^24 = 16M tokens)
            hub_capacity: Max hubs to track for SpaceSaving algorithm
        """
        self.num_offset_bins = num_offset_bins
        self.hub_capacity = hub_capacity

    def compute_hubness(self, attention_tokens: List[Dict]) -> float:
        """
        Axis X: Hubness - Are we focused on a few "anchor" tokens?

        Uses weighted in-degree approximation via streaming aggregation.
        Returns Gini coefficient of hub weights (0 = uniform, 1 = concentrated).
        """
        hub_weights = Counter()

        for entry in attention_tokens:
            positions = entry.get("token_positions", [])
            scores = entry.get("attention_scores", [])

            for pos, score in zip(positions, scores):
                hub_weights[pos] += score

        if not hub_weights:
            return 0.0

        # Keep only top hubs (SpaceSaving approximation)
        top_hubs = hub_weights.most_common(self.hub_capacity)
        weights = np.array([w for _, w in top_hubs], dtype=np.float32)

        if len(weights) < 2:
            return 0.0

        # Gini coefficient
        weights = np.sort(weights)
        n = len(weights)
        indices = np.arange(1, n + 1)
        gini = (2 * np.sum(indices * weights)) / (n * np.sum(weights)) - (n + 1) / n
        return float(np.clip(gini, 0, 1))

    def compute_consensus(self, attention_tokens: List[Dict],
                          early_layers: range = range(0, 10),
                          late_layers: range = range(20, 32)) -> float:
        """
        Axis Y: Consensus - Do early/mid layers agree with late layers?

        High consensus = rigid structure (JSON, code)
        Low consensus = fluid generation (creative, chat)
        """
        early_positions = set()
        late_positions = set()

        for entry in attention_tokens:
            layer_id = entry.get("layer_id", -1)
            positions = set(entry.get("token_positions", []))

            if layer_id in early_layers:
                early_positions.update(positions)
            elif layer_id in late_layers:
                late_positions.update(positions)

        if not early_positions or not late_positions:
            # If single layer, use self-consistency over time
            all_positions = [set(e.get("token_positions", [])) for e in attention_tokens]
            if len(all_positions) < 2:
                return 0.5
            # Jaccard similarity between consecutive steps
            similarities = []
            for i in range(1, len(all_positions)):
                intersection = len(all_positions[i-1] & all_positions[i])
                union = len(all_positions[i-1] | all_positions[i])
                if union > 0:
                    similarities.append(intersection / union)
            return float(np.mean(similarities)) if similarities else 0.5

        # Jaccard similarity between layer groups
        intersection = len(early_positions & late_positions)
        union = len(early_positions | late_positions)
        return intersection / union if union > 0 else 0.0

    def compute_spectral(self, attention_tokens: List[Dict]) -> float:
        """
        Axis Z: Spectral - Local syntax (high freq) vs long-range reasoning (low freq)?

        Computes FFT of offset histogram to detect attention "rhythm".
        High score = stable long-range patterns
        Low score = jittery local syntax
        """
        if not HAS_SCIPY:
            # Fallback: use mean log distance as proxy
            return self._compute_spectral_fallback(attention_tokens)

        # Collect all offsets
        offsets = []
        for t, entry in enumerate(attention_tokens):
            positions = entry.get("token_positions", [])
            for pos in positions:
                offset = max(0, t - pos)  # How far back we're looking
                offsets.append(offset)

        if not offsets:
            return 0.5

        # Build log2-binned histogram
        offsets = np.array(offsets, dtype=np.float32)
        # Clip to reasonable range
        offsets = np.clip(offsets, 1, 2**self.num_offset_bins)
        log_offsets = np.floor(np.log2(offsets)).astype(int)

        hist = np.bincount(log_offsets, minlength=self.num_offset_bins)
        hist = hist[:self.num_offset_bins].astype(np.float32)

        if hist.sum() < 1:
            return 0.5

        # Normalize
        hist = hist / hist.sum()

        # FFT
        fft_vals = np.abs(scipy.fft.rfft(hist))

        # Ratio of low frequency (first 3 components) to total
        low_freq_energy = np.sum(fft_vals[:3])
        total_energy = np.sum(fft_vals) + 1e-9

        return float(low_freq_energy / total_energy)

    def _compute_spectral_fallback(self, attention_tokens: List[Dict]) -> float:
        """Fallback spectral score using mean log distance."""
        distances = []
        for t, entry in enumerate(attention_tokens):
            positions = entry.get("token_positions", [])
            scores = entry.get("attention_scores", [])
            for pos, score in zip(positions, scores):
                dist = max(1, t - pos)
                distances.append(np.log(dist) * score)

        if not distances:
            return 0.5

        mean_log_dist = np.mean(distances)
        # Normalize to 0-1 range (log(1)=0 to log(1000)~7)
        return float(np.clip(mean_log_dist / 7.0, 0, 1))

    def compute_entropy(self, attention_tokens: List[Dict]) -> float:
        """
        Bonus: Attention entropy - How focused is each step?

        Low entropy = very focused attention
        High entropy = diffuse attention
        """
        entropies = []
        for entry in attention_tokens:
            scores = np.array(entry.get("attention_scores", []), dtype=np.float32)
            if len(scores) < 2:
                continue
            scores = scores / (scores.sum() + 1e-9)
            entropy = -np.sum(scores * np.log(scores + 1e-9))
            entropies.append(entropy)

        return float(np.mean(entropies)) if entropies else 0.0

    def compute_offset_histogram(self, attention_tokens: List[Dict]) -> np.ndarray:
        """
        Compute the full offset histogram for visualization/clustering.

        Returns log2-binned histogram of attention distances.
        """
        hist = np.zeros(self.num_offset_bins, dtype=np.float32)

        for t, entry in enumerate(attention_tokens):
            positions = entry.get("token_positions", [])
            scores = entry.get("attention_scores", [])

            for pos, score in zip(positions, scores):
                offset = max(1, t - pos)
                bin_idx = min(self.num_offset_bins - 1, int(np.log2(offset)))
                hist[bin_idx] += score

        # Normalize
        if hist.sum() > 0:
            hist = hist / hist.sum()

        return hist

    def fingerprint(self, attention_tokens: List[Dict]) -> Dict:
        """
        Compute full cognitive fingerprint from attention trace.

        Returns dict with all metrics for clustering.
        """
        hubness = self.compute_hubness(attention_tokens)
        consensus = self.compute_consensus(attention_tokens)
        spectral = self.compute_spectral(attention_tokens)
        entropy = self.compute_entropy(attention_tokens)
        histogram = self.compute_offset_histogram(attention_tokens)

        return {
            "hubness": hubness,
            "consensus": consensus,
            "spectral": spectral,
            "entropy": entropy,
            "offset_histogram": histogram.tolist(),
            "vector": [hubness, consensus, spectral],  # For clustering
            "n_tokens": len(attention_tokens),
        }

    def fingerprint_vector(self, attention_tokens: List[Dict]) -> np.ndarray:
        """
        Compute fingerprint as a feature vector for clustering.

        Returns: [hubness, consensus, spectral, entropy, histogram...]
        """
        fp = self.fingerprint(attention_tokens)
        base = [fp["hubness"], fp["consensus"], fp["spectral"], fp["entropy"]]
        return np.array(base + fp["offset_histogram"], dtype=np.float32)


def test_fingerprint():
    """Test with synthetic attention data."""
    # Simulate attention tokens from a focused code generation
    attention_tokens = []
    for t in range(20):
        # Focused attention: mostly looking at positions 0-3 (anchors)
        if t < 5:
            positions = [0, 1, 2, 3, 4]
        else:
            # Later tokens look back at anchors + recent tokens
            positions = [0, 1, t-1, t-2, t-3]

        scores = [0.4, 0.3, 0.15, 0.1, 0.05]

        attention_tokens.append({
            "token_positions": positions,
            "attention_scores": scores,
            "layer_id": 23,
        })

    fp = AttentionFingerprint()
    result = fp.fingerprint(attention_tokens)

    print("Fingerprint for focused code generation:")
    print(f"  Hubness:   {result['hubness']:.3f} (high = focused on anchors)")
    print(f"  Consensus: {result['consensus']:.3f} (high = stable pattern)")
    print(f"  Spectral:  {result['spectral']:.3f} (high = long-range)")
    print(f"  Entropy:   {result['entropy']:.3f} (low = concentrated)")
    print(f"  Vector:    {result['vector']}")
    print()

    # Simulate diffuse chat-like attention
    attention_tokens_chat = []
    for t in range(20):
        # Random-ish positions, recent tokens
        positions = list(range(max(0, t-4), t+1))[:5]
        scores = [0.2] * len(positions)
        if len(positions) < 5:
            positions = positions + [0] * (5 - len(positions))
            scores = scores + [0.0] * (5 - len(scores))

        attention_tokens_chat.append({
            "token_positions": positions,
            "attention_scores": scores,
            "layer_id": 23,
        })

    result_chat = fp.fingerprint(attention_tokens_chat)

    print("Fingerprint for diffuse chat:")
    print(f"  Hubness:   {result_chat['hubness']:.3f}")
    print(f"  Consensus: {result_chat['consensus']:.3f}")
    print(f"  Spectral:  {result_chat['spectral']:.3f}")
    print(f"  Entropy:   {result_chat['entropy']:.3f}")
    print(f"  Vector:    {result_chat['vector']}")


def main():
    parser = argparse.ArgumentParser(description="Compute attention fingerprints")
    parser.add_argument("--input", "-i", help="JSON file with attention trace")
    parser.add_argument("--server", "-s", default="http://localhost:8000",
                        help="SGLang server URL")
    parser.add_argument("--prompt", "-p", help="Prompt to test with server")
    parser.add_argument("--test", action="store_true", help="Run test with synthetic data")

    args = parser.parse_args()

    if args.test:
        test_fingerprint()
        return

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        attention_tokens = data.get("attention_tokens", data)
    elif args.prompt:
        import requests
        resp = requests.post(
            f"{args.server}/v1/completions",
            json={
                "model": "default",
                "prompt": args.prompt,
                "max_tokens": 50,
                "return_attention_tokens": True,
                "top_k_attention": 10,
            }
        )
        resp.raise_for_status()
        data = resp.json()
        attention_tokens = data["choices"][0].get("attention_tokens", [])
    else:
        print("Provide --input, --prompt, or --test")
        sys.exit(1)

    fp = AttentionFingerprint()
    result = fp.fingerprint(attention_tokens)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
