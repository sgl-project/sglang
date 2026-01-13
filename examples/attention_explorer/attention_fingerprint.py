#!/usr/bin/env python3
"""
Attention Fingerprinting for Self-Discovery Loop

Computes compact "cognitive fingerprints" from sparse attention data
for UMAP/HDBSCAN manifold discovery without storing full matrices.

Feature axes (8-16 dimensions):
- Hubness (X): Concentration of attention on "anchor" tokens (Gini coefficient)
- Consensus (Y): Cross-layer agreement via weighted pairwise Jaccard
- Spectral (Z): FFT of offset histogram (local syntax vs long-range reasoning)
- Offset histogram: Log2-binned attention distance distribution (8-16 bins)

Supports multi-layer format:
{
  "layers": {
    "6": {"token_positions": [...], "attention_scores": [...], ...},
    "12": {...},
    "18": {...},
    "23": {...}
  },
  "layer_id": 23,  # backward compat
  "token_positions": [...]
}

Usage:
    python attention_fingerprint.py --input attention_trace.json
    python attention_fingerprint.py --server http://localhost:8000 --prompt "Hello"
"""

import argparse
import json
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import scipy.fft

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def normalize_attention_data(
    attention_tokens: Union[List[Dict], Dict]
) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """
    Normalize attention data to both flat list and per-layer dict.

    Handles both:
    - List of entries (old format)
    - Single entry with "layers" dict (multi-layer format)

    Returns:
        flat_list: List of entries (for backward compat)
        per_layer: Dict of layer_id -> list of entries
    """
    flat_list = []
    per_layer: Dict[int, List[Dict]] = {}

    if isinstance(attention_tokens, dict):
        # Single multi-layer entry or single entry
        if "layers" in attention_tokens:
            layers = attention_tokens["layers"]
            for layer_id_str, entry in layers.items():
                layer_id = int(layer_id_str)
                entry_with_layer = {**entry, "layer_id": layer_id}
                flat_list.append(entry_with_layer)
                per_layer.setdefault(layer_id, []).append(entry_with_layer)
        else:
            # Single entry without layers
            flat_list.append(attention_tokens)
            layer_id = attention_tokens.get("layer_id", 0)
            per_layer.setdefault(layer_id, []).append(attention_tokens)
    elif isinstance(attention_tokens, list):
        # List of entries (could be list of multi-layer entries)
        for item in attention_tokens:
            if isinstance(item, dict) and "layers" in item:
                # Multi-layer entry in list
                layers = item["layers"]
                for layer_id_str, entry in layers.items():
                    layer_id = int(layer_id_str)
                    entry_with_layer = {**entry, "layer_id": layer_id}
                    flat_list.append(entry_with_layer)
                    per_layer.setdefault(layer_id, []).append(entry_with_layer)
            else:
                # Regular entry
                flat_list.append(item)
                layer_id = item.get("layer_id", 0)
                per_layer.setdefault(layer_id, []).append(item)

    return flat_list, per_layer


class AttentionFingerprint:
    """Compute compact fingerprints from attention token traces."""

    def __init__(self, num_offset_bins: int = 16, hub_capacity: int = 64):
        """
        Args:
            num_offset_bins: Number of log2 bins for offset histogram (16 covers 2^16 = 64K tokens)
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

    def compute_consensus(
        self,
        attention_tokens: List[Dict],
        per_layer: Optional[Dict[int, List[Dict]]] = None,
    ) -> float:
        """
        Axis Y: Consensus - Weighted pairwise Jaccard across layers.

        Formula: For layer pair (i,j):
            weighted_jaccard(A_i, A_j) = sum(min(w_i[k], w_j[k])) / sum(max(w_i[k], w_j[k]))
            where w_i[k] is attention weight for position k in layer i

        High consensus = rigid structure (JSON, code)
        Low consensus = fluid generation (creative, chat)
        """
        if per_layer is None:
            _, per_layer = normalize_attention_data(attention_tokens)

        layer_ids = sorted(per_layer.keys())

        if len(layer_ids) < 2:
            # Single layer: use temporal consistency
            all_positions = [
                set(e.get("token_positions", [])) for e in attention_tokens
            ]
            if len(all_positions) < 2:
                return 0.5
            similarities = []
            for i in range(1, len(all_positions)):
                intersection = len(all_positions[i - 1] & all_positions[i])
                union = len(all_positions[i - 1] | all_positions[i])
                if union > 0:
                    similarities.append(intersection / union)
            return float(np.mean(similarities)) if similarities else 0.5

        # Build weighted position dict for each layer
        layer_weights: Dict[int, Dict[int, float]] = {}
        for layer_id in layer_ids:
            weights: Dict[int, float] = {}
            for entry in per_layer[layer_id]:
                positions = entry.get("token_positions", [])
                scores = entry.get("attention_scores", [])
                for pos, score in zip(positions, scores):
                    weights[pos] = weights.get(pos, 0) + score
            layer_weights[layer_id] = weights

        # Weighted pairwise Jaccard
        pairwise_scores = []
        for i, layer_i in enumerate(layer_ids):
            for layer_j in layer_ids[i + 1 :]:
                w_i = layer_weights[layer_i]
                w_j = layer_weights[layer_j]
                all_positions = set(w_i.keys()) | set(w_j.keys())

                if not all_positions:
                    continue

                min_sum = sum(min(w_i.get(k, 0), w_j.get(k, 0)) for k in all_positions)
                max_sum = sum(max(w_i.get(k, 0), w_j.get(k, 0)) for k in all_positions)

                if max_sum > 0:
                    pairwise_scores.append(min_sum / max_sum)

        return float(np.mean(pairwise_scores)) if pairwise_scores else 0.5

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
        hist = hist[: self.num_offset_bins].astype(np.float32)

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

    def fingerprint(self, attention_tokens: Union[List[Dict], Dict]) -> Dict:
        """
        Compute full cognitive fingerprint from attention trace.

        Args:
            attention_tokens: Either list of attention entries or multi-layer dict

        Returns dict with all metrics for clustering.
        """
        flat_list, per_layer = normalize_attention_data(attention_tokens)

        hubness = self.compute_hubness(flat_list)
        consensus = self.compute_consensus(flat_list, per_layer)
        spectral = self.compute_spectral(flat_list)
        entropy = self.compute_entropy(flat_list)
        histogram = self.compute_offset_histogram(flat_list)

        return {
            "hubness": hubness,
            "consensus": consensus,
            "spectral": spectral,
            "entropy": entropy,
            "offset_histogram": histogram.tolist(),
            "vector": [hubness, consensus, spectral],  # 3D for visualization
            "full_vector": [hubness, consensus, spectral, entropy]
            + histogram.tolist(),  # Full for clustering
            "n_tokens": len(flat_list),
            "n_layers": len(per_layer),
            "layer_ids": sorted(per_layer.keys()),
        }

    def fingerprint_vector(
        self, attention_tokens: Union[List[Dict], Dict], include_histogram: bool = True
    ) -> np.ndarray:
        """
        Compute fingerprint as a feature vector for clustering.

        Args:
            attention_tokens: Attention data
            include_histogram: If True, include offset histogram (20D total)
                             If False, just core metrics (4D)

        Returns: numpy array suitable for HDBSCAN/UMAP
        """
        fp = self.fingerprint(attention_tokens)
        if include_histogram:
            return np.array(fp["full_vector"], dtype=np.float32)
        else:
            return np.array(
                [fp["hubness"], fp["consensus"], fp["spectral"], fp["entropy"]],
                dtype=np.float32,
            )


def test_fingerprint():
    """Test with synthetic and multi-layer attention data."""
    print("=" * 60)
    print("Fingerprint Tests")
    print("=" * 60)

    # Test 1: Focused code generation (single layer)
    attention_tokens = []
    for t in range(20):
        if t < 5:
            positions = [0, 1, 2, 3, 4]
        else:
            positions = [0, 1, t - 1, t - 2, t - 3]
        scores = [0.4, 0.3, 0.15, 0.1, 0.05]
        attention_tokens.append(
            {
                "token_positions": positions,
                "attention_scores": scores,
                "layer_id": 23,
            }
        )

    fp = AttentionFingerprint()
    result = fp.fingerprint(attention_tokens)

    print("\n1. Focused code generation (single layer):")
    print(f"   Hubness:   {result['hubness']:.3f} (high = focused on anchors)")
    print(f"   Consensus: {result['consensus']:.3f} (temporal consistency)")
    print(f"   Spectral:  {result['spectral']:.3f} (high = long-range)")
    print(f"   Entropy:   {result['entropy']:.3f} (low = concentrated)")
    print(f"   N_layers:  {result['n_layers']}")

    # Test 2: Multi-layer format (simulating real server output)
    multi_layer_entry = {
        "layers": {
            "6": {
                "token_positions": [0, 1, 2, 5, 8],
                "attention_scores": [0.35, 0.25, 0.2, 0.12, 0.08],
                "topk_logits": [2.5, 2.0, 1.8, 1.2, 0.9],
                "logsumexp_candidates": 3.2,
            },
            "12": {
                "token_positions": [0, 1, 3, 5, 10],
                "attention_scores": [0.40, 0.25, 0.15, 0.12, 0.08],
                "topk_logits": [2.8, 2.1, 1.5, 1.2, 0.8],
                "logsumexp_candidates": 3.5,
            },
            "18": {
                "token_positions": [0, 2, 4, 7, 12],
                "attention_scores": [0.30, 0.28, 0.20, 0.14, 0.08],
                "topk_logits": [2.3, 2.2, 1.9, 1.4, 0.9],
                "logsumexp_candidates": 3.3,
            },
            "23": {
                "token_positions": [0, 1, 2, 8, 15],
                "attention_scores": [0.38, 0.22, 0.18, 0.14, 0.08],
                "topk_logits": [2.6, 1.9, 1.7, 1.3, 0.8],
                "logsumexp_candidates": 3.4,
            },
        },
        "layer_id": 23,
        "token_positions": [0, 1, 2, 8, 15],
        "attention_scores": [0.38, 0.22, 0.18, 0.14, 0.08],
    }

    result_multi = fp.fingerprint(multi_layer_entry)

    print("\n2. Multi-layer format (4 layers: 6, 12, 18, 23):")
    print(f"   Hubness:   {result_multi['hubness']:.3f}")
    print(
        f"   Consensus: {result_multi['consensus']:.3f} (weighted cross-layer Jaccard)"
    )
    print(f"   Spectral:  {result_multi['spectral']:.3f}")
    print(f"   Entropy:   {result_multi['entropy']:.3f}")
    print(f"   N_layers:  {result_multi['n_layers']}")
    print(f"   Layer IDs: {result_multi['layer_ids']}")

    # Test 3: Stream of multi-layer entries
    stream_entries = []
    for t in range(10):
        entry = {
            "layers": {
                "6": {
                    "token_positions": [0, max(0, t - 1)],
                    "attention_scores": [0.6, 0.4],
                },
                "23": {
                    "token_positions": [0, max(0, t - 2)],
                    "attention_scores": [0.5, 0.5],
                },
            }
        }
        stream_entries.append(entry)

    result_stream = fp.fingerprint(stream_entries)

    print("\n3. Stream of multi-layer entries (10 tokens, 2 layers each):")
    print(f"   Hubness:   {result_stream['hubness']:.3f}")
    print(f"   Consensus: {result_stream['consensus']:.3f}")
    print(f"   Spectral:  {result_stream['spectral']:.3f}")
    print(f"   N_tokens:  {result_stream['n_tokens']}")
    print(f"   N_layers:  {result_stream['n_layers']}")

    # Test 4: Full vector for clustering
    vec = fp.fingerprint_vector(multi_layer_entry, include_histogram=True)
    print(f"\n4. Full fingerprint vector dimensions: {len(vec)}")
    print(f"   (4 core metrics + {fp.num_offset_bins} histogram bins)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compute attention fingerprints")
    parser.add_argument("--input", "-i", help="JSON file with attention trace")
    parser.add_argument(
        "--server", "-s", default="http://localhost:8000", help="SGLang server URL"
    )
    parser.add_argument("--prompt", "-p", help="Prompt to test with server")
    parser.add_argument(
        "--test", action="store_true", help="Run test with synthetic data"
    )

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
            },
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
