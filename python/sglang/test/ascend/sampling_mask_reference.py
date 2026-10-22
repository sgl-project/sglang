"""CPU reference for sampling-mask tests; never consumes captured mask weights."""

import math


def reference_distribution(logprobs, top_k, top_p=1.0, min_p=0.0, *, cutoff_ties=False):
    """Filter full-vocabulary log probabilities and normalize the retained mass.

    Nucleus filtering retains the token that crosses top_p. Fused Ascend top-k
    also retains ties at its cutoff, unlike the rank-based torch fallback.
    Callers supplying a prefix must establish that it covers the top-k cutoff.
    """
    ranked = sorted(logprobs.items(), key=lambda item: item[1], reverse=True)
    cutoff = ranked[min(top_k, len(ranked)) - 1][1]
    candidates = [
        (token, math.exp(value))
        for rank, (token, value) in enumerate(ranked)
        if rank < top_k or (cutoff_ties and value == cutoff)
    ]
    retained = {}
    mass = 0.0
    maximum = candidates[0][1]
    for token, weight in candidates:
        if top_p < 1.0 and mass > top_p:
            break
        mass += weight
        if weight >= maximum * min_p:
            retained[token] = weight
    denominator = math.fsum(retained.values())
    return {token: math.log(weight / denominator) for token, weight in retained.items()}


def assert_distribution(test, token, mask, actual, expected, mode, tolerance):
    """Compare IDs and token-aligned values for either return mode."""
    test.assertEqual(len(mask), len(expected))
    test.assertEqual(set(mask), set(expected))
    test.assertIn(token, expected)
    if mode == "selected":
        test.assertIsInstance(actual, float)
    pairs = [(token, actual)] if mode == "selected" else zip(mask, actual, strict=True)
    for token_id, value in pairs:
        test.assertAlmostEqual(float(value), expected[token_id], delta=tolerance)
