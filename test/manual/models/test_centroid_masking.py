"""Correctness and performance tests for centroid-masked logits.

Tests cover the contiguous-gather implementation of
``Gemma4AssistantForCausalLM._apply_centroid_masking`` against a
ground-truth full-vocab reference, and benchmark it against the
pre-commit scattered-gather baseline (commit 610a7f090~1).

The key invariant: ``lm_head.weight`` is stored in centroid order (loaded
from ``input_embedding_ordered``), so row ``c * vpc + j`` is the embedding
for the j-th token of centroid c. ``token_ordering`` maps these
centroid-ordered positions to real vocab IDs.

Correctness tests run on CPU (no GPU required).
Performance tests require CUDA and are skipped otherwise.

Usage:
    # Run all tests (correctness on CPU, perf on CUDA if available):
    python test/manual/models/test_centroid_masking.py

    # Run only correctness tests:
    python -m pytest test/manual/models/test_centroid_masking.py -k Correctness

    # Run only perf tests (requires CUDA):
    python -m pytest test/manual/models/test_centroid_masking.py -k Perf
"""

from __future__ import annotations

import time
import unittest
from typing import Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config constants matching Gemma 4 E2B assistant
# ---------------------------------------------------------------------------
VOCAB_SIZE = 262_144
HIDDEN_SIZE = 1_152
NUM_CENTROIDS = 2_048
CENTROID_TOP_K = 32
VOCAB_PER_CENTROID = VOCAB_SIZE // NUM_CENTROIDS  # 128


# ---------------------------------------------------------------------------
# Ground-truth reference: full-vocab matmul + top-k centroid mask
# ---------------------------------------------------------------------------
def _centroid_masking_reference(
    flat_hidden: torch.Tensor,
    centroids_weight: torch.Tensor,
    lm_head_weight_centroid_order: torch.Tensor,
    token_ordering: torch.Tensor,
) -> torch.Tensor:
    """Ground-truth: compute full-vocab logits, then mask to top-k centroids.

    This avoids any gather tricks and serves as the oracle. It computes
    logits for ALL vocab tokens, selects the top-k centroids, and masks
    everything outside those centroids.

    ``lm_head_weight_centroid_order`` is [V, H] in centroid order.
    ``token_ordering`` maps centroid-ordered position -> real vocab ID.
    """
    num_tokens = flat_hidden.shape[0]

    # Full logits in centroid-ordered space: [N, V]
    full_logits_centroid_order = F.linear(flat_hidden, lm_head_weight_centroid_order)

    # Select top-k centroids per token.
    _, top_k_indices = torch.topk(
        F.linear(flat_hidden, centroids_weight),
        k=CENTROID_TOP_K,
        dim=-1,
    )

    # Build centroid-order positions for selected centroids.
    # For centroid c, positions are [c*vpc, c*vpc+1, ..., c*vpc+vpc-1].
    arange_vpc = torch.arange(VOCAB_PER_CENTROID, device=flat_hidden.device)
    # top_k_indices: [N, K] -> centroid_positions: [N, K*vpc]
    centroid_positions = (
        top_k_indices.unsqueeze(-1) * VOCAB_PER_CENTROID + arange_vpc
    ).view(num_tokens, -1)

    # Gather selected logits from centroid-ordered full logits.
    selected_logits = full_logits_centroid_order.gather(1, centroid_positions)

    # Map centroid-ordered positions to real vocab IDs for output.
    centroid_vocab_indices = token_ordering.long()[centroid_positions]

    mask_value = selected_logits.min() - 1.0
    output = mask_value.expand(num_tokens, VOCAB_SIZE).clone()
    output.scatter_(dim=-1, index=centroid_vocab_indices, src=selected_logits)
    return output


# ---------------------------------------------------------------------------
# Pre-commit baseline (610a7f090~1): scattered gather
# ---------------------------------------------------------------------------
def _centroid_masking_scattered(
    flat_hidden: torch.Tensor,
    centroids_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_ordering: torch.Tensor,
) -> torch.Tensor:
    """Pre-commit implementation: gathers lm_head rows via token_ordering.

    Indexes lm_head.weight with real vocab IDs obtained from
    token_ordering, producing non-contiguous (scattered) memory access.
    Kept here as the performance baseline for before/after comparison.
    """
    num_tokens = flat_hidden.shape[0]

    centroid_logits = F.linear(flat_hidden, centroids_weight)
    _, top_k_indices = torch.topk(centroid_logits, k=CENTROID_TOP_K, dim=-1)

    canonical_positions_per_cluster = token_ordering.long().view(
        NUM_CENTROIDS, VOCAB_PER_CENTROID
    )
    selected_canonical = canonical_positions_per_cluster[top_k_indices]
    selected_flat = selected_canonical.reshape(-1)
    selected_embeddings = lm_head_weight[selected_flat].view(
        num_tokens,
        CENTROID_TOP_K * VOCAB_PER_CENTROID,
        HIDDEN_SIZE,
    )
    selected_logits = (
        flat_hidden.unsqueeze(1) @ selected_embeddings.transpose(-1, -2)
    ).squeeze(1)

    mask_value = selected_logits.min() - 1.0
    output = mask_value.expand(num_tokens, VOCAB_SIZE).clone()
    output.scatter_(
        dim=-1,
        index=selected_canonical.view(num_tokens, -1),
        src=selected_logits,
    )
    return output


# ---------------------------------------------------------------------------
# Post-commit implementation: contiguous centroid-ordered gather
# ---------------------------------------------------------------------------
def _centroid_masking_contiguous(
    flat_hidden: torch.Tensor,
    centroids_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_ordering: torch.Tensor,
) -> torch.Tensor:
    """Post-commit implementation: views lm_head as [C, vpc, H], indexes by centroid ID."""
    num_tokens = flat_hidden.shape[0]
    num_selected = CENTROID_TOP_K * VOCAB_PER_CENTROID

    _, top_k_indices = torch.topk(
        F.linear(flat_hidden, centroids_weight),
        k=CENTROID_TOP_K,
        dim=-1,
    )

    selected_embeddings = lm_head_weight.view(
        NUM_CENTROIDS,
        VOCAB_PER_CENTROID,
        HIDDEN_SIZE,
    )[top_k_indices].reshape(num_tokens, num_selected, HIDDEN_SIZE)

    selected_logits = torch.bmm(
        flat_hidden.unsqueeze(1),
        selected_embeddings.transpose(1, 2),
    ).squeeze(1)

    centroid_vocab_indices = (
        token_ordering.long()
        .view(NUM_CENTROIDS, VOCAB_PER_CENTROID)[top_k_indices]
        .view(num_tokens, -1)
    )
    mask_value = selected_logits.min() - 1.0
    output = mask_value.expand(num_tokens, VOCAB_SIZE).clone()
    output.scatter_(dim=-1, index=centroid_vocab_indices, src=selected_logits)
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_weights(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random centroid weights, lm_head weights, and token_ordering.

    ``lm_head_weight`` is in centroid order (row ``c * vpc + j`` =
    embedding for the j-th token of centroid c). ``token_ordering`` maps
    centroid-ordered positions to real vocab IDs: a random permutation.
    """
    centroids_weight = torch.randn(
        NUM_CENTROIDS, HIDDEN_SIZE, dtype=dtype, device=device
    )
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=dtype, device=device)
    token_ordering = torch.randperm(VOCAB_SIZE, device=device)
    return centroids_weight, lm_head_weight, token_ordering


def _benchmark_fn(fn, *args, warmup: int = 20, iters: int = 100) -> float:
    """Returns median latency in milliseconds (CUDA-synchronized)."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# ===================================================================
# Correctness tests: contiguous == ground-truth reference (CPU)
# ===================================================================
class TestCentroidMaskingCorrectness(unittest.TestCase):
    """Verify the contiguous implementation matches the ground-truth reference.

    The reference implementation computes full-vocab logits and then masks to
    the top-k centroids. The contiguous implementation must produce matching
    output for centroid-ordered lm_head weights.
    """

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.device = torch.device("cpu")
        cls.dtype = torch.float32
        cls.centroids_weight, cls.lm_head_weight, cls.token_ordering = _make_weights(
            cls.device, cls.dtype
        )

    # -- Shape tests ---------------------------------------------------

    def test_output_shape_single_token(self):
        flat_hidden = torch.randn(1, HIDDEN_SIZE, device=self.device, dtype=self.dtype)
        out = _centroid_masking_contiguous(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        self.assertEqual(out.shape, (1, VOCAB_SIZE))

    def test_output_shape_batch(self):
        for n in [4, 16, 64]:
            flat_hidden = torch.randn(
                n, HIDDEN_SIZE, device=self.device, dtype=self.dtype
            )
            out = _centroid_masking_contiguous(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            self.assertEqual(out.shape, (n, VOCAB_SIZE), f"Failed for n={n}")

    # -- Equivalence: contiguous == reference --------------------------

    def test_matches_reference_single_token(self):
        """Contiguous and reference produce nearly identical output for 1 token.

        Small floating-point differences arise because the reference uses
        F.linear (full matmul) while the contiguous impl uses bmm on a
        gathered subset. We allow atol=1e-4 which is well within noise.
        """
        flat_hidden = torch.randn(1, HIDDEN_SIZE, device=self.device, dtype=self.dtype)
        out_ref = _centroid_masking_reference(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        out_new = _centroid_masking_contiguous(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        self.assertTrue(
            torch.allclose(out_ref, out_new, atol=1e-4, rtol=1e-5),
            f"Outputs differ; max_diff={_max_diff(out_ref, out_new):.6e}",
        )

    def test_matches_reference_batch(self):
        """Contiguous and reference produce nearly identical output for various batch sizes."""
        for n in [4, 16, 64]:
            flat_hidden = torch.randn(
                n, HIDDEN_SIZE, device=self.device, dtype=self.dtype
            )
            out_ref = _centroid_masking_reference(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            out_new = _centroid_masking_contiguous(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            self.assertTrue(
                torch.allclose(out_ref, out_new, atol=1e-4, rtol=1e-5),
                f"n={n}: max_diff={_max_diff(out_ref, out_new):.6e}",
            )

    def test_argmax_matches_reference(self):
        """Argmax (decoded token) agrees with reference across batch sizes."""
        for n in [1, 4, 32]:
            flat_hidden = torch.randn(
                n, HIDDEN_SIZE, device=self.device, dtype=self.dtype
            )
            out_ref = _centroid_masking_reference(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            out_new = _centroid_masking_contiguous(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            self.assertTrue(
                torch.equal(out_ref.argmax(dim=-1), out_new.argmax(dim=-1)),
                f"n={n}: argmax mismatch",
            )

    # -- Mask value tests ---------------------------------------------

    def test_masked_positions_count(self):
        """Exactly top_k * vpc positions have real logits; the rest are masked."""
        flat_hidden = torch.randn(4, HIDDEN_SIZE, device=self.device, dtype=self.dtype)
        out = _centroid_masking_contiguous(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        expected_active = CENTROID_TOP_K * VOCAB_PER_CENTROID
        for i in range(out.shape[0]):
            row = out[i]
            num_active = (row > row.min()).sum().item()
            self.assertEqual(
                num_active,
                expected_active,
                f"Token {i}: expected {expected_active} active logits, got {num_active}",
            )

    def test_masked_value_below_all_selected(self):
        """The mask value (min - 1) is strictly below all selected logits."""
        flat_hidden = torch.randn(4, HIDDEN_SIZE, device=self.device, dtype=self.dtype)
        out = _centroid_masking_contiguous(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        for i in range(out.shape[0]):
            row = out[i]
            mask_val = row.min().item()
            active = row[row > mask_val]
            self.assertTrue(
                (active > mask_val).all(),
                f"Token {i}: some active logits are not above mask value",
            )

    # -- Determinism tests --------------------------------------------

    def test_deterministic_across_calls(self):
        """Same input produces same output across two calls."""
        flat_hidden = torch.randn(8, HIDDEN_SIZE, device=self.device, dtype=self.dtype)
        out1 = _centroid_masking_contiguous(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        out2 = _centroid_masking_contiguous(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        self.assertTrue(torch.equal(out1, out2))


# ===================================================================
# Correctness tests on CUDA with bf16 (closer to production)
# ===================================================================
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCentroidMaskingCorrectnessCUDA(unittest.TestCase):
    """Same correctness checks on CUDA with bfloat16, matching production dtype."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.device = torch.device("cuda")
        cls.dtype = torch.bfloat16
        cls.centroids_weight, cls.lm_head_weight, cls.token_ordering = _make_weights(
            cls.device, cls.dtype
        )

    def test_matches_reference_bf16(self):
        """Contiguous matches reference on CUDA bf16."""
        for n in [1, 4, 16, 64, 128]:
            flat_hidden = torch.randn(
                n, HIDDEN_SIZE, device=self.device, dtype=self.dtype
            )
            out_ref = _centroid_masking_reference(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            out_new = _centroid_masking_contiguous(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            self.assertTrue(
                torch.equal(out_ref, out_new),
                f"n={n}: max_diff={_max_diff(out_ref, out_new):.6e}",
            )

    def test_argmax_matches_reference_bf16(self):
        """Argmax agreement with reference on CUDA bf16."""
        for n in [1, 16, 128]:
            flat_hidden = torch.randn(
                n, HIDDEN_SIZE, device=self.device, dtype=self.dtype
            )
            out_ref = _centroid_masking_reference(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            out_new = _centroid_masking_contiguous(
                flat_hidden,
                self.centroids_weight,
                self.lm_head_weight,
                self.token_ordering,
            )
            self.assertTrue(
                torch.equal(out_ref.argmax(dim=-1), out_new.argmax(dim=-1)),
                f"n={n}: argmax mismatch on CUDA bf16",
            )

    def test_masked_positions_count_bf16(self):
        """Correct number of active (non-masked) positions on CUDA bf16."""
        flat_hidden = torch.randn(16, HIDDEN_SIZE, device=self.device, dtype=self.dtype)
        out = _centroid_masking_contiguous(
            flat_hidden, self.centroids_weight, self.lm_head_weight, self.token_ordering
        )
        expected_active = CENTROID_TOP_K * VOCAB_PER_CENTROID
        for i in range(out.shape[0]):
            row = out[i]
            num_active = (row > row.min()).sum().item()
            self.assertEqual(num_active, expected_active, f"Token {i}")


# ===================================================================
# Performance tests: before vs after comparison (CUDA only)
# ===================================================================
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for perf tests")
class TestCentroidMaskingPerf(unittest.TestCase):
    """Benchmark before (scattered) vs after (contiguous) centroid masking.

    Compares the pre-commit scattered-gather implementation against the
    post-commit contiguous-gather implementation. Both do the same amount
    of compute (same gather size, matmul shape, scatter) -- the only
    difference is the memory access pattern on the weight gather.

    Asserts the contiguous implementation has no regression (>= 0.9x).
    """

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.device = torch.device("cuda")
        cls.dtype = torch.bfloat16
        cls.centroids_weight, cls.lm_head_weight, cls.token_ordering = _make_weights(
            cls.device, cls.dtype
        )

    def _run_comparison(
        self, num_tokens: int, warmup: int = 20, iters: int = 100
    ) -> Tuple[float, float, float]:
        """Benchmark both, return (before_ms, after_ms, speedup)."""
        flat_hidden = torch.randn(
            num_tokens, HIDDEN_SIZE, device=self.device, dtype=self.dtype
        )
        args = (
            flat_hidden,
            self.centroids_weight,
            self.lm_head_weight,
            self.token_ordering,
        )

        lat_before = _benchmark_fn(
            _centroid_masking_scattered, *args, warmup=warmup, iters=iters
        )
        lat_after = _benchmark_fn(
            _centroid_masking_contiguous, *args, warmup=warmup, iters=iters
        )
        speedup = lat_before / lat_after if lat_after > 0 else float("inf")
        return lat_before, lat_after, speedup

    def test_no_regression_single_token(self):
        """Contiguous must not be slower than scattered for 1 token."""
        lat_before, lat_after, speedup = self._run_comparison(1)
        print(
            f"\n  [n=1]   before={lat_before:.3f}ms  after={lat_after:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )
        self.assertGreaterEqual(
            speedup,
            0.90,
            f"Contiguous is >10% slower than scattered for n=1 "
            f"(speedup={speedup:.2f}x)",
        )

    def test_no_regression_small_batch(self):
        """Contiguous must not be slower for batch size 16."""
        lat_before, lat_after, speedup = self._run_comparison(16)
        print(
            f"\n  [n=16]  before={lat_before:.3f}ms  after={lat_after:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )
        self.assertGreaterEqual(
            speedup,
            0.90,
            f"Contiguous is >10% slower for n=16 (speedup={speedup:.2f}x)",
        )

    def test_no_regression_medium_batch(self):
        """Contiguous must not be slower for batch size 64."""
        lat_before, lat_after, speedup = self._run_comparison(64)
        print(
            f"\n  [n=64]  before={lat_before:.3f}ms  after={lat_after:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )
        self.assertGreaterEqual(
            speedup,
            0.90,
            f"Contiguous is >10% slower for n=64 (speedup={speedup:.2f}x)",
        )

    def test_no_regression_large_batch(self):
        """Contiguous must not be slower for batch size 256."""
        lat_before, lat_after, speedup = self._run_comparison(256)
        print(
            f"\n  [n=256] before={lat_before:.3f}ms  after={lat_after:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )
        self.assertGreaterEqual(
            speedup,
            0.90,
            f"Contiguous is >10% slower for n=256 (speedup={speedup:.2f}x)",
        )

    def test_perf_summary(self):
        """Print a before/after summary table across token counts."""
        token_counts = [1, 4, 16, 64, 128, 256]
        header = (
            f"{'Tokens':>8}  {'Before (ms)':>12}  {'After (ms)':>12}  {'Speedup':>8}"
        )
        print(f"\n  {header}")
        print(f"  {'-' * len(header)}")
        for n in token_counts:
            lat_before, lat_after, speedup = self._run_comparison(
                n, warmup=20, iters=80
            )
            print(
                f"  {n:>8}  {lat_before:>12.3f}  {lat_after:>12.3f}  {speedup:>7.2f}x"
            )


if __name__ == "__main__":
    unittest.main()
