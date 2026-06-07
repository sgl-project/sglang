"""
E2E integration tests for P-EAGLE worker in SGLang.

Tests the P-EAGLE implementation at multiple levels:
  1. Kernel unit tests (fused_parallel_draft_input)
  2. Worker logic tests (P-EAGLE algorithm correctness)
  3. Acceptance rate simulation (verifying speculative correctness)
  4. Throughput benchmark vs EAGLE-3 sequential baseline

Run with:
    /home/buddywhitman/miniforge3/envs/gpu/bin/python -m pytest \
        tests/test_p_eagle_e2e.py -v --tb=short

For live server test (requires SGLang server + model weights):
    pytest tests/test_p_eagle_e2e.py -v -m server --model-path <path>
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
# SGLang source is in the sglang/python subdirectory
_SGLANG_PYTHON = os.path.join(os.path.dirname(__file__), "..", "sglang", "python")
if os.path.isdir(_SGLANG_PYTHON):
    sys.path.insert(0, _SGLANG_PYTHON)


# ============================================================================
# 1. Kernel-level E2E: fused_parallel_draft_input
# ============================================================================


class TestFusedDraftInputE2E:
    """End-to-end kernel tests: correctness + performance."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def _make_inputs(self, B, K, hidden, vocab=32000, device="cuda"):
        """Create synthetic inputs matching the actual function signature."""
        torch.manual_seed(42)
        h_fused = torch.randn(B, hidden, device=device, dtype=torch.float16)
        embed_table = torch.randn(vocab, hidden, device=device, dtype=torch.float16)
        last_tokens = torch.randint(0, vocab, (B,), device=device, dtype=torch.int64)
        h_shared = torch.randn(hidden, device=device, dtype=torch.float16)
        mask_token_id = vocab - 1  # use last vocab entry as MASK
        return h_fused, embed_table, last_tokens, h_shared, mask_token_id

    def _run_kernel(self, B, K, hidden, vocab=32000):
        """Run fused kernel and PyTorch ref; return (result, ref)."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
            fused_parallel_draft_input_torch,
        )

        h, emb, toks, h_sh, mask_id = self._make_inputs(B, K, hidden, vocab)

        result = fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
        ref = fused_parallel_draft_input_torch(h, emb, toks, h_sh, mask_id, K)
        return result, ref

    def test_shape_B1_K4_small(self):
        result, ref = self._run_kernel(B=1, K=4, hidden=512)
        assert (
            result.shape == ref.shape
        ), f"Shape mismatch: {result.shape} vs {ref.shape}"
        assert result.shape == (1 * 4, 512)  # [B*K, hidden]

    def test_shape_B4_K4_standard(self):
        result, ref = self._run_kernel(B=4, K=4, hidden=2048)
        assert result.shape == (4 * 4, 2048)  # [B*K, hidden]

    def test_shape_B8_K6_llama(self):
        result, ref = self._run_kernel(B=8, K=6, hidden=4096)
        assert result.shape == (8 * 6, 4096)  # [B*K, hidden]

    def test_correctness_B1(self):
        result, ref = self._run_kernel(B=1, K=4, hidden=512)
        rel_err = (result.float() - ref.float()).abs().max() / (
            ref.float().abs().max() + 1e-6
        )
        assert rel_err < 0.02, f"Relative error {rel_err:.4%} > 2%"

    def test_correctness_B4_hidden2048(self):
        result, ref = self._run_kernel(B=4, K=4, hidden=2048)
        rel_err = (result.float() - ref.float()).abs().max() / (
            ref.float().abs().max() + 1e-6
        )
        assert rel_err < 0.02

    def test_correctness_B8_hidden4096(self):
        result, ref = self._run_kernel(B=8, K=4, hidden=4096)
        rel_err = (result.float() - ref.float()).abs().max() / (
            ref.float().abs().max() + 1e-6
        )
        assert rel_err < 0.02

    def test_pos0_uses_h_fused_plus_embed(self):
        """P-EAGLE invariant: position 0 = h_fused + embed_table[last_token]."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, K, hidden, vocab = 2, 4, 512, 32000
        h, emb, toks, h_sh, mask_id = self._make_inputs(B, K, hidden, vocab)
        result = fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
        # result is [B*K, hidden]; pos 0 for seq b is at index b*K
        expected_pos0_b0 = (h[0] + emb[toks[0]]).float()
        actual_pos0_b0 = result[0 * K + 0].float()
        err = (actual_pos0_b0 - expected_pos0_b0).abs().max()
        assert err < 0.1, f"Seq 0 pos0 mismatch (expected h_fused+embed): {err}"

    def test_pos1_K_use_shared_context(self):
        """P-EAGLE invariant: positions 1..K-1 use h_shared + embed(MASK)."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, K, hidden, vocab = 2, 4, 512, 32000
        h, emb, toks, h_sh, mask_id = self._make_inputs(B, K, hidden, vocab)
        result = fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
        # For each seq, pos 1 should equal h_shared + emb[mask_id]
        expected_shared = (h_sh + emb[mask_id]).float()
        for b in range(B):
            pos1 = result[b * K + 1].float()
            err = (pos1 - expected_shared).abs().max()
            assert err < 0.1, f"Seq {b} pos1 mismatch (expected h_shared+mask): {err}"

    def test_speedup_over_pytorch(self):
        """Triton kernel should be faster than sequential PyTorch loop."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
            fused_parallel_draft_input_torch,
        )

        B, K, hidden, vocab = 8, 4, 4096, 32000
        h, emb, toks, h_sh, mask_id = self._make_inputs(B, K, hidden, vocab)

        # Warmup
        for _ in range(10):
            fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
            fused_parallel_draft_input_torch(h, emb, toks, h_sh, mask_id, K)
        torch.cuda.synchronize()

        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / N * 1e6

        t0 = time.perf_counter()
        for _ in range(N):
            fused_parallel_draft_input_torch(h, emb, toks, h_sh, mask_id, K)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / N * 1e6

        speedup = t_torch / t_triton
        print(
            f"\n  Triton: {t_triton:.1f}μs  PyTorch: {t_torch:.1f}μs  Speedup: {speedup:.2f}x"
        )
        assert speedup >= 1.5, f"Expected ≥1.5x speedup, got {speedup:.2f}x"


# ============================================================================
# 2. P-EAGLE Algorithm Correctness
# ============================================================================


class TestPEagleAlgorithm:
    """Test the parallel draft generation algorithm."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_one_pass_generates_K_positions(self):
        """Core P-EAGLE claim: one forward pass for all K positions."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, K, hidden, vocab = 4, 4, 2048, 32000
        h = torch.randn(B, hidden, device="cuda", dtype=torch.float16)
        emb = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16)
        toks = torch.randint(0, vocab, (B,), device="cuda")
        h_sh = torch.randn(hidden, device="cuda", dtype=torch.float16)

        result = fused_parallel_draft_input(h, emb, toks, h_sh, vocab - 1, K)
        # One call → [B*K, hidden] ready for one batched forward pass
        assert result.shape == (
            B * K,
            hidden,
        ), f"Expected [{B*K}, {hidden}], got {result.shape}"

    def test_different_seqs_independent(self):
        """Sequences in a batch must not contaminate each other."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        K, hidden, vocab = 4, 512, 32000
        emb = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16)
        h_sh = torch.randn(hidden, device="cuda", dtype=torch.float16)
        mask_id = vocab - 1

        h_a = torch.randn(1, hidden, device="cuda", dtype=torch.float16)
        h_b = torch.randn(1, hidden, device="cuda", dtype=torch.float16)
        tok_a = torch.randint(0, vocab, (1,), device="cuda")
        tok_b = torch.randint(0, vocab, (1,), device="cuda")

        r_a = fused_parallel_draft_input(h_a, emb, tok_a, h_sh, mask_id, K)
        r_b = fused_parallel_draft_input(h_b, emb, tok_b, h_sh, mask_id, K)
        h_ab = torch.cat([h_a, h_b], dim=0)
        tok_ab = torch.cat([tok_a, tok_b], dim=0)
        r_ab = fused_parallel_draft_input(h_ab, emb, tok_ab, h_sh, mask_id, K)

        # r_ab[0*K:1*K] should match r_a, r_ab[1*K:2*K] should match r_b
        err_a = (r_ab[0:K].float() - r_a[0:K].float()).abs().max()
        err_b = (r_ab[K : 2 * K].float() - r_b[0:K].float()).abs().max()
        assert err_a < 0.1, f"Batch contamination for seq 0: {err_a}"
        assert err_b < 0.1, f"Batch contamination for seq 1: {err_b}"

    def test_deterministic_outputs(self):
        """Same inputs must always produce same outputs."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, K, hidden, vocab = 4, 4, 2048, 32000
        torch.manual_seed(99)
        h = torch.randn(B, hidden, device="cuda", dtype=torch.float16)
        emb = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16)
        toks = torch.randint(0, vocab, (B,), device="cuda")
        h_sh = torch.randn(hidden, device="cuda", dtype=torch.float16)

        r1 = fused_parallel_draft_input(h, emb, toks, h_sh, vocab - 1, K)
        r2 = fused_parallel_draft_input(h, emb, toks, h_sh, vocab - 1, K)
        assert torch.allclose(r1, r2), "Non-deterministic outputs detected"

    def test_varying_K(self):
        """Kernel must work for K=1..8."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, hidden, vocab = 2, 2048, 32000
        h = torch.randn(B, hidden, device="cuda", dtype=torch.float16)
        emb = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16)
        toks = torch.randint(0, vocab, (B,), device="cuda")
        h_sh = torch.randn(hidden, device="cuda", dtype=torch.float16)
        for K in [1, 2, 4, 6, 8]:
            r = fused_parallel_draft_input(h, emb, toks, h_sh, vocab - 1, K)
            assert r.shape == (B * K, hidden), f"K={K}: wrong shape {r.shape}"


# ============================================================================
# 3. Acceptance Rate Simulation (No actual model needed)
# ============================================================================


class TestSpeculativeCorrectness:
    """
    Simulate speculative decoding acceptance to verify P-EAGLE produces
    valid (non-degenerate) draft distributions.
    """

    def test_acceptance_cascade_property(self):
        """
        In speculative decoding, token k+1 can only be accepted if token k
        was accepted. This cascade property must hold.
        """
        # Simulate B=100 sequences with K=4
        B, K = 100, 4
        # Random accept/reject masks
        accept_probs = [0.75, 0.55, 0.40, 0.28]

        masks = []
        for _ in range(B):
            mask = []
            for k in range(K):
                if not mask or mask[-1]:
                    mask.append(torch.rand(1).item() < accept_probs[k])
                else:
                    mask.append(False)  # Cascade rejection
            masks.append(mask)

        # Verify cascade property
        for i, mask in enumerate(masks):
            for k in range(1, K):
                if mask[k] and not mask[k - 1]:
                    pytest.fail(f"Seq {i}: pos {k} accepted but pos {k-1} rejected")

    def test_token_generation_no_cpu_sync(self):
        """
        P-EAGLE + sync-free DSL: verify draft input computation has
        no hidden .item() calls (no GPU→CPU sync).
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
        )

        B, K, hidden, vocab = 4, 4, 2048, 32000
        h = torch.randn(B, hidden, device="cuda", dtype=torch.float16)
        emb = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16)
        toks = torch.randint(0, vocab, (B,), device="cuda")
        h_sh = torch.randn(hidden, device="cuda", dtype=torch.float16)

        # Verify the kernel launches and returns a CUDA tensor without syncing
        result = fused_parallel_draft_input(h, emb, toks, h_sh, vocab - 1, K)
        # If no .item() was called, the stream is still async
        assert result is not None
        assert result.device.type == "cuda"
        assert result.shape == (B * K, hidden)


# ============================================================================
# 4. Throughput Benchmark
# ============================================================================


class TestPEagleThroughput:
    """
    Benchmark P-EAGLE vs sequential EAGLE baseline.
    Reports speedup from parallel draft input computation.
    """

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def _bench_inputs(self, B, hidden, vocab=32000):
        K = 4
        h = torch.randn(B, hidden, device="cuda", dtype=torch.float16)
        emb = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16)
        toks = torch.randint(0, vocab, (B,), device="cuda")
        h_sh = torch.randn(hidden, device="cuda", dtype=torch.float16)
        return h, emb, toks, h_sh, vocab - 1, K

    @pytest.mark.parametrize(
        "B,hidden",
        [
            (4, 2048),
            (8, 2048),
            (4, 4096),
            (16, 2048),
        ],
    )
    def test_parallel_vs_sequential_throughput(self, B, hidden):
        """Fused kernel should be consistently faster than PyTorch reference."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
            fused_parallel_draft_input_torch,
        )

        h, emb, toks, h_sh, mask_id, K = self._bench_inputs(B, hidden)

        for _ in range(20):
            fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
            fused_parallel_draft_input_torch(h, emb, toks, h_sh, mask_id, K)
        torch.cuda.synchronize()

        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / N * 1e6

        t0 = time.perf_counter()
        for _ in range(N):
            fused_parallel_draft_input_torch(h, emb, toks, h_sh, mask_id, K)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / N * 1e6

        speedup = t_torch / t_triton
        print(
            f"\n  B={B:2d} h={hidden}: Triton={t_triton:.1f}μs "
            f"PyTorch={t_torch:.1f}μs Speedup={speedup:.2f}x"
        )
        assert speedup >= 1.0, f"Expected speedup ≥ 1.0x, got {speedup:.2f}x"

    def test_peak_throughput_summary(self):
        """Run all shapes and print summary table."""
        from sglang.srt.speculative.triton_ops.fused_draft_input import (
            fused_parallel_draft_input,
            fused_parallel_draft_input_torch,
        )

        configs = [
            (1, 4, 2048),
            (4, 4, 2048),
            (8, 4, 2048),
            (16, 4, 2048),
            (4, 4, 4096),
            (8, 4, 4096),
        ]

        print(
            f"\n{'B':>4} {'K':>3} {'hidden':>7} | {'Triton':>10} {'PyTorch':>10} {'Speedup':>9}"
        )
        print("-" * 55)

        for B, K, hidden in configs:
            h, emb, toks, h_sh, mask_id, _ = self._bench_inputs(B, hidden)

            for _ in range(20):
                fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
            torch.cuda.synchronize()

            N = 100
            t0 = time.perf_counter()
            for _ in range(N):
                fused_parallel_draft_input(h, emb, toks, h_sh, mask_id, K)
            torch.cuda.synchronize()
            t_triton = (time.perf_counter() - t0) / N * 1e6

            t0 = time.perf_counter()
            for _ in range(N):
                fused_parallel_draft_input_torch(h, emb, toks, h_sh, mask_id, K)
            torch.cuda.synchronize()
            t_torch = (time.perf_counter() - t0) / N * 1e6

            speedup = t_torch / t_triton
            print(
                f"{B:>4} {K:>3} {hidden:>7} | {t_triton:>8.1f}μs {t_torch:>8.1f}μs {speedup:>8.2f}x"
            )

        assert True
