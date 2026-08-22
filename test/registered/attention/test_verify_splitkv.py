"""Correctness tests for the split-KV EAGLE-verify attention kernel.

``verify_splitkv_fwd`` is a drop-in for ``extend_attention_fwd`` on the topk=1
causal (target-verify) path. These tests check:
  (a) numerical parity with ``extend_attention_fwd`` on the pure-causal path
      (the operative case at topk=1), across head dims / GQA ratios / prefix
      lengths / extend lengths / KV scales;
  (b) numerical parity on the bidirectional (``is_causal=False``) draft
      block, which is what ENCODER_ONLY draft layers need;
  (c) ``can_handle()`` rejects cases the kernel cannot serve bit-equivalently
      (explicit custom_mask, sinks, sliding-window, logit-cap, ragged extend),
      so the backend falls back to ``extend_attention_fwd``;
  (d) ``choose_n_splits`` keeps its tuned ladder bit-identical when the base
      grid already fills the device, and only raises the cap when it does not.

The topk>1 case is gated off in the backend (TritonAttnBackend enables this path
only when ``self.topk == 1``), since the kernel ignores the tree custom_mask;
that gate is exercised end-to-end by the nightly ROCm spec accuracy test.

GPU + Triton required. Runs on the CUDA PR lane and both AMD lanes
(MI30x/gfx942 and MI35x/gfx950 -- the kernel is enabled on both).
"""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import (
    extend_attention_fwd,
)
from sglang.kernels.ops.attention.verify_splitkv import (
    HARD_MAX_N_SPLITS,
    MAX_N_SPLITS,
    can_handle,
    choose_n_splits,
    verify_splitkv_fwd,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

# Split-KV accumulates the prefix in parallel splits and merges via log-sum-exp;
# it differs from the single-pass extend kernel only by reduction order. On-GPU
# (gfx950) the max abs diff across these shapes was ~2e-3 (rising to ~6e-2 only
# at much longer ctx); 2e-2 keeps a ~10x margin over the observed noise.
ATOL = 2e-2
RTOL = 1e-2


def _build_verify_inputs(
    prefix_lens, l_ext, h_q, h_kv, head_dim, v_head_dim, dtype, device
):
    """Build a verify-shaped problem: a constant extend length ``l_ext`` per
    sequence, with the prefix cache addressed contiguously by ``kv_indices``.
    Returns the positional args shared by extend_attention_fwd / verify_splitkv_fwd.
    """
    B = len(prefix_lens)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    total_prefix = int(prefix_lens_t.sum())

    # Prefix KV cache laid out contiguously; kv_indices is just arange over it.
    # kv_indices is int64 in production (TritonAttnBackend allocates int64) -- match it.
    k_buffer = torch.randn(total_prefix, h_kv, head_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(total_prefix, h_kv, v_head_dim, dtype=dtype, device=device)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(prefix_lens_t, 0)
    kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=device)

    # Draft (extend) tensors: constant l_ext rows per sequence.
    n_ext = B * l_ext
    q_extend = torch.randn(n_ext, h_q, head_dim, dtype=dtype, device=device)
    k_extend = torch.randn(n_ext, h_kv, head_dim, dtype=dtype, device=device)
    v_extend = torch.randn(n_ext, h_kv, v_head_dim, dtype=dtype, device=device)
    qo_indptr = torch.arange(0, n_ext + 1, l_ext, dtype=torch.int32, device=device)

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        l_ext,
    )


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestVerifySplitKV(CustomTestCase):
    def _run_parity(
        self,
        prefix_lens,
        l_ext=4,
        h_q=16,
        h_kv=2,
        head_dim=256,
        v_head_dim=256,
        k_scale=1.0,
        v_scale=1.0,
        dtype=torch.bfloat16,
        is_causal=True,
    ):
        device = "cuda"
        q, k, v, kb, vb, qo, kvp, kvi, mle = _build_verify_inputs(
            prefix_lens, l_ext, h_q, h_kv, head_dim, v_head_dim, dtype, device
        )
        sm_scale = 1.0 / (head_dim**0.5)

        # Reference: extend_attention_fwd with custom_mask=None. is_causal=True is
        # the topk=1 target-verify case; is_causal=False is the bidirectional
        # draft block (ENCODER_ONLY draft layers). Same KV scales in both.
        o_ref = torch.empty(q.shape[0], h_q, v_head_dim, dtype=dtype, device=device)
        extend_attention_fwd(
            q,
            k,
            v,
            o_ref,
            kb,
            vb,
            qo,
            kvp,
            kvi,
            None,
            is_causal,
            None,
            mle,
            k_scale,
            v_scale,
            sm_scale=sm_scale,
        )

        o_split = torch.empty_like(o_ref)
        ran = verify_splitkv_fwd(
            q,
            k,
            v,
            o_split,
            kb,
            vb,
            qo,
            kvp,
            kvi,
            None,
            is_causal,
            None,
            mle,
            k_scale,
            v_scale,
            sm_scale=sm_scale,
        )
        self.assertTrue(
            ran,
            f"verify_splitkv_fwd must handle the topk=1 case (is_causal={is_causal})",
        )
        torch.testing.assert_close(o_split, o_ref, atol=ATOL, rtol=RTOL)

    def test_numerics_head_dim_256(self):
        # head_dim=256 is the validated Qwen3 value (the tuned block config).
        for prefix_lens in ([512, 512, 512], [768, 1536, 3072], [4096, 8192]):
            with self.subTest(prefix_lens=prefix_lens):
                self._run_parity(prefix_lens)

    def test_numerics_head_dim_128(self):
        # A head_dim without a tuned block entry must still be correct (default).
        self._run_parity([1024, 2048], head_dim=128, v_head_dim=128)

    def test_numerics_gqa_ratios(self):
        # Sweep GQA group sizes incl. MQA (h_kv=1); the kv_group_num arithmetic
        # in the kernel must be correct across ratios.
        for h_q, h_kv in ((16, 1), (8, 1), (8, 2), (8, 4), (8, 8)):
            with self.subTest(h_q=h_q, h_kv=h_kv):
                self._run_parity([1024, 2048], h_q=h_q, h_kv=h_kv)

    def test_numerics_extend_len_variants(self):
        for l_ext in (1, 2, 4, 8):
            with self.subTest(l_ext=l_ext):
                self._run_parity([1024, 1024], l_ext=l_ext)

    def test_numerics_with_kv_scales(self):
        # Exercise the k_scale/v_scale dequant-multiplier path (same multipliers
        # the fp8 KV-cache path applies); both kernels must apply them identically.
        self._run_parity([1024, 2048], k_scale=0.5, v_scale=0.25)

    def test_numerics_non_causal_draft_block(self):
        # ENCODER_ONLY draft layers are bidirectional inside the draft block:
        # every valid draft token sees every other, and the whole prefix stays
        # visible. That is exactly extend_attention_fwd with is_causal=False and
        # custom_mask=None, so the same parity check applies.
        for l_ext in (2, 4, 8):
            with self.subTest(l_ext=l_ext):
                self._run_parity([1024, 2048], l_ext=l_ext, is_causal=False)

    def test_causal_and_non_causal_specializations_differ(self):
        # Guard against the two stage2 specializations aliasing in _VK_CACHE:
        # with l_ext > 1 the bidirectional result must not equal the causal one.
        device = "cuda"
        h_q, h_kv, head_dim, l_ext = 8, 1, 128, 4
        q, k, v, kb, vb, qo, kvp, kvi, mle = _build_verify_inputs(
            [512, 512], l_ext, h_q, h_kv, head_dim, head_dim, torch.bfloat16, device
        )
        sm_scale = 1.0 / (head_dim**0.5)
        outs = []
        for is_causal in (True, False):
            o = torch.empty(q.shape[0], h_q, head_dim, dtype=q.dtype, device=device)
            self.assertTrue(
                verify_splitkv_fwd(
                    q,
                    k,
                    v,
                    o,
                    kb,
                    vb,
                    qo,
                    kvp,
                    kvi,
                    None,
                    is_causal,
                    None,
                    mle,
                    1.0,
                    1.0,
                    sm_scale=sm_scale,
                )
            )
            outs.append(o)
        self.assertFalse(torch.allclose(outs[0], outs[1], atol=ATOL, rtol=RTOL))

    # --- fallback: can_handle() must reject what the kernel can't serve --------
    # (topk>1 is gated off in the backend, not here -- can_handle never inspects
    #  the tree custom_mask; see verify_splitkv.can_handle docstring.)
    def _inputs(self):
        return _build_verify_inputs(
            [512, 512], 4, 16, 2, 256, 256, torch.bfloat16, "cuda"
        )

    def test_accepts_non_causal_without_custom_mask(self):
        # Bidirectional draft blocks are served by the INTRA_BLOCK_CAUSAL=False
        # stage2 specialization, so is_causal=False alone is no longer a reason
        # to fall back.
        q, k, v, kb, vb, qo, kvp, kvi, mle = self._inputs()
        self.assertTrue(
            can_handle(q, k, v, kb, vb, qo, kvp, kvi, None, False, None, mle)
        )

    def test_fallback_non_causal_with_custom_mask(self):
        # An explicit mask on a non-causal call is a shape neither stage2
        # specialization reproduces (the kernel never reads custom_mask), so it
        # must fall back. The causal + custom_mask case is unchanged: it is
        # gated off in the backend by topk == 1, not here -- see the can_handle
        # docstring.
        q, k, v, kb, vb, qo, kvp, kvi, mle = self._inputs()
        n_ext = q.shape[0]
        custom_mask = torch.ones(n_ext * n_ext, dtype=torch.bool, device="cuda")
        self.assertFalse(
            can_handle(q, k, v, kb, vb, qo, kvp, kvi, custom_mask, False, None, mle)
        )

    def test_fallback_exotic_features(self):
        q, k, v, kb, vb, qo, kvp, kvi, mle = self._inputs()
        self.assertFalse(
            can_handle(
                q,
                k,
                v,
                kb,
                vb,
                qo,
                kvp,
                kvi,
                None,
                True,
                None,
                mle,
                sinks=torch.zeros(16, device="cuda"),
            )
        )
        self.assertFalse(
            can_handle(
                q,
                k,
                v,
                kb,
                vb,
                qo,
                kvp,
                kvi,
                None,
                True,
                None,
                mle,
                sliding_window_size=128,
            )
        )
        self.assertFalse(
            can_handle(
                q, k, v, kb, vb, qo, kvp, kvi, None, True, None, mle, logit_cap=30.0
            )
        )

    def test_fallback_ragged_extend(self):
        # q rows (bs*l_ext) inconsistent with the claimed max_len_extend -> reject.
        q, k, v, kb, vb, qo, kvp, kvi, mle = self._inputs()
        self.assertFalse(
            can_handle(q, k, v, kb, vb, qo, kvp, kvi, None, True, None, mle + 1)
        )

    def test_fallback_mla_head_dim_mismatch(self):
        # MLA (DeepSeek) has head_dim != v_head_dim (576 vs 512): the shared
        # latent KV / absorbed layout is not something the split-KV verify
        # kernel is built for -- it GPU-faults on that shape. can_handle() must
        # reject it so the backend falls back to extend_attention_fwd.
        q, k, v, kb, vb, qo, kvp, kvi, mle = _build_verify_inputs(
            [512, 512], 4, 16, 1, 576, 512, torch.bfloat16, "cuda"
        )
        self.assertEqual(q.shape[2], 576)
        self.assertEqual(v.shape[2], 512)
        self.assertFalse(
            can_handle(q, k, v, kb, vb, qo, kvp, kvi, None, True, None, mle)
        )


class TestChooseNSplits(CustomTestCase):
    """The CU-aware split cap. Host-side arithmetic only -- no GPU needed for
    the invariance checks; the raise itself reads the device CU count."""

    def test_legacy_call_is_unchanged(self):
        # No base_programs -> the flat, previously tuned ladder, bit-identical.
        for avg_seqlen, expected in ((1024, 4), (4095, 4), (4096, 8), (8191, 8)):
            with self.subTest(avg_seqlen=avg_seqlen):
                self.assertEqual(choose_n_splits(avg_seqlen), expected)
        self.assertEqual(choose_n_splits(1 << 20), MAX_N_SPLITS)

    @unittest.skipIf(not torch.cuda.is_available(), "GPU required")
    def test_full_base_grid_keeps_tuned_value(self):
        # A base grid that already saturates the device must not be boosted:
        # the boost factor is max_splits // MAX_N_SPLITS == 1 there.
        n_cu = torch.cuda.get_device_properties(0).multi_processor_count
        big = 8 * n_cu
        for avg_seqlen in (1024, 4096, 1 << 20):
            with self.subTest(avg_seqlen=avg_seqlen):
                self.assertEqual(
                    choose_n_splits(avg_seqlen, base_programs=big),
                    choose_n_splits(avg_seqlen),
                )

    @unittest.skipIf(not torch.cuda.is_available(), "GPU required")
    def test_underfilled_base_grid_raises_cap(self):
        # bs=1, h_q=8 is a per-TP-rank draft shard: 8 workgroups on ~132-304 CUs.
        n = choose_n_splits(1 << 20, base_programs=8)
        self.assertGreater(n, MAX_N_SPLITS)
        self.assertLessEqual(n, HARD_MAX_N_SPLITS)
        self.assertEqual(n & (n - 1), 0, "n_splits must stay a power of two")
        # Monotone in the shortfall: a smaller base grid never asks for less.
        self.assertGreaterEqual(n, choose_n_splits(1 << 20, base_programs=64))


if __name__ == "__main__":
    unittest.main()
