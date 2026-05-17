"""Stage-2 merge Triton kernel tests (v1.1-5).

Pins:
  - Triton stage-2 == torch reference for max_abs / mean / soq stage-1
    inputs (the merge step is GQA-reduction-agnostic; we just exercise
    realistic stage-1 outputs).
  - Sentinel handling: stage-1 emits -1/NEG_INF for past-history block
    slots; stage-2 must drop those and not let them survive as picks.
  - Capacity guard: caller passing `num_candidates > merge_safe_threshold`
    raises a clear error (v1.1.x will replace with a chunked path).
"""

import unittest

import torch

from sglang.srt.mem_cache.sparsity.triton_ops.select_kernels import (
    GQA_REDUCTION_MAX_ABS,
    GQA_REDUCTION_MEAN,
    GQA_REDUCTION_SOQ,
)
from sglang.srt.mem_cache.sparsity.triton_ops.select_triton import (
    ds_select_stage1_block_topk,
    ds_select_stage2_merge,
    ds_select_stage2_merge_torch_ref,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")


def _build_stage1_outputs(
    *,
    bs,
    h_q,
    h_kv,
    d,
    s,
    max_ctx,
    T_pool,
    seq_lens,
    block_t,
    k_block,
    gqa,
    device,
    seed=21,
):
    """Run stage-1 (Triton) and return its outputs to feed stage-2."""
    g = torch.Generator(device=device).manual_seed(seed)
    queries = torch.randn(bs, h_q, d, generator=g, device=device, dtype=torch.float32)
    channel_idx = torch.stack(
        [torch.randperm(d, generator=g, device=device)[:s] for _ in range(h_kv)]
    ).to(torch.int32)
    k_label = torch.randn(
        T_pool, h_kv, s, generator=g, device=device, dtype=torch.float32
    )
    req_to_token = torch.zeros(bs, max_ctx, dtype=torch.int32, device=device)
    for b in range(bs):
        perm = torch.randperm(T_pool, generator=g, device=device)[:max_ctx]
        req_to_token[b] = perm.to(torch.int32)
    rpi = torch.arange(bs, dtype=torch.int64, device=device)
    sl = seq_lens.to(device)

    log, scr = ds_select_stage1_block_topk(
        queries=queries,
        channel_idx=channel_idx,
        k_label=k_label,
        req_to_token=req_to_token,
        req_pool_indices=rpi,
        seq_lens=sl,
        num_kv_heads=h_kv,
        block_t=block_t,
        k_block=k_block,
        gqa_reduction_id=gqa,
    )
    return log, scr


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStage2MergeParity(CustomTestCase):
    """Triton merge == torch reference, given the same stage-1 inputs."""

    def _parity(
        self,
        gqa,
        *,
        bs=2,
        h_q=8,
        h_kv=4,
        d=32,
        s=8,
        max_ctx=64,
        T=256,
        block_t=16,
        k_block=4,
        seq_lens=None,
    ):
        device = torch.device("cuda")
        if seq_lens is None:
            seq_lens = torch.tensor([max_ctx] * bs, dtype=torch.int64)
        log, scr = _build_stage1_outputs(
            bs=bs,
            h_q=h_q,
            h_kv=h_kv,
            d=d,
            s=s,
            max_ctx=max_ctx,
            T_pool=T,
            seq_lens=seq_lens,
            block_t=block_t,
            k_block=k_block,
            gqa=gqa,
            device=device,
            seed=42 + gqa,
        )
        num_blocks = log.shape[2]
        effective_budget = min(8, num_blocks * k_block)

        ref_log, ref_scr = ds_select_stage2_merge_torch_ref(
            block_topk_logical=log,
            block_topk_scores=scr,
            effective_budget=effective_budget,
        )
        tri_log, tri_scr = ds_select_stage2_merge(
            block_topk_logical=log,
            block_topk_scores=scr,
            effective_budget=effective_budget,
        )

        # Compare as ordered (since both impls use stable sort by score
        # then by logical-position-within-block; same input → same output).
        # Compare as sets per (bs, kv_head); if they differ, fail loudly.
        for b in range(bs):
            for h in range(h_kv):
                ref_set = sorted(int(x) for x in ref_log[b, h].tolist() if int(x) >= 0)
                tri_set = sorted(int(x) for x in tri_log[b, h].tolist() if int(x) >= 0)
                self.assertEqual(
                    ref_set,
                    tri_set,
                    f"merge mismatch (b={b}, h={h}, gqa={gqa}):\n"
                    f"  ref={ref_set}\n  tri={tri_set}",
                )

    def test_max_abs(self):
        self._parity(GQA_REDUCTION_MAX_ABS)

    def test_mean(self):
        self._parity(GQA_REDUCTION_MEAN)

    def test_soq(self):
        self._parity(GQA_REDUCTION_SOQ)

    def test_partial_seq_len(self):
        # seq_lens shorter than max_ctx — stage-1 emits sentinels for
        # past-history blocks; stage-2 must not surface them as picks.
        sl = torch.tensor([24, 48], dtype=torch.int64)
        self._parity(GQA_REDUCTION_MAX_ABS, seq_lens=sl, max_ctx=64, T=256)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStage2SentinelHandling(CustomTestCase):
    def test_no_sentinels_in_output_when_stage1_partial(self):
        """If stage-1 emits sentinels (-1, NEG_INF) for past-history slots,
        stage-2 must drop them, not surface them as picks."""
        device = torch.device("cuda")
        bs, h_q, h_kv, d, s = 1, 4, 2, 16, 4
        max_ctx, T = 32, 128
        block_t, k_block = 8, 2

        # seq_len so most blocks are past history.
        sl = torch.tensor([5], dtype=torch.int64)
        log, scr = _build_stage1_outputs(
            bs=bs,
            h_q=h_q,
            h_kv=h_kv,
            d=d,
            s=s,
            max_ctx=max_ctx,
            T_pool=T,
            seq_lens=sl,
            block_t=block_t,
            k_block=k_block,
            gqa=GQA_REDUCTION_MAX_ABS,
            device=device,
        )

        # Use the full candidate pool as the budget — this is the
        # "potentially admits sentinels if the merge logic is buggy"
        # regime: more budget than valid picks. Stage-2 must produce
        # sentinels (-1, NEG_INF) in the unfilled slots, never surface
        # past-history positions.
        num_blocks = log.shape[2]
        num_candidates = num_blocks * k_block  # = budget
        tri_log, tri_scr = ds_select_stage2_merge(
            block_topk_logical=log,
            block_topk_scores=scr,
            effective_budget=num_candidates,
        )
        # All non-sentinel picks must be < seq - 1 = 4 (history-only).
        # The remaining slots must be -1 / NEG_INF.
        for h in range(h_kv):
            picks = tri_log[0, h].cpu().tolist()
            scores = tri_scr[0, h].cpu().tolist()
            n_valid = 0
            for p, s in zip(picks, scores):
                if p >= 0:
                    n_valid += 1
                    self.assertLess(
                        p,
                        4,
                        f"sentinel-or-out-of-history pick survived: pos={p}",
                    )
                    self.assertGreater(
                        s,
                        -1e30,
                        f"valid pick has NEG_INF score: pos={p}",
                    )
                else:
                    self.assertLess(
                        s,
                        -1e30,
                        f"sentinel pick has non-NEG_INF score: score={s}",
                    )
            # Stage-1 found only `k_block` valid picks per (b, h) (only
            # block 0 had any history); stage-2 must not surface more than
            # that.
            self.assertLessEqual(
                n_valid,
                k_block,
                f"stage-2 surfaced {n_valid} picks for h={h} but stage-1 "
                f"only had {k_block} valid candidates",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStage2CapacityGuard(CustomTestCase):
    def test_exceeding_threshold_raises(self):
        device = torch.device("cuda")
        # Construct stage-1 outputs with num_candidates > 4096:
        # num_blocks=128, k_block=64 → 8192 candidates.
        bs, h_kv = 1, 1
        block_topk_logical = torch.full(
            (bs, h_kv, 128, 64), -1, dtype=torch.int32, device=device
        )
        block_topk_scores = torch.full(
            (bs, h_kv, 128, 64), float("-inf"), dtype=torch.float32, device=device
        )
        with self.assertRaisesRegex(RuntimeError, "merge_safe_threshold"):
            ds_select_stage2_merge(
                block_topk_logical=block_topk_logical,
                block_topk_scores=block_topk_scores,
                effective_budget=512,
            )

    def test_threshold_overridable(self):
        device = torch.device("cuda")
        bs, h_kv = 1, 1
        # 8x4 = 32 candidates; well under any reasonable threshold.
        block_topk_logical = torch.zeros(
            bs, h_kv, 8, 4, dtype=torch.int32, device=device
        )
        block_topk_scores = torch.zeros(
            bs, h_kv, 8, 4, dtype=torch.float32, device=device
        )
        # Set some valid picks
        block_topk_logical[0, 0, 0, 0] = 1
        block_topk_logical[0, 0, 0, 1] = 2
        # Lower the threshold below candidate count to trigger the guard.
        with self.assertRaisesRegex(RuntimeError, "merge_safe_threshold"):
            ds_select_stage2_merge(
                block_topk_logical=block_topk_logical,
                block_topk_scores=block_topk_scores,
                effective_budget=4,
                merge_safe_threshold=16,  # < 32 candidates
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStage2NonPow2Shapes(CustomTestCase):
    """Triton's `tl.arange` requires a power-of-2 length. Real workloads
    produce non-pow2 `num_candidates = num_blocks * k_block` (e.g. 9 * 64 =
    576 at 9K-cap context with BLOCK_T=1024) and non-pow2 `effective_budget
    = min(token_budget, num_candidates)` (576 < 1024). The kernel must pad
    via mask, not crash."""

    def _run(self, *, num_blocks, k_block, effective_budget):
        device = torch.device("cuda")
        bs, h_kv = 2, 2
        # Build stage-1-shaped inputs with deterministic, distinct logicals
        # so the merge has unambiguous ordering.
        log = torch.full(
            (bs, h_kv, num_blocks, k_block), -1, dtype=torch.int32, device=device
        )
        scr = torch.full(
            (bs, h_kv, num_blocks, k_block),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        n = num_blocks * k_block
        for b in range(bs):
            for h in range(h_kv):
                # Place logicals 0..n-1 with score = logical (highest = top).
                pos = torch.arange(n, dtype=torch.int32, device=device)
                log[b, h] = pos.view(num_blocks, k_block)
                scr[b, h] = pos.float().view(num_blocks, k_block)

        out_log, out_scr = ds_select_stage2_merge(
            block_topk_logical=log,
            block_topk_scores=scr,
            effective_budget=effective_budget,
        )
        ref_log, ref_scr = ds_select_stage2_merge_torch_ref(
            block_topk_logical=log,
            block_topk_scores=scr,
            effective_budget=effective_budget,
        )
        # Stage-2 emits top-effective_budget by score, then sorted by score
        # descending (the merge's internal ordering). For deterministic
        # input both impls must agree exactly on the surviving set.
        for b in range(bs):
            for h in range(h_kv):
                tri = sorted(int(x) for x in out_log[b, h].tolist() if int(x) >= 0)
                ref = sorted(int(x) for x in ref_log[b, h].tolist() if int(x) >= 0)
                self.assertEqual(tri, ref, f"non-pow2 mismatch at (b={b}, h={h})")

    def test_non_pow2_num_candidates(self):
        # num_blocks=9 → 9*64=576 candidates (not pow2).
        self._run(num_blocks=9, k_block=64, effective_budget=128)

    def test_non_pow2_effective_budget(self):
        # 9*64=576; effective_budget=min(token_budget=1024, 576)=576 (not pow2).
        self._run(num_blocks=9, k_block=64, effective_budget=576)


if __name__ == "__main__":
    unittest.main()
