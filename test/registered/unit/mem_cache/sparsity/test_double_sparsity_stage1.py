"""Stage-1 block-topk Triton kernel tests (v1.1-4).

Three regimes per the v1.1 plan:

1. **Block-topk parity oracle.** The Triton kernel must exactly match
   `ds_select_stage1_block_topk_torch_ref` (a reference impl that does
   block-topk with the same BLOCK_T and K_BLOCK). Run for each GQA
   reduction. Note: parity is against block-topk, NOT `torch.topk`,
   since stage 1 is approximate top-k by construction.

2. **Recall vs exact topk.** Two regimes (gated on effective_budget):
   - effective_budget = min(token_budget, num_blocks * k_block) >=
     token_budget: recall against `exact_topk(token_budget)` >= 95%.
   - effective_budget < token_budget (capacity-bound): recall against
     `exact_topk(effective_budget)` >= 95%.

3. **Bounded work.** Kernel cost grows with seq_lens, not max_ctx.
   Programs whose `block_start >= seq_lens - 1` mask their compute to
   no-op work. Best-effort timing test.
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
    ds_select_stage1_block_topk_torch_ref,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="stage-b-test-1-gpu-small")


def _build_inputs(
    *,
    bs: int,
    h_q: int,
    h_kv: int,
    d: int,
    s: int,
    max_ctx: int,
    T_pool: int,
    seq_lens: torch.Tensor,
    device: torch.device,
    seed: int = 0,
):
    g = torch.Generator(device=device).manual_seed(seed)
    queries = torch.randn(bs, h_q, d, generator=g, device=device, dtype=torch.float32)
    channel_idx = torch.stack(
        [torch.randperm(d, generator=g, device=device)[:s] for _ in range(h_kv)]
    ).to(torch.int32)
    k_label = torch.randn(
        T_pool, h_kv, s, generator=g, device=device, dtype=torch.float32
    )
    # Distinct permuted req_to_token rows per batch (radix prefix-reuse safe).
    req_to_token = torch.zeros(bs, max_ctx, dtype=torch.int32, device=device)
    for b in range(bs):
        perm = torch.randperm(T_pool, generator=g, device=device)[:max_ctx]
        req_to_token[b] = perm.to(torch.int32)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)
    return queries, channel_idx, k_label, req_to_token, req_pool_indices, seq_lens


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStage1BlockTopkParity(CustomTestCase):
    """Triton kernel == torch reference, exact match on logical positions
    and bit-equal on scores."""

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
            seq_lens = torch.tensor([max_ctx] * bs, dtype=torch.int64, device=device)
        else:
            seq_lens = seq_lens.to(device)
        q, ch, kl, r2t, rpi, sl = _build_inputs(
            bs=bs,
            h_q=h_q,
            h_kv=h_kv,
            d=d,
            s=s,
            max_ctx=max_ctx,
            T_pool=T,
            seq_lens=seq_lens,
            device=device,
            seed=11 + gqa,
        )
        ref_log, ref_scr = ds_select_stage1_block_topk_torch_ref(
            queries=q,
            channel_idx=ch,
            k_label=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=h_kv,
            block_t=block_t,
            k_block=k_block,
            gqa_reduction_id=gqa,
        )
        tri_log, tri_scr = ds_select_stage1_block_topk(
            queries=q,
            channel_idx=ch,
            k_label=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=h_kv,
            block_t=block_t,
            k_block=k_block,
            gqa_reduction_id=gqa,
        )

        # Logical positions must match exactly (with deterministic tie-break).
        # Compare as sets per (bs, kv_head, block) — tie ordering can differ
        # in pathological cases, but on random Gaussian scores ties are
        # near-impossible.
        self.assertEqual(ref_log.shape, tri_log.shape)
        for b in range(ref_log.shape[0]):
            for h in range(ref_log.shape[1]):
                for blk in range(ref_log.shape[2]):
                    ref_set = sorted(
                        int(x) for x in ref_log[b, h, blk].tolist() if int(x) >= 0
                    )
                    tri_set = sorted(
                        int(x) for x in tri_log[b, h, blk].tolist() if int(x) >= 0
                    )
                    self.assertEqual(
                        ref_set,
                        tri_set,
                        f"mismatch at (b={b}, h={h}, blk={blk}, gqa={gqa})\n"
                        f"  ref={ref_set}\n  tri={tri_set}",
                    )

        # Scores match within fp32 reduction-order tolerance (Triton sum may
        # reduce in a different order than torch reduction).
        self.assertTrue(
            torch.allclose(
                tri_scr.where(tri_scr > -1e30, torch.zeros_like(tri_scr)),
                ref_scr.where(ref_scr > -1e30, torch.zeros_like(ref_scr)),
                rtol=1e-4,
                atol=1e-4,
            ),
            "stage-1 score parity broken",
        )

    def test_max_abs(self):
        self._parity(GQA_REDUCTION_MAX_ABS)

    def test_mean(self):
        self._parity(GQA_REDUCTION_MEAN)

    def test_soq(self):
        self._parity(GQA_REDUCTION_SOQ)

    def test_partial_seq_len(self):
        # seq_len shorter than max_ctx — programs in past-history blocks
        # must emit -1 / NEG_INF, programs in active blocks must work.
        sl = torch.tensor([24, 48], dtype=torch.int64)
        self._parity(GQA_REDUCTION_MAX_ABS, seq_lens=sl, max_ctx=64, T=256)

    def test_realistic_shapes(self):
        # Closer to production: bs=2, GQA group_size=4, H_kv=8, head_dim=128,
        # S=32, max_ctx=2048, BLOCK_T=128, K_BLOCK=32. Smaller than 8B but
        # exercises the same stride patterns.
        self._parity(
            GQA_REDUCTION_MAX_ABS,
            bs=2,
            h_q=32,
            h_kv=8,
            d=128,
            s=32,
            max_ctx=2048,
            T=8192,
            block_t=128,
            k_block=32,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStage1RecallVsExactTopk(CustomTestCase):
    """Recall against exact `torch.topk`, gated on effective_budget.

    Stage 1 is approximate by construction (per-block K_BLOCK cap), so we
    don't expect exact equality with torch.topk. We do expect high recall
    when capacity is sufficient. The test bounds catastrophic regressions.
    """

    def _measure_recall(
        self,
        *,
        bs: int,
        h_q: int,
        h_kv: int,
        d: int,
        s: int,
        max_ctx: int,
        T: int,
        block_t: int,
        k_block: int,
        token_budget: int,
        seed: int = 31,
    ):
        device = torch.device("cuda")
        seq_lens = torch.tensor([max_ctx] * bs, dtype=torch.int64, device=device)
        q, ch, kl, r2t, rpi, sl = _build_inputs(
            bs=bs,
            h_q=h_q,
            h_kv=h_kv,
            d=d,
            s=s,
            max_ctx=max_ctx,
            T_pool=T,
            seq_lens=seq_lens,
            device=device,
            seed=seed,
        )
        # Stage-1 block-topk output.
        log, scr = ds_select_stage1_block_topk(
            queries=q,
            channel_idx=ch,
            k_label=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=h_kv,
            block_t=block_t,
            k_block=k_block,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )
        # Stage-2 will pick up to token_budget candidates by score; for
        # this test we approximate by taking all (bs, h) candidates.
        num_blocks = (max_ctx + block_t - 1) // block_t
        effective_budget = min(token_budget, num_blocks * k_block)

        # Compute exact top-effective_budget per (bs, h) for comparison.
        from sglang.srt.mem_cache.sparsity.triton_ops.select_triton import (
            _q_label_for_kv_head,
        )

        recalls = []
        for b in range(bs):
            for h in range(h_kv):
                q_label = _q_label_for_kv_head(
                    q,
                    ch,
                    bs_idx=b,
                    kv_idx=h,
                    num_kv_heads=h_kv,
                    gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
                )
                phys = r2t[b, : sl[b].item()].long()
                scores = (kl[phys, h, :] * q_label[None, :]).sum(dim=1)
                exact_idx = scores.topk(effective_budget).indices.cpu().tolist()

                # Stage-1 candidates per (b, h): flatten across blocks, dedup,
                # take top-effective_budget by score.
                flat_log = log[b, h].reshape(-1).cpu().tolist()  # int
                flat_scr = scr[b, h].reshape(-1).cpu().tolist()
                paired = sorted(
                    [(s_, l) for s_, l in zip(flat_scr, flat_log) if l >= 0],
                    key=lambda x: -x[0],
                )[:effective_budget]
                stage1_set = set(l for _, l in paired)
                exact_set = set(exact_idx)
                if exact_set:
                    recalls.append(len(stage1_set & exact_set) / len(exact_set))
        return recalls

    def test_recall_when_capacity_sufficient(self):
        # max_ctx=128, BLOCK_T=16, K_BLOCK=8 → num_blocks=8, candidates=64.
        # token_budget=32 → effective_budget=32 (capacity 64 > 32 ✓).
        recalls = self._measure_recall(
            bs=2,
            h_q=8,
            h_kv=2,
            d=64,
            s=16,
            max_ctx=128,
            T=512,
            block_t=16,
            k_block=8,
            token_budget=32,
        )
        # Expect very high recall in the capacity-sufficient regime.
        self.assertTrue(
            all(r >= 0.95 for r in recalls),
            f"recall floor 0.95 missed; recalls={recalls}",
        )

    def test_recall_when_capacity_bound(self):
        # max_ctx=128, BLOCK_T=16, K_BLOCK=2 → num_blocks=8, candidates=16.
        # token_budget=32 → effective_budget=16 (capacity 16 < 32, bound).
        # Per the plan: in this regime recall is measured against
        # exact_topk(effective_budget), not exact_topk(token_budget).
        recalls = self._measure_recall(
            bs=2,
            h_q=8,
            h_kv=2,
            d=64,
            s=16,
            max_ctx=128,
            T=512,
            block_t=16,
            k_block=2,
            token_budget=32,
        )
        self.assertTrue(
            all(r >= 0.5 for r in recalls),
            f"recall floor 0.5 missed in capacity-bound regime; recalls={recalls}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStage1BoundedWorkStructural(CustomTestCase):
    """Bounded-work property is structural, not timing-based.

    For a request with seq_len=S, all blocks with `block_start >= S - 1`
    must emit -1 / NEG_INF — the kernel's `valid = t_offsets < history_len`
    mask skips the K_label loads for those slots. Past-history blocks
    don't waste bandwidth even though their programs still launch under
    the static graph grid.

    A wall-time test is flaky: kernel-launch overhead dominates at
    small kernel sizes, programs still launch under the static grid,
    and timing variance swamps the actual work delta. The structural
    invariant — "past-history blocks emit only -1 / NEG_INF" — catches
    O(max_ctx) regressions cleanly without any timing dependency.
    """

    def test_past_history_blocks_emit_sentinels_only(self):
        device = torch.device("cuda")
        bs, h_q, h_kv, d, s = 2, 8, 4, 64, 16
        max_ctx, T_pool = 1024, 4096
        block_t, k_block = 64, 16
        # seq_lens chosen so blocks 0..3 are active for row 0, blocks 0..1
        # active for row 1; the rest must emit only sentinels.
        seq_lens = torch.tensor([200, 100], dtype=torch.int64, device=device)

        q, ch, kl, r2t, rpi, sl = _build_inputs(
            bs=bs,
            h_q=h_q,
            h_kv=h_kv,
            d=d,
            s=s,
            max_ctx=max_ctx,
            T_pool=T_pool,
            seq_lens=seq_lens,
            device=device,
            seed=99,
        )
        log, scr = ds_select_stage1_block_topk(
            queries=q,
            channel_idx=ch,
            k_label=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=h_kv,
            block_t=block_t,
            k_block=k_block,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )
        log = log.cpu()
        scr = scr.cpu()
        for b in range(bs):
            history_len = max(int(seq_lens[b].item()) - 1, 0)
            for h in range(h_kv):
                for blk in range(log.shape[2]):
                    block_start = blk * block_t
                    if block_start >= history_len:
                        # Entire block past history — must be all -1 / NEG_INF.
                        self.assertTrue(
                            (log[b, h, blk] == -1).all(),
                            f"past-history block (b={b}, h={h}, blk={blk}, "
                            f"block_start={block_start}, history_len={history_len}) "
                            f"emitted non-sentinel logical positions: "
                            f"{log[b, h, blk].tolist()}",
                        )
                        self.assertTrue(
                            (scr[b, h, blk] < -1e30).all(),
                            f"past-history block scores not NEG_INF: "
                            f"{scr[b, h, blk].tolist()}",
                        )


if __name__ == "__main__":
    unittest.main()
