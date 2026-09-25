"""Numerical checks for kernels 1 and 4 against torch.mm and the standard
top-k path respectively.

Kernel 3 runs here only as the fixture that produces kernel 4's inputs; nothing
asserts on its logits. Kernel 2 is NOT covered -- it is the kernel that *writes*
the preshuffled FP8 cache, so testing it means comparing that cache byte for
byte against indexer_k_quant_and_cache, which this file does not do."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

HEAD_DIM = 128
N_HEADS = 32
PAGE_SIZE = 64
TOPK = 2048
Q_LORA_RANK = 2048
HIDDEN_SIZE = 6144
KW_ROWS = HEAD_DIM + N_HEADS  # [wk ; weights_proj] merged output width
FP8 = torch.float8_e4m3fn


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    if arch != "gfx950":
        return f"kernels are gfx950-only, got {arch}"
    try:
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader
    except ImportError as exc:  # pragma: no cover - build-environment dependent
        return f"extension package unavailable: {exc}"
    if loader.modules_or_none() is None:
        return "gfx950 fused indexer modules failed to build"
    return None


SKIP = _skip_reason()


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float()
    err = got.float() - ref
    power = ref.pow(2).mean()
    noise = err.pow(2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10(power / noise))


# Lives under the amd vendor tree rather than test/registered/kernel/ so it can
# name a gfx950 suite: the taxonomy check requires a *-kernel-* suite for the
# kernel kind, and every AMD one of those dispatches to
# linux-{mi300,mi325}-1gpu-sglang -- gfx942, where _skip_reason() below would
# skip the whole class. stage-b runs on linux-mi35x-gpu-1, which is the hardware
# these kernels are written for.
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipIf(SKIP is not None, SKIP or "")
class TestGfx950FusedIndexerKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader

        cls.loader = loader
        cls.gemv, cls.qk, cls.logits_mod, cls.topk_mod = loader.modules()
        cls.dev = torch.device("cuda", 0)
        torch.manual_seed(0)

    # -- node 1 --------------------------------------------------------------
    def test_dual_gemv_matches_torch_mm(self):
        """One launch, two independent projections; torch.mm on the same operands is
        the reference."""
        for rows in (1, 2, 4, 8):
            with self.subTest(rows=rows):
                q_lora = torch.randn(
                    rows, Q_LORA_RANK, dtype=torch.bfloat16, device=self.dev
                )
                w_q_b = torch.randn(
                    N_HEADS * HEAD_DIM,
                    Q_LORA_RANK,
                    dtype=torch.bfloat16,
                    device=self.dev,
                )
                x = torch.randn(
                    rows, HIDDEN_SIZE, dtype=torch.bfloat16, device=self.dev
                )
                w_kw = torch.randn(
                    KW_ROWS, HIDDEN_SIZE, dtype=torch.bfloat16, device=self.dev
                )
                q_proj = torch.empty(
                    rows, N_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=self.dev
                )
                kw = torch.empty(rows, KW_ROWS, dtype=torch.bfloat16, device=self.dev)

                self.gemv.dual_gemv_bf16(q_lora, w_q_b, q_proj, x, w_kw, kw)
                torch.cuda.synchronize()

                self.assertGreater(
                    _snr_db(q_lora.float() @ w_q_b.float().t(), q_proj), 35.0
                )
                self.assertGreater(_snr_db(x.float() @ w_kw.float().t(), kw), 35.0)

    # -- node 4 --------------------------------------------------------------
    def _fused_logits(self, rows: int, ctx: int, pt_width: int = 0):
        """Logits and the paired histogram from node 3, for the node-4 tests.  The cache
        content is synthetic; only node 4's selection is asserted, which is sound
        because both sides of that comparison read the same logits."""
        pages_per_row = ctx // PAGE_SIZE
        total_pages = rows * pages_per_row
        q_fp8 = torch.randn(
            rows, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=self.dev
        ).to(FP8)
        kv = torch.randn(
            total_pages, PAGE_SIZE * 132, dtype=torch.bfloat16, device=self.dev
        ).to(FP8)
        gate = torch.rand(rows, N_HEADS, dtype=torch.float32, device=self.dev)
        seqlens = torch.full((rows,), ctx, dtype=torch.int32, device=self.dev)
        page_table_64 = (
            torch.arange(total_pages, dtype=torch.int32, device=self.dev)
            .reshape(rows, pages_per_row)
            .contiguous()
        )
        if pt_width > pages_per_row:
            # As in a captured graph: the table is as wide as the graph, the row
            # is as long as seqlens says.  Columns past the row are never read.
            wide = torch.zeros(rows, pt_width, dtype=torch.int32, device=self.dev)
            wide[:, :pages_per_row] = page_table_64
            page_table_64 = wide.contiguous()
        logits = torch.zeros(rows, ctx, dtype=torch.float32, device=self.dev)
        ghist = torch.zeros(
            rows, self.topk_mod.hist_stride(), dtype=torch.int32, device=self.dev
        )
        self.logits_mod.logits_hist(
            q_fp8,
            kv,
            gate,
            seqlens,
            page_table_64,
            logits,
            ghist,
            self.loader.LOGITS_BLOCKS_PER_ROW,
        )
        torch.cuda.synchronize()
        return logits, ghist, (q_fp8, kv, gate, seqlens, page_table_64, ctx)

    def _assert_matches_standard_path(self, rows, ctx, pt_width=0):
        """Node 4 against the path it replaces, on identical logits.

        The reference is sglang's own v2 paged transform -- the one
        ``_topk_transform_v2_paged`` routes to -- rather than
        ``aiter.dsa_topk_transform``: both emit physical page_size=1 slots from
        the same compact table, but aiter only grew that op recently, so a
        container with an older aiter skipped this check instead of running it.
        Compared as sets: neither side fixes the order.
        """
        from sglang.kernels.ops.attention.dsv4.topk import (
            plan_topk_v2,
            topk_transform_paged_v2,
        )

        logits, ghist, (_, _, _, seqlens, page_table_64, _) = self._fused_logits(
            rows, ctx, pt_width=pt_width
        )
        if pt_width:
            self.assertEqual(page_table_64.shape[1], pt_width)

        cap = max(TOPK, ctx)
        out = torch.empty(rows, TOPK, dtype=torch.int32, device=self.dev)
        self.topk_mod.topk_transform(
            logits,
            seqlens,
            page_table_64,
            out,
            ghist,
            torch.zeros(rows, 32, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.float32, device=self.dev),
            self.loader.TOPK_G,
            PAGE_SIZE,
        )
        ref = logits.new_full((rows, TOPK), -1, dtype=torch.int32)
        topk_transform_paged_v2(
            logits, seqlens, page_table_64, ref, PAGE_SIZE, plan_topk_v2(seqlens)
        )
        torch.cuda.synchronize()

        for r in range(rows):
            got_r = set(out[r].tolist()) - {-1}
            ref_r = set(ref[r].tolist()) - {-1}
            self.assertEqual(
                len(got_r), min(TOPK, ctx), f"row {r}: {len(got_r)} distinct slots"
            )
            self.assertEqual(got_r, ref_r, f"row {r}: different slots from the v2 path")

    def test_topk_transform_matches_standard_path(self):
        self._assert_matches_standard_path(rows=8, ctx=8192)

    def test_topk_transform_at_graph_width(self):
        """k_scatter is selected on the page-table width, and GLM-5.2 captures wider
        than the LDS window, so this is the form serving traffic."""
        self._assert_matches_standard_path(rows=8, ctx=8192, pt_width=16384)

    def test_topk_transform_restores_its_histogram(self):
        """The zero-in/zero-out invariant: one ghist serves all 79 layers back to back,
        so a kernel that leaves it dirty corrupts the next layer, not this test."""
        rows, ctx = 4, 4096
        logits, ghist, (_, _, _, seqlens, page_table_64, _) = self._fused_logits(
            rows, ctx
        )
        cap = max(TOPK, ctx)
        self.topk_mod.topk_transform(
            logits,
            seqlens,
            page_table_64,
            torch.empty(rows, TOPK, dtype=torch.int32, device=self.dev),
            ghist,
            torch.zeros(rows, 32, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.float32, device=self.dev),
            self.loader.TOPK_G,
            PAGE_SIZE,
        )
        torch.cuda.synchronize()
        self.assertEqual(
            int(ghist.abs().sum()), 0, "topk_transform left its histogram dirty"
        )


if __name__ == "__main__":
    unittest.main()
