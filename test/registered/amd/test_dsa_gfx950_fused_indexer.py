"""Numerical checks for kernels 1 and 4 against torch.mm and the standard
top-k path respectively, and for the block-FP8 node 1 (the layers' own GEMMs
feeding kernel 2 by row stride) against the packed layout kernel 1 writes, plus
the weight-dtype gate that picks between the bf16 and block-FP8 paths.

Kernel 3 runs here only as the fixture that produces kernel 4's inputs; nothing
asserts on its logits. Kernel 2 runs in the block-FP8 test, where its q, head
gate and cache bytes are compared between the two input layouts only. Its cache
is never compared against indexer_k_quant_and_cache, so a format mismatch with
the standard writer would not be caught here."""

import types
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
        for rows in (1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 33, 48):
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

    # -- block-FP8 wq_b / wk ---------------------------------------------------
    def test_block_fp8_projections_match_packed_layout(self):
        """Block-FP8 wq_b / wk: node 1 is the layers' own GEMMs and node 2 reads
        their outputs by row stride. Feeding node 2 the same outputs packed the
        way the bf16 GEMV writes them must give identical bytes and top-k."""
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import fused_decode as fd

        max_cols = 16384
        fd.prealloc_workspace(device=self.dev, max_cols=max_cols, fp8_dtype=FP8)
        g = torch.Generator(device=self.dev).manual_seed(0)
        for rows in (1, 6, 24, 48):
            with self.subTest(rows=rows):
                ix = _fp8_indexer_stub(self.dev, g)
                state = fd.Gfx950FusedIndexer(ix)
                self.assertTrue(
                    state.ensure_workspace(
                        device=self.dev, max_cols=max_cols, fp8_dtype=FP8
                    )
                )
                inp = _decode_inputs(self.dev, g, rows, max_cols)
                x_bf16 = torch.randn(
                    rows, HIDDEN_SIZE, device=self.dev, generator=g
                ).to(torch.bfloat16)
                # The gfx95 fused RMSNorm + quant hands the indexer this tuple.
                x = (x_bf16.to(FP8), None, x_bf16)
                q_lora = torch.randn(
                    rows, Q_LORA_RANK, device=self.dev, generator=g
                ).to(torch.bfloat16)

                kv_got = inp["kv"].clone()
                got = state.run(
                    x=x,
                    q_lora=q_lora,
                    positions=inp["pos"],
                    out_cache_loc=inp["loc"],
                    kv_cache=kv_got.view(FP8).view(-1, PAGE_SIZE, 132),
                    kv_cache_read=kv_got.view(FP8),
                    seqlens_int32=inp["seq"],
                    row_ends_int32=inp["seq"],
                    page_table_64=inp["pt"],
                    rows=rows,
                )
                q_fp8_got = state.workspace.q_fp8[:rows].clone()
                gate_got = state.workspace.head_gate[:rows].clone()

                kv_ref = inp["kv"].clone()
                ref, q_fp8_ref, gate_ref = self._packed_reference(
                    state, ix, x, q_lora, inp, kv_ref, rows
                )
                torch.cuda.synchronize()

                self.assertTrue(torch.equal(q_fp8_got.view(torch.uint8), q_fp8_ref))
                self.assertTrue(torch.equal(gate_got, gate_ref))
                self.assertTrue(torch.equal(kv_got, kv_ref))
                self.assertTrue(torch.equal(got.sort(-1).values, ref.sort(-1).values))

    def _packed_reference(self, state, ix, x, q_lora, inp, kv, rows):
        """Nodes 2-4 on the layers' outputs copied into a packed [k ; weights]."""
        _, _, norm_w, norm_b, cos2d, sin2d = state._constants()
        ws = state.workspace
        q_proj, _ = ix.wq_b(q_lora)
        k, _ = ix.wk(x)
        kw = torch.cat([k, ix._weights_proj_bf16_in_fp32_out(x)], dim=1)
        self.qk.indexer_qk_rope_hadamard_quant_and_cache(
            q_proj.view(rows, N_HEADS, HEAD_DIM),
            ws.q_fp8[:rows],
            kw[:, HEAD_DIM:],
            ws.head_gate[:rows],
            kw[:, :HEAD_DIM],
            kv.view(FP8).view(-1, PAGE_SIZE, 132),
            inp["loc"],
            norm_w,
            norm_b,
            inp["pos"],
            cos2d,
            sin2d,
            float(ix.k_norm.variance_epsilon),
            float(state.weights_scale),
        )
        q_fp8 = ws.q_fp8[:rows].view(torch.uint8).clone()
        gate = ws.head_gate[:rows].clone()
        cols = inp["pt"].shape[1] * PAGE_SIZE
        out = torch.empty(rows, TOPK, dtype=torch.int32, device=self.dev)
        self.logits_mod.logits_hist(
            ws.q_fp8[:rows],
            kv.view(FP8),
            ws.head_gate[:rows],
            inp["seq"],
            inp["pt"],
            ws.logits[:rows, :cols],
            ws.ghist[:rows],
            self.loader.LOGITS_BLOCKS_PER_ROW,
        )
        self.topk_mod.topk_transform(
            ws.logits[:rows, :cols],
            inp["seq"],
            inp["pt"],
            out,
            ws.ghist[:rows],
            ws.cursor[:rows],
            ws.cand_idx[:rows],
            ws.cand_val[:rows],
            self.loader.TOPK_G,
            PAGE_SIZE,
        )
        return out, q_fp8, gate


class TestGfx950FusedIndexerWeightGate(unittest.TestCase):
    def test_weight_dtypes(self):
        from sglang.srt.layers.attention.dsa.dsa_indexer import (
            _gfx950_fused_weights_ok,
            fp8_dtype,
        )

        bf16 = types.SimpleNamespace(dtype=torch.bfloat16)
        fp8 = types.SimpleNamespace(dtype=fp8_dtype)
        fp32 = types.SimpleNamespace(dtype=torch.float32)
        # (wk, weights_proj, wq_b) -> accepted
        cases = {
            "bf16": ((bf16, bf16, bf16), True),
            "block_fp8": ((fp8, bf16, fp8), True),
            "fp8_wq_b_bf16_wk": ((bf16, bf16, fp8), False),
            "bf16_wq_b_fp8_wk": ((fp8, bf16, bf16), False),
            "fp8_weights_proj": ((fp8, fp8, fp8), False),
            "fp32_weights_proj": ((bf16, fp32, bf16), False),
            "fp32_wk_wq_b": ((fp32, bf16, fp32), False),
            "missing_wk": ((None, bf16, bf16), False),
            "missing_weights_proj": ((bf16, None, bf16), False),
            "missing_wq_b": ((fp8, bf16, None), False),
        }
        for name, (weights, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(_gfx950_fused_weights_ok(*weights), expected)


class _Fp8Linear:
    """A block-FP8 ReplicatedLinear as the fused path sees it: an fp8 `weight`
    (the dispatch keys on its dtype), (out, bias) returned, and the
    (fp8, scale, bf16) activation tuple accepted."""

    def __init__(self, w: torch.Tensor):
        self.w = w
        self.weight = w.to(FP8)

    def __call__(self, x):
        x = x[2] if isinstance(x, tuple) else x
        return x @ self.w.t(), None


def _fp8_indexer_stub(dev, g, max_pos: int = 65536):
    def rand(*shape):
        return (torch.randn(*shape, device=dev, generator=g) * 0.02).to(torch.bfloat16)

    wp = rand(N_HEADS, HIDDEN_SIZE)
    ang = torch.arange(max_pos, device=dev, dtype=torch.float32)[:, None] * (
        10000.0 ** (-torch.arange(0, 64, 2, device=dev, dtype=torch.float32) / 64)
    )
    ix = types.SimpleNamespace(
        n_heads=N_HEADS,
        softmax_scale=HEAD_DIM**-0.5,
        wq_b=_Fp8Linear(rand(N_HEADS * HEAD_DIM, Q_LORA_RANK)),
        wk=_Fp8Linear(rand(HEAD_DIM, HIDDEN_SIZE)),
        weights_proj=types.SimpleNamespace(weight=wp),
        k_norm=types.SimpleNamespace(
            weight=1 + rand(HEAD_DIM), bias=rand(HEAD_DIM), variance_epsilon=1e-6
        ),
        rotary_emb=types.SimpleNamespace(
            cos_sin_cache=torch.cat([ang.cos(), ang.sin()], -1).to(torch.bfloat16)
        ),
    )
    ix._weights_proj_bf16_in_fp32_out = lambda x: (
        (x[2] if isinstance(x, tuple) else x) @ wp.t()
    )
    return ix


def _decode_inputs(dev, g, rows: int, max_cols: int):
    """One decode row per request over a shuffled paged cache with random fp8 K."""
    blocks = max_cols // PAGE_SIZE
    n_pages = rows * blocks + 8
    pt = torch.randperm(n_pages, device=dev, generator=g)[: rows * blocks]
    pt = pt.view(rows, blocks).int().contiguous()
    seq = torch.randint(TOPK + 1, max_cols + 1, (rows,), device=dev, generator=g).int()
    pos = (seq - 1).long()
    loc = pt.gather(1, (pos // PAGE_SIZE)[:, None]).squeeze(1).long() * PAGE_SIZE
    loc += pos % PAGE_SIZE
    k = torch.randn(n_pages, PAGE_SIZE * HEAD_DIM, device=dev, generator=g)
    scale = torch.full((n_pages, PAGE_SIZE), 2.0**-6, device=dev)
    kv = torch.cat([k.to(FP8).view(torch.uint8), scale.view(torch.uint8)], 1)
    return dict(pt=pt, seq=seq, pos=pos, loc=loc.contiguous(), kv=kv.contiguous())


if __name__ == "__main__":
    unittest.main()
