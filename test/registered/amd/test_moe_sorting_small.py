"""Small-batch MoE sorting patch (sglang.kernels.ops.moe.moe_sorting_small) vs stock aiter sorting."""

import os
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950():
    return (
        bool(torch.version.hip)
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestMoeSortingSmall(CustomTestCase):
    # Qwen3.5: 512 routed experts + fused shared expert, top-10 + shared slot
    E, TOPK, DIM = 513, 11, 4096

    @classmethod
    def setUpClass(cls):
        import aiter.fused_moe as fm

        from sglang.kernels.ops.moe import moe_sorting_small as S

        cls.S = S
        cls.fm = fm
        orig_sort = fm._moe_sorting_impl
        orig_quant = fm.fused_dynamic_mxfp8_quant_moe_sort
        S.apply_aiter_small_moe_sort_patch()
        if fm._moe_sorting_impl is orig_sort:
            # an earlier import already patched aiter; recover the originals
            orig_sort = fm._moe_sorting_impl.__wrapped__
            orig_quant = fm.fused_dynamic_mxfp8_quant_moe_sort.__wrapped__
        cls.orig_sort = staticmethod(orig_sort)
        cls.orig_quant = staticmethod(orig_quant)
        cls.dev = torch.device("cuda", 0)

    def _routing(self, m, seed, E=E, topk=TOPK):
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.stack(
            [torch.randperm(E - 1, generator=g)[: topk - 1] for _ in range(m)]
        )
        ids = torch.cat([ids, torch.full((m, 1), E - 1)], 1)
        ids = ids.to(torch.int32).to(self.dev).contiguous()
        return ids, torch.rand(m, topk, generator=g).to(self.dev)

    def _sort(self, fn, ids, w, bs, E=E):
        return fn(ids, w, E, self.DIM, torch.bfloat16, bs, None, None, 0, True)

    def _emitted_quant(self, ids, w, x, bs, E):
        S = self.S
        tok = S._pending_quant_input.set(x)
        etok = S._emitted_quant.set(None)
        try:
            self._sort(self.fm._moe_sorting_impl, ids, w, bs, E)
            return S._emitted_quant.get()
        finally:
            S._emitted_quant.reset(etok)
            S._pending_quant_input.reset(tok)

    def _assert_same_sort(self, a, b, bs, msg):
        na, nb = int(a[3][0]), int(b[3][0])
        self.assertTrue(torch.equal(a[3].cpu(), b[3].cpu()), msg)
        self.assertTrue(
            torch.equal(a[2][: na // bs].cpu(), b[2][: nb // bs].cpu()), msg
        )
        ia, wa = a[0][:na].tolist(), a[1][:na].tolist()
        ib, wb = b[0][:nb].tolist(), b[1][:nb].tolist()
        # the order of pairs inside one expert block is not part of the contract
        for i in range(0, na, bs):
            self.assertEqual(
                sorted(zip(ia[i : i + bs], wa[i : i + bs])),
                sorted(zip(ib[i : i + bs], wb[i : i + bs])),
                f"{msg} block {i // bs}",
            )

    def test_sort_matches_aiter(self):
        # compact at E=513; distributed at MiniMax-M3's E=129 / top-4 + shared and at
        # DeepSeek-V4's E=385 / top-6 + shared, at its 112-pair cutoff
        cases = (
            [(513, 11, m) for m in (1, 2, 4, 5)]
            + [(129, 5, m) for m in (13, 25, 51)]
            + [(385, 7, 16)]
        )
        for bs in (32, 64):
            for E, topk, m in cases:
                ids, w = self._routing(m, m, E, topk)
                self._assert_same_sort(
                    self._sort(self.orig_sort, ids, w, bs, E),
                    self._sort(self.fm._moe_sorting_impl, ids, w, bs, E),
                    bs,
                    f"E={E} m={m} bs={bs}",
                )

    def test_fused_mxfp8_quant_matches_aiter(self):
        bs = 32
        # compact at E=513; distributed at MiniMax-M3's E=129 / top-4 + shared
        for E, topk, m in ((513, 11, 1), (513, 11, 5), (129, 5, 16), (129, 5, 51)):
            ids, w = self._routing(m, m, E, topk)
            x = torch.randn(m, self.DIM, dtype=torch.bfloat16, device=self.dev)
            sid, sw, _, nv, _ = self._sort(self.orig_sort, ids, w, bs, E)
            ref_q, _ = self.orig_quant(
                x,
                sorted_ids=sid,
                num_valid_ids=nv,
                token_num=m,
                topk=topk,
                block_size=bs,
                sorted_weights=sw,
                num_experts_upper_bound=E,
            )
            emitted = self._emitted_quant(ids, w, x, bs, E)
            self.assertIsNotNone(emitted, f"E={E} m={m}")
            self.assertTrue(
                torch.equal(ref_q.view(torch.uint8), emitted[0].view(torch.uint8)),
                f"E={E} m={m}",
            )

    def _valid_scale_bytes(self, scale, sorted_ids, num_valid, m, n_cols):
        """The e8m0 bytes of every real sorted row, at aiter's mx_scale_shuffle_idx address."""
        n = int(num_valid[0])
        rows = torch.arange(n, device=self.dev)
        rows = rows[(sorted_ids[:n] & 0xFFFFFF) < m]
        scalen_pad = ((n_cols // 32 + 7) // 8) * 8
        y = torch.arange(n_cols // 32, device=self.dev)
        base = (rows // 32) * (scalen_pad * 32) + (rows % 16) * 4 + (rows % 32) // 16
        addr = base[:, None] + (y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2
        return scale.view(torch.uint8).flatten()[addr]

    def test_fused_mxfp8_quant_matches_aiter_dsv4_hidden(self):
        """A hidden size off a 2048 multiple must still quantize each row and real-row scale
        bitwise like aiter's `fused_dynamic_mxfp8_quant_moe_sort`."""
        bs, dim, E, topk = 32, 5120, 385, 7
        # the widest compact sort (63 pairs) and the distributed one at the 112-pair cutoff
        for m in (9, 16):
            ids, w = self._routing(m, m, E, topk)
            x = torch.randn(m, dim, dtype=torch.bfloat16, device=self.dev)
            x[:, :64] *= 1e-3  # spread the per-group scales
            tok = self.S._pending_quant_input.set(x)
            etok = self.S._emitted_quant.set(None)
            try:
                sid, sw, _, nv, _ = self.fm._moe_sorting_impl(
                    ids, w, E, dim, torch.bfloat16, bs, None, None, 0, True
                )
                emitted = self.S._emitted_quant.get()
            finally:
                self.S._emitted_quant.reset(etok)
                self.S._pending_quant_input.reset(tok)
            msg = f"m={m}"
            self.assertIsNotNone(emitted, msg)
            ref_q, ref_s = self.orig_quant(
                x,
                sorted_ids=sid,
                num_valid_ids=nv,
                token_num=m,
                topk=topk,
                block_size=bs,
                sorted_weights=sw,
                num_experts_upper_bound=E,
            )
            self.assertTrue(
                torch.equal(ref_q.view(torch.uint8), emitted[0].view(torch.uint8)),
                msg,
            )
            self.assertEqual(ref_s.shape, emitted[1].shape, msg)
            self.assertTrue(
                torch.equal(
                    self._valid_scale_bytes(ref_s, sid, nv, m, dim),
                    self._valid_scale_bytes(emitted[1], sid, nv, m, dim),
                ),
                msg,
            )

    def test_falls_back_to_aiter_past_compact_at_513_experts(self):
        m = 6
        ids, _ = self._routing(m, m)
        self.assertFalse(self.S._small_sort_supported(ids, 32, None, None, self.E))

    def test_falls_back_to_aiter_past_112_pairs_at_385_experts(self):
        """Past 112 pairs at E=385 the distributed sort is slower than aiter's."""
        for m, small in ((16, True), (17, False)):
            ids, _ = self._routing(m, m, 385, 7)
            self.assertEqual(
                self.S._small_sort_supported(ids, 32, None, None, 385), small, m
            )


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestDsv4MoePrologue(CustomTestCase):
    """The sort-time quant must hand DeepSeek-V4.1's decode stage1 GEMM (TP4 afp8_wfp4)
    operands bitwise equal to stock aiter's."""

    H, INTER = 5120, 640

    @classmethod
    def setUpClass(cls):
        import aiter.fused_moe as fm

        from sglang.kernels.ops.moe import moe_sorting_small as S

        S.apply_aiter_small_moe_sort_patch()
        cls.S, cls.fm = S, fm
        cls.dev = torch.device("cuda", 0)
        cls._bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
        # V4.1 serves with fp8 activations at every row count
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"

    @classmethod
    def tearDownClass(cls):
        if cls._bound is None:
            os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
        else:
            os.environ["AITER_BF16_FP8_MOE_BOUND"] = cls._bound

    def _experts(self, E, gen):
        from aiter import dtypes
        from aiter.ops.shuffle import shuffle_scale

        H, I = self.H, self.INTER

        def fp4(*shape):
            w = torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)
            w = w.to(self.dev).view(dtypes.fp4x2)
            w.is_shuffled = True
            return w

        def e8m0(rows, cols, gate_up):
            s = torch.randint(118, 124, (rows, cols), dtype=torch.uint8, generator=gen)
            return shuffle_scale(
                s.to(self.dev).view(dtypes.fp8_e8m0), E, False, gate_up
            )

        return (
            fp4(E, 2 * I, H // 2),
            fp4(E, H, I // 2),
            e8m0(E * 2 * I, H // 32, True),
            e8m0(E * H, I // 32, False),
        )

    def _fused_moe(self, x, experts, weights, ids, stock):
        """fused_moe output, the stage1 output the stage2 quant reads, and whether the
        stage1 quant came from the sort."""
        from aiter import ActivationType, QuantType
        from aiter.ops.flydsl.moe_common import GateMode

        fm, S = self.fm, self.S
        w1, w2, w1_scale, w2_scale = experts
        stage1_out, sort_time_quant = [], []
        patched_quant = fm.fused_dynamic_mxfp8_quant_moe_sort
        patched_supported = S._small_sort_supported

        def recording_quant(input, *args, **kwargs):
            if input.shape[-1] == self.INTER:
                stage1_out.append(input.clone())
            else:
                sort_time_quant.append(S._emitted_quant.get() is not None)
            return patched_quant(input, *args, **kwargs)

        fm.fused_dynamic_mxfp8_quant_moe_sort = recording_quant
        if stock:
            S._small_sort_supported = lambda *args: False
        try:
            run = fm.fused_moe.__wrapped__ if stock else fm.fused_moe
            out = run(
                x,
                w1,
                w2,
                weights,
                ids,
                quant_type=QuantType.per_1x32,
                activation=ActivationType.Silu,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                swiglu_limit=10.0,
                gate_mode=GateMode.INTERLEAVE.value,
            )
        finally:
            fm.fused_dynamic_mxfp8_quant_moe_sort = patched_quant
            S._small_sort_supported = patched_supported
        self.assertEqual(len(stage1_out), 1)
        self.assertEqual(len(sort_time_quant), 1)
        return out, stage1_out[0], sort_time_quant[0]

    def _route(self, x, gate_w, bias):
        """V4.1's decode router: split-K GEMV + sqrtsoftplus gate + the shared slot."""
        from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k
        from sglang.kernels.ops.moe.rocm_router_gate import rocm_router_gate

        logits = torch.empty(x.shape[0], gate_w.shape[0], device=self.dev)
        partials = rocm_router_gemv_split_k(x, gate_w)
        return rocm_router_gate(
            logits, bias, 6, True, 1.5, partials=partials, num_shared=1
        )

    def test_target_moe_385_experts_top7(self):
        E, topk = 385, 7
        gen = torch.Generator().manual_seed(E)
        experts = self._experts(E, gen)
        gate_w = (torch.randn(E - 1, self.H, generator=gen) * 0.02).to(torch.bfloat16)
        gate_w = gate_w.to(self.dev)
        bias = (torch.randn(E - 1, generator=gen) * 0.5).to(torch.bfloat16).to(self.dev)
        # compact sort, and the distributed sort at the 112-pair cutoff
        for m in (1, 16):
            msg = f"M={m}"
            x = torch.randn(m, self.H, generator=gen).to(torch.bfloat16).to(self.dev)
            weights, ids = self._route(x, gate_w, bias)
            self.assertEqual(ids.shape, (m, topk), msg)
            ref_out, ref_stage1, _ = self._fused_moe(
                x, experts, weights, ids, stock=True
            )
            out, stage1, fused = self._fused_moe(x, experts, weights, ids, stock=False)
            self.assertTrue(fused, msg)
            self.assertTrue(torch.equal(stage1, ref_stage1), msg)
            # bf16 atomics round partial sums of up to topk rows: a few ulp of the row scale
            err = (out.float() - ref_out.float()).abs().max().item()
            self.assertLessEqual(err, 2e-2 * ref_out.float().abs().max().item(), msg)

    def test_draft_moe_129_experts_top4(self):
        """The draft MoE's sort-time quant must hand stage1 what stock aiter builds."""
        E, topk, m = 129, 4, 1
        gen = torch.Generator().manual_seed(E)
        experts = self._experts(E, gen)
        x = torch.randn(m, self.H, generator=gen).to(torch.bfloat16).to(self.dev)
        ids = torch.cat(
            [torch.randperm(E - 1, generator=gen)[: topk - 1], torch.tensor([E - 1])]
        )
        ids = ids.view(m, topk).to(torch.int32).to(self.dev).contiguous()
        weights = torch.rand(m, topk, generator=gen).to(self.dev)
        ref_out, ref_stage1, _ = self._fused_moe(x, experts, weights, ids, stock=True)
        out, stage1, fused = self._fused_moe(x, experts, weights, ids, stock=False)
        self.assertTrue(fused)
        self.assertTrue(torch.equal(stage1, ref_stage1))
        err = (out.float() - ref_out.float()).abs().max().item()
        self.assertLessEqual(err, 2e-2 * ref_out.float().abs().max().item())


if __name__ == "__main__":
    unittest.main()
