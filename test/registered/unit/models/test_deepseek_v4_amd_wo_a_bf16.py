"""`_apply_wo_a_bf16_matmul` on ROCm must gate the aiter reroute, fall back to einsum, and match einsum on the gfx950 routes."""

import unittest
from unittest import mock

import torch

from sglang.srt.utils.common import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip(), "wo_a batched_gemm_bf16 routing requires ROCm")
class TestWoABf16BatchedGemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # import the heavy model module only on a GPU runner
        from sglang.srt.models import deepseek_v4 as dsv4
        from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense

        cls.dsv4 = dsv4
        cls.gfx95_dense = deepseek_v4_gfx95_dense
        cls.device = "cuda"  # torch maps "cuda" onto the ROCm HIP device

    def setUp(self):
        torch.manual_seed(0)
        # the gfx950 wo_a route reads the exec bag (deterministic gate): publish one
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        # The one-shot runtime-disable flag is process-global; reset it so a
        # failure case in one test cannot leak into another.
        self.dsv4._wo_a_aiter_batched_gemm_disabled = False

    def _rand(self, T, G, D, R):
        o = torch.randn(T, G, D, device=self.device, dtype=torch.bfloat16)
        wo_a = torch.randn(G, R, D, device=self.device, dtype=torch.bfloat16)
        return o, wo_a

    @staticmethod
    def _einsum(o, wo_a):
        return torch.einsum("tgd,grd->tgr", o, wo_a)

    def _require_gfx95_aiter(self):
        if not self.dsv4._is_gfx95_supported:
            self.skipTest("aiter batched_gemm_bf16 is gfx95-only")
        try:
            from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (  # noqa: F401
                batched_gemm_bf16,
            )
        except Exception as err:  # pragma: no cover - env-dependent
            self.skipTest(f"aiter batched_gemm_bf16 unavailable: {err}")

    # ------------------------------------------------------ static eligibility

    def test_eligibility_gating(self):
        # The reroute activates only when the opt-in flag, the global aiter
        # switch, HIP, and gfx95 are ALL set -- resolved once at import.
        eligible = self.dsv4._wo_a_aiter_gemm_eligible
        base = dict(flag=True, use_aiter=True, is_hip=True, is_gfx95=True)
        self.assertTrue(eligible(**base))
        for off in ("flag", "use_aiter", "is_hip", "is_gfx95"):
            with self.subTest(disabled=off):
                self.assertFalse(eligible(**{**base, off: False}))

    # ---------------------------------------------------------- dispatch gating

    def test_dispatch_gating(self):
        # Given the cached ``enabled`` bool and the forward mode, the kernel is
        # hit only on decode + enabled; every other case takes the einsum.
        o, wo_a = self._rand(8, 4, 128, 32)
        ref = self._einsum(o, wo_a)

        def _fake_kernel(xq, w, dtype):  # [G,T,D] @ [G,R,D]^T -> [G,T,R]
            fake_kernel.calls += 1
            return torch.einsum("gtd,grd->gtr", xq, w).to(dtype)

        fake_kernel = _fake_kernel
        for enabled, is_decode, expect_kernel in (
            (True, True, True),
            (True, False, False),  # prefill keeps the einsum
            (False, True, False),  # reroute off -> einsum
            (False, False, False),
        ):
            with self.subTest(enabled=enabled, is_decode=is_decode):
                fake_kernel.calls = 0
                with (
                    mock.patch.object(
                        self.dsv4, "_wo_a_aiter_batched_gemm_enabled", enabled
                    ),
                    # the gfx950 fp8-grid fork would run before the aiter kernel
                    mock.patch.object(self.gfx95_dense, "_wo_a_fp8_grid_gemm", None),
                    mock.patch.object(
                        self.dsv4, "_wo_a_batched_gemm_bf16", fake_kernel
                    ),
                    mock.patch("torch.einsum", wraps=torch.einsum) as spy,
                ):
                    out = self.dsv4._apply_wo_a_bf16_matmul(
                        o, wo_a, is_decode=is_decode
                    )
                self.assertEqual(out.shape, (8, 4, 32))
                if expect_kernel:
                    self.assertEqual(fake_kernel.calls, 1)
                    # einsum only ran here as the fake kernel's own impl.
                else:
                    self.assertEqual(fake_kernel.calls, 0)
                    spy.assert_called_once()
                    self.assertTrue(torch.equal(out, ref))

    # ---------------------------------------------------------------- fallback

    def test_runtime_failure_falls_back_and_disables_reroute(self):
        o, wo_a = self._rand(8, 4, 128, 32)

        call_count = {"n": 0}

        def _boom(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("kernel missing")

        with (
            mock.patch.object(self.dsv4, "_wo_a_aiter_batched_gemm_enabled", True),
            mock.patch.object(self.gfx95_dense, "_wo_a_fp8_grid_gemm", None),
            mock.patch.object(self.dsv4, "_wo_a_batched_gemm_bf16", _boom),
        ):
            out = self.dsv4._apply_wo_a_bf16_matmul(o, wo_a, is_decode=True)

            # Failure inside the aiter branch must not raise and must match einsum.
            self.assertEqual(out.shape, (8, 4, 32))
            self.assertTrue(torch.equal(out, self._einsum(o, wo_a)))

            # The reroute is disabled after the first failure, so a subsequent
            # decode step does not retry the broken kernel (no per-call log spam
            # on the critical path).
            self.assertTrue(self.dsv4._wo_a_aiter_batched_gemm_disabled)
            out2 = self.dsv4._apply_wo_a_bf16_matmul(o, wo_a, is_decode=True)
            self.assertTrue(torch.equal(out2, self._einsum(o, wo_a)))

        self.assertEqual(call_count["n"], 1)  # kernel attempted exactly once

    # ---------------------------------------------------------------- numerics

    def test_aiter_matches_einsum_across_shapes(self):
        self._require_gfx95_aiter()

        from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

        # (T tokens, G groups, D head_dim, R o_lora_rank)
        shapes = [
            (1, 4, 128, 32),
            (8, 8, 128, 64),
            (37, 4, 192, 32),
            (128, 2, 128, 16),
        ]
        for T, G, D, R in shapes:
            with self.subTest(T=T, G=G, D=D, R=R):
                o, wo_a = self._rand(T, G, D, R)
                ref = self._einsum(o, wo_a).float()

                with (
                    mock.patch.object(
                        self.dsv4, "_wo_a_aiter_batched_gemm_enabled", True
                    ),
                    mock.patch.object(
                        self.dsv4, "_wo_a_batched_gemm_bf16", batched_gemm_bf16
                    ),
                    mock.patch("torch.einsum", wraps=torch.einsum) as spy,
                ):
                    out = self.dsv4._apply_wo_a_bf16_matmul(o, wo_a, is_decode=True)
                # The aiter kernel -- not the einsum fallback -- must have run,
                # otherwise this check would be trivially satisfied.
                spy.assert_not_called()

                self.assertEqual(out.shape, (T, G, R))
                self.assertEqual(out.dtype, torch.bfloat16)
                rel = ((out.float() - ref).abs() / (ref.abs() + 1e-6)).max().item()
                self.assertLessEqual(rel, 5e-4)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestWoABf16PrefillAndVerifyRoutes(CustomTestCase):
    """gfx950 prefill rows write the token-major layout directly and verify rows keep the
    GEMV / split-K / strided-bmm regimes bit-exact under graph replay."""

    def setUp(self):
        from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.project = _apply_wo_a_bf16_matmul
        torch.manual_seed(39186)

    def operands(self, rows, *, strided=False, dtype=torch.bfloat16, width=4096):
        x = torch.randn(rows, 4 if strided else 2, width, device="cuda", dtype=dtype)
        if strided:
            x = x[:, 1:3]
        w = torch.randn(2, 1024, width, device="cuda", dtype=dtype) * 0.015625
        return x, w

    def test_prefill_and_mutable_graph(self):
        for rows, strided in (
            (4096, False),
            (4097, True),
            (65536, False),
        ):
            with self.subTest(rows=rows, strided=strided):
                x, w = self.operands(rows, strided=strided)
                y = self.project(x, w, is_decode=False, is_prefill=True)
                self.assertTrue(y.is_contiguous())
                torch.testing.assert_close(
                    y, torch.einsum("tgd,grd->tgr", x, w), atol=0, rtol=0
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = self.project(x, w, is_decode=False, is_prefill=True)
                for _ in range(2):
                    x.normal_()
                    w.normal_(std=0.015625)
                    graph.replay()
                    torch.testing.assert_close(
                        y, torch.einsum("tgd,grd->tgr", x, w), atol=0, rtol=0
                    )
                del graph, x, w, y

    def test_decode_verify_and_mutable_graph(self):

        for rows in (1, 2, 8, 64, 129):
            with self.subTest(rows=rows):
                x, w = self.operands(rows, strided=rows == 8)
                kwargs = dict(is_decode=True, is_target_verify=rows > 1)
                self.project(x, w, **kwargs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = self.project(x, w, **kwargs)
                for _ in range(2):
                    x.normal_()
                    w.normal_(std=0.015625)
                    graph.replay()
                    ref = torch.einsum("tgd,grd->tgr", x, w)
                    self.assertTrue(y.is_contiguous())
                    if rows > 8:
                        torch.testing.assert_close(y, ref, atol=0, rtol=0)
                    else:
                        # split-K sums in its own fixed order: occasional one-ulp bf16
                        # flips against einsum, far below one ulp (2^-8) on average
                        error = (y.float() - ref.float()).square().mean()
                        self.assertLess(
                            (error / ref.float().square().mean()).sqrt().item(), 1e-3
                        )


if __name__ == "__main__":
    unittest.main()
