"""Tests for the DeepSeek-V4 decode ``wo_a`` bf16 batched-matmul routing.

Covers ``deepseek_v4._apply_wo_a_bf16_matmul``, which (opt-in via
``SGLANG_OPT_USE_AITER_BATCHED_GEMM`` on HIP/gfx95) routes the MLA output-absorb
bf16 GEMM off the rocBLAS/Tensile ``Cijk_*`` batched GEMM onto aiter's tuned
``batched_gemm_bf16``, with an einsum fallback.

The opt-in flag and the aiter kernel are resolved ONCE at module import
(``_wo_a_aiter_gemm_eligible`` -> ``_wo_a_aiter_batched_gemm_enabled`` and the
pre-imported ``_wo_a_batched_gemm_bf16``) so the decode critical path pays no
``EnvBool.get()`` or function-local import per token. The tests therefore drive
those cached module globals directly.

Validates:
1. Static eligibility -- ``_wo_a_aiter_gemm_eligible`` is true only when the
   flag, the global ``SGLANG_USE_AITER`` switch, HIP, and gfx95 are all set
   (table-driven).
2. Dispatch/gating -- with the reroute disabled, or on prefill
   (``is_decode=False``), the plain ``torch.einsum("tgd,grd->tgr", ...)`` path
   runs (bit-identical to the old code); only decode + enabled hits the kernel.
3. Fallback -- a runtime kernel failure degrades to the einsum and disables the
   reroute for the rest of the process (no per-call retry / log spam on the
   decode critical path).
4. Numerics -- on gfx95 with aiter, the aiter kernel is genuinely used and its
   result matches the einsum within bf16 tolerance across ``T/G/D/R`` shapes
   (the PR's model-free bit-check).

deepseek_v4 pulls in the full model stack, so this is registered as an AMD GPU
test and only imported behind ``is_hip()`` -- matching the existing AMD aiter
op tests.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.utils.common import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip(), "wo_a batched_gemm_bf16 routing requires ROCm")
class TestWoABf16BatchedGemm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the heavy model module only on a GPU runner (see module docstring).
        from sglang.srt.models import deepseek_v4 as dsv4

        cls.dsv4 = dsv4
        cls.device = "cuda"  # torch maps "cuda" onto the ROCm HIP device

    def setUp(self):
        torch.manual_seed(0)
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


if __name__ == "__main__":
    unittest.main()
