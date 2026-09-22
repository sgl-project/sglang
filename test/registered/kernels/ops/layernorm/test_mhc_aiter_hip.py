"""The AITER mHC route on gfx950: gate, fallback latch, and kernel numerics vs the Torch oracle."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestAiterMHCGLM53Flash(CustomTestCase):
    hidden_size = 4096
    hc_mult = 4
    rms_eps = 1e-6
    hc_eps = 1e-6

    def setUp(self):
        mhc._AITER_MHC_RUNTIME_DISABLED = False

    def _inputs(self, tokens: int, seed: int = 0):
        torch.manual_seed(seed)
        device = torch.device("cuda")
        mix_size = 2 * self.hc_mult + self.hc_mult**2
        residual = (
            torch.randn(
                tokens,
                self.hc_mult,
                self.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        fn = (
            torch.randn(
                mix_size,
                self.hc_mult * self.hidden_size,
                device=device,
                dtype=torch.float32,
            )
            * 0.01
        )
        scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
        base = torch.zeros(mix_size, device=device, dtype=torch.float32)
        return residual, fn, scale, base

    def _rmsnorm(self, x, weight):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.rms_eps)
            * weight.float()
        ).to(x.dtype)

    def test_gate_selects_aiter_on_gfx950(self):
        """A gate that resolves False on real hardware silently serves the Torch path."""
        with envs.SGLANG_USE_AITER.override(True):
            self.assertTrue(mhc._use_aiter_mhc())

    def test_hip_without_aiter_stays_on_torch_and_never_loads_tilelang(self):
        """The TileLang/DeepGEMM flags default on; only the HIP gate keeps them off this device."""
        residual, fn, scale, base = self._inputs(8)
        x = residual.reshape(8, self.hc_mult * self.hidden_size)
        _, _, layer_ref = mhc._mhc_pre_torch(
            residual, fn, scale, base, self.rms_eps, self.hc_eps, self.hc_eps, 2.0, 4
        )
        with (
            envs.SGLANG_USE_AITER.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
            patch.object(
                mhc, "_load_tilelang", side_effect=AssertionError("TileLang imported")
            ),
        ):
            self.assertFalse(mhc._use_aiter_mhc())
            self.assertFalse(mhc._use_tilelang_mhc_pre())
            self.assertFalse(mhc._use_tilelang_mhc_post())
            self.assertFalse(mhc._use_deep_gemm_hc_prenorm())
            layer_input, h_res, h_post, norm_fused = mhc.hc_pre(
                x, fn, scale, base, self.hc_mult, self.rms_eps, self.hc_eps, 4
            )
            out = mhc.hc_post(layer_input, x, h_post, h_res, self.hc_mult)
        self.assertFalse(norm_fused)
        torch.testing.assert_close(layer_input, layer_ref)
        self.assertTrue(torch.isfinite(out).all())

    def test_aiter_import_and_runtime_failures_latch_to_torch(self):
        """A missing symbol or a raising kernel must disable the route once, not fail the request."""
        residual, fn, scale, base = self._inputs(8)
        x = residual.reshape(8, self.hc_mult * self.hidden_size)
        modules = {
            "aiter": types.ModuleType("aiter"),
            "aiter.ops": types.ModuleType("aiter.ops"),
            "aiter.ops.mhc": types.ModuleType("aiter.ops.mhc"),
        }
        with patch.dict(sys.modules, modules):
            result = mhc._try_aiter_mhc_pre(
                residual,
                fn,
                scale,
                base,
                self.rms_eps,
                self.hc_eps,
                self.hc_eps,
                2.0,
                4,
                None,
                None,
            )
        self.assertIsNone(result)
        self.assertTrue(mhc._AITER_MHC_RUNTIME_DISABLED)

        mhc._AITER_MHC_RUNTIME_DISABLED = False

        def fail_post(*_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        failing = types.ModuleType("aiter.ops.mhc")
        failing.mhc_post = fail_post
        modules["aiter.ops.mhc"] = failing
        with envs.SGLANG_USE_AITER.override(False):
            layer_input, h_res, h_post, _ = mhc.hc_pre(
                x, fn, scale, base, self.hc_mult, self.rms_eps, self.hc_eps, 4
            )
        with (
            patch.dict(sys.modules, modules),
            envs.SGLANG_USE_AITER.override(True),
        ):
            out = mhc.hc_post(layer_input, x, h_post, h_res, self.hc_mult)
        self.assertTrue(mhc._AITER_MHC_RUNTIME_DISABLED)
        self.assertTrue(torch.isfinite(out).all())

    def test_aiter_pre_post_match_torch_oracle(self):
        """A positional or kwarg mixup in the AITER call shows up only against the real kernel."""
        norm_weight = torch.linspace(
            0.75, 1.25, self.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        for tokens in (1, 8, 17, 32, 64, 128):
            for sinkhorn_iters in (2, 20):
                for fused_norm in (False, True):
                    with self.subTest(
                        tokens=tokens, sinkhorn_iters=sinkhorn_iters, norm=fused_norm
                    ):
                        residual, fn, scale, base = self._inputs(tokens)
                        post_ref, comb_ref, layer_ref = mhc._mhc_pre_torch(
                            residual,
                            fn,
                            scale,
                            base,
                            self.rms_eps,
                            self.hc_eps,
                            self.hc_eps,
                            2.0,
                            sinkhorn_iters,
                        )
                        result = mhc._try_aiter_mhc_pre(
                            residual,
                            fn,
                            scale,
                            base,
                            self.rms_eps,
                            self.hc_eps,
                            self.hc_eps,
                            2.0,
                            sinkhorn_iters,
                            norm_weight if fused_norm else None,
                            self.rms_eps if fused_norm else None,
                        )
                        self.assertIsNotNone(result, "AITER mHC pre fell back")
                        post_out, comb_out, layer_out = result
                        if fused_norm:
                            layer_ref = self._rmsnorm(layer_ref, norm_weight)
                        torch.cuda.synchronize()

                        torch.testing.assert_close(
                            post_out, post_ref, atol=2e-3, rtol=2e-3
                        )
                        torch.testing.assert_close(
                            comb_out, comb_ref, atol=2e-3, rtol=2e-3
                        )
                        torch.testing.assert_close(
                            layer_out, layer_ref, atol=2e-2, rtol=2e-2
                        )

                        x = (layer_ref.float() * 0.75).to(layer_ref.dtype)
                        post_ref_out = mhc._mhc_post_torch(
                            x, residual, post_ref, comb_ref
                        )
                        post_out_actual = mhc._try_aiter_mhc_post(
                            x, residual, post_out, comb_out
                        )
                        self.assertIsNotNone(
                            post_out_actual, "AITER mHC post fell back"
                        )
                        torch.testing.assert_close(
                            post_out_actual, post_ref_out, atol=2e-2, rtol=2e-2
                        )


if __name__ == "__main__":
    unittest.main()
