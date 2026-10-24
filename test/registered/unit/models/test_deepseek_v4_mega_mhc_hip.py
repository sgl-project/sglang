import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cfg(*, mult=4, hidden=5120, cp_prefill=False):
    return mhc.HcConfig(
        mult=mult,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-6,
        hidden=hidden,
        pre_from_prev=True,
        cp_prefill=cp_prefill,
    )


class _Norm:
    def __init__(self):
        self.weight = torch.ones(4)
        self.variance_epsilon = 1e-6


def _sublayer(cfg):
    return mhc.HcSubLayer(
        cfg=cfg,
        fn=torch.zeros(1),
        scale=torch.zeros(1),
        base=torch.zeros(1),
        norm=_Norm(),
    )


class TestMegaMhcGate(unittest.TestCase):
    def test_off_by_default_on_gfx950(self):
        with (
            mock.patch.object(mhc, "_is_hip", True),
            mock.patch.object(mhc, "_is_gfx95_supported", True),
        ):
            self.assertFalse(mhc._can_use_mega_mhc(_cfg()))

    def test_hip_gfx950_dsv41_shape(self):
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(mhc, "_is_hip", True),
            mock.patch.object(mhc, "_is_gfx95_supported", True),
        ):
            self.assertTrue(mhc._can_use_mega_mhc(_cfg()))
            self.assertFalse(mhc._can_use_mega_mhc(_cfg(mult=2)))
            self.assertFalse(mhc._can_use_mega_mhc(_cfg(hidden=4096)))
            self.assertFalse(mhc._can_use_mega_mhc(_cfg(cp_prefill=True)))

    def test_hip_without_gfx95_stays_off(self):
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(mhc, "_is_hip", True),
            mock.patch.object(mhc, "_is_gfx95_supported", False),
        ):
            self.assertFalse(mhc._can_use_mega_mhc(_cfg()))


class TestHipFlydslDispatch(unittest.TestCase):
    def _boundary(self, cfg):
        hc = _sublayer(cfg)
        return mhc.HcNextBoundary(
            norm=hc.norm,
            accepts_mxfp8=False,
            norm_fusable=True,
            hc=hc,
        )

    def test_calls_flydsl_bf16_and_packs_next_stats(self):
        cfg = _cfg()
        y = torch.zeros(2, 4)
        residual = torch.zeros(2, 4, 4)
        pre = torch.zeros(2, 4)
        post_mix = torch.zeros(2, 4)
        comb = torch.zeros(2, 4, 4)
        updated = torch.ones(2, 4, 4)
        normalized = torch.full((2, 4), 3.0)
        next_post = torch.full((2, 4, 1), 4.0)
        next_comb = torch.full((2, 4, 4), 5.0)
        next_pre = torch.full((2, 4), 6.0)
        seen = {}

        def fake_flydsl(*args, **kwargs):
            seen["out_dtype"] = kwargs["out_dtype"]
            seen["post_ndim"] = kwargs["post_layer_mix"].ndim
            return updated, next_post, next_comb, normalized, next_pre

        flydsl = types.ModuleType("aiter.ops.flydsl")
        flydsl.flydsl_mega_mhc = fake_flydsl
        ops = types.ModuleType("aiter.ops")
        ops.flydsl = flydsl
        aiter = types.ModuleType("aiter")
        aiter.ops = ops
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(mhc, "_is_hip", True),
            mock.patch.object(mhc, "_is_gfx95_supported", True),
            mock.patch.dict(
                sys.modules,
                {
                    "aiter": aiter,
                    "aiter.ops": ops,
                    "aiter.ops.flydsl": flydsl,
                },
            ),
        ):
            state = mhc._post_fusion(
                _sublayer(cfg), y, residual, (pre, post_mix, comb), self._boundary(cfg)
            )

        self.assertEqual(seen["out_dtype"], "bf16")
        self.assertEqual(seen["post_ndim"], 3)
        self.assertIsInstance(state.input, mhc.HcNormed)
        self.assertTrue(torch.equal(state.input.rows, normalized))
        self.assertTrue(torch.equal(state.streams, updated))
        self.assertEqual(state.stats[0].shape, (2, 4))
        self.assertEqual(state.stats[1].shape, (2, 4))
        self.assertTrue(torch.equal(state.pre, pre))


class TestFusedBoundaryYieldsToMega(unittest.TestCase):
    def test_env_disables_fused_boundary(self):
        layer = mock.Mock()
        layer.hc_pre_from_prev_sublayer = True
        layer.hc_mult = 4
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(deepseek_v4_hip, "_is_gfx95_supported", True),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fp8_quant_eligible",
                return_value=False,
            ),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fake_quant_eligible",
                return_value=False,
            ),
        ):
            deepseek_v4_hip.init_decoder_layer(layer, quant_config=None)
        self.assertFalse(layer.hc_boundary_fused)

    def test_env_off_keeps_fused_boundary(self):
        layer = mock.Mock()
        layer.hc_pre_from_prev_sublayer = True
        layer.hc_mult = 4
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(False),
            mock.patch.object(deepseek_v4_hip, "_is_gfx95_supported", True),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fp8_quant_eligible",
                return_value=False,
            ),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fake_quant_eligible",
                return_value=False,
            ),
        ):
            deepseek_v4_hip.init_decoder_layer(layer, quant_config=None)
        self.assertTrue(layer.hc_boundary_fused)


if __name__ == "__main__":
    unittest.main()
