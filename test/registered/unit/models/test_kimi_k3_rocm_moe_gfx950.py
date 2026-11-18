"""Real-AITER check of the Kimi-K3 merged MoE front on gfx950.

The CPU test in ``test_kimi_k3_rocm_quant.py`` fakes ``tgemm.mm``;
only this file checks that AITER's tuned row serves the TP8 front and its output.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3MoE, _k3_bf16_gemm
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

# TP8 merged front rows [shared gate-up | router | latent down]; K = hidden size.
_FRONT_ROWS, _FRONT_K = (1536, 896, 3584), 7168
# Its tuned row in the AITER pinned in docker/rocm.Dockerfile is a split-K kernel, the noisiest one.
_NUM_TOKENS = 64
# Measured 4e-3..8e-3 over 5 seeds on AITER 285579b62, from bf16 output rounding;
# a transposed weight gives ~1.5.
_MAX_REL_ERR = 2e-2


@unittest.skipUnless(is_hip(), "AITER's tuned front rows are gfx950 only")
class TestTunedMoeFrontGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        # A partitioned gfx950 has fewer CUs, and AITER keys its rows on the CU count.
        if (get_gfx(), get_cu_num()) != ("gfx950", 256):
            raise unittest.SkipTest("AITER's tuned front rows are 256-CU gfx950 only")

    def _merged_front(self, gen: torch.Generator):
        class Owner(SimpleNamespace):
            @property
            def _front_shard(self):
                return None

            @property
            def _front_fp32(self):
                return False

        def module(rows):
            weight = torch.randn(rows, _FRONT_K, device="cuda", generator=gen) * 0.02
            return SimpleNamespace(weight=weight.to(torch.bfloat16))

        gate_up, router, down = _FRONT_ROWS
        owner = Owner(
            use_latent_moe=True,
            shared_experts=SimpleNamespace(gate_up_proj=module(gate_up)),
            gate=module(router),
            routed_expert_down_proj=module(down),
            _front_w=None,
            _front_quant_method=UnquantizedLinearMethod(),
            _front_quant_layer=None,
            _moe_front_needs_dense_bf16=False,
        )
        with mock.patch.object(
            kimi_k3,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: True),
        ):
            KimiK3MoE._merge_front_weights(owner)
        return owner

    def test_tuned_row_serves_the_front_and_matches_the_default_gemm(self):
        import aiter.tuned_gemm as tuned_gemm

        gen = torch.Generator(device="cuda").manual_seed(0)
        owner = self._merged_front(gen)
        x = torch.randn(_NUM_TOKENS, _FRONT_K, device="cuda", generator=gen).to(
            torch.bfloat16
        )
        configs = []
        get_config = tuned_gemm.get_GEMM_A16W16_config

        def lookup(**key):
            configs.append((key, get_config(**key)))
            return configs[-1][1]

        with (
            mock.patch.object(unquant, "_use_aiter", True),
            mock.patch.object(unquant, "_skip_aiter_unquant", False),
            mock.patch.object(unquant, "tgemm", tuned_gemm.tgemm, create=True),
            mock.patch.object(tuned_gemm, "get_GEMM_A16W16_config", lookup),
        ):
            out = torch.cat(KimiK3MoE._fused_front(owner, x), dim=-1)
        ref = _k3_bf16_gemm(x, owner._front_w)

        [(key, config)] = configs
        self.assertEqual(
            (key["M"], key["N"], key["K"]), (_NUM_TOKENS, sum(_FRONT_ROWS), _FRONT_K)
        )
        # "torch" is AITER's fallback for a shape without a tuned row, i.e. F.linear.
        self.assertNotEqual(config["libtype"], "torch")
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, ref.shape)
        rel_err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
        self.assertLess(rel_err.item(), _MAX_REL_ERR)


if __name__ == "__main__":
    unittest.main()
