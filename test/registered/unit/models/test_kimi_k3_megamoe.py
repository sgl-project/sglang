"""Unit tests for Kimi-K3's DeepGEMM MegaMoE call contract."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKimiK3MegaMoE(CustomTestCase):
    def test_deep_gemm_receives_direct_situ_activation(self):
        calls = []

        def fp8_fp4_mega_moe(y, l1_weights, l2_weights, buffer, **kwargs):
            calls.append(kwargs)
            y.fill_(7)

        deep_gemm = SimpleNamespace(fp8_fp4_mega_moe=fp8_fp4_mega_moe)
        buffer = SimpleNamespace(
            x=torch.empty((8, 4)),
            x_sf=torch.empty((8, 1), dtype=torch.int32),
            topk_idx=torch.empty((8, 1), dtype=torch.int32),
            topk_weights=torch.empty((8, 1)),
        )
        owner = SimpleNamespace(
            experts=SimpleNamespace(
                num_experts=1,
                mega_l1_weights=object(),
                mega_l2_weights=object(),
                should_fuse_routed_scaling_factor_in_topk=True,
            ),
            _mega_top_k=1,
            _mega_intermediate_size=8,
            moe_hidden_size=4,
        )
        routed_input = torch.ones((2, 4))
        topk_output = SimpleNamespace(
            topk_ids=torch.zeros((2, 1), dtype=torch.int64),
            topk_weights=torch.ones((2, 1)),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch(
                "sglang.kernels.ops.attention.dsv4.mega_moe_pre_dispatch",
            ),
            patch(
                "sglang.srt.layers.moe.mega_moe._get_mega_moe_symm_buffer",
                return_value=buffer,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(device_group=object()),
            ),
            patch(
                "sglang.srt.environ.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get",
                return_value=8,
            ),
        ):
            output = KimiK3MoE._forward_mega_experts(owner, routed_input, topk_output)

        self.assertTrue(
            torch.equal(output, torch.full((2, 4), 7.0, dtype=torch.bfloat16))
        )
        self.assertEqual(
            calls,
            [
                {
                    "recipe": (1, 1, 32),
                    "activation": "situ",
                    "fast_math": True,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
