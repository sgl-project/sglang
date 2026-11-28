"""EP keeps NVFP4 MoE input scales at the global expert count in one branch.

create_weights registers w13_input_scale/w2_input_scale with
`_sglang_require_global_experts = True`, so they carry `num_experts` rows while
g1_alphas / g2_alphas are read per local expert. Every runner that reaches
process_weights_after_loading without its own scale handling -- the branch that
is not cutlass/trtllm, MegaMoE, CuteDSL, or marlin -- used to keep the global
width, so with ep_size > 1 the first multiply in _compute_gemm1_alphas raises
"The size of tensor a (512) must match the size of tensor b (64)".
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.runtime_context import get_flags
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 8
NUM_LOCAL_EXPERTS = 4
EP_RANK = 1


def _ep_layer():
    """A layer with this rank's expert shards and global-width input scales."""
    # The platform gate at the top of __init__ refuses a non-Blackwell device
    # unless marlin is selected; the failing launch was on sm_120, which is
    # Blackwell, so this mock has to say True to reach the scale branch.
    with mock.patch.object(
        modelopt_quant, "get_platform", return_value=SimpleNamespace(is_blackwell=True)
    ):
        method = ModelOptNvFp4FusedMoEMethod(
            SimpleNamespace(use_per_token_activation=False)
        )
    config = MoeRunnerConfig(is_gated=True)
    method.moe_runner_config = config

    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.quant_method = method
    layer.quant_config = method.quant_config
    layer.moe_runner_config = config
    layer.scheme = None
    layer._has_fused_shared = False
    layer.use_flashinfer_trtllm_moe = method.enable_flashinfer_trtllm_moe
    layer.use_presharded_weights = False
    layer.moe_ep_size = 2
    layer.moe_ep_rank = EP_RANK
    layer.num_experts = NUM_EXPERTS
    layer.num_local_experts = NUM_LOCAL_EXPERTS
    layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.hidden_size = 4096
    layer.dispatcher = SimpleNamespace(set_quant_config=lambda config: None)

    # Distinct per-expert scales so a wrong row set cannot pass by symmetry.
    w13_input_scale = torch.arange(NUM_EXPERTS * 2, dtype=torch.float32).reshape(
        NUM_EXPERTS, 2
    )
    w2_input_scale = torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float32)
    for name, data in {
        "w13_weight": torch.zeros(NUM_LOCAL_EXPERTS, 128, 32, dtype=torch.uint8),
        "w13_weight_scale": torch.zeros(
            NUM_LOCAL_EXPERTS, 128, 4, dtype=torch.float8_e4m3fn
        ),
        "w13_weight_scale_2": torch.full((NUM_LOCAL_EXPERTS, 2), 0.25),
        "w13_input_scale": w13_input_scale,
        "w2_weight": torch.zeros(NUM_LOCAL_EXPERTS, 64, 32, dtype=torch.uint8),
        "w2_weight_scale": torch.zeros(
            NUM_LOCAL_EXPERTS, 64, 4, dtype=torch.float8_e4m3fn
        ),
        "w2_weight_scale_2": torch.ones(NUM_LOCAL_EXPERTS),
        "w2_input_scale": w2_input_scale,
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    return layer


class TestNvFp4EpInputScaleSlicing(CustomTestCase):
    def _process(self, layer):
        with (
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON, a2a_backend=MoeA2ABackend.NONE
            ),
            mock.patch.object(
                modelopt_quant, "swizzle_blockscale", side_effect=lambda w: w
            ),
            mock.patch.object(
                modelopt_quant, "_use_nvfp4_dispatch", return_value=False
            ),
        ):
            layer.quant_method.process_weights_after_loading(layer)

    def test_alphas_are_local_expert_width(self):
        layer = _ep_layer()
        self._process(layer)
        # w13_input_scale rows max to [1,3,5,7,9,11,13,15]; rank 1 owns the last
        # four, and w13_weight_scale_2 carries 0.25 in both gate and up columns.
        local_input_scale = torch.tensor([9.0, 11.0, 13.0, 15.0])
        torch.testing.assert_close(
            layer.g1_alphas, local_input_scale * 0.25, rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer.g1_alphas_up, local_input_scale * 0.25, rtol=0, atol=0
        )
        # w2_input_scale is [1..8] globally; this rank's rows are [5,6,7,8].
        torch.testing.assert_close(
            layer.g2_alphas, torch.tensor([5.0, 6.0, 7.0, 8.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer.w13_input_scale_quant, 1 / local_input_scale, rtol=0, atol=0
        )
        self.assertEqual(layer.g1_alphas.shape, (NUM_LOCAL_EXPERTS,))
        self.assertEqual(layer.g2_alphas.shape, (NUM_LOCAL_EXPERTS,))

    def test_ep_size_one_is_unchanged(self):
        layer = _ep_layer()
        layer.moe_ep_size = 1
        layer.moe_ep_rank = 0
        layer.num_local_experts = NUM_EXPERTS
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, 128, 32, dtype=torch.uint8), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, 128, 4, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            torch.full((NUM_EXPERTS, 2), 0.25), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, 64, 32, dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, 64, 4, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(NUM_EXPERTS), requires_grad=False
        )
        self._process(layer)
        self.assertEqual(layer.g1_alphas.shape, (NUM_EXPERTS,))


if __name__ == "__main__":
    unittest.main()
