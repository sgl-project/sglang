"""The weight checker must leave the NVFP4 MoE activation input scales alone: no weight update carries them, so
a reset before an update leaves them random and the comparison after it fails whatever the update wrote.

The platform check is stubbed so the layer is built on CPU, as in test_modelopt_nvfp4_moe_dispatch.py.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

# Import modelopt_quant first (see test_modelopt_nvfp4_moe_scales.py for the circular-import reason).
# isort: off
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod

# isort: on
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils.weight_checker import _is_skip_weight_check
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 8
HIDDEN = 64
INTERMEDIATE = 32


def _nvfp4_moe_layer() -> torch.nn.Module:
    with (
        mock.patch.object(
            modelopt_quant,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        ),
        mock.patch.object(
            modelopt_quant,
            "get_platform",
            return_value=SimpleNamespace(is_blackwell=True),
        ),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=False),
    ):
        method = ModelOptNvFp4FusedMoEMethod(
            SimpleNamespace(
                group_size=16,
                is_checkpoint_nvfp4_serialized=True,
                use_per_token_activation=False,
                get_name=lambda: "modelopt_fp4",
            )
        )
        layer = torch.nn.Module()
        layer.num_experts = NUM_EXPERTS
        layer.num_local_experts = NUM_EXPERTS
        layer.moe_ep_rank = 0
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        method.create_weights(
            layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
    return layer


class TestNvFp4MoeWeightCheck(CustomTestCase):
    def test_the_checker_skips_the_input_scales_and_checks_the_weight_scales(self):
        params = dict(_nvfp4_moe_layer().named_parameters())

        for name in ("w13_input_scale", "w2_input_scale"):
            self.assertTrue(_is_skip_weight_check(name, params[name]), name)
        for name in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2"):
            self.assertFalse(_is_skip_weight_check(name, params[name]), name)


if __name__ == "__main__":
    unittest.main()
