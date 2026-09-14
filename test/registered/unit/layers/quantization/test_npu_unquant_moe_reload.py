import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.hardware_backend.npu.quantization.moe_methods as npu_moe_methods
import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNPUUnquantMoEReload(CustomTestCase):
    def test_postprocess_preserves_canonical_reload_layout(self):
        num_experts, hidden_size, intermediate_size = 1, 8, 3
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros(num_experts, 2 * intermediate_size, hidden_size),
                requires_grad=False,
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, intermediate_size),
                requires_grad=False,
            ),
            w13_kernel=npu_moe_methods.NPUUnquantMoEMethod(),
            w2_kernel=npu_moe_methods.NPUUnquantMoEMethod(),
            dispatcher=MagicMock(),
        )
        backend = SimpleNamespace(is_flashinfer_cutlass=lambda: False)

        with (
            patch.object(unquant, "_is_cpu", False),
            patch.object(unquant, "_is_npu", True),
            patch.object(unquant, "_use_aiter", False),
            patch.object(unquant, "get_moe_runner_backend", return_value=backend),
            patch.object(
                npu_moe_methods,
                "npu_format_cast",
                side_effect=lambda tensor: tensor.detach().clone(),
            ),
        ):
            method = unquant.UnquantizedFusedMoEMethod()
            method.process_weights_after_loading(layer)

        self.assertEqual(
            tuple(layer.w13_weight.shape),
            (num_experts, 2 * intermediate_size, hidden_size),
        )
        self.assertEqual(
            tuple(layer.w2_weight.shape),
            (num_experts, hidden_size, intermediate_size),
        )

        loader = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=True),
            quant_method=SimpleNamespace(load_up_proj_weight_first=False),
            use_padded_loading=False,
            use_presharded_weights=False,
            use_triton_kernels=False,
        )
        gate = torch.full((intermediate_size, hidden_size), 1.0)
        up = torch.full((intermediate_size, hidden_size), 2.0)
        down = torch.full((hidden_size, intermediate_size), 3.0)

        FusedMoE._load_w13(loader, layer.w13_weight[0], 0, "w1", gate, 0)
        FusedMoE._load_w13(loader, layer.w13_weight[0], 0, "w3", up, 0)
        FusedMoE._load_w2(loader, layer.w2_weight[0], 1, "w2", down, 0)

        torch.testing.assert_close(layer.w13_weight[0, :intermediate_size], gate)
        torch.testing.assert_close(layer.w13_weight[0, intermediate_size:], up)
        torch.testing.assert_close(layer.w2_weight[0], down)


if __name__ == "__main__":
    unittest.main()
