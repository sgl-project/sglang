"""Regression contracts for DeepSeek/GLM ModelSlim shared experts on NPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _run_weight_loader_with_context,
)
from sglang.srt.models.deepseek_v2 import _get_shared_expert_fp8_block_size
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestModelSlimSharedExpertsNPU(CustomTestCase):
    def test_npu_mxfp8_does_not_require_generic_quant_config(self):
        # ModelSlimLinearMethod intentionally has no ``quant_config`` member.
        modelslim_method = SimpleNamespace()
        with patch.object(deepseek_v2, "_is_npu", True):
            block_size = _get_shared_expert_fp8_block_size(
                modelslim_method, modelslim_method
            )

        self.assertIsNone(block_size)

    def test_non_npu_preserves_matching_block_size_validation(self):
        gate_up = SimpleNamespace(
            quant_config=SimpleNamespace(weight_block_size=[128, 128])
        )
        down = SimpleNamespace(
            quant_config=SimpleNamespace(weight_block_size=[128, 128])
        )
        with patch.object(deepseek_v2, "_is_npu", False):
            block_size = _get_shared_expert_fp8_block_size(gate_up, down)

        self.assertEqual(block_size, [128, 128])

    def test_async_weight_loader_error_preserves_tensor_context(self):
        class FailingColumnLoader:
            tp_rank = 2
            tp_size = 4
            scheme = SimpleNamespace()
            quant_method = SimpleNamespace()

            def load(self, param, loaded_weight):
                raise ValueError("narrow failed")

        param = SimpleNamespace(
            shape=(6144, 2048),
            dtype="float8_e4m3fn",
            output_dim=0,
            input_dim=1,
        )
        loaded_weight = SimpleNamespace(
            shape=(16384, 2048), dtype="float8_e4m3fn"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint tensor 'model.layers.0.self_attn.q_b_proj.weight'.*"
            "param_shape=\\(6144, 2048\\).*loaded_shape=\\(16384, 2048\\).*"
            "tp_rank=2, tp_size=4",
        ) as error:
            _run_weight_loader_with_context(
                FailingColumnLoader().load,
                param,
                loaded_weight,
                checkpoint_name="model.layers.0.self_attn.q_b_proj.weight",
                parameter_name="model.layers.0.self_attn.q_b_proj.weight",
            )

        self.assertIsInstance(error.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
