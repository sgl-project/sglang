"""CPU regressions for W4AFP8 DeepEP dispatcher dtypes."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    DispatcherOutputDtype,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization import w4afp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestW4AFP8DeepEPDispatcherDtype(CustomTestCase):
    def test_w4afp8_sets_mode_specific_dispatcher_dtypes(self):
        dispatcher = Mock()
        layer = SimpleNamespace(
            dispatcher=dispatcher,
            w2_weight=torch.empty(0),
            w13_weight_scale_inv=torch.ones((1, 1, 4)),
            w2_weight_scale_inv=torch.ones((1, 1, 4)),
            w13_input_scale=torch.ones(1),
            w2_input_scale=torch.ones(1),
        )

        w4afp8.W4AFp8MoEMethod(SimpleNamespace()).process_weights_after_loading(layer)

        dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "fp8",
            }
        )

    def test_mode_specific_dtype_selection(self):
        quant_config = {
            "normal_dispatcher_output_dtype": "bf16",
            "low_latency_dispatcher_output_dtype": "fp8",
        }

        with (
            patch.object(moe_utils, "get_server_args", return_value=None),
            patch.object(
                moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
            patch.object(
                moe_utils,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
        ):
            normal_dtype = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(
                    quant_config=quant_config,
                    dispatch_mode=DeepEPMode.NORMAL,
                )
            )
            low_latency_dtype = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(
                    quant_config=quant_config,
                    dispatch_mode=DeepEPMode.LOW_LATENCY,
                )
            )

        self.assertEqual(
            deepep._DeepEPDispatcherImplNormal.dispatch_mode, DeepEPMode.NORMAL
        )
        self.assertEqual(
            deepep._DeepEPDispatcherImplLowLatency.dispatch_mode,
            DeepEPMode.LOW_LATENCY,
        )
        self.assertEqual(normal_dtype, DispatcherOutputDtype.BF16)
        self.assertEqual(low_latency_dtype, DispatcherOutputDtype.FP8)

    def test_normal_rejects_fp8_and_preserves_empty_bf16(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        empty_topk_ids = torch.empty((0, 1), dtype=torch.int64)
        empty_topk_weights = torch.empty((0, 1), dtype=torch.float32)

        fp8_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.float8_e4m3fn),
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        with self.assertRaisesRegex(RuntimeError, "requires BF16"):
            method.apply_deepep_normal(SimpleNamespace(), fp8_dispatch_output)

        bf16_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.bfloat16),
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        output = method.apply_deepep_normal(SimpleNamespace(), bf16_dispatch_output)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (0, 128))

    def test_low_latency_requires_fp8_scales(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        dispatch_output = (
            torch.empty((1, 1, 128), dtype=torch.bfloat16),
            None,
            torch.empty((0, 1), dtype=torch.int64),
            torch.empty((0, 1), dtype=torch.float32),
            torch.zeros(1, dtype=torch.int32),
            0,
        )

        with self.assertRaisesRegex(RuntimeError, "requires FP8"):
            method.apply_deepep_ll(SimpleNamespace(), dispatch_output)


if __name__ == "__main__":
    unittest.main()
