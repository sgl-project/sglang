"""CPU regressions for A5 MXFP8 DeepEP low-latency dispatch wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DeepEPMode, DispatcherOutputDtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNPUMXFP8DeepEPDispatch(CustomTestCase):
    def test_mode_specific_dtype_selection(self):
        quant_config = {
            "normal_dispatcher_output_dtype": "bf16",
            "low_latency_dispatcher_output_dtype": "mxfp8",
        }
        common = dict(
            quant_config=quant_config,
        )
        with (
            patch.object(moe_utils, "get_server_args", return_value=None),
            patch.object(
                moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
        ):
            normal = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(**common, dispatch_mode=DeepEPMode.NORMAL)
            )
            low_latency = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(**common, dispatch_mode=DeepEPMode.LOW_LATENCY)
            )

        self.assertEqual(normal, DispatcherOutputDtype.BF16)
        self.assertEqual(low_latency, DispatcherOutputDtype.MXFP8)

    def test_mxfp8_config_selects_explicit_deepep_quant_mode(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        dispatcher.quant_config = {
            "low_latency_dispatcher_output_dtype": "mxfp8"
        }
        with patch.object(
            deepep, "get_deepep_output_dtype", return_value=DispatcherOutputDtype.MXFP8
        ):
            dispatcher.set_deepep_dispatcher_dtype()

        self.assertTrue(dispatcher.use_fp8)
        self.assertFalse(dispatcher.use_nvfp4)
        self.assertEqual(dispatcher.quant_mode, "mx_fp8_e4m3")

    def test_low_latency_forwards_explicit_quant_mode(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        dispatcher.quant_config = {}
        dispatcher.num_max_dispatch_tokens_per_rank = 64
        dispatcher.num_experts = 512
        dispatcher.use_fp8 = True
        dispatcher.use_nvfp4 = False
        dispatcher.quant_mode = "mx_fp8_e4m3"
        dispatcher.return_recv_hook = False

        event, hook = Mock(), Mock()
        buffer = Mock()
        buffer.low_latency_dispatch.return_value = (
            (Mock(), Mock()),
            Mock(),
            Mock(),
            event,
            hook,
        )
        dispatcher._get_buffer = Mock(return_value=buffer)

        hidden_states = torch.empty((8, 128), dtype=torch.bfloat16)
        topk_ids = torch.empty((8, 2), dtype=torch.int64)
        topk_weights = torch.empty((8, 2), dtype=torch.float32)
        with patch.object(deepep, "_deepep_precompile_tp_barrier"):
            dispatcher._dispatch_core(hidden_states, topk_ids, topk_weights)

        kwargs = buffer.low_latency_dispatch.call_args.kwargs
        self.assertEqual(kwargs["quant_mode"], "mx_fp8_e4m3")


if __name__ == "__main__":
    unittest.main()
