"""CPU regressions for W4AFP8 DeepEP dispatcher dtypes."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    DispatcherOutputDtype,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization import w4afp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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

        with patch.object(w4afp8, "get_moe_a2a_backend", create=True) as backend:
            backend.return_value.is_deepep.return_value = True
            w4afp8.W4AFp8MoEMethod(SimpleNamespace()).process_weights_after_loading(
                layer
            )

        dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "fp8",
                "low_latency_dispatcher_output_dtype": "fp8",
            }
        )

    def test_w4afp8_preserves_bf16_normal_dtype_for_non_deepep(self):
        dispatcher = Mock()
        layer = SimpleNamespace(
            dispatcher=dispatcher,
            w2_weight=torch.empty(0),
            w13_weight_scale_inv=torch.ones((1, 1, 4)),
            w2_weight_scale_inv=torch.ones((1, 1, 4)),
            w13_input_scale=torch.ones(1),
            w2_input_scale=torch.ones(1),
        )

        with patch.object(w4afp8, "get_moe_a2a_backend", create=True) as backend:
            backend.return_value.is_deepep.return_value = False
            w4afp8.W4AFp8MoEMethod(SimpleNamespace()).process_weights_after_loading(
                layer
            )

        dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "fp8",
            }
        )

    def test_mode_specific_dtype_selection(self):
        quant_config = {
            "normal_dispatcher_output_dtype": "fp8",
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
        self.assertEqual(normal_dtype, DispatcherOutputDtype.FP8)
        self.assertEqual(low_latency_dtype, DispatcherOutputDtype.FP8)

    def test_static_scale_is_taken_from_current_w4afp8_deepep_layer(self):
        scale = torch.ones(1, dtype=torch.float32)
        layer = SimpleNamespace(use_w4afp8=True, w13_input_scale=scale)

        with patch("sglang.srt.layers.moe.ep_moe.layer.get_moe_a2a_backend") as backend:
            backend.return_value.is_deepep.return_value = True
            actual = DeepEPMoE._get_normal_dispatch_static_scale(layer)

        self.assertIs(actual, scale)

    def test_static_scale_is_none_outside_target_path(self):
        scale = torch.ones(1, dtype=torch.float32)
        layer = SimpleNamespace(use_w4afp8=True, w13_input_scale=scale)

        with patch("sglang.srt.layers.moe.ep_moe.layer.get_moe_a2a_backend") as backend:
            backend.return_value.is_deepep.return_value = False
            self.assertIsNone(DeepEPMoE._get_normal_dispatch_static_scale(layer))

        layer.use_w4afp8 = False
        with patch("sglang.srt.layers.moe.ep_moe.layer.get_moe_a2a_backend") as backend:
            backend.return_value.is_deepep.return_value = True
            self.assertIsNone(DeepEPMoE._get_normal_dispatch_static_scale(layer))

    def test_forward_impl_passes_static_scale_to_dispatcher(self):
        hidden_states = Mock()
        topk_output = Mock()
        scale = Mock()
        dispatch_output = Mock()
        combine_input = Mock()
        dispatcher = Mock()
        dispatcher.dispatch.return_value = dispatch_output
        layer = SimpleNamespace(
            deprecate_flag=False,
            dispatcher=dispatcher,
            run_moe_core=Mock(return_value=combine_input),
            _get_normal_dispatch_static_scale=Mock(return_value=scale),
        )

        DeepEPMoE.forward_impl(layer, hidden_states, topk_output)

        dispatcher.dispatch.assert_called_once_with(
            hidden_states=hidden_states,
            topk_output=topk_output,
            static_scale=scale,
        )
        dispatcher.combine.assert_called_once_with(combine_input=combine_input)

    def test_tbo_forwards_only_non_none_static_scale(self):
        wrapper = SimpleNamespace(_execute=Mock(return_value=Mock()))
        scale = torch.ones(1, dtype=torch.float32)

        MaybeTboDeepEPDispatcher.dispatch(
            wrapper,
            static_scale=scale,
            hidden_states=Mock(),
            topk_output=Mock(),
        )
        self.assertIs(wrapper._execute.call_args.kwargs["static_scale"], scale)

        wrapper._execute.reset_mock()
        MaybeTboDeepEPDispatcher.dispatch(
            wrapper,
            static_scale=None,
            hidden_states=Mock(),
            topk_output=Mock(),
        )
        self.assertNotIn("static_scale", wrapper._execute.call_args.kwargs)

    def test_normal_dispatch_a_uses_static_per_tensor_fp8_before_comm(self):
        impl = object.__new__(deepep._DeepEPDispatcherImplNormal)
        impl.async_finish = False
        impl.use_fp8 = True
        hidden_states = torch.ones((2, 8), dtype=torch.bfloat16)
        scale = torch.ones(1, dtype=torch.float32)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=torch.zeros((2, 1), dtype=torch.int64),
            router_logits=None,
        )

        with (
            patch.object(deepep, "per_tensor_quant_fp8", create=True) as quant,
            patch.object(deepep.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False),
        ):
            state = impl.dispatch_a(
                hidden_states,
                topk_output,
                static_scale=scale,
            )

        quant.assert_called_once_with(hidden_states, state[0], scale, True)
        self.assertEqual(state[0].dtype, torch.float8_e4m3fn)
        self.assertFalse(isinstance(state[0], tuple))

    def test_normal_dispatch_a_preserves_bf16_without_static_scale(self):
        impl = object.__new__(deepep._DeepEPDispatcherImplNormal)
        impl.async_finish = False
        impl.use_fp8 = False
        hidden_states = torch.ones((2, 8), dtype=torch.bfloat16)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=torch.zeros((2, 1), dtype=torch.int64),
            router_logits=None,
        )

        state = impl.dispatch_a(
            hidden_states,
            topk_output,
            static_scale=None,
        )

        self.assertIs(state[0], hidden_states)

    def test_normal_requires_static_fp8_and_preserves_empty_bf16(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        empty_topk_ids = torch.empty((0, 1), dtype=torch.int64)
        empty_topk_weights = torch.empty((0, 1), dtype=torch.float32)

        empty_fp8_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.float8_e4m3fn),
            hidden_states_scale=None,
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        output = method.apply_deepep_normal(
            SimpleNamespace(), empty_fp8_dispatch_output
        )
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (0, 128))

        empty_bf16_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        output = method.apply_deepep_normal(
            SimpleNamespace(), empty_bf16_dispatch_output
        )
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (0, 128))

        nonempty_bf16_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((1, 128), dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=torch.zeros((1, 1), dtype=torch.int64),
            topk_weights=torch.ones((1, 1), dtype=torch.float32),
        )
        with (
            patch.object(w4afp8, "get_moe_a2a_backend", create=True) as backend,
            self.assertRaisesRegex(RuntimeError, "requires static FP8"),
        ):
            backend.return_value.is_deepep.return_value = True
            method.apply_deepep_normal(SimpleNamespace(), nonempty_bf16_dispatch_output)

    def test_normal_rejects_fp8_with_per_token_group_scales(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((1, 128), dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.ones((1, 1), dtype=torch.float32),
            topk_ids=torch.zeros((1, 1), dtype=torch.int64),
            topk_weights=torch.ones((1, 1), dtype=torch.float32),
        )

        with (
            patch.object(w4afp8, "get_moe_a2a_backend", create=True) as backend,
            self.assertRaisesRegex(RuntimeError, "without per-token-group scales"),
        ):
            backend.return_value.is_deepep.return_value = True
            method.apply_deepep_normal(SimpleNamespace(), dispatch_output)

    def test_non_deepep_normal_preserves_bf16_cutlass_path(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        for name in (
            "a_strides1",
            "b_strides1",
            "c_strides1",
            "a_strides2",
            "b_strides2",
            "c_strides2",
            "s_strides13",
            "s_strides2",
            "expert_offsets",
            "problem_sizes1",
            "problem_sizes2",
        ):
            setattr(method, name, Mock())
        layer = SimpleNamespace(
            w13_weight=Mock(),
            w2_weight=Mock(),
            w13_weight_scale_inv=Mock(),
            w2_weight_scale_inv=Mock(),
            w13_input_scale=Mock(),
            w2_input_scale=Mock(),
        )
        dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((1, 128), dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=torch.zeros((1, 1), dtype=torch.int64),
            topk_weights=torch.ones((1, 1), dtype=torch.float32),
        )
        expected = torch.empty((1, 128), dtype=torch.bfloat16)

        with (
            patch.object(w4afp8, "get_moe_a2a_backend", create=True) as backend,
            patch(
                "sglang.srt.layers.moe.cutlass_w4a8_moe."
                "cutlass_w4a8_moe_deepep_normal",
                return_value=expected,
            ),
        ):
            backend.return_value.is_deepep.return_value = False
            actual = method.apply_deepep_normal(layer, dispatch_output)

        self.assertIs(actual, expected)

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
