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

_QUANT_MODULE = "sglang.kernels.ops.quantization.per_tensor_quant_fp8"


def _make_layer(dispatcher):
    return SimpleNamespace(
        dispatcher=dispatcher,
        w2_weight=torch.empty(0),
        w13_weight_scale_inv=torch.ones((1, 1, 4)),
        w2_weight_scale_inv=torch.ones((1, 1, 4)),
        w13_input_scale=torch.ones(1),
        w2_input_scale=torch.ones(1),
    )


class TestW4AFP8DeepEPDispatcherDtype(CustomTestCase):
    def test_w4afp8_sets_mode_specific_dispatcher_dtypes(self):
        """DeepEP backend: BF16 a2a + the static scale for the FP8-view hack."""
        dispatcher = Mock()
        layer = _make_layer(dispatcher)

        with patch("sglang.srt.layers.moe.get_moe_a2a_backend") as backend:
            backend.return_value.is_deepep.return_value = True
            w4afp8.W4AFp8MoEMethod(SimpleNamespace()).process_weights_after_loading(
                layer
            )

        dispatcher.set_quant_config.assert_called_once()
        quant_config = dispatcher.set_quant_config.call_args.args[0]
        self.assertEqual(quant_config["normal_dispatcher_output_dtype"], "bf16")
        self.assertEqual(quant_config["low_latency_dispatcher_output_dtype"], "fp8")
        # The static scale handed to the dispatcher is the collapsed one.
        self.assertIs(quant_config["normal_static_fp8_scale"], layer.w13_input_scale)
        self.assertEqual(quant_config["normal_static_fp8_scale"].dtype, torch.float32)

    def test_w4afp8_omits_static_scale_for_non_deepep(self):
        dispatcher = Mock()
        layer = _make_layer(dispatcher)

        with patch("sglang.srt.layers.moe.get_moe_a2a_backend") as backend:
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

    def test_normal_accepts_bf16_and_static_fp8(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        empty_topk_ids = torch.empty((0, 1), dtype=torch.int64)
        empty_topk_weights = torch.empty((0, 1), dtype=torch.float32)

        bf16_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.bfloat16),
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        output = method.apply_deepep_normal(SimpleNamespace(), bf16_dispatch_output)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (0, 128))

        fp8_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.float8_e4m3fn),
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        output = method.apply_deepep_normal(SimpleNamespace(), fp8_dispatch_output)
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.shape, (0, 128))

        fp32_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.float32),
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        with self.assertRaisesRegex(RuntimeError, "requires BF16"):
            method.apply_deepep_normal(SimpleNamespace(), fp32_dispatch_output)

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


class TestStaticFp8NormalDispatch(CustomTestCase):
    """The sender quantizes per-tensor and ships FP8 in a BF16 view."""

    def _make_impl(self, scale):
        impl = object.__new__(deepep._DeepEPDispatcherImplNormal)
        impl.async_finish = False
        impl.use_fp8 = False
        impl.quant_config = (
            {"normal_static_fp8_scale": scale} if scale is not None else {}
        )
        return impl

    @staticmethod
    def _fake_quant(input, output_q, scale, is_static):
        output_q.copy_((input.float() * scale.float()).to(torch.float8_e4m3fn))

    def test_dispatch_a_quantizes_and_views_as_bf16(self):
        scale = torch.ones(1)
        impl = self._make_impl(scale)
        hidden_states = torch.ones((2, 8), dtype=torch.bfloat16)
        topk_output = SimpleNamespace(
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=torch.zeros((2, 1), dtype=torch.int64),
        )

        with patch(f"{_QUANT_MODULE}.per_tensor_quant_fp8") as quant_mock:
            hidden_out, topk_ids, topk_weights, previous_event = impl.dispatch_a(
                hidden_states, topk_output
            )

        quant_mock.assert_called_once()
        quant_input, quant_output, quant_scale, _ = quant_mock.call_args.args
        self.assertIs(quant_input, hidden_states)
        self.assertEqual(quant_output.dtype, torch.float8_e4m3fn)
        self.assertEqual(quant_output.shape, hidden_states.shape)
        self.assertIs(quant_scale, scale)
        # The payload shipped to DeepEP is a BF16 view with half the columns.
        self.assertEqual(hidden_out.dtype, torch.bfloat16)
        self.assertEqual(hidden_out.shape, (2, 4))
        self.assertIsNone(previous_event)

    def test_dispatch_a_without_static_scale_keeps_bf16(self):
        impl = self._make_impl(scale=None)
        hidden_states = torch.ones((2, 8), dtype=torch.bfloat16)
        topk_output = SimpleNamespace(
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=torch.zeros((2, 1), dtype=torch.int64),
        )

        hidden_out, _, _, _ = impl.dispatch_a(hidden_states, topk_output)

        self.assertIs(hidden_out, hidden_states)
        self.assertEqual(hidden_out.dtype, torch.bfloat16)
        self.assertEqual(hidden_out.shape, (2, 8))

    def test_dispatch_b_restores_fp8_view(self):
        impl = self._make_impl(torch.ones(1))
        bf16_view = torch.ones((3, 4), dtype=torch.bfloat16)
        recv = (bf16_view, None, None, "counts", Mock())

        with patch.object(impl, "_dispatch_core", return_value=recv):
            output = impl.dispatch_b(bf16_view, None, None, None)

        self.assertEqual(output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.hidden_states.shape, (3, 8))
        self.assertIsNone(output.hidden_states_scale)

    def test_dispatch_b_without_static_scale_passes_through(self):
        impl = self._make_impl(scale=None)
        bf16_recv = torch.ones((3, 8), dtype=torch.bfloat16)
        recv = (bf16_recv, None, None, "counts", Mock())

        with patch.object(impl, "_dispatch_core", return_value=recv):
            output = impl.dispatch_b(bf16_recv, None, None, None)

        self.assertIs(output.hidden_states, bf16_recv)
        self.assertIsNone(output.hidden_states_scale)

    def test_roundtrip_preserves_fp8_bytes(self):
        """dispatch_a's BF16 view bit-exactly restores to FP8 in dispatch_b."""
        scale = torch.full((1,), 0.02)
        impl = self._make_impl(scale)
        hidden_states = torch.randn((4, 8), dtype=torch.bfloat16)
        topk_output = SimpleNamespace(
            topk_weights=torch.ones((4, 1), dtype=torch.float32),
            topk_ids=torch.zeros((4, 1), dtype=torch.int64),
        )
        expected_fp8 = (hidden_states.float() * scale.float()).to(torch.float8_e4m3fn)

        with patch(
            f"{_QUANT_MODULE}.per_tensor_quant_fp8", side_effect=self._fake_quant
        ):
            payload, _, _, _ = impl.dispatch_a(hidden_states, topk_output)
        self.assertEqual(payload.dtype, torch.bfloat16)

        recv = (payload, None, None, "counts", Mock())
        with patch.object(impl, "_dispatch_core", return_value=recv):
            output = impl.dispatch_b(payload, None, None, None)

        self.assertEqual(output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.hidden_states.shape, hidden_states.shape)
        self.assertTrue(
            torch.equal(
                output.hidden_states.view(torch.uint8), expected_fp8.view(torch.uint8)
            )
        )


if __name__ == "__main__":
    unittest.main()
