"""Unit tests for the experimental DSV4 ROCm multi-stream schedule."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4RocmMultiStream(CustomTestCase):
    @staticmethod
    def _layer(**overrides):
        values = {
            "alt_streams": [object(), object()],
            "compressor": object(),
            "compress_ratio": 4,
            "fuse_wqa_wkv": True,
            "use_fused_qk_norm_rope": True,
            "attn_tp_size": 1,
            "n_local_heads": 128,
            "dsa_enable_prefill_cp": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _enabled(
        layer,
        mode,
        rows=8,
        *,
        breakable=False,
        moe_ep_size=1,
        parallel_overrides=None,
        use_cp=False,
    ):
        forward_batch = SimpleNamespace(forward_mode=mode)
        parallel_values = dict(
            tp_size=8,
            pp_size=1,
            attn_dp_size=8,
            attn_tp_size=1,
            moe_ep_size=moe_ep_size,
            moe_tp_size=8,
        )
        parallel_values.update(parallel_overrides or {})
        parallel = SimpleNamespace(**parallel_values)
        x = torch.empty(rows, 16)
        with (
            envs.SGLANG_DSV4_ROCM_ATTN_MULTI_STREAM.override(True),
            envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.override(True),
            envs.SGLANG_HACK_FLASHMLA_BACKEND.override("unified_kv_triton"),
            patch.object(deepseek_v4, "_is_hip", True),
            patch.object(deepseek_v4, "_is_gfx95_supported", True),
            patch.object(deepseek_v4, "_use_aiter", True),
            patch.object(deepseek_v4, "get_parallel", return_value=parallel),
            patch.object(deepseek_v4, "get_is_capture_mode", return_value=True),
            patch.object(
                deepseek_v4,
                "is_in_breakable_cuda_graph",
                return_value=breakable,
            ),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=use_cp),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=True,
            ),
        ):
            return deepseek_v4.MQALayer._use_hip_after_shared_multi_stream(
                layer, x, forward_batch
            )

    def test_defaults_off_and_accepts_only_validated_rows(self):
        layer = self._layer()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        with (
            envs.SGLANG_DSV4_ROCM_ATTN_MULTI_STREAM.override(False),
            patch.object(deepseek_v4, "_is_hip", True),
        ):
            self.assertFalse(
                deepseek_v4.MQALayer._use_hip_after_shared_multi_stream(
                    layer, torch.empty(8, 16), forward_batch
                )
            )

        self.assertTrue(self._enabled(layer, ForwardMode.TARGET_VERIFY, rows=4))
        self.assertTrue(self._enabled(layer, ForwardMode.TARGET_VERIFY, rows=8))
        self.assertFalse(self._enabled(layer, ForwardMode.TARGET_VERIFY, rows=3))
        self.assertFalse(self._enabled(layer, ForwardMode.TARGET_VERIFY, rows=16))

    def test_rejects_unvalidated_runtime_configurations(self):
        layer = self._layer()
        self.assertFalse(self._enabled(layer, ForwardMode.DECODE))
        self.assertFalse(
            self._enabled(self._layer(compressor=None), ForwardMode.TARGET_VERIFY)
        )
        self.assertFalse(
            self._enabled(self._layer(compress_ratio=128), ForwardMode.TARGET_VERIFY)
        )
        self.assertFalse(
            self._enabled(self._layer(attn_tp_size=2), ForwardMode.TARGET_VERIFY)
        )
        self.assertFalse(
            self._enabled(
                self._layer(n_local_heads=64), ForwardMode.TARGET_VERIFY, rows=64
            )
        )
        self.assertFalse(
            self._enabled(
                self._layer(dsa_enable_prefill_cp=True),
                ForwardMode.TARGET_VERIFY,
                use_cp=True,
            )
        )
        self.assertFalse(
            self._enabled(layer, ForwardMode.TARGET_VERIFY, breakable=True)
        )

        self.assertFalse(
            self._enabled(
                layer,
                ForwardMode.TARGET_VERIFY,
                moe_ep_size=8,
            )
        )

        for topology_override in (
            {"tp_size": 4, "attn_dp_size": 4, "moe_tp_size": 4},
            {"pp_size": 2},
            {"attn_dp_size": 4, "attn_tp_size": 2},
        ):
            with self.subTest(topology_override=topology_override):
                self.assertFalse(
                    self._enabled(
                        layer,
                        ForwardMode.TARGET_VERIFY,
                        parallel_overrides=topology_override,
                    )
                )

    def test_dedicated_flag_disables_generic_rocm_fallback(self):
        layer = self._layer(
            _multi_stream_bs_limit=64,
            dsa_enable_prefill_cp=False,
        )
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        with (
            envs.SGLANG_DSV4_ROCM_ATTN_MULTI_STREAM.override(True),
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.override(True),
            patch.object(deepseek_v4, "_is_hip", True),
            patch.object(deepseek_v4, "get_is_capture_mode", return_value=True),
            patch.object(deepseek_v4, "is_in_breakable_cuda_graph", return_value=False),
        ):
            enabled = deepseek_v4.MQALayer._use_generic_multi_stream(
                layer,
                torch.empty(16, 16),
                forward_batch,
            )

        self.assertFalse(enabled)

    def test_after_shared_helper_waits_for_side_and_tracks_outputs(self):
        expected_q = torch.empty(1)
        expected_kv = torch.empty(1)
        call_order = []
        main_stream = Mock()
        side_stream = Mock()
        shared_ready = Mock()
        compressor = object()
        layer = SimpleNamespace(
            alt_streams=[side_stream],
            compressor=compressor,
            compress_ratio=4,
            layer_id=7,
            _forward_prepare=Mock(
                side_effect=lambda *args, **kwargs: (
                    call_order.append("side"),
                    (expected_q, expected_kv),
                )[1]
            ),
        )
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        backend = SimpleNamespace(
            forward_core_compressor=Mock(
                side_effect=lambda *args, **kwargs: call_order.append("core")
            )
        )

        with (
            patch.object(torch.cuda, "current_stream", return_value=main_stream),
            patch.object(torch.cuda, "stream", return_value=MagicMock()),
            patch.object(torch.cuda, "Event", return_value=shared_ready),
            patch.object(torch.Tensor, "record_stream") as record_stream,
        ):
            actual = (
                deepseek_v4.MQALayer._forward_prepare_multi_stream_hip_after_shared(
                    layer,
                    object(),
                    object(),
                    forward_batch,
                    backend,
                    object(),
                    x_quant=object(),
                )
            )

        self.assertIs(actual[0], expected_q)
        self.assertIs(actual[1], expected_kv)
        self.assertEqual(call_order, ["side", "core"])
        side_stream.wait_stream.assert_called_once_with(main_stream)
        main_stream.wait_event.assert_called_once_with(shared_ready)
        main_stream.wait_stream.assert_called_once_with(side_stream)
        self.assertEqual(
            record_stream.call_args_list, [call(main_stream), call(main_stream)]
        )
        backend.forward_core_compressor.assert_called_once()
        layer._forward_prepare.assert_called_once()
        helper_kwargs = layer._forward_prepare.call_args.kwargs
        self.assertTrue(helper_kwargs["skip_core_compressor"])
        self.assertIs(helper_kwargs["core_compressor_ready"], shared_ready)

    def test_after_shared_helper_joins_side_stream_on_failure(self):
        main_stream = Mock()
        side_stream = Mock()
        layer = SimpleNamespace(
            alt_streams=[side_stream],
            compressor=object(),
            compress_ratio=4,
            _forward_prepare=Mock(side_effect=RuntimeError("capture failed")),
        )

        with (
            patch.object(torch.cuda, "current_stream", return_value=main_stream),
            patch.object(torch.cuda, "stream", return_value=MagicMock()),
            patch.object(torch.cuda, "Event", return_value=Mock()),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            deepseek_v4.MQALayer._forward_prepare_multi_stream_hip_after_shared(
                layer,
                object(),
                object(),
                SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
                object(),
            )

        side_stream.wait_stream.assert_called_once_with(main_stream)
        main_stream.wait_stream.assert_called_once_with(side_stream)


if __name__ == "__main__":
    unittest.main()
