"""Unit tests for the experimental DSV4 ROCm multi-stream schedule."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.kernels.ops.attention.nsa_triton_decode.triton_mla_kernels_decode_fused import (
    SplitKBufferPool,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4RocmMultiStream(CustomTestCase):
    def tearDown(self):
        SplitKBufferPool.clear()

    @staticmethod
    def _layer(**overrides):
        values = {
            "alt_streams": [object(), object()],
            "compressor": object(),
            "indexer": object(),
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
        *,
        use_cp=False,
        capture=True,
        is_hip=True,
    ):
        forward_batch = SimpleNamespace(forward_mode=mode)
        with (
            envs.SGLANG_DSV4_ROCM_ATTN_MULTI_STREAM.override(True),
            patch.object(deepseek_v4, "_is_hip", is_hip),
            patch.object(deepseek_v4, "get_is_capture_mode", return_value=capture),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=use_cp),
        ):
            return deepseek_v4.MQALayer._use_hip_multi_stream(layer, forward_batch)

    def test_user_flag_controls_decode_and_verify(self):
        layer = self._layer()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        with (
            envs.SGLANG_DSV4_ROCM_ATTN_MULTI_STREAM.override(False),
            patch.object(deepseek_v4, "_is_hip", True),
        ):
            self.assertFalse(
                deepseek_v4.MQALayer._use_hip_multi_stream(layer, forward_batch)
            )

        self.assertTrue(self._enabled(layer, ForwardMode.DECODE))
        self.assertTrue(self._enabled(layer, ForwardMode.TARGET_VERIFY))
        self.assertTrue(
            self._enabled(
                self._layer(
                    attn_tp_size=8,
                    n_local_heads=16,
                    fuse_wqa_wkv=False,
                    use_fused_qk_norm_rope=False,
                ),
                ForwardMode.DECODE,
            )
        )

    def test_rejects_only_unsafe_runtime_configurations(self):
        layer = self._layer()
        self.assertFalse(
            self._enabled(self._layer(compressor=None), ForwardMode.DECODE)
        )
        self.assertFalse(
            self._enabled(
                self._layer(indexer=None, compress_ratio=128),
                ForwardMode.DECODE,
            )
        )
        self.assertFalse(
            self._enabled(self._layer(alt_streams=None), ForwardMode.DECODE)
        )
        self.assertFalse(
            self._enabled(self._layer(alt_streams=[object()]), ForwardMode.DECODE)
        )
        self.assertFalse(self._enabled(layer, ForwardMode.DECODE, capture=False))
        self.assertFalse(self._enabled(layer, ForwardMode.DECODE, is_hip=False))
        self.assertFalse(
            self._enabled(
                self._layer(dsa_enable_prefill_cp=True),
                ForwardMode.DECODE,
                use_cp=True,
            )
        )
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode):
                self.assertFalse(self._enabled(layer, mode))

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

    def test_hip_helper_launches_compressors_before_prepare(self):
        expected_q = torch.empty(1)
        expected_kv = torch.empty(1)
        q_rope_out = object()
        k_nope_out = object()
        k_rope_out = object()
        call_order = []
        main_stream = Mock()
        core_stream = Mock()
        indexer_stream = Mock()
        compressor = object()
        indexer_compressor = object()
        layer = SimpleNamespace(
            alt_streams=[core_stream, indexer_stream],
            compressor=compressor,
            indexer=SimpleNamespace(layer_id=9, compressor=indexer_compressor),
            compress_ratio=4,
            layer_id=7,
            _forward_prepare=Mock(
                side_effect=lambda *args, **kwargs: (
                    call_order.append("prepare"),
                    (expected_q, expected_kv),
                )[1]
            ),
        )
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        backend = SimpleNamespace(
            forward_core_compressor=Mock(
                side_effect=lambda *args, **kwargs: call_order.append("core")
            ),
            forward_indexer_compressor=Mock(
                side_effect=lambda *args, **kwargs: call_order.append("indexer")
            ),
        )

        with (
            patch.object(torch.cuda, "current_stream", return_value=main_stream),
            patch.object(torch.cuda, "stream", return_value=MagicMock()),
        ):
            actual = deepseek_v4.MQALayer._forward_prepare_multi_stream_hip(
                layer,
                object(),
                object(),
                forward_batch,
                backend,
                object(),
                x_quant=object(),
                q_rope_out=q_rope_out,
                k_nope_out=k_nope_out,
                k_rope_out=k_rope_out,
            )

        self.assertIs(actual[0], expected_q)
        self.assertIs(actual[1], expected_kv)
        self.assertEqual(call_order, ["core", "indexer", "prepare"])
        core_stream.wait_stream.assert_called_once_with(main_stream)
        indexer_stream.wait_stream.assert_called_once_with(main_stream)
        backend.forward_core_compressor.assert_called_once()
        backend.forward_indexer_compressor.assert_called_once()
        layer._forward_prepare.assert_called_once()
        helper_kwargs = layer._forward_prepare.call_args.kwargs
        self.assertIs(helper_kwargs["q_rope_out"], q_rope_out)
        self.assertIs(helper_kwargs["k_nope_out"], k_nope_out)
        self.assertIs(helper_kwargs["k_rope_out"], k_rope_out)
        self.assertTrue(helper_kwargs["skip_core_compressor"])
        self.assertTrue(helper_kwargs["skip_indexer_compressor"])
        self.assertEqual(
            helper_kwargs["pre_indexer_streams"], [core_stream, indexer_stream]
        )

    def test_hip_helper_joins_side_streams_on_failure(self):
        main_stream = Mock()
        core_stream = Mock()
        indexer_stream = Mock()
        layer = SimpleNamespace(
            alt_streams=[core_stream, indexer_stream],
            compressor=object(),
            indexer=SimpleNamespace(layer_id=9, compressor=object()),
            compress_ratio=4,
            layer_id=7,
            _forward_prepare=Mock(side_effect=RuntimeError("capture failed")),
        )
        backend = SimpleNamespace(
            forward_core_compressor=Mock(),
            forward_indexer_compressor=Mock(),
        )

        with (
            patch.object(torch.cuda, "current_stream", return_value=main_stream),
            patch.object(torch.cuda, "stream", return_value=MagicMock()),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            deepseek_v4.MQALayer._forward_prepare_multi_stream_hip(
                layer,
                object(),
                object(),
                SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
                backend,
            )

        core_stream.wait_stream.assert_called_once_with(main_stream)
        indexer_stream.wait_stream.assert_called_once_with(main_stream)
        self.assertEqual(
            main_stream.wait_stream.call_args_list,
            [call(core_stream), call(indexer_stream)],
        )

    def test_split_k_buffers_are_isolated_by_stream(self):
        stream_a = SimpleNamespace(cuda_stream=11)
        stream_b = SimpleNamespace(cuda_stream=12)
        allocated = []

        def fake_empty(*shape, **kwargs):
            tensor = Mock()
            tensor.shape = shape
            tensor.stride.return_value = tuple(range(len(shape), 0, -1))
            allocated.append(tensor)
            return tensor

        with (
            patch.object(
                torch.cuda,
                "current_stream",
                side_effect=[stream_a, stream_a, stream_b],
            ),
            patch.object(torch, "empty", side_effect=fake_empty),
        ):
            first = SplitKBufferPool.get_buffers(2, 8, 16, 512, torch.device("cuda"))
            same_stream = SplitKBufferPool.get_buffers(
                2, 8, 16, 512, torch.device("cuda")
            )
            other_stream = SplitKBufferPool.get_buffers(
                2, 8, 16, 512, torch.device("cuda")
            )

        self.assertIs(first, same_stream)
        self.assertIsNot(first, other_stream)
        self.assertEqual(len(allocated), 4)


if __name__ == "__main__":
    unittest.main()
