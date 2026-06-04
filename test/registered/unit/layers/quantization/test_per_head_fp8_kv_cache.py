"""Unit tests for per-head FP8 KV cache scale handling."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPerHeadFP8KVCacheScales(CustomTestCase):
    @staticmethod
    def _method(strategy=None):
        scheme = None if strategy is None else {"strategy": strategy}
        return BaseKVCacheMethod(SimpleNamespace(kv_cache_scheme=scheme))

    @staticmethod
    def _layer():
        return SimpleNamespace(
            tp_k_head_num=2,
            is_cross_attention=False,
            attn_type=SimpleNamespace(value="decoder"),
            head_dim=128,
        )

    def test_per_tensor_scales_remain_scalar(self):
        layer = self._layer()
        self._method().create_weights(layer)

        self.assertEqual(layer.k_scale.shape, torch.Size([]))
        self.assertTrue(layer.k_scale._skip_weight_check)
        self.assertTrue(layer.v_scale._skip_weight_check)

    @patch("sglang.srt.server_args.get_global_server_args")
    def test_per_head_scales_are_loaded_and_duplicated(self, get_server_args):
        get_server_args.return_value.get_attention_backends.return_value = (
            "fa3",
            "fa3",
        )
        layer = self._layer()
        method = self._method("attn_head")
        method.create_weights(layer)

        self.assertEqual(layer.k_scale.shape, torch.Size([2]))
        layer.k_scale.weight_loader(layer.k_scale, torch.tensor([0.25, 0.5]))
        method.process_weights_after_loading(layer)

        expected = torch.tensor([0.25, 0.5])
        torch.testing.assert_close(layer.k_scale_float, expected)
        torch.testing.assert_close(layer.v_scale_float, expected)

    @patch("sglang.srt.server_args.get_global_server_args")
    def test_per_head_scales_reject_unsupported_backend(self, get_server_args):
        get_server_args.return_value.get_attention_backends.return_value = (
            "triton",
            "fa3",
        )

        with self.assertRaisesRegex(ValueError, "only supported with"):
            self._method("attn_head").create_weights(self._layer())
