"""Unit tests for srt/model_executor/model_runner_components/load_model_utils.py"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_kv_cache_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER_NAME = "sglang.srt.model_executor.model_runner_components.load_model_utils"
FALLBACK_WARNING_SNIPPET = "no scaling factors"


def _server_args(kv_cache_dtype="fp8_e4m3", quantization_param_path=None):
    return SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
    )


def _attn_layer(k_scale_float=None, v_scale_float=None):
    layer = RadixAttention(
        num_heads=1, head_dim=8, scaling=1.0, num_kv_heads=1, layer_id=0
    )
    # Mimic BaseKVCacheMethod.process_weights_after_loading, which sets these
    # floats after weight load (1.0 = no checkpoint scales, else loaded values).
    layer.k_scale_float = k_scale_float
    layer.v_scale_float = v_scale_float
    return layer


class _Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class TestLoadKVCacheScales(CustomTestCase):
    def test_warns_when_scales_fell_back_to_default(self):
        model = _Model([_attn_layer(1.0, 1.0), _attn_layer(1.0, 1.0)])
        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            load_kv_cache_scales(model=model, server_args=_server_args())
        self.assertTrue(
            any(FALLBACK_WARNING_SNIPPET in message for message in logs.output)
        )

    def test_no_warning_when_checkpoint_scales_loaded(self):
        """Bug regression (#31224): the "no scaling factors provided" warning
        used to be gated only on --quantization-param-path, so it fired even
        when per-layer checkpoint k/v scales were present and loaded, falsely
        suggesting the scales fell back to 1.0.
        """
        # One layer without a kv-cache quant method (floats stay None) plus one
        # layer with calibrated checkpoint scales: the warning must not fire.
        model = _Model([_attn_layer(None, None), _attn_layer(0.028, 0.0118)])
        with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
            load_kv_cache_scales(model=model, server_args=_server_args())
        self.assertFalse(
            any(FALLBACK_WARNING_SNIPPET in message for message in logs.output)
        )
        self.assertTrue(
            any(
                "loaded from the model checkpoint" in message for message in logs.output
            )
        )

    def test_warns_when_no_layer_has_kv_quant_method(self):
        model = _Model([_attn_layer(None, None)])
        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            load_kv_cache_scales(model=model, server_args=_server_args())
        self.assertTrue(
            any(FALLBACK_WARNING_SNIPPET in message for message in logs.output)
        )

    def test_no_logs_for_non_fp8_kv_cache_dtype(self):
        model = _Model([_attn_layer(1.0, 1.0)])
        with self.assertNoLogs(LOGGER_NAME):
            load_kv_cache_scales(
                model=model, server_args=_server_args(kv_cache_dtype="auto")
            )

    def test_quantization_param_path_uses_legacy_loader(self):
        model = MagicMock()
        load_kv_cache_scales(
            model=model,
            server_args=_server_args(quantization_param_path="/tmp/scales.json"),
        )
        model.load_kv_cache_scales.assert_called_once_with("/tmp/scales.json")


if __name__ == "__main__":
    unittest.main()
