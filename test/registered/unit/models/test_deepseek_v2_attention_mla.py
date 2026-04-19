import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    _dispatch_mla_subtype,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestDeepseekV2AttentionMLA(unittest.TestCase):
    def _make_attn(self, fused_proj=None):
        attn = object.__new__(DeepseekV2AttentionMLA)
        nn.Module.__init__(attn)
        attn.has_fused_proj = fused_proj is not None
        attn.is_packed_weight = False
        if fused_proj is not None:
            attn.fused_qkv_a_proj_with_mqa = fused_proj
        return attn

    def test_get_fused_qkv_a_proj_weight_returns_none_when_missing(self):
        attn = self._make_attn(SimpleNamespace())

        self.assertIsNone(attn._get_fused_qkv_a_proj_weight())

    @patch("sglang.srt.models.deepseek_v2._device_sm", 90)
    @patch("sglang.srt.models.deepseek_v2._is_cuda", True)
    def test_can_use_min_latency_fused_a_gemm_preserves_bf16_path(self):
        fused_proj = SimpleNamespace(
            weight=torch.empty((2112, 7168), dtype=torch.bfloat16)
        )
        attn = self._make_attn(fused_proj)

        self.assertTrue(attn._can_use_min_latency_fused_a_gemm())

    @patch("sglang.srt.models.deepseek_v2._device_sm", 90)
    @patch("sglang.srt.models.deepseek_v2._is_cuda", True)
    def test_can_use_min_latency_fused_a_gemm_disables_when_weight_missing(self):
        attn = self._make_attn(SimpleNamespace())

        self.assertFalse(attn._can_use_min_latency_fused_a_gemm())

    @patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_fused_rope_cpu._is_cpu_amx_available",
        False,
    )
    @patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_fused_rope_cpu._is_cpu",
        False,
    )
    def test_init_mla_fused_rope_cpu_forward_tolerates_missing_weight(self):
        attn = self._make_attn(SimpleNamespace())

        attn.init_mla_fused_rope_cpu_forward()

        self.assertFalse(attn.qkv_proj_with_rope_is_int8)
        self.assertFalse(attn.qkv_proj_with_rope_is_fp8)
        self.assertIsNone(attn.weight_block_size)

    @patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_fused_rope_cpu._is_cpu_amx_available",
        False,
    )
    @patch(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_fused_rope_cpu._is_cpu",
        False,
    )
    def test_init_mla_fused_rope_cpu_forward_preserves_int8_detection(self):
        fused_proj = SimpleNamespace(weight=torch.empty((8, 8), dtype=torch.int8))
        attn = self._make_attn(fused_proj)

        attn.init_mla_fused_rope_cpu_forward()

        self.assertTrue(attn.qkv_proj_with_rope_is_int8)
        self.assertFalse(attn.qkv_proj_with_rope_is_fp8)

    @patch(
        "sglang.srt.models.deepseek_common.attention_backend_handler.use_intel_amx_backend",
        return_value=True,
    )
    @patch("sglang.srt.models.deepseek_common.attention_backend_handler._is_hip", False)
    def test_dispatch_mla_subtype_falls_back_without_weight(self, *_):
        attn = SimpleNamespace(fused_qkv_a_proj_with_mqa=SimpleNamespace())

        self.assertEqual(_dispatch_mla_subtype(attn, None), AttnForwardMethod.MLA)

    @patch(
        "sglang.srt.models.deepseek_common.attention_backend_handler.use_intel_amx_backend",
        return_value=True,
    )
    @patch("sglang.srt.models.deepseek_common.attention_backend_handler._is_hip", False)
    def test_dispatch_mla_subtype_keeps_cpu_fused_rope_with_weight(self, *_):
        attn = SimpleNamespace(fused_qkv_a_proj_with_mqa=SimpleNamespace(weight=object()))

        self.assertEqual(
            _dispatch_mla_subtype(attn, None),
            AttnForwardMethod.MLA_FUSED_ROPE_CPU,
        )


if __name__ == "__main__":
    unittest.main()
