# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for #28808: TRTLLMHAAttnBackend derives from
AttentionBackend directly, so members formerly inherited from
FlashInferAttnBackend must resolve locally."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestTrtllmMhaBackendInheritance(CustomTestCase):
    def test_direct_attention_backend_parent(self):
        self.assertTrue(issubclass(TRTLLMHAAttnBackend, AttentionBackend))
        self.assertFalse(issubclass(TRTLLMHAAttnBackend, FlashInferAttnBackend))

    def test_kv_write_scales_resolves_and_matches_parent_semantics(self):
        # Regression: forward_decode/forward_extend call
        # self._kv_write_scales(layer) on the common (non-fused) KV-store
        # path; after #28808 it must resolve without FlashInferAttnBackend.
        self.assertTrue(hasattr(TRTLLMHAAttnBackend, "_kv_write_scales"))

        layer = SimpleNamespace(k_scale="k", v_scale="v")
        quant_method = Mock()
        quant_method.needs_global_scale.return_value = False
        owner = SimpleNamespace(kv_cache_quant_method=quant_method)
        self.assertEqual(TRTLLMHAAttnBackend._kv_write_scales(owner, layer), ("k", "v"))
        quant_method.needs_global_scale.return_value = True
        self.assertEqual(
            TRTLLMHAAttnBackend._kv_write_scales(owner, layer), (None, None)
        )


if __name__ == "__main__":
    unittest.main()
