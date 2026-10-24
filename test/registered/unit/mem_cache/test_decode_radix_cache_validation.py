"""Decode-side radix cache admits DeepSeek-V4 only on the unified KV layout.

Both the bf16 and fp8 unified rows are admitted; the paged layout stays rejected.

    python -m pytest test/registered/unit/mem_cache/test_decode_radix_cache_validation.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.kv_cache_builder import validate_decode_radix_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_DSV4 = SimpleNamespace(is_deepseek_v4_arch=True, is_hybrid_swa_compress=False)


def _validate(token_to_kv_pool):
    validate_decode_radix_cache(
        model_config=_DSV4,
        token_to_kv_pool=token_to_kv_pool,
        is_hybrid_swa=True,
        enable_hierarchical_cache=False,
    )


class TestValidateDecodeRadixCacheDSV4(CustomTestCase):
    def test_unified_kv_bf16_is_admitted(self):
        _validate(SimpleNamespace(_unified_kv=True, _unified_kv_fp8=False))

    def test_unified_kv_fp8_is_admitted(self):
        _validate(SimpleNamespace(_unified_kv=True, _unified_kv_fp8=True))

    def test_paged_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unified KV layout"):
            _validate(SimpleNamespace(_unified_kv=False, _unified_kv_fp8=False))


if __name__ == "__main__":
    unittest.main()
