import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _backend():
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.page_size = 1
    backend.kv_cache_dtype = torch.float16
    backend.kv_cache_dtype_str = "float8_e4m3fn"
    backend.kv_cache_is_mxfp8 = False
    backend.fa_impl_ver = 3
    backend.num_splits = 4
    return backend


class TestFlashAttentionPagedMHA(CustomTestCase):
    def test_get_paged_mha_kv_cache_supports_head_groups(self):
        backend = _backend()
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(
                return_value=(
                    torch.empty(8, 2, 16),
                    torch.empty(8, 2, 16),
                )
            )
        )
        layer = SimpleNamespace(
            layer_id=3,
            tp_k_head_num=2,
            tp_v_head_num=2,
            head_dim=16,
            v_head_dim=16,
        )

        key_cache, value_cache = backend.get_paged_mha_kv_cache(
            layer,
            head_group_num=2,
        )

        self.assertEqual(key_cache.shape, (16, 1, 1, 16))
        self.assertEqual(value_cache.shape, (16, 1, 1, 16))

    def test_fa4_bf16_kv_does_not_use_checkpoint_scales(self):
        for is_prefill in (True, False):
            for dtype_str in ("bf16", "bfloat16", "auto"):
                with self.subTest(is_prefill=is_prefill, dtype_str=dtype_str):
                    backend = _backend()
                    backend.fa_impl_ver = 4
                    backend.kv_cache_dtype = torch.bfloat16
                    backend.kv_cache_dtype_str = dtype_str
                    layer = SimpleNamespace(
                        head_dim=16,
                        k_scale=torch.tensor(2.0),
                        v_scale=torch.tensor(4.0),
                    )
                    q = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
                    q_rope = q + 1
                    k_rope = q + 2

                    result = backend.prepare_paged_mha_query(
                        q,
                        q_rope,
                        k_rope,
                        layer,
                        logical_batch_size=2,
                        kv_head_num=1,
                        is_prefill=is_prefill,
                    )

                    for actual, expected in zip(result[:3], (q, q_rope, k_rope)):
                        torch.testing.assert_close(actual, expected)
                    self.assertIsNone(result[3])
                    self.assertIsNone(result[4])

    def test_fp8_query_preserves_existing_scaling_policy(self):
        for fa_impl_ver, is_prefill in ((3, True), (3, False), (4, False)):
            for dtype_str, dtype in (
                ("fp8_e4m3", torch.float8_e4m3fn),
                ("fp8_e5m2", torch.float8_e5m2),
            ):
                with self.subTest(
                    fa_impl_ver=fa_impl_ver, is_prefill=is_prefill, dtype_str=dtype_str
                ):
                    backend = _backend()
                    backend.fa_impl_ver = fa_impl_ver
                    backend.kv_cache_dtype = dtype
                    backend.kv_cache_dtype_str = dtype_str
                    layer = SimpleNamespace(
                        head_dim=16,
                        k_scale=torch.tensor(2.0),
                        v_scale=torch.tensor(4.0),
                    )
                    q = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)

                    result = backend.prepare_paged_mha_query(
                        q,
                        q + 1,
                        q + 2,
                        layer,
                        logical_batch_size=2,
                        kv_head_num=1,
                        is_prefill=is_prefill,
                    )

                    for actual, expected in zip(result[:3], (q, q + 1, q + 2)):
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(
                            actual.float(), expected.to(dtype).float()
                        )
                    torch.testing.assert_close(result[3], torch.full((2, 1), 2.0))
                    torch.testing.assert_close(result[4], torch.full((2, 1), 4.0))


if __name__ == "__main__":
    unittest.main()
