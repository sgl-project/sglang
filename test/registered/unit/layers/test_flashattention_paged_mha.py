import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

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

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend():
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.page_size = 1
    backend.kv_cache_dtype = torch.float16
    backend.kv_cache_dtype_str = "float8_e4m3fn"
    backend.kv_cache_is_mxfp8 = False
    backend.fa_impl_ver = 3
    backend.num_splits = 4
    return backend


class TestFlashAttentionPagedMHA(unittest.TestCase):
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

    def test_prepare_paged_mha_query_reuses_fa_scaling_policy(self):
        backend = _backend()
        layer = SimpleNamespace(
            head_dim=16,
            k_scale=torch.tensor(2.0),
            v_scale=torch.tensor(4.0),
        )
        q = torch.ones(2, 16, dtype=torch.bfloat16)

        q, _, _, k_descale, v_descale = backend.prepare_paged_mha_query(
            q,
            None,
            None,
            layer,
            logical_batch_size=2,
            kv_head_num=1,
            is_prefill=True,
        )

        self.assertEqual(q.dtype, torch.float16)
        self.assertEqual(k_descale.shape, (2, 1))
        self.assertEqual(v_descale.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
