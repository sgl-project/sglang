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
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        UnquantizedKVCacheMethod,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

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


def _mha_pool(k_buffer, v_buffer, *, page_size):
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.page_size = page_size
    pool.start_layer = 0
    pool.layer_transfer_counter = None
    pool.quant_method = UnquantizedKVCacheMethod()
    pool.dtype = pool.store_dtype = k_buffer.dtype
    pool.k_buffer = [k_buffer]
    pool.v_buffer = [v_buffer]
    return pool


def _layer():
    return SimpleNamespace(
        layer_id=0,
        tp_k_head_num=2,
        tp_v_head_num=2,
        head_dim=16,
        v_head_dim=16,
    )


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

    def test_get_paged_mha_kv_cache_refuses_head_groups_on_a_strided_pool(self):
        # A unified pool's per-layer view is a whole ENTRY apart; without the
        # guard the head re-chunk raised PyTorch's "view size is not compatible".
        H, D, slots = 2, 16, 8
        entry = 64 * H * D  # every layer's K and V share the slot
        backing = torch.zeros(slots * entry)
        strided = backing.as_strided((slots, H, D), (entry, D, 1))
        self.assertFalse(strided.is_contiguous())

        backend = _backend()
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(strided, strided))
        )
        layer = SimpleNamespace(
            layer_id=3,
            tp_k_head_num=H,
            tp_v_head_num=H,
            head_dim=D,
            v_head_dim=D,
        )

        with self.assertRaises(AssertionError) as caught:
            backend.get_paged_mha_kv_cache(layer, head_group_num=2)
        self.assertIn("contiguous paged KV", str(caught.exception))

        key_cache, _ = backend.get_paged_mha_kv_cache(layer)
        self.assertEqual(key_cache.shape, (slots, 1, H, D))

    def test_get_paged_mha_kv_cache_pages_by_the_backend_page_size(self):
        # Pages follow the backend's page size even when the pool pages at a
        # multiple of it, as a draft pool does under decode context parallelism.
        backend = _backend()
        backend.page_size = 2
        k = torch.arange(8 * 2 * 16, dtype=torch.float16).view(8, 2, 16)
        backend.token_to_kv_pool = _mha_pool(k, k + 1, page_size=4)

        key_cache, value_cache = backend.get_paged_mha_kv_cache(_layer())

        self.assertEqual(key_cache.shape, (4, 2, 2, 16))
        self.assertEqual(value_cache.shape, (4, 2, 2, 16))
        self.assertTrue(torch.equal(key_cache[1, 0], k[2]))

    def test_get_paged_mha_kv_cache_keeps_the_hnd_pool_view(self):
        # An HND pool hands over [num_pages, heads, page_size, head_dim]
        # buffers; they reach the kernel through their plain view.
        backend = _backend()
        backend.page_size = 4
        k = torch.arange(3 * 2 * 4 * 16, dtype=torch.float16).view(3, 2, 4, 16)
        backend.token_to_kv_pool = _mha_pool(k, k + 1, page_size=4)

        key_cache, _ = backend.get_paged_mha_kv_cache(_layer())

        want = k.view(-1, 4, 2, 16)
        self.assertEqual(key_cache.shape, want.shape)
        self.assertEqual(key_cache.stride(), want.stride())
        self.assertEqual(key_cache.data_ptr(), want.data_ptr())

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
