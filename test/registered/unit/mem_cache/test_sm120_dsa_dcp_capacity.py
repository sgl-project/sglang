"""Replicated Index-K covers the real allocator's final widened page."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_storage(size, *, page_size, index_buf_size=None, **kwargs):
    # Use the production IndexKeyCache allocation; substitute only the large
    # GPU-only model pool constructor with a tiny CPU storage owner.
    pool = SimpleNamespace(
        size=size,
        page_size=page_size,
        device="cpu",
        layer_num=1,
        index_head_dim=128,
        quant_block_size=128,
        index_k_with_scale_buffer_dtype=torch.uint8,
        custom_mem_pool=None,
        skip_topk_layers=[False],
    )
    pool.index_cache = IndexKeyCache(
        pool, size if index_buf_size is None else index_buf_size
    )
    pool.latent = torch.zeros((size + page_size, 656), dtype=torch.uint8)
    return pool


class TestSM120DSACapacity(CustomTestCase):
    is_draft_worker = False

    def test_last_allocator_page_is_addressable(self):
        for size in (1, 2, 4, 8):
            with self.subTest(dcp_size=size):
                n, page = 128, 64
                allocator = PagedTokenToKVPoolAllocator(
                    size=n * size,
                    page_size=page * size,
                    dtype=torch.float8_e4m3fn,
                    device="cpu",
                    kvcache=None,
                    need_sort=False,
                )
                virtual_ids = allocator.alloc(n * size)
                self.assertIsNotNone(virtual_ids)
                self.assertEqual(allocator.available_size(), 0)
                last_id = int(virtual_ids.max())
                hf = SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLM"],
                    index_head_dim=128,
                    index_topk=2048,
                )
                kvc = SimpleNamespace(
                    is_draft_worker=self.is_draft_worker,
                    pool_page_size=page,
                    kv_cache_dtype=torch.float8_e4m3fn,
                    device="cpu",
                    model_config=SimpleNamespace(
                        hf_config=hf, kv_lora_rank=512, qk_rope_head_dim=64
                    ),
                    layer_info=SimpleNamespace(
                        start_layer=0, end_layer=1, num_effective_layers=1
                    ),
                )
                with (
                    get_context().override_server_args(
                        dsa_prefill_backend="flashinfer_sparse_mla",
                        dsa_decode_backend="flashinfer_sparse_mla",
                        enable_hisparse=False,
                        page_size=page,
                    ),
                    get_parallel().override(attn_dcp_size=size),
                    patch(
                        "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                        return_value=(None, 1),
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator._should_elide_dsa_index_k",
                        return_value=False,
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
                        return_value=656,
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.DSATokenToKVPool",
                        side_effect=make_storage,
                    ),
                ):
                    kvc.loc_space_scale = KVCacheConfigurator.loc_space_scale.fget(kvc)
                    kvc.pool_page_size = KVCacheConfigurator.pool_page_size.fget(kvc)
                    self.assertEqual(kvc.pool_page_size, page)
                    pool = KVCacheConfigurator._build_dsa_kv_pool(
                        kvc,
                        max_total_num_tokens=n * size if self.is_draft_worker else n,
                        max_running_requests=1,
                    )
                self.assertEqual(
                    pool.size,
                    n * size + (size - 1) * page if self.is_draft_worker else n,
                )
                # Verify both key and scale byte addresses in the actual buffer.
                index_buffer = pool.index_cache.buffer[0]
                last_page, offset = divmod(last_id, page)
                index_buffer[last_page, offset * 128] = 17
                index_buffer[last_page, page * 128 + offset * 4] = 29
                self.assertEqual(int(index_buffer[last_page, offset * 128]), 17)
                self.assertEqual(
                    int(index_buffer[last_page, page * 128 + offset * 4]), 29
                )
                latent_id = last_id if self.is_draft_worker else last_id // size
                pool.latent[latent_id, 0] = 31
                self.assertEqual(int(pool.latent[latent_id, 0]), 31)


class TestSM120DSADraftCapacity(TestSM120DSACapacity):
    is_draft_worker = True


if __name__ == "__main__":
    unittest.main()
