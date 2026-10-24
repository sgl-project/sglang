import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiCacheAutoSize(CustomTestCase):
    @staticmethod
    def _qsa_pool(size):
        pool = object.__new__(QSATokenToKVPool)
        pool.size = size
        pool.full_kv_pool = Mock(
            spec=MHATokenToKVPool,
            size=size,
            host_capacity_bytes=None,
            get_kv_size_bytes=Mock(return_value=(size * 6, size * 4)),
        )
        pool.qsa_compressed_k_buffer_pool = [torch.empty(size * 2, dtype=torch.uint8)]
        # Request-local key/rope states are not mirrored by QSAIndexerPoolHost.
        pool.qsa_key_state_buffer_pool = [torch.empty(size * 100)]
        pool.qsa_rope_position_buffer = torch.empty(size * 100, dtype=torch.int64)
        return pool

    def test_qsa_target_and_draft_count_only_mirrored_state(self):
        target, draft = self._qsa_pool(100), self._qsa_pool(10)
        for mode in ("packed", "sidecar"):
            with self.subTest(mode=mode):
                params = CacheInitParams(
                    disable=False,
                    req_to_token_pool=Mock(
                        spec=HybridReqToTokenPool,
                        mamba_pool=Mock(
                            host_capacity_bytes=None,
                            get_kv_size_bytes=Mock(return_value=400),
                        ),
                    ),
                    token_to_kv_pool_allocator=Mock(
                        get_kvcache=Mock(return_value=target)
                    ),
                    page_size=4,
                    mtp_draft_device_pools=(draft,) if mode == "packed" else (),
                )
                plan = (
                    Mock(mode="sidecar", device_pools=(draft,))
                    if mode == "sidecar"
                    else None
                )
                # Draft host slots follow target capacity, not draft device capacity.
                self.assertEqual(sizing._estimate_hicache_bytes(params, plan), 2800)

    def test_qsa_auto_ratio_leaves_room_for_compressed_index(self):
        target = self._qsa_pool(100)
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=4,
        )
        host_free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 1000
        with (
            get_context().override_server_args(enable_hierarchical_cache=True),
            patch.object(base, "ranks_per_host", return_value=1),
            patch.object(base, "available_host_memory_bytes", return_value=host_free),
            sizing.auto_size_hicache(params, None, enabled=True),
        ):
            ratio = get_memory().hicache_ratio
            budget = 1000 * get_memory().hicache_host_memory_fraction
            self.assertAlmostEqual(ratio, budget * 0.95 / 1200)
            self.assertLessEqual(ratio * 1200, budget)

    def test_hybrid_target_draft_uses_its_sidecar_slot_capacity(self):
        # EAGLE3 can pair a hybrid target with a plain MHA draft. Full drafts
        # follow full target slots; SWA drafts follow the smaller SWA capacity.
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=4096),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=1024),
            ),
        )
        draft_mha = Mock(
            spec=MHATokenToKVPool,
            size=16,
            host_capacity_bytes=None,
            get_kv_size_bytes=Mock(return_value=(128, 128)),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        for draft, expected_sidecar_bytes in (
            (draft_mha, 2048),
            (Mock(spec=BaseSWAKVPool, swa_kv_pool=draft_mha), 512),
        ):
            with self.subTest(
                draft_type=type(draft).__name__, bytes=expected_sidecar_bytes
            ):
                plan = Mock(mode="sidecar", device_pools=(draft,))
                self.assertEqual(
                    sizing._estimate_hicache_bytes(params, plan),
                    4096 + 1024 + expected_sidecar_bytes,
                )

    def test_unified_views_are_sized_from_host_capacity(self):
        """Unified sub-pools answer get_kv_size_bytes with zero (UnifiedKVPool
        logs the shared buffer once) and publish host_capacity_bytes instead,
        the weight the explicit host-size split already uses. An estimate taken
        from get_kv_size_bytes is zero device bytes, and the default ratio then
        divides by it."""
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=4096,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=1024,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        self.assertEqual(sizing._estimate_hicache_bytes(params, None), 4096 + 1024)

    def test_default_ratio_fits_host_budget_and_pools_book_it(self):
        """With only --enable-hierarchical-cache the default ratio shrinks to the
        per-rank budget, pools book one snapshot, and an explicit ratio opts out."""
        pool = MHATokenToKVPool(
            size=128,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=pool)),
            page_size=2,
        )
        rank_budget = 10_000
        # Four ranks per host (e.g. TP8 over two 4-GPU nodes) share what psutil reports.
        host_free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 4 * rank_budget
        with (
            get_context().override_server_args(enable_hierarchical_cache=True),
            patch.object(base, "ranks_per_host", return_value=4),
            patch.object(base, "available_host_memory_bytes", return_value=host_free),
            sizing.auto_size_hicache(params, None, enabled=True),
        ):
            ratio = get_memory().hicache_ratio
            self.assertLess(ratio, 2.0)
            host = MHATokenToKVPoolHost(
                pool.get_device_pool_infos()[0].buffer_info,
                ratio,
                0,
                2,
                "layer_first",
                pin_memory=False,
                device="cpu",
                device_capacity_tokens=pool.size
                if pool.host_capacity_tokens is None
                else pool.host_capacity_tokens,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                write_back_staging=pool.hicache_write_back_staging,
            )
            self.assertLessEqual(host.size * host.size_per_token, 0.8 * rank_budget)
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                MHATokenToKVPoolHost(
                    pool.get_device_pool_infos()[0].buffer_info,
                    ratio,
                    0,
                    2,
                    "layer_first",
                    pin_memory=False,
                    device="cpu",
                    device_capacity_tokens=pool.size
                    if pool.host_capacity_tokens is None
                    else pool.host_capacity_tokens,
                    start_layer=pool.start_layer,
                    end_layer=pool.end_layer,
                    write_back_staging=pool.hicache_write_back_staging,
                )
        self.assertIsNone(base._host_memory_budget.get())

        explicit = ServerArgs(model_path="dummy", hicache_ratio=2.0)
        explicit.resolve_once()
        self.assertIsNone(
            resolution_result(explicit, "hicache_host_memory_fraction", 0.8)
        )


if __name__ == "__main__":
    unittest.main()
