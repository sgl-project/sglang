import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import KVLayerRouter, PlainPoolRouter
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import base, rocm_budget
from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context, get_memory, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _leaf(pool):
    """Route a fake leaf pool as its own one sub-pool."""
    pool.start_layer, pool.layer_num = 0, 1
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool


def _windowed(pool, full_layers=None, swa_layers=None):
    """Route a fake pool over its full and sliding-window sub-pool views."""
    pool.layer_router = KVLayerRouter.from_layer_maps(
        pool,
        [
            (IdSpaceKind.FULL, pool.full_kv_pool, full_layers or {}),
            (IdSpaceKind.SLIDING_WINDOW, pool.swa_kv_pool, swa_layers or {}),
        ],
    )
    return pool


def _cpu_params():
    return SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(device="cpu")
        )
    )


def _plain_left():
    # Plain RAM left in the active snapshot scope.
    return base._host_memory_budget.get()[0]


class TestHiCacheAutoSize(CustomTestCase):
    def test_hybrid_target_draft_uses_its_sidecar_slot_capacity(self):
        # EAGLE3 can pair a hybrid target with a plain MHA draft. Full drafts
        # follow full target slots; SWA drafts follow the smaller SWA capacity.
        target = _windowed(
            Mock(
                spec=SWAKVPool,
                size=128,
                full_kv_pool=_leaf(
                    Mock(
                        size=128,
                        host_capacity_bytes=None,
                        get_kv_size_bytes=Mock(return_value=4096),
                    )
                ),
                swa_kv_pool=_leaf(
                    Mock(
                        size=32,
                        host_capacity_bytes=None,
                        get_kv_size_bytes=Mock(return_value=1024),
                    )
                ),
            )
        )
        draft_mha = _leaf(
            Mock(
                spec=MHATokenToKVPool,
                size=16,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=(128, 128)),
            )
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(holds_state_slots=False),
            token_to_kv_pool_allocator=Mock(
                get_kvcache=Mock(return_value=target), storage=None
            ),
            page_size=2,
        )
        for draft, expected_sidecar_bytes in (
            (draft_mha, 2048),
            (
                _windowed(
                    Mock(
                        spec=BaseSWAKVPool,
                        full_kv_pool=draft_mha,
                        swa_kv_pool=draft_mha,
                    )
                ),
                512,
            ),
            # A draft with routed full and SWA layers gets both sidecars.
            (
                _windowed(
                    Mock(
                        spec=BaseSWAKVPool,
                        full_kv_pool=draft_mha,
                        swa_kv_pool=draft_mha,
                    ),
                    full_layers={2: 0},
                    swa_layers={0: 0},
                ),
                2048 + 512,
            ),
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
        target = _windowed(
            Mock(
                spec=SWAKVPool,
                size=128,
                full_kv_pool=_leaf(
                    Mock(
                        size=128,
                        host_capacity_bytes=4096,
                        get_kv_size_bytes=Mock(return_value=(0, 0)),
                    )
                ),
                swa_kv_pool=_leaf(
                    Mock(
                        size=32,
                        host_capacity_bytes=1024,
                        get_kv_size_bytes=Mock(return_value=(0, 0)),
                    )
                ),
            )
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(holds_state_slots=False),
            token_to_kv_pool_allocator=Mock(
                get_kvcache=Mock(return_value=target), storage=None
            ),
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
            req_to_token_pool=SimpleNamespace(holds_state_slots=False),
            token_to_kv_pool_allocator=Mock(
                get_kvcache=Mock(return_value=pool), storage=None
            ),
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

    def test_explicit_size_books_a_snapshot_taken_before_peers_allocate(self):
        """An explicit --hicache-size is checked against host memory sampled
        before any rank allocates, not after its peers pinned their pools."""
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
        # 1e-5 GB is 79 pages of 2 tokens at 64 bytes per token on every rank.
        pool_bytes = 79 * 2 * 64
        rank_budget = 12_000
        host_free = {"bytes": base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 4 * rank_budget}
        free_now = Mock(side_effect=lambda **_: host_free["bytes"])
        with (
            get_context().override_server_args(
                enable_hierarchical_cache=True, hicache_host_memory_fraction=None
            ),
            patch.object(base, "ranks_per_host", return_value=4),
            patch.object(sizing, "ranks_per_host", return_value=4),
            patch.object(torch.distributed, "barrier"),
            get_parallel().override(world_group=SimpleNamespace(cpu_group="cpu")),
            patch.object(base, "available_host_memory_bytes", free_now),
            sizing.auto_size_hicache(
                params=_cpu_params(), draft_plan=None, enabled=True
            ),
        ):
            # Three peers on this host pin their pools before this rank builds its own.
            host_free["bytes"] -= 3 * pool_bytes
            host = MHATokenToKVPoolHost(
                buffer_info=pool.get_device_pool_infos()[0].buffer_info,
                host_to_device_ratio=2.0,
                host_size=1e-5,
                page_size=2,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                device_capacity_tokens=pool.size,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                write_back_staging=pool.hicache_write_back_staging,
            )
            self.assertEqual(host.size * host.size_per_token, pool_bytes)
            self.assertEqual(_plain_left(), rank_budget - pool_bytes)
        self.assertIsNone(base._host_memory_budget.get())
        # One snapshot per rank: building the pool books it, never re-samples.
        free_now.assert_called_once()

    def test_explicit_size_samples_host_memory_before_the_rank_barrier(self):
        """Ranks sharing a host sample before any of them allocates."""
        calls = Mock()
        calls.free.return_value = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 2000
        with (
            get_context().override_server_args(
                enable_hierarchical_cache=True, hicache_host_memory_fraction=None
            ),
            patch.object(base, "ranks_per_host", return_value=2),
            patch.object(sizing, "ranks_per_host", return_value=2),
            patch.object(base, "available_host_memory_bytes", calls.free),
            patch.object(torch.distributed, "barrier", calls.barrier),
            get_parallel().override(world_group=SimpleNamespace(cpu_group="cpu")),
            sizing.auto_size_hicache(
                params=_cpu_params(), draft_plan=None, enabled=True
            ),
        ):
            self.assertEqual(_plain_left(), 1000)
        self.assertEqual([name for name, _, _ in calls.mock_calls], ["free", "barrier"])
        calls.barrier.assert_called_once_with(group="cpu")

    def test_explicit_size_on_a_lone_rank_still_sees_other_replicas_pools(self):
        """A lone rank samples per pool and takes no collective, so another replica's pins stay visible."""
        host_free = {"bytes": base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 1000}
        with (
            get_context().override_server_args(
                enable_hierarchical_cache=True, hicache_host_memory_fraction=None
            ),
            patch.object(base, "ranks_per_host", return_value=1),
            patch.object(sizing, "ranks_per_host", return_value=1),
            patch.object(base, "is_gfx95_supported", return_value=False),
            patch.object(rocm_budget, "rocm_pinnable_share_bytes", return_value=None),
            patch.object(
                base, "available_host_memory_bytes", lambda **_: host_free["bytes"]
            ),
            patch.object(torch.distributed, "barrier") as barrier,
            sizing.auto_size_hicache(
                params=_cpu_params(), draft_plan=None, enabled=True
            ),
        ):
            self.assertIsNone(base._host_memory_budget.get())
            self.assertEqual(base.host_memory_preflight([200]).available_bytes, 1000)
            # Another replica pins 600 bytes after this rank's first pool was checked.
            host_free["bytes"] -= 600
            self.assertEqual(base.host_memory_preflight([600]), (600, 400))
        barrier.assert_not_called()

    def _explicit_size_scope(
        self, *, mode, available, hugetlb, gfx950, mappings, ranks
    ):
        # Returns the scope budget (None without a snapshot) and the preflight for the mappings.
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(device="cuda")
            )
        )
        with (
            get_context().override_server_args(
                enable_hierarchical_cache=True, hicache_host_memory_fraction=None
            ),
            patch.object(base, "ranks_per_host", return_value=ranks),
            patch.object(sizing, "ranks_per_host", return_value=ranks),
            patch.object(torch.distributed, "barrier") as barrier,
            get_parallel().override(world_group=SimpleNamespace(cpu_group="cpu")),
            patch.object(base, "available_host_memory_bytes", return_value=available),
            patch.object(base, "hugepage_size_requested", return_value=2 << 20),
            patch.object(base, "hugepage_mode", return_value=mode),
            patch.object(base, "device_uses_allocator", return_value=True),
            patch.object(base, "is_gfx95_supported", return_value=gfx950),
            patch.object(rocm_budget, "is_gfx95_supported", return_value=gfx950),
            patch.object(
                rocm_budget,
                "rocm_pinnable_share_bytes",
                return_value=(20 << 30) if gfx950 else None,
            ),
            patch.object(
                HostTensorAllocator,
                "free_hugetlb_bytes",
                classmethod(lambda cls: hugetlb),
            ),
            patch.object(sizing, "get_allocator_type", return_value="default"),
            sizing.auto_size_hicache(params=params, draft_plan=None, enabled=True),
        ):
            scope = base._host_memory_budget.get()
            result = base.host_memory_preflight(mappings, HostTensorAllocator, "cuda")
        # Only ranks sharing a host meet at the barrier; a lone rank has no process group to wait on.
        if ranks > 1:
            barrier.assert_called_once_with(group="cpu")
        else:
            barrier.assert_not_called()
        return scope, result

    def test_explicit_size_admits_a_hugepage_pool_when_plain_ram_is_below_the_reserve(
        self,
    ):
        scope, (allocation, available) = self._explicit_size_scope(
            mode=base.HUGEPAGE_MODE_REQUIRED,
            available=9 << 30,
            hugetlb=16 << 30,
            gfx950=False,
            mappings=[4 << 30],
            ranks=2,
        )
        self.assertLessEqual(allocation, available)
        self.assertEqual(scope, (0, 8 << 30))

    def test_a_lone_rank_admits_a_hugepage_pool_when_plain_ram_is_below_the_reserve(
        self,
    ):
        scope, (allocation, available) = self._explicit_size_scope(
            mode=base.HUGEPAGE_MODE_PREFER,
            available=9 << 30,
            hugetlb=8 << 30,
            gfx950=False,
            mappings=[4 << 30],
            ranks=1,
        )
        self.assertIsNone(scope)
        self.assertLessEqual(allocation, available)

    def test_explicit_size_on_gfx950_books_hugepage_pools_against_the_pinnable_share(
        self,
    ):
        scope, (allocation, available) = self._explicit_size_scope(
            mode=base.HUGEPAGE_MODE_PREFER,
            available=200 << 30,
            hugetlb=64 << 30,
            gfx950=True,
            mappings=[40 << 30],
            ranks=1,
        )
        self.assertEqual(scope, (20 << 30, 0))
        self.assertGreater(allocation, available)


if __name__ == "__main__":
    unittest.main()
