import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    InsertResult,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.chunk_cache import ChunkCache  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pure_swa_radix_cache import (  # noqa: E402
    PureSWARadixCache,
)
from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.swa_component import (  # noqa: E402
    SWAComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentData,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCachedTokensByComponent(unittest.TestCase):
    @staticmethod
    def _unified_cache(*component_types: ComponentType) -> UnifiedRadixCache:
        cache = object.__new__(UnifiedRadixCache)
        cache.components = {
            component_type: object() for component_type in component_types
        }
        return cache

    def test_base_cache_component_accounting_is_neutral(self):
        cache = object.__new__(ChunkCache)
        self.assertEqual(
            cache.build_cached_tokens_by_component(
                SimpleNamespace(),
                device=10,
                host=20,
                storage=30,
            ),
            {},
        )

    def test_radix_cache_reports_full_tokens_only(self):
        cache = RadixCache.create_simulated()
        self.assertEqual(
            cache.build_cached_tokens_by_component(
                SimpleNamespace(),
                device=10,
                host=20,
                storage=30,
            ),
            {"full": {"device": 10, "host": 20, "storage": 30}},
        )

    def test_pure_swa_cache_keeps_component_accounting_neutral(self):
        cache = object.__new__(PureSWARadixCache)
        self.assertEqual(
            cache.build_cached_tokens_by_component(
                SimpleNamespace(),
                device=10,
                host=20,
                storage=30,
            ),
            {},
        )

    def test_unified_cache_splits_full_and_swa_sources(self):
        cache = self._unified_cache(
            ComponentType.FULL,
            ComponentType.SWA,
        )
        req = SimpleNamespace(
            matched_tokens_by_component={
                "swa": {"device": 16, "host": 12},
            },
            storage_hit_tokens_by_component={"swa": 4},
        )
        self.assertEqual(
            cache.build_cached_tokens_by_component(
                req,
                device=10,
                host=20,
                storage=30,
            ),
            {
                "full": {"device": 10, "host": 20, "storage": 30},
                "swa": {"device": 16, "host": 8, "storage": 4},
            },
        )

    def test_storage_prefetch_counts_only_committed_swa_tokens(self):
        cache = self._unified_cache(ComponentType.SWA)
        cache.page_size = 2
        cache.host_lru_lists = {
            ComponentType.SWA: SimpleNamespace(
                in_list=lambda node: False,
                insert_mru=lambda node: None,
            )
        }
        cache._update_evictable_leaf_sets = lambda node: None
        cache._split_node = lambda *args: self.fail("unexpected node split")
        released = []
        cache.cache_controller = SimpleNamespace(
            append_host_mem_release=lambda **kwargs: released.append(kwargs)
        )
        component = object.__new__(SWAComponent)
        component.cache = cache
        component.sliding_window_size = 4
        cache.components[ComponentType.SWA] = component
        anchor = SimpleNamespace()
        transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(4),
        )
        pool_result = PoolTransferResult(
            kv_hit_pages=2,
            extra_pool_hit_pages={PoolName.SWA: 2},
        )

        self.assertEqual(
            cache._commit_prefetched_component_tokens(
                anchor,
                comp_xfers={
                    ComponentType.SWA: [transfer],
                },
                insert_result=InsertResult(
                    prefix_len=0,
                    total_len=4,
                    inserted_host_node=SimpleNamespace(
                        key=torch.arange(4),
                        parent=anchor,
                        component_data={ComponentType.SWA: ComponentData()},
                    ),
                ),
                pool_storage_result=pool_result,
            ),
            {"swa": 4},
        )

        self.assertEqual(
            cache._commit_prefetched_component_tokens(
                anchor,
                comp_xfers={
                    ComponentType.SWA: [transfer],
                },
                insert_result=InsertResult(
                    prefix_len=4,
                    total_len=4,
                    inserted_host_node=SimpleNamespace(
                        key=torch.arange(4),
                        parent=anchor,
                        component_data={
                            ComponentType.SWA: ComponentData(host_value=torch.arange(4))
                        },
                    ),
                ),
                pool_storage_result=pool_result,
            ),
            {},
        )
        self.assertEqual(len(released), 1)

    def test_storage_prefetch_counts_only_tokens_in_active_swa_window(self):
        cache = self._unified_cache(ComponentType.SWA)
        cache.page_size = 2
        cache.host_lru_lists = {
            ComponentType.SWA: SimpleNamespace(
                in_list=lambda node: False,
                insert_mru=lambda node: None,
            )
        }
        cache._update_evictable_leaf_sets = lambda node: None
        cache._split_node = lambda *args: self.fail("unexpected node split")
        cache.cache_controller = SimpleNamespace(
            append_host_mem_release=lambda **kwargs: None
        )
        component = object.__new__(SWAComponent)
        component.cache = cache
        component.sliding_window_size = 3
        cache.components[ComponentType.SWA] = component

        anchor = SimpleNamespace()
        storage_node = SimpleNamespace(
            key=torch.arange(2),
            parent=anchor,
            component_data={ComponentType.SWA: ComponentData()},
        )
        existing_host_node = SimpleNamespace(
            key=torch.arange(2),
            parent=storage_node,
            component_data={
                ComponentType.SWA: ComponentData(host_value=torch.arange(2))
            },
        )

        committed = cache._commit_prefetched_component_tokens(
            anchor,
            comp_xfers={
                ComponentType.SWA: [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=torch.arange(4),
                    )
                ],
            },
            insert_result=InsertResult(
                prefix_len=2,
                total_len=4,
                inserted_host_node=existing_host_node,
            ),
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=2,
                extra_pool_hit_pages={PoolName.SWA: 2},
            ),
        )

        self.assertEqual(committed, {"swa": 1})
        self.assertEqual(
            cache.build_cached_tokens_by_component(
                SimpleNamespace(
                    matched_tokens_by_component={"swa": {"host": 3}},
                    storage_hit_tokens_by_component=committed,
                ),
                device=0,
                host=0,
                storage=0,
            ),
            {"swa": {"host": 2, "storage": 1}},
        )

    def test_storage_prefetch_excludes_tokens_served_from_device(self):
        cache = self._unified_cache(ComponentType.SWA)
        cache.page_size = 2
        cache.host_lru_lists = {
            ComponentType.SWA: SimpleNamespace(
                in_list=lambda node: False,
                insert_mru=lambda node: None,
            )
        }
        cache._update_evictable_leaf_sets = lambda node: None
        cache._split_node = lambda *args: self.fail("unexpected node split")
        cache.cache_controller = SimpleNamespace(
            append_host_mem_release=lambda **kwargs: None
        )
        component = object.__new__(SWAComponent)
        component.cache = cache
        component.sliding_window_size = 4
        cache.components[ComponentType.SWA] = component

        anchor = SimpleNamespace()
        existing_host_node = SimpleNamespace(
            key=torch.arange(2),
            parent=anchor,
            component_data={
                ComponentType.SWA: ComponentData(host_value=torch.arange(2))
            },
        )
        device_node = SimpleNamespace(
            key=torch.arange(2),
            parent=existing_host_node,
            component_data={ComponentType.SWA: ComponentData(value=torch.arange(2))},
        )

        committed = cache._commit_prefetched_component_tokens(
            anchor,
            comp_xfers={
                ComponentType.SWA: [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=torch.arange(4),
                    )
                ],
            },
            insert_result=InsertResult(
                prefix_len=2,
                total_len=4,
                inserted_host_node=device_node,
            ),
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=2,
                extra_pool_hit_pages={PoolName.SWA: 2},
            ),
        )

        self.assertEqual(committed, {})
        self.assertEqual(
            cache.build_cached_tokens_by_component(
                SimpleNamespace(
                    matched_tokens_by_component={
                        "swa": {"device": 2, "host": 2},
                    },
                    storage_hit_tokens_by_component=committed,
                ),
                device=0,
                host=0,
                storage=0,
            ),
            {"swa": {"device": 2, "host": 2}},
        )

    def test_storage_token_attribution_is_popped_and_cleared_on_abort(self):
        cache = self._unified_cache(ComponentType.FULL, ComponentType.SWA)
        cache.prefetch_loaded_tokens_by_reqid = {"aborted": 8}
        cache.prefetch_loaded_tokens_by_component_by_reqid = {
            "done": {"swa": 4},
            "aborted": {"swa": 4},
        }
        cache.ongoing_prefetch = {}

        self.assertEqual(
            cache.pop_prefetch_loaded_tokens_by_component("done"),
            {"swa": 4},
        )
        self.assertEqual(cache.pop_prefetch_loaded_tokens_by_component("done"), {})

        cache.release_aborted_request("aborted")
        self.assertNotIn("aborted", cache.prefetch_loaded_tokens_by_reqid)
        self.assertNotIn("aborted", cache.prefetch_loaded_tokens_by_component_by_reqid)

    def test_force_miss_clears_matched_component_tokens(self):
        root = object()
        tree_cache = SimpleNamespace(
            is_chunk_cache=lambda: False,
            root_node=root,
        )
        result = zero_match_result(
            tree_cache,
            MatchResult(
                device_indices=torch.tensor([1]),
                last_device_node=object(),
                last_host_node=object(),
                best_match_node=object(),
                matched_tokens_by_component={"swa": {"device": 1}},
            ),
        )

        self.assertIsNone(result.matched_tokens_by_component)
        self.assertEqual(result.device_indices.numel(), 0)

    def test_swa_component_caps_window_tokens_but_preserves_load_back_length(self):
        component = object.__new__(SWAComponent)
        component.sliding_window_size = 4
        root = SimpleNamespace()
        host_node = SimpleNamespace(
            parent=root,
            component_data={
                ComponentType.SWA: ComponentData(host_value=torch.tensor([3, 4, 5]))
            },
        )
        device_node = SimpleNamespace(
            parent=host_node,
            component_data={
                ComponentType.SWA: ComponentData(value=torch.tensor([1, 2]))
            },
        )
        component.cache = SimpleNamespace(root_node=root)

        result = component.finalize_match_result(
            result=MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64),
                last_device_node=device_node,
                last_host_node=host_node,
                best_match_node=device_node,
            ),
            params=MatchPrefixParams(key=None),
            value_chunks=[],
            best_value_len=0,
        )

        self.assertEqual(
            result.matched_tokens_by_component,
            {"swa": {"device": 2, "host": 2}},
        )
        self.assertEqual(result.swa_host_hit_length, 3)


if __name__ == "__main__":
    unittest.main()
