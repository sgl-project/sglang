"""UnifiedRadixCache.load_back(kv_only=True) restores base KV without components."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams
from sglang.srt.mem_cache.unified_cache.components.base import BASE_COMPONENT_TYPE
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _lock_result(params: str) -> Mock:
    return Mock(to_dec_params=Mock(return_value=params), delta=0)


def _cache() -> UnifiedRadixCache:
    # Bypass __init__: the constructor wires device pools and a controller.
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = Mock(load=Mock(return_value=torch.arange(128)))
    cache.inc_host_lock_ref = Mock(return_value=_lock_result("host"))
    cache.inc_lock_ref = Mock(return_value=_lock_result("anchor"))
    cache.dec_lock_ref = Mock()
    cache.dec_host_lock_ref = Mock()
    return cache


class TestLoadBackKvOnly(CustomTestCase):
    def test_kv_only_skips_component_hooks(self):
        cache = _cache()
        comp = Mock(component_type="mamba")
        cache._components_tuple = (comp,)
        cache._load_back_transfers = Mock(return_value=True)

        self.assertTrue(cache.load_back(7, req=SimpleNamespace(), kv_only=True))

        comp.prepare_load_back.assert_not_called()
        comp.finalize_load_back.assert_not_called()
        self.assertTrue(cache._load_back_transfers.call_args.kwargs["kv_only"])

    def test_default_runs_component_hooks(self):
        cache = _cache()
        comp = Mock(component_type="mamba")
        cache._components_tuple = (comp,)
        cache._load_back_transfers = Mock(return_value=True)
        req = SimpleNamespace()

        cache.load_back(7, req=req)

        comp.prepare_load_back.assert_called_once_with(7, req=req)
        comp.finalize_load_back.assert_called_once_with(
            req, comp.prepare_load_back.return_value, True
        )

    def test_kv_only_builds_no_component_transfers(self):
        # The spec itself must be KV-only: building an SWA transfer for a node
        # whose SWA state is tombstoned (neither host nor device) asserts.
        cache = _cache()
        kv_xfer = SimpleNamespace(host_indices=torch.arange(128))
        cache.tree_core = Mock(
            build_load_back_spec=Mock(return_value=(kv_xfer, {})),
            commit_load_back=Mock(return_value=[]),
        )
        cache._build_sidecar_transfers = Mock(return_value=[])
        cache.load_back_threshold = 1
        cache.token_to_kv_pool_allocator = SimpleNamespace()
        cache._component_available_size = Mock(return_value=10**6)
        cache._apply_cache_actions = Mock()
        cache.ongoing_load_back = {}

        ok = cache._load_back_transfers(
            node_id=7,
            mem_quota=None,
            req=SimpleNamespace(),
            result=_lock_result("anchor"),
            ancestor_lock_params="anchor",
            host_anchor_params="host",
            kv_only=True,
        )

        self.assertTrue(ok)
        self.assertTrue(
            cache.tree_core.build_load_back_spec.call_args.kwargs["kv_only"]
        )
        self.assertIsNone(cache.cache_controller.load.call_args.kwargs["extra_pools"])
        commit_args = cache.tree_core.commit_load_back.call_args.args
        self.assertEqual(commit_args[3], {})

    def test_kv_only_resident_full_kv_needs_no_dma(self):
        # After a KV-only restore the node's FULL KV is on device while its
        # Mamba / SWA state stays host-only, so a rematch still reports a host
        # hit. A KV-only consumer must get the resident indices, not a failed
        # load_back (reported on the PR by HZY-Wade).
        cache = _cache()
        cache.buffer_pipeline = None
        cache.linker = None
        cache.tree_components = (BASE_COMPONENT_TYPE, "mamba")
        cache.ongoing_load_back = {}
        cache.tree_core = Mock(
            is_full_device_evicted=Mock(return_value=False),
            collect_full_device_indices=Mock(return_value=torch.tensor([20, 21])),
        )
        cache.load_back = Mock()
        req = SimpleNamespace(
            rid="req-0", last_node=3, swa_host_hit_length=0, mamba_host_hit_length=1
        )

        indices, node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=7, host_hit_length=2, req=req, kv_only=True
            )
        )

        self.assertEqual(indices.tolist(), [20, 21])
        self.assertEqual(node, 7)
        cache.load_back.assert_not_called()
        self.assertFalse(cache.has_ongoing_load_back(7))


if __name__ == "__main__":
    unittest.main()
