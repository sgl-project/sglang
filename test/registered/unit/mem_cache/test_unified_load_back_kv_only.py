"""UnifiedRadixCache.load_back(kv_only=True) restores base KV without components."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

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

    def test_kv_only_drops_component_transfers(self):
        cache = _cache()
        kv_xfer = SimpleNamespace(host_indices=torch.arange(128))
        cache.tree_core = Mock(
            build_load_back_spec=Mock(return_value=(kv_xfer, {"swa": [Mock()]})),
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
        self.assertIsNone(cache.cache_controller.load.call_args.kwargs["extra_pools"])
        commit_args = cache.tree_core.commit_load_back.call_args.args
        self.assertEqual(commit_args[3], {})


if __name__ == "__main__":
    unittest.main()
