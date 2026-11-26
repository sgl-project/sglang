"""The scheduling policy's prefix matches only order the waiting queue (probes): an
external store answers them from the request's last lookup while its key and device
hit are unchanged, and they leave no hit marker. Every other match can be followed by
a load, so it looks the store up again and leaves the marker init_load_back loads,
whether or not the model has Mamba state, and drops the request's last lookup."""

import unittest
from array import array
from collections import defaultdict

import test_unified_cache_linker as base
import torch

from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="base-b", runner_config="1-gpu-small-amd")

TOKENS = list(range(1, 9))


class _ProbeReuse:
    def _cache(self, metrics=None):
        """A cache with nothing on the device whose store holds TOKENS (FULL KV
        only: admission matches without cow_mamba)."""
        self.cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)
        stored_keys = defaultdict(set)
        producer, allocator, req_pool = build_fixture(self.cfg)
        producer_store = base._InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_store)
        self._insert(producer, allocator, req_pool, TOKENS)
        producer_store.complete_next_offload(True)
        producer.check_hicache_events()

        self.cache, self.allocator, self.req_pool = build_fixture(self.cfg)
        if metrics is not None:
            self.cache.metrics_collector = metrics
        self.store = base._InMemoryUnifiedCacheLinker(stored_keys)
        self.cache.init_cache_linker(self.store)
        req = self._make_req(self.req_pool)
        req.origin_input_ids = array("q", TOKENS)
        req.output_ids = array("q")
        req.extra_key = None
        return req

    def _match(self, req, tokens, probe):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req, probe=probe)
        )

    def _load(self, req, match):
        self._apply_match_to_req(req, match)
        loaded, _ = self.cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        return loaded

    def test_probes_reuse_the_last_lookup_and_leave_no_marker(self):
        req = self._cache()
        for _ in range(3):
            self.assertEqual(self._match(req, TOKENS, True).host_hit_length, 8)
        self.assertEqual(len(self.store.lookup_calls), 1)
        self.assertFalse(self.cache.linker.has_hit(req.rid))

    def test_admission_after_probes_looks_up_again_and_loads(self):
        req = self._cache()
        self._match(req, TOKENS, True)
        match = self._match(req, TOKENS, False)
        self.assertEqual(len(self.store.lookup_calls), 2)
        self.assertTrue(self.cache.linker.has_hit(req.rid))
        self.assertEqual(self._load(req, match), 8)
        self.assertIn(req.rid, self.store.queued_loads)

    def test_admission_drops_the_last_lookup(self):
        # An admitted request ends without release_request.
        req = self._cache()
        self._match(req, TOKENS, True)
        self._match(req, TOKENS, False)
        self.assertNotIn(req.rid, self.cache.linker._last_lookups)
        # A request admission turned away probes the store again.
        self._match(req, TOKENS, True)
        self.assertEqual(len(self.store.lookup_calls), 3)

    def test_probe_looks_up_again_when_the_key_grows(self):
        req = self._cache()
        self.assertEqual(self._match(req, TOKENS[:4], True).host_hit_length, 4)
        self.assertEqual(self._match(req, TOKENS, True).host_hit_length, 8)
        self.assertEqual(len(self.store.lookup_calls), 2)

    def test_probe_looks_up_again_when_the_device_hit_changes(self):
        req = self._cache()
        self.assertEqual(self._match(req, TOKENS, True).host_hit_length, 8)
        self._insert(self.cache, self.allocator, self.req_pool, TOKENS[:4])
        self.store.complete_next_offload(True)
        self.cache.check_hicache_events()
        match = self._match(req, TOKENS, True)
        self.assertEqual(match.device_prefix_len, 4)
        self.assertEqual(match.host_hit_length, 4)
        self.assertEqual(len(self.store.lookup_calls), 2)

    def test_queue_refresh_probes_through_init_next_round_input(self):
        req = self._cache()
        for _ in range(2):
            req.init_next_round_input(self.cache, probe=True)
        self.assertEqual(len(self.store.lookup_calls), 1)
        self.assertFalse(self.cache.linker.has_hit(req.rid))
        req.init_next_round_input(self.cache)
        self.assertEqual(len(self.store.lookup_calls), 2)
        self.assertTrue(self.cache.linker.has_hit(req.rid))

    def test_probe_after_release_looks_up_again(self):
        req = self._cache()
        self._match(req, TOKENS, True)
        self.cache.linker.release_request(req.cache_request_handle)
        self._match(req, TOKENS, True)
        self.assertEqual(len(self.store.lookup_calls), 2)

    def test_admission_the_device_covers_drops_the_marker(self):
        req = self._cache()
        self._match(req, TOKENS, False)
        self.assertTrue(self.cache.linker.has_hit(req.rid))
        self._insert(self.cache, self.allocator, self.req_pool, TOKENS)
        self.store.complete_next_offload(True)
        self.cache.check_hicache_events()
        match = self._match(req, TOKENS, False)
        self.assertEqual(match.device_prefix_len, 8)
        self.assertFalse(self.cache.linker.has_hit(req.rid))

    def test_metrics_count_lookups_and_probe_reuses(self):
        metrics = base._RecordingRadixCacheMetrics()
        req = self._cache(metrics)
        value = metrics.value
        for _ in range(3):
            self._match(req, TOKENS, True)
        self.assertEqual(value(metrics.linker_lookup_tokens), 8)
        self.assertEqual(value(metrics.linker_probe_reuses), 2)
        self._match(req, TOKENS, False)
        self.assertEqual(value(metrics.linker_lookup_tokens), 16)
        self.assertEqual(value(metrics.linker_probe_reuses), 2)

    def test_policy_probes_then_admission_loads(self):
        req = self._cache()
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=self.cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        for _ in range(3):
            policy.calc_priority([req])
        self.assertEqual(len(self.store.lookup_calls), 1)
        self.assertFalse(self.cache.linker.has_hit(req.rid))
        # As init_next_round_input admits a request of a model without Mamba.
        match = match_kv_cache(req, self.cache)
        self.assertEqual(len(self.store.lookup_calls), 2)
        self.assertEqual(self._load(req, match), 8)


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestProbeReusePythonBackend(
    _ProbeReuse,
    base._TreeCoreBackendTestMixin,
    UnifiedRadixCacheFixtures,
    CustomTestCase,
):
    pass


class TestProbeReuseRustBackend(TestProbeReusePythonBackend):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
