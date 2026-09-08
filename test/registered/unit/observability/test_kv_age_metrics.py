"""Pure-CPU unit tests for the KV age metrics on RadixCacheMetricsCollector.

``sglang:kv_age_seconds`` / ``sglang:kv_age_tokens_total`` record how long a
radix node had gone untouched when it was matched again (``event="hit"``) or
removed from a tier (``event="evict"``); ``sglang:kv_lifetime_seconds`` and
``sglang:kv_reuses`` record total residency and TreeNode.hit_count at removal.
These tests cover the collector-level contract (label routing, the
token-weighted bucket label, one observation per call) and an end-to-end CPU
RadixCache insert / match / re-insert / evict sequence.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import (
    KV_AGE_BUCKETS,
    KV_REUSE_BUCKETS,
    RadixCacheMetricsCollector,
    kv_age_bucket,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _req(rid: str, tokens) -> Req:
    """A bare Req: hit ages are observed once per request, on its first match."""
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(tokens),
        sampling_params=SamplingParams(),
    )


class _BoundRecordingMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = labels

    def inc(self, value=1):
        self.metric.increments.append((self.labels, value))

    def observe(self, value):
        self.metric.observations.append((self.labels, value))


class _RecordingMetric:
    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.increments = []
        self.observations = []

    def labels(self, *values, **labels):
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _BoundRecordingMetric(self, labels)


class _RecordingRadixCacheMetricsCollector(RadixCacheMetricsCollector):
    _counter_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class TestKvAgeBucket(unittest.TestCase):
    def test_edges_map_to_their_own_label(self):
        for edge in KV_AGE_BUCKETS:
            self.assertEqual(kv_age_bucket(edge), str(int(edge)))

    def test_interior_values_round_up(self):
        self.assertEqual(kv_age_bucket(0.0), "1")
        self.assertEqual(kv_age_bucket(0.5), "1")
        self.assertEqual(kv_age_bucket(1.5), "5")
        self.assertEqual(kv_age_bucket(59.9), "60")
        self.assertEqual(kv_age_bucket(1800.1), "3600")

    def test_beyond_last_edge_is_inf(self):
        self.assertEqual(kv_age_bucket(KV_AGE_BUCKETS[-1] + 1), "+Inf")
        self.assertEqual(kv_age_bucket(1e9), "+Inf")


class TestObserveKvAge(unittest.TestCase):
    def setUp(self):
        self.collector = _RecordingRadixCacheMetricsCollector(
            labels={"cache_type": "RadixCache"}
        )

    def test_metric_names_and_labelnames(self):
        self.assertEqual(self.collector.kv_age_seconds.name, "sglang:kv_age_seconds")
        self.assertEqual(
            self.collector.kv_age_tokens.name, "sglang:kv_age_tokens_total"
        )
        self.assertEqual(
            self.collector.kv_age_seconds.labelnames,
            ("cache_type", "event", "tier", "outcome"),
        )
        self.assertEqual(
            self.collector.kv_age_tokens.labelnames,
            ("cache_type", "event", "tier", "outcome", "age_le"),
        )

    def test_hit_routes_labels_and_weights_tokens(self):
        self.collector.observe_kv_age(
            42.0, 128, event="hit", tier="device", outcome="hit"
        )
        self.assertEqual(
            self.collector.kv_age_seconds.observations,
            [
                (
                    {
                        "cache_type": "RadixCache",
                        "event": "hit",
                        "tier": "device",
                        "outcome": "hit",
                    },
                    42.0,
                )
            ],
        )
        self.assertEqual(
            self.collector.kv_age_tokens.increments,
            [
                (
                    {
                        "cache_type": "RadixCache",
                        "event": "hit",
                        "tier": "device",
                        "outcome": "hit",
                        "age_le": "60",
                    },
                    128,
                )
            ],
        )

    def test_evict_outcomes_are_distinct_series(self):
        self.collector.observe_kv_age(
            700.0, 64, event="evict", tier="device", outcome="demoted"
        )
        self.collector.observe_kv_age(
            700.0, 64, event="evict", tier="device", outcome="dropped"
        )
        self.collector.observe_kv_age(
            9000.0, 64, event="evict", tier="host", outcome="dropped"
        )
        seen = [
            (lab["tier"], lab["outcome"], lab["age_le"])
            for lab, _ in self.collector.kv_age_tokens.increments
        ]
        self.assertEqual(
            seen,
            [
                ("device", "demoted", "1200"),
                ("device", "dropped", "1200"),
                ("host", "dropped", "+Inf"),
            ],
        )
        self.assertEqual(len(self.collector.kv_age_seconds.observations), 3)

    def test_eviction_records_age_lifetime_and_reuses(self):
        self.collector.observe_kv_eviction(
            age_seconds=30.0,
            lifetime_seconds=900.0,
            reuses=4,
            num_tokens=256,
            tier="device",
            outcome="demoted",
        )
        want = {"cache_type": "RadixCache", "tier": "device", "outcome": "demoted"}
        self.assertEqual(
            self.collector.kv_age_seconds.observations,
            [({**want, "event": "evict"}, 30.0)],
        )
        self.assertEqual(
            self.collector.kv_age_tokens.increments,
            [({**want, "event": "evict", "age_le": "30"}, 256)],
        )
        self.assertEqual(
            self.collector.kv_lifetime_seconds.observations, [(want, 900.0)]
        )
        self.assertEqual(self.collector.kv_reuses.observations, [(want, 4)])
        self.assertEqual(self.collector.kv_reuses.name, "sglang:kv_reuses")
        self.assertEqual(
            self.collector.kv_lifetime_seconds.name, "sglang:kv_lifetime_seconds"
        )
        self.assertEqual(KV_REUSE_BUCKETS[0], 0.0)


class TestRadixCacheEmitsKvAge(unittest.TestCase):
    """End-to-end on a CPU RadixCache: a re-match records a hit age for the
    matched tokens, and an eviction records an evict age for the freed tokens."""

    PAGE_SIZE = 1

    def _build_cache(self):
        req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        kv_pool = MHATokenToKVPool(
            size=64,
            page_size=self.PAGE_SIZE,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=64,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=self.PAGE_SIZE,
            )
        )
        cache.metrics_collector = _RecordingRadixCacheMetricsCollector(
            labels={"cache_type": "RadixCache"}
        )
        return cache, allocator

    def test_match_then_evict_records_hit_and_evict_ages(self):
        cache, allocator = self._build_cache()
        collector = cache.metrics_collector
        tokens = array("q", [1, 2, 3, 4])
        indices = allocator.alloc(len(tokens))
        cache.insert(InsertParams(key=RadixKey(token_ids=tokens), value=indices))
        self.assertEqual(collector.kv_age_seconds.observations, [])

        req = _req("r1", tokens)
        cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens), req=req))
        # Re-matches of the same request (scheduling rounds, post-insert
        # re-anchor) and request-less matches must not count again.
        cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens), req=req))
        cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens)))
        hits = [
            (lab, v)
            for lab, v in collector.kv_age_seconds.observations
            if lab["event"] == "hit"
        ]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0]["tier"], "device")
        self.assertEqual(hits[0][0]["outcome"], "hit")
        self.assertGreaterEqual(hits[0][1], 0.0)
        hit_tokens = [
            v for lab, v in collector.kv_age_tokens.increments if lab["event"] == "hit"
        ]
        self.assertEqual(hit_tokens, [len(tokens)])

        # hit_count counts inserts including the creating one: a second finished
        # request re-inserting the same prefix takes it from 1 to 2.
        cache.insert(InsertParams(key=RadixKey(token_ids=tokens), value=indices))

        cache.evict(EvictParams(num_tokens=len(tokens)))
        evicts = [
            (lab, v)
            for lab, v in collector.kv_age_seconds.observations
            if lab["event"] == "evict"
        ]
        self.assertEqual(len(evicts), 1)
        self.assertEqual(evicts[0][0]["tier"], "device")
        self.assertEqual(evicts[0][0]["outcome"], "dropped")
        want = {"cache_type": "RadixCache", "tier": "device", "outcome": "dropped"}
        self.assertEqual(len(collector.kv_lifetime_seconds.observations), 1)
        self.assertEqual(collector.kv_lifetime_seconds.observations[0][0], want)
        self.assertGreaterEqual(
            collector.kv_lifetime_seconds.observations[0][1], evicts[0][1]
        )
        self.assertEqual(collector.kv_reuses.observations, [(want, 2)])
        evict_tokens = [
            v
            for lab, v in collector.kv_age_tokens.increments
            if lab["event"] == "evict"
        ]
        self.assertEqual(evict_tokens, [len(tokens)])


class TestUnifiedRadixCacheEmitsKvAge(unittest.TestCase):
    """Same sequence on the default UnifiedRadixCache (Python tree core): the
    tree core reports ages through the observer the cache installs."""

    PAGE_SIZE = 1

    def _build_cache(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=self.PAGE_SIZE)
        )
        req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        kv_pool = MHATokenToKVPool(
            size=64,
            page_size=self.PAGE_SIZE,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=64,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=self.PAGE_SIZE,
                tree_components=(ComponentType.FULL,),
            )
        )
        # What UnifiedRadixCache.__init__ does when metrics are enabled.
        cache.metrics_collector = _RecordingRadixCacheMetricsCollector(
            labels={"cache_type": "UnifiedRadixCache"}
        )
        cache.tree_core.kv_age_observer = cache._observe_kv_age_event
        return cache, allocator

    def test_match_then_evict_records_hit_and_evict_ages(self):
        cache, allocator = self._build_cache()
        collector = cache.metrics_collector
        tokens = array("q", [1, 2, 3, 4])
        indices = allocator.alloc(len(tokens))
        cache.insert(InsertParams(key=RadixKey(token_ids=tokens), value=indices))
        self.assertEqual(collector.kv_age_seconds.observations, [])

        req = _req("r1", tokens)
        cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens), req=req))
        # Re-matches of the same request (scheduling rounds, post-insert
        # re-anchor) and request-less matches must not count again.
        cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens), req=req))
        cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens)))
        hits = [
            (lab, v)
            for lab, v in collector.kv_age_seconds.observations
            if lab["event"] == "hit"
        ]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0]["tier"], "device")
        self.assertGreaterEqual(hits[0][1], 0.0)
        self.assertEqual(
            [
                v
                for lab, v in collector.kv_age_tokens.increments
                if lab["event"] == "hit"
            ],
            [len(tokens)],
        )

        cache.evict(EvictParams(num_tokens=len(tokens)))
        evicts = [
            (lab, v)
            for lab, v in collector.kv_age_seconds.observations
            if lab["event"] == "evict"
        ]
        self.assertEqual(len(evicts), 1)
        self.assertEqual(evicts[0][0]["tier"], "device")
        self.assertEqual(evicts[0][0]["outcome"], "dropped")
        want = {
            "cache_type": "UnifiedRadixCache",
            "tier": "device",
            "outcome": "dropped",
        }
        self.assertEqual(len(collector.kv_lifetime_seconds.observations), 1)
        self.assertEqual(collector.kv_lifetime_seconds.observations[0][0], want)
        self.assertEqual(len(collector.kv_reuses.observations), 1)
        self.assertEqual(collector.kv_reuses.observations[0][0], want)
        self.assertEqual(
            [
                v
                for lab, v in collector.kv_age_tokens.increments
                if lab["event"] == "evict"
            ],
            [len(tokens)],
        )


if __name__ == "__main__":
    unittest.main()
