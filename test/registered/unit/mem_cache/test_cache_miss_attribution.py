"""CPU-only unit tests for --enable-cache-miss-attribution."""

import random
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.miss_attribution import MISS_PARTS, CacheMissAttribution
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _tokens(seed, n):
    rng = random.Random(seed)
    return array("q", (rng.randrange(1, 150_000) for _ in range(n)))


class TestCacheMissAttributionParts(CustomTestCase):
    """Each part on a hand-built history; block 256, the default."""

    def setUp(self):
        self.index = CacheMissAttribution(page_size=256, records_states=True)
        self.history = _tokens(0, 600)
        self.index.record_computed(self.history, 600)
        for position in (256, 512):
            self.index.record_state(self.history, position)
        self.prompt = self.history + _tokens(1, 100)

    def split(self, hit, kv):
        return self.index.attribute(self.prompt, len(self.prompt), hit, kv)

    def test_placement_is_what_lies_past_the_last_saved_state(self):
        self.assertEqual(
            self.split(hit=512, kv=512),
            {"new": 100, "placement": 88, "dependency": 0, "capacity": 0},
        )

    def test_dependency_is_cached_kv_whose_state_is_gone(self):
        self.assertEqual(
            self.split(hit=256, kv=600),
            {"new": 100, "placement": 88, "dependency": 256, "capacity": 0},
        )

    def test_capacity_is_kv_and_state_both_gone(self):
        self.assertEqual(
            self.split(hit=256, kv=256),
            {"new": 100, "placement": 88, "dependency": 0, "capacity": 256},
        )
        self.assertEqual(
            self.split(hit=0, kv=300),
            {"new": 100, "placement": 88, "dependency": 256, "capacity": 256},
        )

    def test_never_computed_prompts_are_new(self):
        fresh = CacheMissAttribution(page_size=256, records_states=True)
        prompt = _tokens(2, 300)
        self.assertEqual(
            fresh.attribute(prompt, 300, 0, 0),
            {"new": 300, "placement": 0, "dependency": 0, "capacity": 0},
        )

    def test_a_divergence_inside_a_block_is_found_to_the_token(self):
        prompt = self.history[:290] + _tokens(3, 50)
        split = self.index.attribute(prompt, len(prompt), 256, 290)
        self.assertEqual(split["new"], len(prompt) - 290)
        self.assertEqual(split["placement"], 290 - 256)

    def test_a_state_inside_a_block_counts_only_on_its_own_prefix(self):
        self.index.record_state(self.history, 580)
        self.assertEqual(self.split(hit=580, kv=580)["placement"], 20)
        other = self.history[:579] + array("q", [7]) + _tokens(4, 30)
        split = self.index.attribute(other, len(other), 512, 579)
        self.assertEqual(split["placement"], 579 - 512)

    def test_a_state_on_a_branch_that_parted_earlier_does_not_count(self):
        self.index.record_state(self.history, 580)
        branch = self.history[:550] + _tokens(6, 50)
        self.index.record_computed(branch, 600)
        prompt = branch + _tokens(7, 20)
        split = self.index.attribute(prompt, len(prompt), 512, 512)
        self.assertEqual(split["placement"], 600 - 512)

    def test_every_continuation_of_a_shared_prefix_counts(self):
        index = CacheMissAttribution(page_size=256, records_states=True)
        shared = _tokens(8, 256)
        index.record_computed(shared + _tokens(9, 100), 356)
        second = shared + _tokens(10, 100)
        index.record_computed(second, 356)
        prompt = second[:306] + _tokens(11, 30)
        self.assertEqual(index.attribute(prompt, len(prompt), 0, 0)["new"], 30)

    def test_a_shared_prefix_keeps_its_latest_continuations(self):
        """Eight continuations and saved positions per prefix at most
        (max_entries // 8): the oldest prompt's continuation is new again."""
        index = CacheMissAttribution(page_size=256, records_states=True, max_entries=64)
        shared = _tokens(8, 256)
        prompts = [shared + _tokens(20 + i, 100) for i in range(10)]
        for prompt in prompts:
            index.record_computed(prompt, 356)
            index.record_state(prompt, 300)
        self.assertEqual(index._num_states, 8)
        latest = prompts[-1][:306] + _tokens(11, 30)
        oldest = prompts[0][:306] + _tokens(11, 30)
        self.assertEqual(
            index.attribute(latest, len(latest), 0, 0),
            {"new": 30, "placement": 6, "dependency": 0, "capacity": 300},
        )
        self.assertEqual(index.attribute(oldest, len(oldest), 0, 0)["new"], 80)

    def test_the_index_stays_within_max_entries(self):
        index = CacheMissAttribution(
            page_size=64, records_states=True, block=64, max_entries=40
        )
        prompts = [_tokens(300 + i, 200) for i in range(60)]
        for prompt in prompts:
            index.record_computed(prompt, 200)
            index.record_state(prompt, 192)
        self.assertLessEqual(index._num_continuations, 40)
        self.assertLessEqual(index._num_states, 40)
        latest = prompts[-1] + _tokens(12, 10)
        self.assertEqual(
            index.attribute(latest, 210, 0, 0),
            {"new": 10, "placement": 8, "dependency": 0, "capacity": 192},
        )

    def test_the_last_prompt_token_is_always_computed(self):
        prompt = self.history[:400]
        split = self.index.attribute(prompt, 400, 256, 399)
        self.assertEqual(split["new"], 1)

    def test_without_states_every_page_boundary_is_saved(self):
        index = CacheMissAttribution(page_size=16, records_states=False)
        index.record_computed(self.history, 600)
        split = index.attribute(self.prompt, len(self.prompt), 592, 592)
        self.assertEqual(split["placement"], 600 - 592)
        self.assertEqual(split["dependency"], 0)

    def test_a_request_extends_its_hashes_instead_of_rehashing(self):
        tokens = _tokens(5, 2000)
        index = CacheMissAttribution(page_size=256, records_states=True)
        reference = CacheMissAttribution(page_size=256, records_states=True)
        for end in (700, 1300, 2000):
            index.record_computed(tokens, end, request_id="r", finished=end == 2000)
            reference.record_computed(tokens, end)
        self.assertEqual(index._continuations, reference._continuations)
        self.assertNotIn("r", index._request_chains)

    def test_parts_tile_the_miss_on_random_histories(self):
        rng = random.Random(7)
        index = CacheMissAttribution(page_size=64, records_states=True, block=64)
        sequences = [_tokens(100 + i, 400) for i in range(4)]
        for step in range(300):
            seq = sequences[rng.randrange(4)]
            end = rng.randint(1, len(seq))
            if rng.random() < 0.5:
                index.record_computed(seq, end)
            else:
                index.record_state(seq, end)
            base = sequences[rng.randrange(4)]
            cut = rng.randint(1, len(base))
            prompt = base[:cut] + _tokens(1000 + step, rng.randint(1, 50))
            kv = rng.randint(0, len(prompt) - 1)
            hit = rng.randint(0, kv)
            split = index.attribute(prompt, len(prompt), hit, kv)
            self.assertEqual(tuple(split), MISS_PARTS)
            self.assertTrue(all(v >= 0 for v in split.values()), split)
            self.assertEqual(sum(split.values()), len(prompt) - hit, split)


class TestCacheMissAttributionInTheTree(CustomTestCase):
    def _cache(self, enable):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.MAMBA), mamba_cache_size=8
        )
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            cache, allocator, pool = build_fixture(cfg)
        if enable:
            cache.miss_attribution = CacheMissAttribution(
                page_size=1, records_states=True
            )
        return cache, allocator, pool

    def _req(self, pool, rid, tokens=()):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        acquire_rows(pool, pool.state, [req])
        return req

    def test_an_insert_with_a_state_records_its_position(self):
        cache, allocator, pool = self._cache(enable=True)
        tokens = [1, 2, 3, 4]
        req = self._req(pool, "state")
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=allocator.alloc(4),
                mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
            )
        )
        index = cache.miss_attribution
        index.record_computed(array("q", tokens), 4)
        prompt = array("q", tokens + [9])
        self.assertEqual(index.attribute(prompt, 5, 0, 0)["capacity"], 4)

    def test_release_forgets_the_request(self):
        cache, _, pool = self._cache(enable=True)
        req = self._req(pool, "released")
        cache.miss_attribution.record_computed(array("q", [1, 2]), 2, "released")
        cache.on_release(req, checkpointed=False)
        self.assertNotIn("released", cache.miss_attribution._request_chains)

    def test_a_match_keeps_the_cached_kv_length(self):
        cache, allocator, pool = self._cache(enable=False)
        self.assertIsNone(cache.miss_attribution)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=allocator.alloc(4),
                allow_missing_mamba=True,
            )
        )
        req = self._req(pool, "match", [1, 2, 3, 4, 5])
        match_kv_cache(req, cache, array("q", [1, 2, 3, 4, 5]))
        self.assertEqual(req.prefix_len, 0)
        self.assertEqual(req.full_kv_hit_length, 4)

    def test_the_first_extend_splits_the_miss_and_later_chunks_keep_it(self):
        """The scheduler splits a prompt's miss at its first extend, from the
        hit and the KV the match found beyond it; later chunks keep it."""
        cache, allocator, pool = self._cache(enable=True)
        for tokens in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6]):
            state = self._req(pool, f"turn-{len(tokens)}")
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)),
                    value=allocator.alloc(len(tokens)),
                    mamba_value=state.kv.mamba_pool_idx.unsqueeze(0),
                )
            )
            cache.miss_attribution.record_computed(array("q", tokens), len(tokens))
        # The older state goes; the KV under it stays for the deeper one.
        cache.evict(EvictParams(mamba_num=1))
        req = self._req(pool, "resumed", [1, 2, 3, 4, 9, 9])
        req.init_next_round_input(cache)
        req.extend_end = len(req.origin_input_ids)
        self.assertEqual((req.prefix_len, req.full_kv_hit_length), (0, 4))
        batch = ScheduleBatch(
            reqs=[req],
            device="cpu",
            tree_cache=cache,
            model_config=SimpleNamespace(is_encoder_decoder=False, vocab_size=128),
        )
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=False),
            mamba=SimpleNamespace(enable_mamba_extra_buffer=False),
        )
        with (
            patch(
                "sglang.srt.managers.schedule_batch.get_exec", return_value=execution
            ),
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_extend",
                return_value=(torch.arange(6), torch.tensor([0]), torch.tensor([0])),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.SamplingBatchInfo.from_schedule_batch"
            ),
        ):
            batch.prepare_for_extend()
            split = {"new": 2, "placement": 0, "dependency": 4, "capacity": 0}
            self.assertEqual(req.cache_miss_tokens, split)
            req.full_kv_hit_length = 0
            batch.prepare_for_extend()
            self.assertEqual(req.cache_miss_tokens, split)

    def test_server_args_refuse_it_without_the_radix_cache(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility

        args = ServerArgs(
            model_path="dummy",
            enable_cache_miss_attribution=True,
            disable_radix_cache=True,
        )
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            handle_cache_compatibility(args)


class _RecordingMetric:
    """A prometheus_client-shaped metric that records its increments."""

    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.increments = []

    def labels(self, **labels):
        metric = self

        class _Bound:
            def inc(self, value=1):
                metric.increments.append((labels, value))

            def observe(self, value):
                pass

            def set(self, value):
                pass

        return _Bound()


class _RecordingTokenizerMetrics(TokenizerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class TestCacheMissMetrics(CustomTestCase):
    def _collector(self, enable):
        args = ServerArgs(model_path="dummy", enable_cache_miss_attribution=enable)
        return _RecordingTokenizerMetrics(server_args=args, labels={"model": "m"})

    def test_the_counter_exists_only_with_the_switch(self):
        self.assertIsNone(self._collector(False).cache_miss_tokens_total)
        self.assertIsNotNone(self._collector(True).cache_miss_tokens_total)

    def test_a_finished_request_adds_its_parts(self):
        collector = self._collector(True)
        parts = {"new": 3, "placement": 2, "dependency": 5, "capacity": 0}
        collector.observe_one_finished_request(
            {"model": "m"}, 20, 4, 10, 1.0, False, cache_miss_tokens=parts
        )
        self.assertEqual(
            collector.cache_miss_tokens_total.increments,
            [({"model": "m", "cause": cause}, n) for cause, n in parts.items()],
        )


if __name__ == "__main__":
    unittest.main()
