"""CPU-only unit tests for --enable-cache-miss-attribution."""

import random
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.miss_attribution import CacheMissAttribution
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

    def test_each_cause_on_a_known_history(self):
        """600 tokens computed with states saved at 256 and 512, then 100 new
        tokens: placement is what lies past the last saved state, dependency
        cached KV whose state is gone, capacity KV and state both gone."""
        for hit, kv, dependency, capacity in (
            (512, 512, 0, 0),
            (256, 600, 256, 0),
            (256, 256, 0, 256),
            (0, 300, 256, 256),
        ):
            with self.subTest(hit=hit, kv=kv):
                self.assertEqual(
                    self.index.attribute(self.prompt, len(self.prompt), hit, kv),
                    {
                        "new": 100,
                        "placement": 88,
                        "dependency": dependency,
                        "capacity": capacity,
                    },
                )
        fresh = CacheMissAttribution(page_size=256, records_states=True)
        self.assertEqual(
            fresh.attribute(self.prompt, len(self.prompt), 0, 0),
            {"new": 700, "placement": 0, "dependency": 0, "capacity": 0},
        )

    def test_a_divergence_inside_a_block_is_found_to_the_token(self):
        prompt = self.history[:290] + _tokens(3, 50)
        split = self.index.attribute(prompt, len(prompt), 256, 290)
        self.assertEqual(split["new"], len(prompt) - 290)
        self.assertEqual(split["placement"], 290 - 256)

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


class TestCacheMissAttributionInTheTree(CustomTestCase):
    def _cache(self):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.MAMBA), mamba_cache_size=8
        )
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            cache, allocator, pool = build_fixture(cfg)
        cache.miss_attribution = CacheMissAttribution(page_size=1, records_states=True)
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

    def test_the_first_extend_splits_the_miss_and_later_chunks_keep_it(self):
        """The scheduler splits a prompt's miss at its first extend, from the
        hit and the KV the match found beyond it; later chunks keep it."""
        cache, allocator, pool = self._cache()
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
        args = ServerArgs(
            model_path="dummy",
            enable_cache_miss_attribution=True,
            disable_radix_cache=True,
        )
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            handle_cache_compatibility(args)


class TestCacheMissMetrics(CustomTestCase):
    def test_the_counter_exists_only_with_the_switch_and_adds_the_parts(self):
        class _Collector(TokenizerMetricsCollector):
            _counter_cls = _gauge_cls = _histogram_cls = staticmethod(
                lambda *args, **kwargs: MagicMock()
            )

        def collector(enable):
            args = ServerArgs(model_path="dummy", enable_cache_miss_attribution=enable)
            return _Collector(server_args=args, labels={"model": "m"})

        self.assertIsNone(collector(False).cache_miss_tokens_total)
        on = collector(True)
        parts = {"new": 3, "placement": 2, "dependency": 5, "capacity": 0}
        on.observe_one_finished_request(
            {"model": "m"}, 20, 4, 10, 1.0, False, cache_miss_tokens=parts
        )
        counter = on.cache_miss_tokens_total
        self.assertEqual(
            counter.labels.call_args_list,
            [call(model="m", cause=cause) for cause in parts],
        )
        self.assertEqual(
            counter.labels.return_value.inc.call_args_list,
            [call(n) for n in parts.values()],
        )


if __name__ == "__main__":
    unittest.main()
