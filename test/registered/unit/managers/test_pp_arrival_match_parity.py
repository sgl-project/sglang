"""Arrival matching must mutate each PP stage's cache before grammar waiting."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures require CUDA")
class TestArrivalMatchEviction(UnifiedRadixCacheFixtures, CustomTestCase):
    cfg = CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=1024,
        max_context_len=1024,
    )

    def _stage(self, rank, *, match_on_arrival=True, ticket_enabled=True):
        cache, allocator, pool = build_fixture(self.cfg)
        shared = self._make_seq(1, 2)
        self._insert(cache, allocator, pool, shared)
        # Keep the shared checkpoint internal so losing Mamba need not lose FULL.
        self._insert(cache, allocator, pool, shared + self._make_seq(9000, 2))
        self._insert(cache, allocator, pool, self._make_seq(1000, 2))
        req = self._make_req(pool)
        req.origin_input_ids = array("q", shared + self._make_seq(5000, 1))
        cache.cache_controller = SimpleNamespace(
            pp_prefetch_command_group=object() if ticket_enabled else None
        )
        scheduler = SimpleNamespace(
            tree_cache=cache,
            _prefetch_kvcache=lambda r: r.init_next_round_input(cache, cow_mamba=False),
        )
        # The request stays out of the enqueue path while its grammar is pending.
        if match_on_arrival:
            with (
                patch(
                    "sglang.srt.managers.scheduler.get_parallel",
                    return_value=SimpleNamespace(pp_rank=rank, pp_size=4),
                ),
                patch.object(cache, "bind_prefetch_ticket"),
            ):
                Scheduler._bind_pp_arrival_prefetch(scheduler, req, SimpleNamespace())
        cache.cache_controller = None
        cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        result = match_prefix(cache, shared)
        return result.device_prefix_len, result.full_kv_hit_length

    def test_missing_arrival_match_changes_eviction(self):
        self.assertEqual(self._stage(0), (128, 128))
        self.assertEqual(self._stage(1, match_on_arrival=False), (0, 128))
        self.assertEqual(self._stage(1, ticket_enabled=False), (0, 128))

    def test_every_stage_preserves_the_same_checkpoint(self):
        self.assertEqual([self._stage(rank) for rank in range(4)], [(128, 128)] * 4)


if __name__ == "__main__":
    unittest.main()
