"""A request left unadmitted after a Mamba COW match must not keep a stale COW source."""

import unittest
from array import array

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import (
    node_device_value,
    node_host_value,
)
from sglang.test.kits.unified_radix_cache.hicache import (
    UnifiedRadixCacheHiCacheFixtures,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "mamba pool fixtures need CUDA")
class TestMambaStaleCowSource(UnifiedRadixCacheHiCacheFixtures, CustomTestCase):
    cfg = CacheConfig(components=(ComponentType.FULL, ComponentType.MAMBA))

    def test_rematch_after_device_mamba_eviction_drops_cow_source(self):
        # M = seq_a's internal node, with a device Mamba state.
        cache, req_to_token_pool, seq_a = self._build_internal_mamba_fixture(
            "write_back"
        )
        token_ids = array("q", seq_a)

        # B is COW-matched against M (init_next_round_input), then the
        # admission loop `continue`s: B keeps its slot and its COW source.
        req = self._make_req(req_to_token_pool)
        first = match_kv_cache(req, cache, token_ids, cow_mamba=True)
        node_m = first.best_match_node
        stale_src = int(req.kv.mamba_cow_src_index)
        self.assertEqual(
            stale_src, int(node_device_value(cache, node_m, ComponentType.MAMBA))
        )

        # M's device state is evicted; write_back demotes it to host.
        cache.evict(EvictParams(num_tokens=0, mamba_num=10))
        self.assertIsNone(node_device_value(cache, node_m, ComponentType.MAMBA))
        self.assertIsNotNone(node_host_value(cache, node_m, ComponentType.MAMBA))
        src_freed = stale_src in req_to_token_pool.mamba_allocator.free_slots

        # Next admission pass matches B again.
        second = match_kv_cache(req, cache, token_ids, cow_mamba=True)
        src = req.kv.mamba_cow_src_index
        if src is not None:
            live = node_device_value(cache, second.best_match_node, ComponentType.MAMBA)
            self.assertTrue(
                live is not None and int(live) == int(src),
                f"stale COW source slot {int(src)} survives the rematch: "
                f"best_match_node device mamba={live}, "
                f"slot back in free list={src_freed}",
            )


if __name__ == "__main__":
    unittest.main()
