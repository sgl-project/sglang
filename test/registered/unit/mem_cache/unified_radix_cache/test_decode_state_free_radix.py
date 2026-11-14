"""State-free PD decode radix tree (SGLANG_DECODE_STATE_FREE_RADIX): the tree
keeps FULL KV only while each request keeps its own Mamba state slot. A
finished request's prefix is still cached and matched in full, its state slot
goes back to the pool instead of into the tree, and nothing ever asks the tree
for a Mamba state (no copy-on-write, no eviction for state slots)."""

import unittest
from array import array
from dataclasses import replace
from unittest import mock

from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocation import ensure_mamba_capacity
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.kv.plan import TREE_STATE
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

# The fixture builds the hybrid pools through a no-buffer Mamba tree first,
# which needs page 1; the FULL-only tree itself is page-agnostic.
PAGE = 1


def _state_free_cache():
    """Pools of a hybrid-SSM model, tree of FULL KV only."""
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA),
        page_size=PAGE,
        mamba_cache_size=8,
    )
    hybrid, allocator, req_to_token_pool = build_fixture(cfg)
    params = replace(
        hybrid.cache_init_params,
        tree_slots=tuple(
            slot
            for slot in hybrid.cache_init_params.tree_slots
            if slot.family != TREE_STATE
        ),
        request_mamba_only=True,
        enable_mamba_extra_buffer=False,
    )
    return UnifiedRadixCache(params=params), allocator, req_to_token_pool


def _request(req_to_token_pool, allocator, tokens, rid):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    acquire_rows(req_to_token_pool, req_to_token_pool.state, [req])
    req.output_ids = array("q")
    kv = allocator.alloc(len(tokens))
    req_to_token_pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), kv)
    req.kv.kv_committed_len = len(tokens)
    req.kv.kv_allocated_len = len(tokens)
    req.kv.cache_protected_len = 0
    req.lock = None
    req.extra_key = None
    return req


class TestDecodeStateFreeRadix(CustomTestCase):
    def test_finished_request_is_cached_and_frees_its_slot(self):
        cache, allocator, req_to_token_pool = _state_free_cache()
        mamba_free = req_to_token_pool.mamba_allocator.available_size()
        tokens = list(range(1, 13))
        req = _request(req_to_token_pool, allocator, tokens, "finished")
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), mamba_free - 1
        )
        req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(req, cache, checkpoint=True)
        cache.unlock(req.lock)
        self.assertFalse(req.kv.holds_mamba)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), mamba_free)
        # Every page of the prompt is cached; no checkpoint caps the insert.
        self.assertEqual(match_prefix(cache, tokens).device_prefix_len, len(tokens))
        cache.sanity_check()

    def test_request_released_before_kv_frees_its_slot(self):
        cache, _, req_to_token_pool = _state_free_cache()
        mamba_free = req_to_token_pool.mamba_allocator.available_size()
        req = Req(
            rid="early",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3]),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        # the decode pool hands out a state slot before any KV
        req.kv.mamba_pool_idx = req_to_token_pool.mamba_allocator.alloc(1)[0]
        self.assertFalse(req.kv.holds_kv)
        release_kv_cache(req, cache, checkpoint=False)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), mamba_free)

    def test_state_shortfall_never_evicts_the_tree(self):
        cache, _, req_to_token_pool = _state_free_cache()
        allocator = req_to_token_pool.mamba_allocator
        held = allocator.alloc(allocator.available_size())
        req = Req(
            rid="short",
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        with mock.patch.object(
            cache, "evict_for_alloc", side_effect=AssertionError("evicted")
        ):
            self.assertFalse(
                ensure_mamba_capacity(
                    req_to_token_pool, req_to_token_pool.state, [req], cache
                )
            )
        allocator.free(held)


if __name__ == "__main__":
    unittest.main()
