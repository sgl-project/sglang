"""Tests for --allow-subagent-keepalive on UnifiedRadixCache."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

TURN_LEN = 4
PARENT_TOKENS = [10, 11, 12, 13]
CHILD_TOKENS = [20, 21, 22, 23]
THIRD_TOKENS = [30, 31, 32, 33]


def make_cache(allow_subagent_keepalive: bool):
    """FULL-only, page_size=1, CPU cache with room for a handful of turns."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="lru",
        allow_subagent_keepalive=allow_subagent_keepalive,
        tree_components=(ComponentType.FULL,),
    )
    return UnifiedRadixCache(params), allocator, req_pool


class SubagentKeepaliveTestBase(CustomTestCase):
    allow_subagent_keepalive = True

    def setUp(self):
        self.cache, self.allocator, self.req_pool = make_cache(
            self.allow_subagent_keepalive
        )
        self._rid = 0

    def finish_turn(self, tokens, session_id):
        """Run one request of ``session_id`` to completion through the cache."""
        req = Req(
            rid=str(self._rid),
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            session_id=session_id,
        )
        self._rid += 1
        self.req_pool.alloc([req])
        req.output_ids = array("q")

        kv_indices = self.allocator.alloc(len(tokens))
        self.req_pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), kv_indices)
        req.kv.kv_committed_len = len(tokens)
        req.last_node = self.cache.root_node.id
        req.kv.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )

        self.cache.cache_finished_req(
            req, is_insert=True, kv_len_to_handle=req.effective_kv_committed_len()
        )
        return req

    def match_len(self, tokens) -> int:
        return len(
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            ).device_indices
        )


class TestSubagentKeepaliveEnabled(SubagentKeepaliveTestBase):
    allow_subagent_keepalive = True

    def test_keepalive_promotes_the_parent_over_a_newer_session(self):
        """A bumped parent outranks the subagent turn that came after it."""
        self.finish_turn(PARENT_TOKENS, "parent")
        self.finish_turn(CHILD_TOKENS, "child")

        self.assertTrue(self.cache.bump_session_keepalive("parent"))

        self.cache.evict(EvictParams(num_tokens=TURN_LEN))

        self.assertEqual(self.match_len(PARENT_TOKENS), TURN_LEN)
        self.assertEqual(self.match_len(CHILD_TOKENS), 0)

    def test_without_a_keepalive_the_parent_is_evicted_first(self):
        """Control for the test above: plain LRU drops the older parent."""
        self.finish_turn(PARENT_TOKENS, "parent")
        self.finish_turn(CHILD_TOKENS, "child")

        self.cache.evict(EvictParams(num_tokens=TURN_LEN))

        self.assertEqual(self.match_len(PARENT_TOKENS), 0)
        self.assertEqual(self.match_len(CHILD_TOKENS), TURN_LEN)

    def test_keepalive_does_not_promote_past_a_later_bump(self):
        """Repeated keepalives re-age; the least recently kept session goes first."""
        self.finish_turn(PARENT_TOKENS, "parent")
        self.finish_turn(CHILD_TOKENS, "child")
        self.finish_turn(THIRD_TOKENS, "third")

        self.cache.bump_session_keepalive("parent")
        self.cache.bump_session_keepalive("child")

        self.cache.evict(EvictParams(num_tokens=TURN_LEN))

        self.assertEqual(self.match_len(THIRD_TOKENS), 0)
        self.assertEqual(self.match_len(PARENT_TOKENS), TURN_LEN)
        self.assertEqual(self.match_len(CHILD_TOKENS), TURN_LEN)

    def test_unknown_session_is_a_miss(self):
        self.finish_turn(PARENT_TOKENS, "parent")
        self.assertFalse(self.cache.bump_session_keepalive("never-seen"))

    def test_blank_session_id_is_a_miss(self):
        self.assertFalse(self.cache.bump_session_keepalive(""))

    def test_reclaimed_path_drops_the_session_entry(self):
        """Once a session's KV is gone the map must not keep pointing at it."""
        self.finish_turn(PARENT_TOKENS, "parent")
        self.assertIn("parent", self.cache._session_tail_node)

        self.cache.evict(EvictParams(num_tokens=TURN_LEN))

        self.assertFalse(self.cache.bump_session_keepalive("parent"))
        self.assertNotIn("parent", self.cache._session_tail_node)

    def test_request_without_a_session_id_is_not_tracked(self):
        self.finish_turn(PARENT_TOKENS, None)
        self.assertEqual(len(self.cache._session_tail_node), 0)

    def test_later_turn_replaces_the_recorded_tail(self):
        """The map tracks the session's newest tail, not its first."""
        self.finish_turn(PARENT_TOKENS, "parent")
        first_tail = self.cache._session_tail_node["parent"]

        self.finish_turn(PARENT_TOKENS + THIRD_TOKENS, "parent")
        second_tail = self.cache._session_tail_node["parent"]

        self.assertNotEqual(first_tail, second_tail)
        self.assertTrue(self.cache.bump_session_keepalive("parent"))


class TestSubagentKeepaliveDisabled(SubagentKeepaliveTestBase):
    allow_subagent_keepalive = False

    def test_nothing_is_recorded(self):
        self.finish_turn(PARENT_TOKENS, "parent")
        self.assertEqual(len(self.cache._session_tail_node), 0)

    def test_bump_is_inert(self):
        self.finish_turn(PARENT_TOKENS, "parent")
        self.finish_turn(CHILD_TOKENS, "child")

        self.assertFalse(self.cache.bump_session_keepalive("parent"))

        self.cache.evict(EvictParams(num_tokens=TURN_LEN))

        # Same verdict as plain LRU: the flag changes nothing when it is off.
        self.assertEqual(self.match_len(PARENT_TOKENS), 0)
        self.assertEqual(self.match_len(CHILD_TOKENS), TURN_LEN)


if __name__ == "__main__":
    unittest.main()
