"""Tests for session references on UnifiedRadixCache."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import dataclasses
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase


def make_params(enable_session: bool) -> CacheInitParams:
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
    return CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="lru",
        enable_session_radix_cache=enable_session,
        tree_components=(ComponentType.FULL,),
    )


def insert(cache, token_ids):
    """Insert and return the tail node; the cache boundary hands back a NodeId."""
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    node_id = cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=indices.to(torch.int64),
        )
    ).last_device_node
    return cache.tree_core.node_by_id(node_id)


def match_len(cache, token_ids) -> int:
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids)))
    ).device_prefix_len


def register(cache, token_ids, session_id, generation=None):
    if generation is None:
        generation = cache.ensure_session_generation(session_id)
    leaf = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids)))
    ).last_device_node
    cache.session_refs.register_session_ref(
        SimpleNamespace(
            session_id=session_id,
            session_generation=generation,
            session=None,
            origin_input_ids=array("q", token_ids),
            output_ids=array("q"),
            extra_key=None,
        ),
        leaf=leaf,
    )


class TestSessionUnifiedRadixCache(CustomTestCase):
    def setUp(self):
        self.cache = UnifiedRadixCache(make_params(enable_session=True))
        self.full = self.cache.components[ComponentType.FULL]

    def test_explicit_rust_selection_uses_python_session_semantics(self):
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
            cache = UnifiedRadixCache(make_params(enable_session=True))
        self.assertEqual(cache._tree_core_backend, "python")
        self.assertIsInstance(cache.tree_core, UnifiedTreeCore)
        leaf = insert(cache, [1, 2, 3, 4])
        generation = cache.open_radix_session("fallback")
        register(cache, [1, 2, 3, 4], "fallback", generation)
        full = cache.components[ComponentType.FULL]
        self.assertEqual(full.session_ref(leaf), 1)
        cache.release_radix_session("fallback")
        self.assertEqual(full.session_ref(leaf), 0)
        cache.sanity_check()

    def test_register_and_release_update_full_component_reference(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        generation = self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", generation)
        self.assertEqual(self.full.session_ref(leaf), 1)

        self.cache.release_radix_session("s1")
        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_reopen_rejects_stale_generation(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        old_generation = self.cache.open_radix_session("s1")
        self.cache.release_radix_session("s1")
        self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", old_generation)

        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_eviction_prefers_unreferenced_full_kv(self):
        referenced = insert(self.cache, [1, 2, 3, 4])
        insert(self.cache, [7, 8, 9])
        register(self.cache, [1, 2, 3, 4], "s1")

        self.cache.evict(EvictParams(num_tokens=3))

        self.assertEqual(match_len(self.cache, [7, 8, 9]), 0)
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 4)
        self.assertEqual(self.full.session_ref(referenced), 1)


FULL, SWA = ComponentType.FULL, ComponentType.SWA


class TestSessionRefPromptTail(UnifiedRadixCacheFixtures, CustomTestCase):
    """SGLANG_OPT_SESSION_REF_PROMPT_TAIL, for clients whose next turn resends the
    history (a re-rendered assistant turn) instead of extending the generated
    output: every finished output is then a dead branch of its session."""

    OUT = 6  # longer than window + page, so a stock SWA ref lands in the output
    cfg = CacheConfig(
        components=(FULL, SWA),
        sliding_window_size=4,
        kv_size=1024,
        max_context_len=1024,
    )

    def _cache(self, *, prompt_tail: bool, page_size: int = 1):
        # The kit's helpers read self.cfg.
        self.cfg = dataclasses.replace(type(self).cfg, page_size=page_size)
        with envs.SGLANG_OPT_SESSION_REF_PROMPT_TAIL.override(prompt_tail):
            fx = build_fixture(self.cfg, enable_session_radix_cache=True)
        # Host tier simulated at the tree level; host frees are bookkeeping only.
        fx[0].tree_core.set_hicache_enabled()
        fx[0].tree_core.has_swa_host_pool = True
        return fx

    def _finish(self, fx, session_id, prompt, output, managed=False):
        """The post-prefill checkpoint, then the finished insert and the session
        registration cache_finished_req makes; both inserts are page-aligned.
        ``managed`` makes it a server-managed (session_params) session's request."""
        cache, page = fx[0], self.cfg.page_size
        if len(prompt) >= page:
            self._insert(*fx, prompt[: len(prompt) // page * page])
        tokens = prompt + output
        leaf = self._insert(*fx, tokens[: len(tokens) // page * page]).last_device_node
        req = SimpleNamespace(
            session_id=None if managed else session_id,
            session_generation=cache.ensure_session_generation(session_id),
            session=(
                SimpleNamespace(session_id=session_id, streaming=False)
                if managed
                else None
            ),
            origin_input_ids=array("q", prompt),
            output_ids=array("q", output),
            extra_key=None,
        )
        cache.session_refs.register_session_ref(req, leaf=leaf)

    def _turns(self, fx, session_id, prompt, turns):
        """Run ``turns`` turns; return the last prompt."""
        for k in range(turns):
            if k:
                rerendered = [7000 + 100 * k + i for i in range(3)]
                prompt = prompt + rerendered + [8000 + 100 * k + i for i in range(5)]
            output = [9000 + 100 * k + i for i in range(self.OUT)]
            self._finish(fx, session_id, prompt, output)
        return prompt

    def _nodes(self, cache):
        root = cache.tree_core.root_node
        return [n for n in cache.tree_core._node_arena.values() if n is not root]

    def _end(self, cache, node):
        end = 0
        while node is not cache.tree_core.root_node:
            end += len(node.key)
            node = node.parent
        return end

    def _hit(self, cache, tokens):
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        return m.device_prefix_len + m.host_hit_length

    def test_one_ref_per_session_at_its_last_prompt_end(self):
        # Off: stock refs pile up, one per finished output branch.
        stock = self._cache(prompt_tail=False)
        self._turns(stock, "s", list(range(1, 13)), 3)
        self.assertEqual(len(stock[0].components[FULL]._session_leaves["s"]), 3)

        fx = self._cache(prompt_tail=True)
        cache = fx[0]
        prompt = self._turns(fx, "s", list(range(1, 13)), 3)
        (leaf,) = cache.components[FULL]._session_leaves["s"]
        self.assertEqual(cache.components[SWA]._session_leaves["s"], {leaf})
        self.assertEqual(self._end(cache, leaf), len(prompt))
        for node in self._nodes(cache):
            if min(node.key.token_ids) >= 9000:  # live outputs
                for ct in (FULL, SWA):
                    self.assertEqual(cache.components[ct].session_ref(node), 0)
        cache.sanity_check()

        cache.release_radix_session("s")
        for node in self._nodes(cache):
            for ct in (FULL, SWA):
                self.assertEqual(cache.components[ct].session_ref(node), 0)
        cache.sanity_check()

    def test_host_eviction_keeps_each_sessions_next_window(self):
        fx = self._cache(prompt_tail=True)
        cache = fx[0]
        idle = self._turns(fx, "idle", list(range(5000, 5016)), 1)
        busy = self._turns(fx, "busy", list(range(1, 13)), 3)
        self._simulate_backup_tree(cache)
        cache.evict(EvictParams(num_tokens=10**6, swa_num_tokens=10**6))
        unreferenced = sum(
            len(n.component_data[SWA].host_value)
            for n in cache.tree_core.host_lru_lists[SWA].cache.values()
            if n.component_data[SWA].session_ref == 0
        )
        self.assertGreater(unreferenced, 0)

        cache.evict_host(unreferenced, SWA)

        # The next turn of each session still hits its whole previous prompt.
        self.assertEqual(self._hit(cache, idle + [7100, 8100]), len(idle))
        self.assertEqual(self._hit(cache, busy + [7300, 8300]), len(busy))
        cache.sanity_check()

    def test_server_managed_session_keeps_its_ref_on_the_output(self):
        """A session_params session builds its next prompt from the previous
        output, so the output it will extend must stay referenced."""
        fx = self._cache(prompt_tail=True)
        cache = fx[0]
        prompt, output = list(range(1, 13)), [9000 + i for i in range(self.OUT)]
        self._finish(fx, "s", prompt, output, managed=True)
        (leaf,) = cache.components[FULL]._session_leaves["s"]
        self.assertEqual(self._end(cache, leaf), len(prompt + output))

    def test_prompt_shorter_than_a_node_releases_the_previous_ref(self):
        """A turn anchored at the root must still drop the session's older ref,
        or that dead branch stays protected until the session's next turn."""
        fx = self._cache(prompt_tail=True, page_size=4)
        cache = fx[0]
        self._turns(fx, "s", list(range(1, 13)), 1)
        self.assertIn("s", cache.components[FULL]._session_leaves)

        # Under one page, so no node ends within this prompt.
        self._finish(fx, "s", [1, 2, 99], [9500 + i for i in range(5)])

        for ct in (FULL, SWA):
            self.assertNotIn("s", cache.components[ct]._session_leaves)
        for node in self._nodes(cache):
            for ct in (FULL, SWA):
                self.assertEqual(cache.components[ct].session_ref(node), 0)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
