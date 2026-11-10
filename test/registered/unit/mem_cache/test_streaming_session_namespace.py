"""A streaming session's KV row stays in the namespace it was computed under.

Aborting a later turn whose ``extra_key`` or ``cache_salt`` differs must not
publish that row into the new namespace. The radix walk there shares no node
with the original one, so the insert would create a second leaf over the same
slots; evicting either leaf then frees slots the other still maps.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of

_POOL = 32


def _make_cache():
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=_POOL,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=_POOL,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=4,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="lru",
            tree_slots=tree_slots_of((ComponentType.FULL,)),
            tree_core_backend="python",
        )
    )
    return cache, allocator, req_pool


def _session():
    return Session(capacity_of_str_len=0, session_id="s", streaming=True)


def _turn(tokens, session, *, extra_key, cache_salt, rid):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=8),
        extra_key=extra_key,
        cache_salt=cache_salt,
        session=session,
    )
    req.refresh_fill_ids()
    return req


def _key(tokens, extra_key, cache_salt):
    return RadixKey(array("q", tokens), extra_key=extra_key, cache_salt=cache_salt)


def _fill_row(pool, allocator, req, start, count):
    indices = allocator.alloc(count)
    pool.write(
        (req.kv.req_pool_idx, slice(start, start + count)),
        indices.to(dtype=torch.int32),
    )
    return indices


def _admit_first_turn(cache, allocator, pool, tokens, session, extra_key, cache_salt):
    req = _turn(tokens, session, extra_key=extra_key, cache_salt=cache_salt, rid="t1")
    pool.alloc([req])
    _fill_row(pool, allocator, req, 0, len(tokens))
    req.kv.kv_committed_len = len(tokens)
    req.kv.kv_allocated_len = len(tokens)
    req.last_node = cache.root_node_handle()
    req.lock = cache.lock(req.last_node)
    cache.maybe_hand_to_session(req)
    cache.checkpoint(req, up_to=len(tokens))
    req.finished_reason = FINISH_LENGTH(length=0)
    req.finished_len = 0
    assert cache.session.try_cache_finished_req(req)
    return req


def _match_len(cache, tokens, extra_key, cache_salt):
    return cache.match_prefix(
        MatchPrefixParams(key=_key(tokens, extra_key, cache_salt))
    ).device_prefix_len


def _evict_all(cache):
    while cache.evictable_size() > 0:
        freed = cache.evict(
            EvictParams(num_tokens=cache.evictable_size())
        ).num_tokens_evicted
        if freed == 0:
            break


class TestStreamingSessionNamespace(CustomTestCase):
    def setUp(self):
        # release_kv_cache reads the serving config, and a rejected turn logs
        # on tp rank 0.
        self._server_args = get_context().override_server_args()
        self._server_args.install()
        self._parallel = get_parallel().override(tp_rank=0)
        self._parallel.__enter__()

    def tearDown(self):
        self._parallel.__exit__(None, None, None)
        self._server_args.restore()

    def _assert_one_owner(self, cache, allocator, initial_available):
        flat = cache.all_values_flatten()
        self.assertEqual(flat.numel(), torch.unique(flat).numel())
        free_ids = set(allocator.free_pages.tolist())
        self.assertTrue(set(flat.tolist()).isdisjoint(free_ids))
        if "s" in cache.session.slots:
            cache.session.release_session("s")
        _evict_all(cache)
        freed = allocator.free_pages
        self.assertEqual(freed.numel(), torch.unique(freed).numel())
        self.assertEqual(allocator.available_size(), initial_available)
        self.assertLessEqual(allocator.available_size(), allocator.size)
        cache.sanity_check()

    def test_abort_under_foreign_namespace_does_not_alias_slots(self):
        """A later streaming turn with a different prefix-cache namespace must
        not publish the session row there. Evicting the tree afterwards frees
        every slot once."""
        cases = (
            ("tenant-a", None, "tenant-b", None),
            (None, "salt-a", None, "salt-b"),
        )
        for extra_key, cache_salt, foreign_key, foreign_salt in cases:
            with self.subTest(extra_key=foreign_key, cache_salt=foreign_salt):
                cache, allocator, pool = _make_cache()
                initial = allocator.available_size()
                session = _session()
                tokens = list(range(1, 9))
                _admit_first_turn(
                    cache, allocator, pool, tokens, session, extra_key, cache_salt
                )
                self.assertEqual(
                    _match_len(cache, tokens, extra_key, cache_salt), len(tokens)
                )

                later = _turn(
                    tokens + [9, 10, 11, 12],
                    session,
                    extra_key=foreign_key,
                    cache_salt=foreign_salt,
                    rid="t2",
                )
                slot = cache.session.slots["s"]
                cache.match_prefix(
                    MatchPrefixParams(
                        key=_key(tokens + [9, 10, 11, 12], foreign_key, foreign_salt),
                        req=later,
                    )
                )
                # The session row must not be this turn's to checkpoint.
                self.assertIsNot(later.kv, slot.kv)
                if later.kv.holds_kv:
                    _fill_row(pool, allocator, later, len(tokens), 4)
                    later.kv.kv_committed_len = len(tokens) + 4
                    later.kv.kv_allocated_len = len(tokens) + 4
                    later.finished_reason = FINISH_ABORT()
                    release_kv_cache(later, cache, checkpoint=True)

                self.assertEqual(
                    _match_len(cache, tokens, foreign_key, foreign_salt), 0
                )
                self._assert_one_owner(cache, allocator, initial)

    def test_namespace_changed_after_borrow_is_not_republished(self):
        """The row was already lent. Changing the turn's namespace before an
        abort must still leave a single owner of those slots."""
        cache, allocator, pool = _make_cache()
        initial = allocator.available_size()
        session = _session()
        tokens = list(range(1, 9))
        _admit_first_turn(cache, allocator, pool, tokens, session, "tenant-a", None)
        slot = cache.session.slots["s"]

        later = _turn(
            tokens + [9, 10],
            session,
            extra_key="tenant-a",
            cache_salt=None,
            rid="t2",
        )
        cache.match_prefix(
            MatchPrefixParams(
                key=_key(tokens + [9, 10], "tenant-a", None),
                req=later,
            )
        )
        self.assertIs(later.kv, slot.kv)
        _fill_row(pool, allocator, later, len(tokens), 2)
        later.kv.kv_committed_len = len(tokens) + 2
        later.kv.kv_allocated_len = len(tokens) + 2
        later.cache_salt = "salt-b"
        later.finished_reason = FINISH_ABORT()
        release_kv_cache(later, cache, checkpoint=True)

        self.assertEqual(_match_len(cache, tokens + [9, 10], "tenant-a", "salt-b"), 0)
        self.assertEqual(_match_len(cache, tokens, "tenant-a", None), len(tokens))
        self._assert_one_owner(cache, allocator, initial)

    def test_same_namespace_abort_publishes_without_duplicating_slots(self):
        """An abort in the session's own namespace still caches the turn, and
        the overlapping prefix is not stored twice."""
        cache, allocator, pool = _make_cache()
        initial = allocator.available_size()
        session = _session()
        tokens = list(range(1, 9))
        _admit_first_turn(cache, allocator, pool, tokens, session, "tenant-a", "s")
        extended = tokens + [9, 10, 11, 12]
        later = _turn(extended, session, extra_key="tenant-a", cache_salt="s", rid="t2")
        cache.match_prefix(
            MatchPrefixParams(key=_key(extended, "tenant-a", "s"), req=later)
        )
        _fill_row(pool, allocator, later, len(tokens), 4)
        later.kv.kv_committed_len = len(extended)
        later.kv.kv_allocated_len = len(extended)
        later.refresh_fill_ids()
        later.finished_reason = FINISH_ABORT()
        release_kv_cache(later, cache, checkpoint=True)

        self.assertEqual(_match_len(cache, extended, "tenant-a", "s"), len(extended))
        self._assert_one_owner(cache, allocator, initial)

    def test_media_adding_turn_is_lent_and_abort_does_not_alias_slots(self):
        """Adding media rewrites cache_salt through bind_multimodal_cache_namespace
        without the caller changing extra_key or the user salt. That turn must
        still borrow the session row, and aborting it must not insert the row
        under the new salt."""
        cases = (
            # Text first turn, then a later turn attaches media.
            (None, "sglang-mm-v1:media-turn-2"),
            # First turn already had media; a later turn adds another item.
            ("sglang-mm-v1:media-turn-1", "sglang-mm-v1:media-turn-2"),
        )
        for first_salt, later_salt in cases:
            with self.subTest(first_salt=first_salt, later_salt=later_salt):
                cache, allocator, pool = _make_cache()
                initial = allocator.available_size()
                session = _session()
                tokens = list(range(1, 9))
                user_salt = "user-salt"
                first = _turn(
                    tokens,
                    session,
                    extra_key="tenant-a",
                    cache_salt=user_salt,
                    rid="t1",
                )
                # bind_multimodal_cache_namespace rewrites cache_salt only.
                if first_salt is not None:
                    first.cache_salt = first_salt
                pool.alloc([first])
                _fill_row(pool, allocator, first, 0, len(tokens))
                first.kv.kv_committed_len = len(tokens)
                first.kv.kv_allocated_len = len(tokens)
                first.last_node = cache.root_node_handle()
                first.lock = cache.lock(first.last_node)
                cache.maybe_hand_to_session(first)
                cache.checkpoint(first, up_to=len(tokens))
                first.finished_reason = FINISH_LENGTH(length=0)
                first.finished_len = 0
                self.assertTrue(cache.session.try_cache_finished_req(first))
                slot = cache.session.slots["s"]
                self.assertEqual(
                    _match_len(cache, tokens, "tenant-a", first.cache_salt),
                    len(tokens),
                )

                later = _turn(
                    tokens + [9, 10],
                    session,
                    extra_key="tenant-a",
                    cache_salt=user_salt,
                    rid="t2",
                )
                later.cache_salt = later_salt
                cache.match_prefix(
                    MatchPrefixParams(
                        key=_key(tokens + [9, 10], "tenant-a", later_salt),
                        req=later,
                    )
                )
                self.assertIs(later.kv, slot.kv)
                self.assertFalse(isinstance(later.finished_reason, FINISH_ABORT))
                _fill_row(pool, allocator, later, len(tokens), 2)
                later.kv.kv_committed_len = len(tokens) + 2
                later.kv.kv_allocated_len = len(tokens) + 2
                later.finished_reason = FINISH_ABORT()
                release_kv_cache(later, cache, checkpoint=True)

                self.assertEqual(
                    _match_len(cache, tokens + [9, 10], "tenant-a", later_salt), 0
                )
                self.assertEqual(
                    _match_len(cache, tokens, "tenant-a", first.cache_salt),
                    len(tokens),
                )
                self._assert_one_owner(cache, allocator, initial)


if __name__ == "__main__":
    unittest.main()
