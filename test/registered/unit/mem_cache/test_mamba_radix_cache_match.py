"""CPU-only match semantics of MambaRadixCache around split tombstones.

A mamba state is a point value at a node's end depth, so a match may only
return KV up to the deepest checkpointed node on the matched path. When a
lookup ends mid-edge, `_split_node` leaves a checkpoint-less internal node and
the match falls back to that safe boundary — possibly 0 tokens
(sgl-project/sglang#22935). These tests pin the contract: returned KV never
exceeds the deepest checkpoint depth, tombstones never block matches that
continue past them, and the branching signal points at the chunk-aligned
re-track position.

A second class covers the no_buffer interior checkpoint (`_insert_interior_checkpoint`,
the fix for #22935): the tracked grid-boundary state must land on the tree as its
own checkpointed node, must revive the tombstone a fresh n-1 lookup created, and
must hand the main insert a prev_prefix_len that keeps the interior node's KV.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 64


class _CountingMambaAllocator:
    """Slot-id allocator standing in for the real mamba allocator: alloc hands
    out increasing ids, free counts what was returned."""

    def __init__(self):
        self.next_id = 0
        self.freed = 0

    def alloc(self, n):
        self.next_id += n
        return torch.arange(self.next_id - n, self.next_id, dtype=torch.int64)

    def free(self, slots):
        self.freed += int(slots.numel())


def _make_cache() -> MambaRadixCache:
    server_args = ServerArgs(model_path="dummy", page_size=1)
    # The property would otherwise load the HF config for the dummy model.
    server_args._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(server_args)
    allocator = PagedTokenToKVPoolAllocator(
        size=4096,
        page_size=1,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    return MambaRadixCache(
        CacheInitParams(
            disable=False,
            # match/insert without cow_mamba only touch the mamba allocator
            # (freeing duplicate/dropped checkpoints)
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=_CountingMambaAllocator(),
                mamba_ckpt_pool=None,
            ),
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )


def _insert(cache: MambaRadixCache, ids: list, prev_prefix_len: int = 0) -> None:
    """Insert one finished request: the full path with a checkpoint at its end."""
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", ids)),
            value=torch.arange(len(ids), dtype=torch.int64),
            mamba_value=torch.zeros(1, 8),
            prev_prefix_len=prev_prefix_len,
        )
    )


def _match(cache: MambaRadixCache, ids: list):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids))))


def _fake_req(
    cache_protected_len: int = 0,
    interior_len=None,
    interior_idx=None,
) -> SimpleNamespace:
    """The req surface `_insert_interior_checkpoint` reads."""
    return SimpleNamespace(
        kv=SimpleNamespace(
            cache_protected_len=cache_protected_len,
            mamba_interior_ckpt_idx=interior_idx,
            mamba_interior_ckpt_seqlen=interior_len,
        ),
        extra_key=None,
        cache_salt=None,
    )


class TestMambaRadixCacheMatch(unittest.TestCase):
    def test_n_minus_1_lookup_after_identical_prompt_lands_on_the_safe_boundary(self):
        # The fresh prefill lookup is capped at input_len - 1, splits the only
        # checkpointed node mid-edge, and no earlier node on the path has a
        # checkpoint: the safe boundary is 0 tokens.
        # https://github.com/sgl-project/sglang/issues/22935
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        result = _match(cache, [1, 2])
        self.assertEqual(result.device_indices.numel(), 0)
        self.assertIsNone(result.mamba_branching_seqlen)

    def test_tombstone_does_not_block_matches_that_continue_past_it(self):
        # A checkpoint-less split node only caps matches that END at it; a
        # lookup covering the full edge still resumes at the checkpointed child.
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        self.assertEqual(_match(cache, [1, 2]).device_indices.numel(), 0)
        self.assertEqual(_match(cache, [1, 2, 3]).device_indices.numel(), 3)

    def test_match_never_returns_past_the_deepest_checkpoint(self):
        # Two checkpointed nodes, as per-chunk inserts leave them: the n-1
        # lookup keeps the first chunk and drops the split partial edge.
        cache = _make_cache()
        _insert(cache, list(range(CHUNK)))
        _insert(cache, list(range(CHUNK + 30)))
        result = _match(cache, list(range(CHUNK + 29)))
        self.assertEqual(result.device_indices.numel(), CHUNK)

    def test_branching_signal_points_at_the_chunk_aligned_boundary(self):
        # The matched path runs past the last checkpoint, so the scheduler is
        # told to re-track a state at the chunk-aligned boundary.
        cache = _make_cache()
        _insert(cache, list(range(CHUNK)))
        _insert(cache, list(range(CHUNK + 30)))
        result = _match(cache, list(range(CHUNK + 29)))
        self.assertEqual(result.mamba_branching_seqlen, CHUNK)

    def test_shared_prefix_divergence_stops_at_the_tombstone(self):
        # A divergent request shares the [1, 2] edge, but that edge carries no
        # checkpoint, so the shared part cannot be reused either.
        # https://github.com/sgl-project/sglang/issues/22935
        cache = _make_cache()
        _insert(cache, [1, 2, 3])
        _insert(cache, [1, 2, 4, 5])
        result = _match(cache, [1, 2, 4])
        self.assertEqual(result.device_indices.numel(), 0)

    def test_full_match_returns_kv_through_the_last_checkpoint(self):
        cache = _make_cache()
        _insert(cache, list(range(CHUNK)))
        _insert(cache, list(range(CHUNK + 30)))
        result = _match(cache, list(range(CHUNK + 30)))
        self.assertEqual(result.device_indices.numel(), CHUNK + 30)


class TestMambaRadixCacheInteriorCheckpoint(unittest.TestCase):
    def test_interior_checkpoint_lets_the_n_minus_1_lookup_reuse(self):
        # Bug regression for sgl-project/sglang#22935: with an interior
        # checkpoint on the path, the fresh n-1 lookup reuses up to it instead
        # of falling to 0 tokens.
        cache = _make_cache()
        ids = list(range(2 * CHUNK))
        req = _fake_req(interior_len=CHUNK, interior_idx=torch.tensor(11))
        prev = cache._insert_interior_checkpoint(
            req, array("q", ids), torch.arange(len(ids))
        )
        self.assertEqual(prev, CHUNK)
        # The helper consumes the stash exactly once, so a later
        # free_mamba_cache cannot free the same slot again.
        self.assertIsNone(req.kv.mamba_interior_ckpt_idx)
        _insert(cache, ids, prev_prefix_len=prev)
        self.assertEqual(_match(cache, ids[:-1]).device_indices.numel(), CHUNK)

    def test_interior_checkpoint_revives_the_split_tombstone(self):
        # The identical-prompt sequence: the first request leaves only the end
        # checkpoint, the second request's n-1 lookup splits it into a
        # tombstone, and the second request's tracked interior checkpoint must
        # land ON that tombstone.
        cache = _make_cache()
        ids = list(range(2 * CHUNK))
        _insert(cache, ids)
        self.assertEqual(_match(cache, ids[:-1]).device_indices.numel(), 0)
        req = _fake_req(interior_len=CHUNK, interior_idx=torch.tensor(11))
        prev = cache._insert_interior_checkpoint(
            req, array("q", ids), torch.arange(len(ids))
        )
        self.assertEqual(prev, CHUNK)
        self.assertEqual(_match(cache, ids[:-1]).device_indices.numel(), CHUNK)

    def test_main_insert_prev_prefix_len_keeps_the_interior_node_kv(self):
        # The interior node owns the KV segment below the checkpoint; a main
        # insert that still claimed that segment as duplicate would free it
        # (double free). The correct prev leaves the token pool untouched.
        cache = _make_cache()
        ids = list(range(2 * CHUNK))
        req = _fake_req(interior_len=CHUNK, interior_idx=torch.tensor(11))
        prev = cache._insert_interior_checkpoint(
            req, array("q", ids), torch.arange(len(ids))
        )
        _insert(cache, ids, prev_prefix_len=prev)
        self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), 4096)

    def test_duplicate_interior_checkpoint_is_freed(self):
        # A checkpointed node already covers the interior depth: the tracked
        # slot is a duplicate, must be freed, and the tree must stay unchanged.
        cache = _make_cache()
        ids = list(range(2 * CHUNK))
        req = _fake_req(interior_len=CHUNK, interior_idx=torch.tensor(11))
        cache._insert_interior_checkpoint(req, array("q", ids), torch.arange(len(ids)))
        _insert(cache, ids, prev_prefix_len=CHUNK)
        duplicate = _fake_req(interior_len=CHUNK, interior_idx=torch.tensor(12))
        prev = cache._insert_interior_checkpoint(
            duplicate, array("q", ids), torch.arange(len(ids))
        )
        self.assertEqual(prev, CHUNK)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, 1)
        self.assertEqual(_match(cache, ids[:-1]).device_indices.numel(), CHUNK)

    def test_stale_interior_checkpoint_is_dropped(self):
        # A checkpoint deeper than what the request still holds cannot be
        # inserted; the slot is freed and the main insert keeps its original
        # prev_prefix_len.
        cache = _make_cache()
        ids = list(range(CHUNK + 30))
        req = _fake_req(
            cache_protected_len=7,
            interior_len=len(ids) + CHUNK,
            interior_idx=torch.tensor(11),
        )
        prev = cache._insert_interior_checkpoint(
            req, array("q", ids), torch.arange(len(ids))
        )
        self.assertEqual(prev, 7)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, 1)


if __name__ == "__main__":
    unittest.main()
