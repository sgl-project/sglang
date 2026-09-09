"""CPU-only match semantics of MambaRadixCache around split tombstones.

A mamba state is a point value at a node's end depth, so a match may only
return KV up to the deepest checkpointed node on the matched path. When a
lookup ends mid-edge, `_split_node` leaves a checkpoint-less internal node and
the match falls back to that safe boundary — possibly 0 tokens
(sgl-project/sglang#22935). These tests pin the contract: returned KV never
exceeds the deepest checkpoint depth, tombstones never block matches that
continue past them, and the branching signal points at the chunk-aligned
re-track position.
"""

import unittest
from array import array

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
            req_to_token_pool=None,  # match/insert without cow_mamba never touch it
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )


def _insert(cache: MambaRadixCache, ids: list) -> None:
    """Insert one finished request: the full path with a checkpoint at its end."""
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", ids)),
            value=torch.arange(len(ids), dtype=torch.int64),
            mamba_value=torch.zeros(1, 8),
        )
    )


def _match(cache: MambaRadixCache, ids: list):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids))))


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


if __name__ == "__main__":
    unittest.main()
