"""Unit tests for the batched retained-block gather used by dLLM FDFO KV reuse.

``_alloc_extend_loc_with_kv_reuse`` re-reads each reuse row's retained KV block
out of ``req_to_token``. It used to do that one row at a time (a slice + cast
kernel per row); the rows share a block length, so they are now fetched with a
single flat index op. The index arithmetic (``row * width + start``, flattened
and reshaped back) is what the first class pins down -- it is the part a
"looks equivalent" rewrite silently gets wrong.

The second class covers the caller instead of the helper. Batching splits the
row loop into two independent cursors (``reuse_ptr`` over gathered rows,
``fresh_ptr`` over freshly allocated tokens); if either advances on the wrong
row the batch is silently mispaired, which no helper-level test can see. Every
case therefore runs the same batch with batching forced off and forced on and
requires the two ``out_cache_loc`` tensors to be identical -- the off path is
the pre-existing behaviour, so agreement is the regression signal.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocation import (
    _alloc_extend_loc_with_kv_reuse,
    _gather_reused_block_locs,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS, WIDTH = 8, 64


def _fingerprint_pool() -> torch.Tensor:
    # Slot at [row, col] is row*1000 + col, so a wrong row or a wrong start
    # offset is visible in the value itself rather than only in the shape.
    rows = torch.arange(ROWS, dtype=torch.int32).unsqueeze(1) * 1000
    return rows + torch.arange(WIDTH, dtype=torch.int32)


def _per_row_reference(req_to_token, req_pool_indices, prefix_lens, block_len, dtype):
    return torch.stack(
        [
            req_to_token[idx, start : start + block_len].to(dtype)
            for idx, start in zip(req_pool_indices, prefix_lens)
        ]
    )


class TestGatherReusedBlockLocs(CustomTestCase):
    def _gather(self, req_pool_indices, prefix_lens, block_len, dtype=torch.int64):
        pool = _fingerprint_pool()
        got = _gather_reused_block_locs(
            req_to_token=pool,
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            block_len=block_len,
            device=torch.device("cpu"),
            out_dtype=dtype,
        )
        want = _per_row_reference(pool, req_pool_indices, prefix_lens, block_len, dtype)
        return got, want

    def test_matches_per_row_slicing_at_distinct_offsets(self):
        # Rows are non-contiguous and each starts at a different offset: the
        # flat index must combine the right row with the right start, not
        # broadcast one row's offset across all of them.
        got, want = self._gather([1, 3, 5], [4, 8, 0], 4)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(
            got.tolist(),
            [
                [1004, 1005, 1006, 1007],
                [3008, 3009, 3010, 3011],
                [5000, 5001, 5002, 5003],
            ],
        )

    def test_preserves_row_order_when_pool_indices_are_unsorted(self):
        # The gather feeds out_cache_loc positionally, so it must follow the
        # caller's row order rather than the pool index order.
        got, want = self._gather([5, 0, 3], [0, 12, 4], 4)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual([row[0] for row in got.tolist()], [5000, 12, 3004])

    def test_block_touching_the_row_end_does_not_bleed_into_the_next_row(self):
        # A block ending exactly at WIDTH is the case a flat index gets wrong
        # by reading past the row boundary.
        got, want = self._gather([2], [WIDTH - 4], 4)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(got.tolist(), [[2060, 2061, 2062, 2063]])


class _FakeAllocator:
    """Hands out consecutive slots so the same batch allocates identically with
    batching on and off, making the two out_cache_loc tensors comparable."""

    def __init__(self, base: int = 10_000):
        self.base = base
        self.cursor = 0

    def alloc(self, num_tokens: int) -> torch.Tensor:
        start = self.base + self.cursor
        self.cursor += num_tokens
        return torch.arange(start, start + num_tokens, dtype=torch.int64)


class _FakeTreeCache:
    # is_chunk_cache() short-circuits evict_from_tree_cache, so the allocator
    # above is the only collaborator alloc_token_slots actually needs.
    def __init__(self):
        self.token_to_kv_pool_allocator = _FakeAllocator()

    def is_chunk_cache(self) -> bool:
        return True


def _make_batch(*, reuse_kv, prefix_lens, extend_lens):
    reqs = [
        SimpleNamespace(
            dllm_incomplete_ids=[0] * extend_lens[i] if reuse_kv[i] else [],
            kv=SimpleNamespace(kv_allocated_len=prefix_lens[i] + extend_lens[i]),
        )
        for i in range(len(reuse_kv))
    ]
    return SimpleNamespace(
        device=torch.device("cpu"),
        req_to_token_pool=SimpleNamespace(req_to_token=_fingerprint_pool()),
        reqs=reqs,
        tree_cache=_FakeTreeCache(),
    )


class TestAllocExtendLocWithKvReuseBatching(CustomTestCase):
    """The caller's reuse/fresh reassembly, with batching off vs on."""

    def _run(self, *, reuse_kv, prefix_lens, extend_lens, min_rows):
        # The env var is what we vary: _auto_min_rows_for_batched_gather is
        # lru_cached and process-sticky, so the auto default must not be the
        # knob under test. -1 disables outright; 0 batches every reuse row.
        with envs.SGLANG_DLLM_BATCHED_GATHER_MIN_ROWS.override(min_rows):
            return _alloc_extend_loc_with_kv_reuse(
                _make_batch(
                    reuse_kv=reuse_kv,
                    prefix_lens=prefix_lens,
                    extend_lens=extend_lens,
                ),
                reuse_kv,
                torch.tensor(list(range(len(reuse_kv))), dtype=torch.int64),
                torch.tensor(prefix_lens, dtype=torch.int64),
                torch.tensor(extend_lens, dtype=torch.int64),
                torch.tensor(list(range(len(reuse_kv))), dtype=torch.int64),
                1,
            )

    def _assert_batching_is_transparent(self, **kwargs):
        off = self._run(min_rows=-1, **kwargs)
        on = self._run(min_rows=0, **kwargs)
        self.assertTrue(
            torch.equal(off, on),
            f"batching changed out_cache_loc:\n off={off.tolist()}\n on ={on.tolist()}",
        )
        return on

    def test_interleaved_reuse_and_fresh_rows_keep_their_positions(self):
        # The failure this guards: a cursor that advances on every row instead
        # of only on its own kind silently shifts rows against each other.
        # Fresh slots come from _FakeAllocator (10000+), reuse rows carry the
        # pool fingerprint (row*1000 + col), so a swap is visible in the values.
        out = self._assert_batching_is_transparent(
            reuse_kv=[False, True, False, True],
            prefix_lens=[0, 8, 0, 12],
            extend_lens=[3, 4, 2, 4],
        )
        self.assertEqual(
            out.tolist(),
            # fmt: off
            [
                10000, 10001, 10002,          # row 0, fresh
                1008, 1009, 1010, 1011,       # row 1, reuse @ pool row 1 off 8
                10003, 10004,                 # row 2, fresh
                3012, 3013, 3014, 3015,       # row 3, reuse @ pool row 3 off 12
            ],
            # fmt: on
        )

    def test_all_reuse_allocates_nothing_and_still_reassembles_in_order(self):
        # No fresh slots at all: fresh_slots stays None and reuse_dtype falls
        # back to int64. The gathered rows must still land in caller order.
        out = self._assert_batching_is_transparent(
            reuse_kv=[True, True, True],
            prefix_lens=[4, 0, 16],
            extend_lens=[4, 4, 4],
        )
        self.assertEqual(
            out.tolist(),
            [4, 5, 6, 7, 1000, 1001, 1002, 1003, 2016, 2017, 2018, 2019],
        )

    def test_reuse_rows_of_unequal_length_fall_back_per_row(self):
        # Rows of different block_len cannot share one flat index, so the
        # caller must decline to batch them. If that guard is dropped the
        # gather reshapes against the wrong width and this diverges.
        self._assert_batching_is_transparent(
            reuse_kv=[True, False, True],
            prefix_lens=[0, 0, 8],
            extend_lens=[4, 3, 8],
        )

    def test_single_reuse_row_below_the_default_threshold(self):
        # min_rows=0 forces batching on for a batch the Ascend default (8)
        # would have left on the per-row path -- the gate must not be the only
        # thing keeping the batched path correct on narrow batches.
        self._assert_batching_is_transparent(
            reuse_kv=[True, False],
            prefix_lens=[12, 0],
            extend_lens=[4, 5],
        )


if __name__ == "__main__":
    unittest.main()
