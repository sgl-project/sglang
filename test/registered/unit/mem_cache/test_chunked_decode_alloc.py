"""Unit tests for the chunked speculative decode KV allocation."""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative import eagle_utils  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(committed, allocated):
    return types.SimpleNamespace(
        decode_batch_idx=0,
        kv=types.SimpleNamespace(
            kv_committed_len=committed, kv_allocated_len=allocated
        ),
    )


def _assign_req_to_token_pool(pool_indices, req_to_token, start, end, locs, bs):
    """Torch reference of the req_to_token write kernel."""
    offset = 0
    for i in range(bs):
        n = int(end[i] - start[i])
        req_to_token[pool_indices[i], start[i] : end[i]] = locs[offset : offset + n]
        offset += n


class TestChunkedDecodeAllocLens(CustomTestCase):
    def test_short_request_is_topped_up_to_twice_the_reserve(self):
        reqs = [_req(10, 12), _req(20, 40)]  # the second still holds its reserve
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            reqs, reserve=8, page_size=1, free_tokens=1000, row_len=1000
        )
        self.assertEqual((cur, nxt, needed), ([12, 40], [26, 40], 14))

    def test_without_pool_info_allocates_one_reserve(self):
        cur, nxt, needed = page_aligned_decode_alloc_lens(
            [_req(10, 12)], reserve=8, page_size=1
        )
        self.assertEqual((cur, nxt, needed), ([12], [18], 6))

    def test_tight_pool_keeps_one_reserve_per_request_free(self):
        reqs = [_req(10, 12), _req(20, 40)]
        for free, nxt_expected, needed_expected in (
            (29, [18, 40], 6),  # topping up would leave 15 < 2 x 8 free
            (30, [26, 40], 14),
        ):
            _, nxt, needed = page_aligned_decode_alloc_lens(
                reqs, reserve=8, page_size=1, free_tokens=free, row_len=1000
            )
            self.assertEqual((nxt, needed), (nxt_expected, needed_expected))

    def test_top_up_stays_page_aligned_and_inside_the_row(self):
        for row_len, nxt_expected in ((1000, [28]), (23, [20])):
            _, nxt, _ = page_aligned_decode_alloc_lens(
                [_req(10, 16)],
                reserve=8,
                page_size=4,
                free_tokens=1000,
                row_len=row_len,
            )
            self.assertEqual(nxt, nxt_expected)

    def test_nothing_needed_skips_the_top_up(self):
        _, nxt, needed = page_aligned_decode_alloc_lens(
            [_req(10, 18)], reserve=8, page_size=1, free_tokens=1000, row_len=1000
        )
        self.assertEqual((nxt, needed), ([18], 0))


class TestEaglePrepareForDecodeAllocation(CustomTestCase):
    """eagle_prepare_for_decode with a fake pool: slots come from a counter."""

    def _prepare(self, reqs):
        allocator = types.SimpleNamespace(page_size=1, available_size=lambda: 1000)
        req_to_token = torch.zeros((len(reqs), 64), dtype=torch.int32)
        batch = types.SimpleNamespace(
            reqs=reqs,
            device="cpu",
            maybe_evict_swa=lambda: None,
            sampling_info=types.SimpleNamespace(
                penalizer_orchestrator=types.SimpleNamespace(is_required=False)
            ),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
            tree_cache=types.SimpleNamespace(token_to_kv_pool_allocator=allocator),
            req_pool_indices=torch.arange(len(reqs)),
            hisparse_coordinator=None,
        )
        allocated = []

        def alloc_token_slots(tree_cache, num_tokens):
            allocated.append(num_tokens)
            return torch.arange(100, 100 + num_tokens, dtype=torch.int64)

        with (
            patch.object(eagle_utils, "get_alloc_reserve_per_decode", return_value=8),
            patch(
                "sglang.srt.mem_cache.allocation.alloc_token_slots", alloc_token_slots
            ),
            patch(
                "sglang.srt.mem_cache.allocation.assign_req_to_token_pool_func",
                _assign_req_to_token_pool,
            ),
            patch.object(
                eagle_utils,
                "alloc_for_spec_decode",
                wraps=eagle_utils.alloc_for_spec_decode,
            ) as alloc_for_spec_decode,
        ):
            eagle_utils.eagle_prepare_for_decode(batch)
        return req_to_token, allocated, alloc_for_spec_decode.call_count

    def test_short_request_gets_a_chunk(self):
        reqs = [_req(10, 12)]
        req_to_token, allocated, _ = self._prepare(reqs)
        self.assertEqual(allocated, [14])
        self.assertEqual(reqs[0].kv.kv_allocated_len, 26)
        self.assertEqual(req_to_token[0, 12:26].tolist(), list(range(100, 114)))
        self.assertEqual(reqs[0].decode_batch_idx, 1)

    def test_held_reserve_allocates_nothing(self):
        reqs = [_req(10, 26)]
        req_to_token, allocated, num_alloc_calls = self._prepare(reqs)
        self.assertEqual(allocated, [])
        # No host tensors, copies or slot launches for a step that allocates nothing.
        self.assertEqual(num_alloc_calls, 0)
        self.assertEqual(reqs[0].kv.kv_allocated_len, 26)
        self.assertEqual(int(req_to_token.count_nonzero()), 0)
        self.assertEqual(reqs[0].decode_batch_idx, 1)


if __name__ == "__main__":
    unittest.main()
