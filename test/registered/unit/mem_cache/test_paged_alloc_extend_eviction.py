"""Regression tests for paged prefill eviction accounting."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend
from sglang.srt.mem_cache.base_prefix_cache import EvictParams

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _call_alloc_extend(
    *,
    page_size: int,
    available_size: int,
    prefix_lens: list[int],
    seq_lens: list[int],
):
    prefix_lens_cpu = torch.tensor(prefix_lens, dtype=torch.int64)
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    extend_num_tokens = sum(seq - prefix for prefix, seq in zip(prefix_lens, seq_lens))
    out = torch.arange(extend_num_tokens, dtype=torch.int64)

    allocator = SimpleNamespace(
        page_size=page_size,
        available_size=MagicMock(return_value=available_size),
        alloc_extend=MagicMock(return_value=out),
    )
    tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        is_chunk_cache=MagicMock(return_value=False),
        evict=MagicMock(),
    )

    result = alloc_paged_token_slots_extend(
        tree_cache=tree_cache,
        prefix_lens=prefix_lens_cpu,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        last_loc=torch.tensor(
            [prefix_len - 1 for prefix_len in prefix_lens], dtype=torch.int64
        ),
        extend_num_tokens=extend_num_tokens,
    )
    return result, out, tree_cache, allocator


class TestPagedAllocExtendEviction(CustomTestCase):
    def test_extend_within_partial_page_does_not_evict(self):
        # Previous target: 1 + 1 * 64 = 65 tokens. The extend reuses the
        # existing tail page, so the exact new-page capacity is zero.
        result, expected, tree_cache, allocator = _call_alloc_extend(
            page_size=64,
            available_size=0,
            prefix_lens=[63],
            seq_lens=[64],
        )

        torch.testing.assert_close(result, expected)
        tree_cache.evict.assert_not_called()
        allocator.alloc_extend.assert_called_once()
        self.assertEqual(allocator.alloc_extend.call_args.kwargs, {})

    def test_mixed_batch_evicts_only_exact_page_shortfall(self):
        # Previous target: 129 + 4 * 64 = 385 tokens, or a 353-token
        # shortfall. The batch consumes one new page, leaving only 32 tokens
        # to evict after accounting for the 32 already available.
        result, expected, tree_cache, _ = _call_alloc_extend(
            page_size=64,
            available_size=32,
            prefix_lens=[1, 63, 64, 65],
            seq_lens=[2, 64, 128, 128],
        )

        torch.testing.assert_close(result, expected)
        tree_cache.evict.assert_called_once_with(EvictParams(num_tokens=32))


if __name__ == "__main__":
    unittest.main()
