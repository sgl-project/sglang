from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocation import (
    alloc_paged_token_slots_decode,
    alloc_paged_token_slots_extend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tree_cache(page_size=8):
    allocator = MagicMock(page_size=page_size)
    allocator.alloc_extend.return_value = torch.tensor([1])
    allocator.alloc_decode.return_value = torch.tensor([1])
    return MagicMock(token_to_kv_pool_allocator=allocator)


def test_extend_evicts_only_new_pages():
    tree_cache = _tree_cache()
    prefix = torch.tensor([5, 7])
    seq = torch.tensor([6, 9])

    with patch("sglang.srt.mem_cache.allocation.evict_from_tree_cache") as evict:
        alloc_paged_token_slots_extend(
            tree_cache,
            prefix,
            prefix,
            seq,
            seq,
            torch.tensor([0, 0]),
            extend_num_tokens=3,
        )

    evict.assert_called_once_with(tree_cache, 8)


def test_single_token_decode_evicts_only_page_crossings():
    tree_cache = _tree_cache()
    seq = torch.tensor([9, 10, 17])

    with patch("sglang.srt.mem_cache.allocation.evict_from_tree_cache") as evict:
        alloc_paged_token_slots_decode(tree_cache, seq, seq, torch.tensor([0, 0, 0]))

    evict.assert_called_once_with(tree_cache, 16)


def test_multi_token_decode_can_stay_within_existing_pages():
    tree_cache = _tree_cache()
    seq = torch.tensor([7, 15])

    with patch("sglang.srt.mem_cache.allocation.evict_from_tree_cache") as evict:
        alloc_paged_token_slots_decode(
            tree_cache,
            seq,
            seq,
            torch.tensor([0, 0]),
            token_per_req=2,
        )

    evict.assert_called_once_with(tree_cache, 0)
