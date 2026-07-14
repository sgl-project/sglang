# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU unit test for DeepseekSparseAttnBackend._dcp_shard_page_table_decode,
the per-rank top-k page-table partition added for TRT-LLM sparse MLA decode
under decode context parallel (DCP).

Pins two derived properties to a brute-force reference:
  1. Per-rank membership: an entry is owned by rank `value % dcp_size ==
     dcp_rank`, and the rank-local physical address is `value // dcp_size`
     (same convention as layers/dcp/layout.py's
     filter_dcp_local_kv_indices) -- confirmed to match the paged
     allocator's actual addressing under DCP (mem_cache/allocator/paged.py
     `PagedTokenToKVPoolAllocator.alloc`: with `page_size` and `size` both
     scaled by dcp_size, `num_pages` is dcp-size-independent, so
     `out_cache_loc` ranges up to `dcp_size` times the physical KV tensor's
     row count -- the cache is address-sharded, not replicated), and to the
     real TRT-LLM kernel via a faithfully-simulated per-rank physical buffer
     (see `_dcp_shard_page_table_decode`'s docstring in dsa_backend.py and
     test/registered/attention/unittests/dsa/test_dsa_dcp_trtllm.py). Owned
     entries are compacted to the front of each row (padded with -1 after).
  2. Partition coverage: across all dcp_size ranks, every valid (non -1) entry
     in the original row is owned by exactly one rank -- i.e. the per-rank
     shards reconstruct the original valid set exactly once, which is the
     precondition the online-softmax LSE merge in forward_mla.py relies on
     to reproduce the unsharded attention output.

Usage:
    python -m pytest test_dcp_dsa_trtllm_shard_unit.py -v
    python test_dcp_dsa_trtllm_shard_unit.py
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeParallel:
    def __init__(self, dcp_size: int, dcp_rank: int):
        self.attn_dcp_size = dcp_size
        self.attn_dcp_rank = dcp_rank


def _shard(page_table_1: torch.Tensor, dcp_size: int, dcp_rank: int):
    with patch(
        "sglang.srt.layers.attention.dsa_backend.get_parallel",
        return_value=_FakeParallel(dcp_size, dcp_rank),
    ):
        return DeepseekSparseAttnBackend._dcp_shard_page_table_decode(
            None, page_table_1
        )


def _brute_force_owned(page_table_1: torch.Tensor, dcp_size: int, dcp_rank: int):
    bs, topk = page_table_1.shape
    owned_per_row = []
    for b in range(bs):
        owned_per_row.append(
            [
                v // dcp_size
                for v in page_table_1[b].tolist()
                if v >= 0 and v % dcp_size == dcp_rank
            ]
        )
    return owned_per_row


class TestDcpDsaTrtllmShard(unittest.TestCase):
    def _random_page_table(self, bs, topk, seed):
        g = torch.Generator().manual_seed(seed)
        vals = torch.randint(0, 5000, (bs, topk), generator=g, dtype=torch.int32)
        valid_counts = torch.randint(0, topk + 1, (bs,), generator=g)
        for b in range(bs):
            vals[b, valid_counts[b] :] = -1
        return vals

    def test_matches_brute_force_membership_and_order(self):
        page_table_1 = self._random_page_table(bs=6, topk=37, seed=0)
        for dcp_size in (1, 2, 4, 8):
            for dcp_rank in range(dcp_size):
                local_table, local_seq_lens = _shard(page_table_1, dcp_size, dcp_rank)
                expected = _brute_force_owned(page_table_1, dcp_size, dcp_rank)
                for b, exp in enumerate(expected):
                    n = local_seq_lens[b].item()
                    self.assertEqual(
                        n,
                        len(exp),
                        msg=f"dcp_size={dcp_size} rank={dcp_rank} row={b}",
                    )
                    self.assertEqual(
                        local_table[b, :n].tolist(),
                        exp,
                        msg=f"dcp_size={dcp_size} rank={dcp_rank} row={b}",
                    )
                    self.assertTrue(
                        (local_table[b, n:] == -1).all(),
                        msg=f"dcp_size={dcp_size} rank={dcp_rank} row={b}: "
                        "padding beyond local_seq_lens must be -1",
                    )

    def test_partition_covers_original_valid_set_exactly_once(self):
        page_table_1 = self._random_page_table(bs=4, topk=53, seed=1)
        valid_count_per_row = (page_table_1 >= 0).sum(dim=-1)
        for dcp_size in (2, 4, 8):
            total_owned_per_row = torch.zeros(page_table_1.shape[0], dtype=torch.int64)
            for dcp_rank in range(dcp_size):
                _, local_seq_lens = _shard(page_table_1, dcp_size, dcp_rank)
                total_owned_per_row += local_seq_lens.to(torch.int64)
            self.assertTrue(
                torch.equal(total_owned_per_row, valid_count_per_row),
                msg=f"dcp_size={dcp_size}: shards don't partition the valid "
                f"set exactly once ({total_owned_per_row.tolist()} != "
                f"{valid_count_per_row.tolist()})",
            )

    def test_dcp_size_one_keeps_all_entries_on_the_single_rank(self):
        page_table_1 = self._random_page_table(bs=3, topk=16, seed=2)
        local_table, local_seq_lens = _shard(page_table_1, dcp_size=1, dcp_rank=0)
        self.assertTrue(
            torch.equal(local_seq_lens, (page_table_1 >= 0).sum(dim=-1).to(torch.int32))
        )


if __name__ == "__main__":
    unittest.main()
