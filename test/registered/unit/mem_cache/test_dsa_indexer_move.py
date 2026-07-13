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
"""Regression tests for DSA-indexer token moves in a page-indexed buffer."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 64
INDEX_HEAD_DIM = 128
QUANT_BLOCK_SIZE = 128
S_BYTES = INDEX_HEAD_DIM // QUANT_BLOCK_SIZE * 4
PER_PAGE = PAGE_SIZE * INDEX_HEAD_DIM + PAGE_SIZE * S_BYTES
S_OFFSET_IN_PAGE = PAGE_SIZE * INDEX_HEAD_DIM


def _pool_stub(num_pages: int) -> DSATokenToKVPool:
    pool = object.__new__(DSATokenToKVPool)
    pool.page_size = PAGE_SIZE
    pool.index_head_dim = INDEX_HEAD_DIM
    pool.quant_block_size = QUANT_BLOCK_SIZE

    buf = torch.zeros((num_pages, PER_PAGE), dtype=torch.uint8)
    for row in range(num_pages):
        for offset in range(PAGE_SIZE):
            slot = row * PAGE_SIZE + offset
            buf[row, offset * INDEX_HEAD_DIM : (offset + 1) * INDEX_HEAD_DIM] = (
                slot % 256
            )
            scale_start = S_OFFSET_IN_PAGE + offset * S_BYTES
            buf[row, scale_start : scale_start + S_BYTES] = (slot % 256) ^ 0xA5
    pool.index_k_with_scale_buffer = [buf]
    return pool


def _token_bytes(buf: torch.Tensor, loc: int) -> torch.Tensor:
    row, offset = divmod(loc, PAGE_SIZE)
    key = buf[row, offset * INDEX_HEAD_DIM : (offset + 1) * INDEX_HEAD_DIM]
    scale_start = S_OFFSET_IN_PAGE + offset * S_BYTES
    scale = buf[row, scale_start : scale_start + S_BYTES]
    return torch.cat([key, scale])


def _move_indexer(
    pool: DSATokenToKVPool, target: torch.Tensor, source: torch.Tensor
) -> None:
    with patch.object(MLATokenToKVPool, "move_kv_cache"):
        pool.move_kv_cache(target, source)


class TestDSAIndexerMove(CustomTestCase):
    def test_scattered_non_page_aligned_tokens_are_relocated(self):
        pool = _pool_stub(num_pages=10)
        before = pool.index_k_with_scale_buffer[0].clone()
        source = torch.tensor([200, 5, 130, 511, 64, 383, 7, 448])
        target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        expected = {
            int(target_loc): _token_bytes(before, int(source_loc))
            for source_loc, target_loc in zip(source, target)
        }

        _move_indexer(pool, target, source)

        after = pool.index_k_with_scale_buffer[0]
        for target_loc in target:
            self.assertTrue(
                torch.equal(
                    _token_bytes(after, int(target_loc)), expected[int(target_loc)]
                )
            )

    def test_tokens_sharing_target_pages_are_not_corrupted(self):
        pool = _pool_stub(num_pages=10)
        before = pool.index_k_with_scale_buffer[0].clone()
        source = torch.tensor([200, 5, 130, 511, 64, 383, 7, 448])
        target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        moved = {int(target_loc) for target_loc in target}

        _move_indexer(pool, target, source)

        after = pool.index_k_with_scale_buffer[0]
        for slot in range(10 * PAGE_SIZE):
            if slot not in moved:
                self.assertTrue(
                    torch.equal(_token_bytes(after, slot), _token_bytes(before, slot))
                )

    def test_empty_move_is_noop(self):
        pool = _pool_stub(num_pages=2)
        before = pool.index_k_with_scale_buffer[0].clone()
        empty = torch.empty(0, dtype=torch.long)

        _move_indexer(pool, empty, empty)

        self.assertTrue(torch.equal(pool.index_k_with_scale_buffer[0], before))


if __name__ == "__main__":
    unittest.main()
