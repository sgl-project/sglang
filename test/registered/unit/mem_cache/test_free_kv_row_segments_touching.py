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
"""`free_kv_row_segments` must coalesce touching segments of one kv row.

`UnifiedRadixCache.cache_finished_req` frees a request truncated mid prefill
(an abort) as two ranges, ``[page_aligned_len, effective_len)`` and
``[effective_len, full_len)``, which touch at ``effective_len``. Handed to
``allocator.free_segments`` as two segments, ``_page_disjoint`` asserts on the
shared boundary page ("segment at N shares a page with the one ending at N")
and every scheduler rank dies with the request. Observed on Kimi-K3 (TP8,
DSPARK, page_size 64) three times in production in two days, each time right
after an abort of a request with a 120-200K-character prompt.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    _coalesce_touching_segments,
    free_kv_row_segments,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64
ROW = torch.arange(10_000, 10_000 + 4096, dtype=torch.int64)


class _PageCheckOnly:
    """Just enough allocator for `BaseTokenToKVPoolAllocator._page_disjoint`."""

    page_size = PAGE


class _RecordingAllocator:
    page_size = PAGE

    def __init__(self):
        self.full_calls: list[list[tuple[torch.Tensor, int]]] = []
        self.alive_calls: list[list[tuple[torch.Tensor, int]]] = []

    def free_full_segments(self, segments):
        self.full_calls.append(list(segments))

    def free_segments(self, segments):
        # Run the real disjointness check so the test fails the way the
        # scheduler did if the segments still touch mid-page.
        list(BaseTokenToKVPoolAllocator._page_disjoint(_PageCheckOnly(), segments))
        self.alive_calls.append(list(segments))


def _abort_shape(page_aligned_len: int, effective_len: int, full_len: int):
    """The two ranges cache_finished_req emits for a truncated request."""
    return [
        (ROW[page_aligned_len:effective_len], page_aligned_len),
        (ROW[effective_len:full_len], effective_len),
    ]


class TestCoalesceTouchingSegments(unittest.TestCase):
    def test_touching_slices_become_the_slice_they_were_cut_from(self):
        out = _coalesce_touching_segments(_abort_shape(1024, 1100, 1216))
        self.assertEqual(len(out), 1)
        indices, start = out[0]
        self.assertEqual(start, 1024)
        self.assertTrue(torch.equal(indices, ROW[1024:1216]))

    def test_gaps_stay_apart_and_empties_are_dropped(self):
        segs = [(ROW[0:8], 0), (ROW[8:8], 8), (ROW[20:30], 20), (ROW[30:33], 30)]
        out = _coalesce_touching_segments(segs)
        self.assertEqual([(s, i.numel()) for i, s in out], [(0, 8), (20, 13)])


class TestFreeKvRowSegmentsAbortShape(unittest.TestCase):
    def test_the_raw_abort_shape_would_assert(self):
        # Documents the failure this fixes: a boundary that is not page-aligned.
        segs = _abort_shape(1024, 1100, 1216)
        with self.assertRaisesRegex(AssertionError, "shares a page"):
            list(BaseTokenToKVPoolAllocator._page_disjoint(_PageCheckOnly(), segs))

    def test_mid_page_boundary_is_freed_as_one_segment(self):
        alloc = _RecordingAllocator()
        free_kv_row_segments(
            alloc, _abort_shape(1024, 1100, 1216), swa_evicted_seqlen=0
        )
        self.assertEqual(alloc.full_calls, [])
        self.assertEqual(len(alloc.alive_calls), 1)
        [(indices, start)] = alloc.alive_calls[0]
        self.assertEqual(start, 1024)
        self.assertTrue(torch.equal(indices, ROW[1024:1216]))

    def test_swa_floor_split_still_applies_after_coalescing(self):
        alloc = _RecordingAllocator()
        free_kv_row_segments(
            alloc, _abort_shape(1024, 1100, 1216), swa_evicted_seqlen=1088
        )
        [(dead, dead_start)] = alloc.full_calls[0]
        [(alive, alive_start)] = alloc.alive_calls[0]
        self.assertEqual((dead_start, dead.numel()), (1024, 64))
        self.assertEqual((alive_start, alive.numel()), (1088, 128))
        self.assertTrue(torch.equal(torch.cat((dead, alive)), ROW[1024:1216]))


if __name__ == "__main__":
    unittest.main()
