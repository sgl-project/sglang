"""Long-running alloc/free round-trip — confirms the per-forward
transient lifecycle does not drift the pool's free-count or leak
sentinel writes in req_to_token (DESIGN_kv_reshard.md §6, Part 4e).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.srt.layers.utils.cp_transient import (
    CpTransientState,
    cp_alloc_forward_transient,
    cp_free_forward_transient,
)
from sglang.test.test_utils import CustomTestCase


class _FakeAllocator:
    def __init__(self, start: int = 100, capacity: int = 4096):
        self._free: List[int] = list(range(start, start + capacity))

    @property
    def available(self) -> int:
        return len(self._free)

    def available_size(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        if n > len(self._free):
            return None
        out = torch.tensor(self._free[:n], dtype=torch.int64)
        self._free = self._free[n:]
        return out

    def free(self, indices: torch.Tensor) -> None:
        self._free.extend(int(x) for x in indices.tolist())


def _make_fb(num_req_rows: int = 8, max_context: int = 32):
    return SimpleNamespace(
        extend_prefix_lens_cpu=[4],
        extend_seq_lens_cpu=[4],
        req_pool_indices=torch.tensor([3], dtype=torch.int64),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((num_req_rows, max_context), dtype=torch.int64)
        ),
        cp_transient=CpTransientState(
            owner_per_pages=[torch.tensor([0, 1, 0, 1], dtype=torch.int8)],
        ),
    )


class TestNoLeak(CustomTestCase):
    def test_thousand_iters_return_to_baseline(self):
        allocator = _FakeAllocator(start=100, capacity=4096)
        baseline = allocator.available

        for _ in range(1000):
            fb = _make_fb()
            cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
            self.assertIsNotNone(fb.cp_transient.rows)
            cp_free_forward_transient(fb, allocator)
            self.assertIsNone(fb.cp_transient.rows)
            self.assertIsNone(fb.cp_transient.prefix_rows)

        self.assertEqual(allocator.available, baseline)

    def test_req_to_token_returns_to_sentinel(self):
        # After alloc + free, non-owned positions in req_to_token are sentinel-0 again.
        allocator = _FakeAllocator(start=200, capacity=64)
        fb = _make_fb()

        cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        rtt = fb.req_to_token_pool.req_to_token
        # Confirm scatter happened: at least one non-owned position is non-zero.
        self.assertTrue((rtt[3] != 0).any().item())

        cp_free_forward_transient(fb, allocator)
        # All transient slots zero again.
        for pos in (2, 3, 6, 7):
            self.assertEqual(rtt[3, pos].item(), 0)


if __name__ == "__main__":
    unittest.main()
