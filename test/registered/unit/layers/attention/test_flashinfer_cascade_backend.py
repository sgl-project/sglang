"""Unit tests for srt/layers/attention/flashinfer_cascade_backend.py

Covers the shared-prefix detection logic (``_detect_common_prefix_from_rpi``)
that gates cascade dispatch in both the eager and CUDA-graph paths. This is the
branch-heavy, correctness-critical core of the backend, so we exercise it in
isolation -- constructing the backend via ``__new__`` and populating only the
two attributes the method reads (``req_to_token`` and ``_cascade_scan_cap``)
rather than launching a server or allocating flashinfer wrappers.
"""

import unittest

import torch

from sglang.srt.layers.attention.flashinfer_cascade_backend import (
    FlashInferCascadeAttnBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Needs a GPU only so the module's flashinfer imports resolve; the detection
# logic under test runs on plain CPU tensors and launches no server.
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _make_backend(req_to_token: torch.Tensor, scan_cap: int = 32768):
    """Build a FlashInferCascadeAttnBackend with only the fields that
    ``_detect_common_prefix_from_rpi`` reads, skipping the heavy ``__init__``."""
    backend = FlashInferCascadeAttnBackend.__new__(FlashInferCascadeAttnBackend)
    backend.req_to_token = req_to_token
    backend._cascade_scan_cap = scan_cap
    return backend


def _req_to_token(rows):
    return torch.tensor(rows, dtype=torch.int32)


class TestCascadePrefixDetection(CustomTestCase):
    def _detect(self, backend, bs, rpi, seq_lens):
        return backend._detect_common_prefix_from_rpi(
            bs,
            torch.tensor(rpi, dtype=torch.int64),
            None if seq_lens is None else torch.tensor(seq_lens, dtype=torch.int32),
        )

    def test_below_min_batch_returns_zero(self):
        # bs < 2 can never share a cross-request prefix.
        backend = _make_backend(_req_to_token([[100, 101, 102, 103]]))
        self.assertEqual(self._detect(backend, 1, [0], [4]), 0)

    def test_missing_seq_lens_returns_zero(self):
        backend = _make_backend(_req_to_token([[1, 2], [1, 2]]))
        self.assertEqual(self._detect(backend, 2, [0, 1], None), 0)

    def test_min_seq_le_one_returns_zero(self):
        # A request with seq_len 1 has no slot to spare for a shared prefix.
        backend = _make_backend(_req_to_token([[5, 6, 7], [5, 6, 7]]))
        self.assertEqual(self._detect(backend, 2, [0, 1], [1, 3]), 0)

    def test_full_share_capped_at_min_seq_minus_one(self):
        # All four requests share every scanned slot; the result is capped so
        # each request keeps >= 1 unique slot for its own decode token.
        rows = [[9, 9, 9, 9, 9, 9, 9, 9] for _ in range(4)]
        backend = _make_backend(_req_to_token(rows))
        # min_seq = 3 -> cap at 2.
        self.assertEqual(self._detect(backend, 4, [0, 1, 2, 3], [3, 3, 3, 3]), 2)

    def test_partial_share_returns_divergence_index(self):
        # First 5 slots identical, slot 5 differs across requests.
        rows = [
            [100, 101, 102, 103, 104, 200, 0, 0],
            [100, 101, 102, 103, 104, 210, 0, 0],
            [100, 101, 102, 103, 104, 220, 0, 0],
            [100, 101, 102, 103, 104, 230, 0, 0],
        ]
        backend = _make_backend(_req_to_token(rows))
        self.assertEqual(self._detect(backend, 4, [0, 1, 2, 3], [20, 20, 20, 20]), 5)

    def test_no_share_returns_zero(self):
        # Requests diverge at the very first slot.
        rows = [[1, 9, 9], [2, 9, 9], [3, 9, 9], [4, 9, 9]]
        backend = _make_backend(_req_to_token(rows))
        self.assertEqual(self._detect(backend, 4, [0, 1, 2, 3], [20, 20, 20, 20]), 0)

    def test_scan_cap_limits_detection(self):
        # Rows share the first 5 slots, but the scan cap stops at 3, so at most
        # 3 shared slots are reported (still bounded by min_seq - 1).
        rows = [[7, 7, 7, 7, 7, 9, 0, 0] for _ in range(4)]
        backend = _make_backend(_req_to_token(rows), scan_cap=3)
        self.assertEqual(self._detect(backend, 4, [0, 1, 2, 3], [20, 20, 20, 20]), 3)

    def test_respects_req_pool_indices_ordering(self):
        # Detection must read the rows named by req_pool_indices (not rows 0..bs).
        rows = [
            [50, 51, 52, 99, 0],  # row 0 (unused in batch)
            [10, 11, 12, 13, 14],  # row 1
            [10, 11, 12, 99, 14],  # row 2 diverges at slot 3
            [10, 11, 12, 77, 14],  # row 3 diverges at slot 3
        ]
        backend = _make_backend(_req_to_token(rows))
        # Batch = rows [1, 2, 3]; shared prefix is slots 0..2 -> 3.
        self.assertEqual(self._detect(backend, 3, [1, 2, 3], [10, 10, 10]), 3)


if __name__ == "__main__":
    unittest.main()
