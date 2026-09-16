"""Tests for the DeepEP v2 expand-path helper kernels.

fill_m_indices_from_psum and scale_expanded_rows_ only run on the do_expand=True
prefill path, which no other CI test exercises. Both are pure and deterministic
with obvious reference implementations, so pin them directly.
"""

import unittest

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    fill_m_indices_from_psum,
    scale_expanded_rows_,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


def _reference_m_indices(counts, align, total_rows):
    """Label rows [start, start+count) with the expert id, aligned per expert;
    uncovered rows stay at the -1 sentinel (matching the kernel's allocation)."""
    m_indices = [-1] * total_rows
    start = 0
    for e, count in enumerate(counts):
        for r in range(start, start + count):
            m_indices[r] = e
        # psum is inclusive: next expert starts at align(prev_end + count).
        start = ((start + count + align - 1) // align) * align
    return m_indices


class TestFillMIndicesFromPsum(CustomTestCase):
    """m_indices must label exactly the aligned per-expert segments."""

    def _run(self, counts, align):
        # psum[i] = align(psum[i-1]) + count_i (DeepEP's inclusive prefix sum).
        psum_vals = []
        prev_end = 0
        for count in counts:
            start = ((prev_end + align - 1) // align) * align
            prev_end = start + count
            psum_vals.append(prev_end)
        num_local_experts = len(counts)
        # do_cpu_sync=True sizes recv_x to align(psum[-1]).
        total_rows = ((psum_vals[-1] + align - 1) // align) * align

        psum = torch.tensor(psum_vals, dtype=torch.int32, device=DEVICE)
        m_indices = fill_m_indices_from_psum(psum, num_local_experts, total_rows, align)

        expected = _reference_m_indices(counts, align, total_rows)
        self.assertEqual(m_indices.tolist(), expected)

    def test_align_128(self):
        self._run([200, 50, 128], align=128)

    def test_align_not_128(self):
        # The kernel this replaces silently assumed expert_alignment == 128; a
        # separate ALIGN parameter is the fix, so cover a non-128 alignment.
        self._run([100, 30, 64], align=64)

    def test_empty_expert_segment(self):
        # An expert with count == 0 (start == seg_end) must label no rows.
        self._run([128, 0, 64], align=128)


class TestScaleExpandedRows(CustomTestCase):
    """In-place row scaling must match x * w[:, None] on any strides."""

    def _check(self, x, weights):
        ref = x.to(torch.float32) * weights.to(torch.float32).unsqueeze(1)
        scale_expanded_rows_(x, weights)
        torch.testing.assert_close(
            x.to(torch.float32), ref.to(x.dtype).to(torch.float32)
        )

    def test_contiguous(self):
        x = torch.randn(37, 128, device=DEVICE)
        weights = torch.rand(37, device=DEVICE)
        self._check(x, weights)

    def test_transposed_view(self):
        # The column-major (non-unit-stride(1)) case is the whole reason this
        # helper exists rather than x *= w[:, None] on a plain tensor.
        base = torch.randn(16, 40, device=DEVICE)
        x = base.t()  # [40, 16], stride(1) != 1
        self.assertNotEqual(x.stride(1), 1)
        weights = torch.rand(40, device=DEVICE)
        self._check(x, weights)

    def test_rows_zero(self):
        x = torch.randn(0, 128, device=DEVICE)
        weights = torch.empty(0, device=DEVICE)
        # Must not launch or raise on an empty dispatch.
        scale_expanded_rows_(x, weights)
        self.assertEqual(x.shape, (0, 128))


if __name__ == "__main__":
    unittest.main()
