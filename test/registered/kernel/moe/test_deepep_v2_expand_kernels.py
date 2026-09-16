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
    """Label each expert's whole *aligned* segment with its id, matching the
    kernel; only rows past the last segment keep the -1 sentinel."""
    m_indices = [-1] * total_rows
    start = 0
    for e, count in enumerate(counts):
        # psum is inclusive: seg_end = align(start + count), and the padding
        # inside the segment gets the expert id too (combine ignores it).
        seg_end = ((start + count + align - 1) // align) * align
        for r in range(start, min(seg_end, total_rows)):
            m_indices[r] = e
        start = seg_end
    return m_indices


class TestFillMIndicesFromPsum(CustomTestCase):
    """m_indices must label exactly the aligned per-expert segments."""

    def _run(self, counts, align, extra_rows=0):
        # psum[i] = align(psum[i-1]) + count_i (DeepEP's inclusive prefix sum).
        psum_vals = []
        prev_end = 0
        for count in counts:
            start = ((prev_end + align - 1) // align) * align
            prev_end = start + count
            psum_vals.append(prev_end)
        num_local_experts = len(counts)
        # do_cpu_sync=True sizes recv_x to align(psum[-1]); extra_rows models a
        # capacity-sized buffer where the tail past the segments stays sentinel.
        total_rows = ((psum_vals[-1] + align - 1) // align) * align + extra_rows

        psum = torch.tensor(psum_vals, dtype=torch.int32, device=DEVICE)
        m_indices = fill_m_indices_from_psum(psum, num_local_experts, total_rows, align)

        expected = _reference_m_indices(counts, align, total_rows)
        self.assertEqual(m_indices.tolist(), expected)
        return m_indices

    def test_align_128(self):
        self._run([200, 50, 128], align=128)

    def test_align_not_128(self):
        # The kernel this replaces silently assumed expert_alignment == 128; a
        # separate ALIGN parameter is the fix, so cover a non-128 alignment.
        self._run([100, 30, 64], align=64)

    def test_empty_expert_segment(self):
        # An expert with count == 0 (start == seg_end) must label no rows.
        self._run([128, 0, 64], align=128)

    def test_segments_tile_exactly(self):
        # do_cpu_sync=True => total_rows == align(psum[-1]), so the aligned
        # segments tile the array and no -1 sentinel survives.
        m_indices = self._run([200, 50, 128], align=128)
        self.assertNotIn(-1, m_indices.tolist())

    def test_sentinel_survives_capacity_tail(self):
        # A capacity-sized buffer leaves rows past the last segment at -1 -- the
        # defensive torch.full(-1) fill that guards the out-of-bounds weight read.
        m_indices = self._run([200, 50, 128], align=128, extra_rows=128)
        self.assertEqual(m_indices[-128:].tolist(), [-1] * 128)


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
