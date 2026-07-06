"""Unit tests for the DeepEP v2 masked-slab repack Triton kernels.

Covers the corner cases flagged in review: empty expert, single hot expert,
per-expert count near / over max_m (overflow -> fail-fast, not silent truncation),
top-k weight fusion on real rows only, expanded<->slab round-trip layout, and the
fp8 activation+scale path.
"""

import unittest

import torch

from sglang.srt.layers.moe.ep_moe.kernels import (
    expand_to_masked_slab,
    masked_slab_to_expand,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"


def _build_layout(counts, align, hidden, dtype, with_scale=False, scale_hidden=4):
    """Build (recv_x, recv_x_scale, psum, starts, total) for given per-expert counts.

    Mirrors the DeepEP v2 expanded layout: expert e occupies rows
    [align(psum[e-1]), psum[e]) with psum[-1] == 0.
    """
    starts, psum = [], []
    prev_end = 0
    for c in counts:
        start = ((prev_end + align - 1) // align) * align
        end = start + c
        starts.append(start)
        psum.append(end)
        prev_end = end
    total = max(((prev_end + align - 1) // align) * align, 1)

    # unique, exactly-representable values per real row (row index, kept small)
    base = torch.zeros((total, hidden), dtype=torch.float32, device=DEVICE)
    for e, (s, c) in enumerate(zip(starts, counts)):
        for j in range(c):
            base[s + j] = float((s + j) % 200 + 1)
    recv_x = base.to(dtype)

    scale = None
    if with_scale:
        scale = torch.zeros((total, scale_hidden), dtype=torch.float32, device=DEVICE)
        for e, (s, c) in enumerate(zip(starts, counts)):
            for j in range(c):
                scale[s + j] = float((s + j) % 50 + 1) * 0.5

    psum_t = torch.tensor(psum, dtype=torch.int32, device=DEVICE)
    return recv_x, scale, psum_t, starts, total


def _real_rows(starts, counts):
    rows = []
    for s, c in zip(starts, counts):
        rows.extend(range(s, s + c))
    return rows


class TestDeepEPv2MaskedSlab(CustomTestCase):
    ALIGN = 16
    HIDDEN = 8
    MAX_M = 32

    def _check_expand_roundtrip(self, counts, dtype, with_scale, topk=False):
        recv_x, scale, psum, starts, total = _build_layout(
            counts, self.ALIGN, self.HIDDEN, dtype, with_scale=with_scale
        )
        E = len(counts)
        slab, slab_scale, masked_m = expand_to_masked_slab(
            recv_x, scale, psum, E, self.MAX_M, self.ALIGN
        )

        # masked_m == real per-expert count
        self.assertEqual(masked_m.tolist(), list(counts))
        self.assertEqual(tuple(slab.shape), (E, self.MAX_M, self.HIDDEN))

        # slab real rows == source expanded rows
        for e, (s, c) in enumerate(zip(starts, counts)):
            for j in range(c):
                torch.testing.assert_close(slab[e, j].float(), recv_x[s + j].float())
                if with_scale:
                    torch.testing.assert_close(
                        slab_scale[e, j].float(), scale[s + j].float()
                    )

        # round-trip back to expanded order
        weights = None
        if topk:
            weights = torch.zeros(total, dtype=torch.float32, device=DEVICE)
            for r in _real_rows(starts, counts):
                weights[r] = 0.25 + (r % 7) * 0.1
        out = masked_slab_to_expand(slab, psum, total, self.ALIGN, topk_weights=weights)
        self.assertEqual(tuple(out.shape), (total, self.HIDDEN))
        for e, (s, c) in enumerate(zip(starts, counts)):
            for j in range(c):
                expected = slab[e, j].float()
                if topk:
                    expected = (expected * weights[s + j]).to(slab.dtype).float()
                torch.testing.assert_close(out[s + j].float(), expected)

    def test_roundtrip_bf16(self):
        self._check_expand_roundtrip([3, 0, 5, 1], torch.bfloat16, with_scale=False)

    def test_roundtrip_bf16_with_topk_weight(self):
        self._check_expand_roundtrip(
            [2, 4, 0, 7], torch.bfloat16, with_scale=False, topk=True
        )

    def test_roundtrip_fp8_with_scale(self):
        self._check_expand_roundtrip([3, 1, 6, 2], torch.float8_e4m3fn, with_scale=True)

    def test_empty_experts(self):
        # all experts empty
        self._check_expand_roundtrip([0, 0, 0, 0], torch.bfloat16, with_scale=False)

    def test_single_hot_expert(self):
        # one expert holds many tokens, the rest empty
        self._check_expand_roundtrip(
            [0, self.MAX_M, 0, 0], torch.bfloat16, with_scale=False, topk=True
        )

    def test_count_at_max_m_boundary(self):
        # exactly max_m must be kept (no overflow, no truncation)
        self._check_expand_roundtrip(
            [self.MAX_M, 1, self.MAX_M], torch.bfloat16, with_scale=False
        )

    def test_overflow_fails_fast(self):
        # one expert exceeds max_m -> must raise, not silently truncate
        counts = [self.MAX_M + 1, 2]
        recv_x, scale, psum, starts, total = _build_layout(
            counts, self.ALIGN, self.HIDDEN, torch.bfloat16
        )
        with self.assertRaises(RuntimeError):
            expand_to_masked_slab(
                recv_x, None, psum, len(counts), self.MAX_M, self.ALIGN
            )


class TestDeepEPv2HandleLifecycle(CustomTestCase):
    """CPU-only guards of the dispatch/combine handle lifecycle.

    The guards are ordered before any DeepEP work, so misuse is testable
    without deep_ep installed and without a GPU. The positive dispatch ->
    combine path needs real ElasticBuffer communication and is covered by the
    GPU accuracy runs instead.
    """

    @staticmethod
    def _bare_impl():
        from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import _DeepEPv2Impl

        impl = object.__new__(_DeepEPv2Impl)
        impl._handle = None
        impl._pad_empty_combine = False
        return impl

    def test_combine_without_dispatch_raises(self):
        impl = self._bare_impl()
        with self.assertRaisesRegex(RuntimeError, "without a valid dispatch handle"):
            impl.combine(None)

    def test_dispatch_with_unconsumed_handle_raises(self):
        impl = self._bare_impl()
        impl._handle = object()
        with self.assertRaisesRegex(RuntimeError, "unconsumed"):
            impl.dispatch(None, None)

    def test_handle_cleared_when_combine_fails(self):
        impl = self._bare_impl()
        impl._handle = object()
        impl._pad_empty_combine = True

        def _boom():
            raise RuntimeError("boom")

        impl._get_buffer = _boom
        with self.assertRaisesRegex(RuntimeError, "boom"):
            impl.combine(None)
        self.assertIsNone(impl._handle)
        self.assertFalse(impl._pad_empty_combine)


if __name__ == "__main__":
    unittest.main()
