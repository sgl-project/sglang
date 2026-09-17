"""Tests for the DeepEP v2 expanded/masked repack kernels."""

import unittest

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    expand_to_masked_slab,
    masked_slab_to_expand,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"


def _build_layout(counts, align, hidden, dtype, with_scale=False, scale_hidden=4):
    """Build synthetic expanded-layout buffers for per-expert counts."""
    starts, psum = [], []
    prev_end = 0
    for c in counts:
        start = ((prev_end + align - 1) // align) * align
        end = start + c
        starts.append(start)
        psum.append(end)
        prev_end = end
    total = max(((prev_end + align - 1) // align) * align, 1)

    # Vary rows and columns to expose broadcast or stride errors.
    base = torch.zeros((total, hidden), dtype=torch.float32, device=DEVICE)
    col_gain = 1.0 + (torch.arange(hidden, device=DEVICE) % 2).float()
    for s, c in zip(starts, counts):
        for j in range(c):
            base[s + j] = float((s + j) % 200 + 1) * col_gain
    recv_x = base.to(dtype)

    scale = None
    if with_scale:
        scale = torch.zeros((total, scale_hidden), dtype=torch.float32, device=DEVICE)
        # Vary scale columns to expose pack-dimension stride errors.
        col = torch.arange(scale_hidden, dtype=torch.float32, device=DEVICE)
        for s, c in zip(starts, counts):
            for j in range(c):
                scale[s + j] = float((s + j) % 50 + 1) * 0.5 + col

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
        masked_x, masked_x_scale, masked_m = expand_to_masked_slab(
            recv_x, scale, psum, E, self.MAX_M, self.ALIGN
        )

        self.assertEqual(masked_m.tolist(), list(counts))
        self.assertEqual(tuple(masked_x.shape), (E, self.MAX_M, self.HIDDEN))

        for e, (s, c) in enumerate(zip(starts, counts)):
            for j in range(c):
                torch.testing.assert_close(
                    masked_x[e, j].float(), recv_x[s + j].float()
                )
                if with_scale:
                    torch.testing.assert_close(
                        masked_x_scale[e, j].float(), scale[s + j].float()
                    )

        weights = None
        if topk:
            weights = torch.zeros(total, dtype=torch.float32, device=DEVICE)
            for r in _real_rows(starts, counts):
                weights[r] = 0.25 + (r % 7) * 0.1
        out = masked_slab_to_expand(
            masked_x, psum, total, self.ALIGN, topk_weights=weights
        )
        self.assertEqual(tuple(out.shape), (total, self.HIDDEN))
        for e, (s, c) in enumerate(zip(starts, counts)):
            for j in range(c):
                expected = masked_x[e, j].float()
                if topk:
                    expected = (expected * weights[s + j]).to(masked_x.dtype).float()
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
        self._check_expand_roundtrip([0, 0, 0, 0], torch.bfloat16, with_scale=False)

    def test_single_hot_expert(self):
        self._check_expand_roundtrip(
            [0, self.MAX_M, 0, 0], torch.bfloat16, with_scale=False, topk=True
        )

    def test_count_at_max_m_boundary(self):
        self._check_expand_roundtrip(
            [self.MAX_M, 1, self.MAX_M], torch.bfloat16, with_scale=False
        )

    def test_overflow_fails_fast(self):
        counts = [self.MAX_M + 1, 2]
        recv_x, scale, psum, starts, total = _build_layout(
            counts, self.ALIGN, self.HIDDEN, torch.bfloat16
        )
        with self.assertRaises(RuntimeError):
            expand_to_masked_slab(
                recv_x, None, psum, len(counts), self.MAX_M, self.ALIGN
            )

    def _production_packed_ue8m0_layout(self, counts):
        """Build expanded rows with the production packed UE8M0 quantizer."""
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        # hidden=1024 ensures the packed scale has multiple columns.
        hidden = 1024
        raw, _, psum, starts, total = _build_layout(
            counts, self.ALIGN, hidden, torch.bfloat16
        )
        recv_x, recv_x_scale = sglang_per_token_group_quant_fp8(
            raw,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.assertEqual(recv_x_scale.dtype, torch.int32)
        self.assertGreater(recv_x_scale.shape[1], 1, "pack dim must be indexed")
        self.assertNotEqual(recv_x_scale.stride(1), 1)
        return recv_x, recv_x_scale, psum, starts, total, hidden

    def test_fp8_packed_ue8m0_scale_from_production_quantizer(self):
        counts = [3, 1, 6, 2]
        recv_x, recv_x_scale, psum, starts, _, hidden = (
            self._production_packed_ue8m0_layout(counts)
        )
        E = len(counts)
        masked_x, masked_x_scale, masked_m = expand_to_masked_slab(
            recv_x, recv_x_scale, psum, E, self.MAX_M, self.ALIGN
        )
        self.assertEqual(masked_m.tolist(), list(counts))
        self.assertEqual(tuple(masked_x.shape), (E, self.MAX_M, hidden))
        for e, (s, c) in enumerate(zip(starts, counts)):
            for j in range(c):
                torch.testing.assert_close(
                    masked_x[e, j].float(), recv_x[s + j].float()
                )
                torch.testing.assert_close(masked_x_scale[e, j], recv_x_scale[s + j])

    def test_expand_under_cuda_graph_capture(self):
        # Exercise replay with the production packed scale layout.
        counts = [3, 1, 6, 2]
        recv_x, recv_x_scale, psum, starts, _, _ = self._production_packed_ue8m0_layout(
            counts
        )
        E = len(counts)
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            expand_to_masked_slab(recv_x, recv_x_scale, psum, E, self.MAX_M, self.ALIGN)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            masked_x, masked_x_scale, masked_m = expand_to_masked_slab(
                recv_x, recv_x_scale, psum, E, self.MAX_M, self.ALIGN
            )
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(masked_m.tolist(), list(counts))
        for e, (s, c) in enumerate(zip(starts, counts)):
            for j in range(c):
                torch.testing.assert_close(
                    masked_x[e, j].float(), recv_x[s + j].float()
                )
                torch.testing.assert_close(masked_x_scale[e, j], recv_x_scale[s + j])


class TestDeepEPv2HandleLifecycle(CustomTestCase):
    """CPU-only dispatch/combine handle guards."""

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
