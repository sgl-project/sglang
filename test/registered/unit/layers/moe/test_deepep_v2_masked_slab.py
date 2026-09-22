"""Tests for the DeepEP v2 expanded/masked repack kernels."""

import unittest
from types import SimpleNamespace

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

    def test_runner_defers_expanded_route_weighting(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            DeepGemmRunnerOutput,
            post_permute_deep_gemm_to_deepep_v2,
        )
        from sglang.srt.layers.moe.token_dispatcher.base import RoutewiseLayout

        counts = [3, 0, 2]
        recv_x, _, psum, starts, total = _build_layout(
            counts, self.ALIGN, self.HIDDEN, torch.bfloat16
        )
        masked_x, _, _ = expand_to_masked_slab(
            recv_x, None, psum, len(counts), self.MAX_M, self.ALIGN
        )
        weights = torch.full((total,), 0.25, device=DEVICE)
        state = {
            "deepep_v2_expanded": True,
            "deepep_v2_masked": True,
            "deepep_v2_psum": psum,
            "deepep_v2_total_expanded": total,
            "deepep_v2_expert_alignment": self.ALIGN,
            "topk_weights": weights,
        }
        rows = _real_rows(starts, counts)
        for no_combine in (False, True):
            with self.subTest(no_combine=no_combine):
                output = post_permute_deep_gemm_to_deepep_v2(
                    DeepGemmRunnerOutput(masked_x),
                    None,
                    MoeRunnerConfig(no_combine=no_combine),
                    state,
                )
                expected = recv_x[rows] if no_combine else recv_x[rows] * 0.25
                self.assertTrue(torch.equal(output.hidden_states[rows], expected))
                self.assertEqual(
                    output.routewise_layout,
                    RoutewiseLayout.EXPANDED if no_combine else None,
                )
                if no_combine:
                    self.assertTrue(torch.equal(output.topk_weights, weights))

    def test_runner_restores_token_topk_routes_and_masks_nonlocal_slots(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            DeepGemmRunnerOutput,
            post_permute_deep_gemm_to_deepep_v2,
        )
        from sglang.srt.layers.moe.token_dispatcher.base import RoutewiseLayout

        hidden = torch.tensor([[3.0], [5.0], [7.0]], device=DEVICE)
        weights = torch.tensor([[0.25, 0.0], [0.5, 0.75]], device=DEVICE)
        state = {
            "topk_ids": torch.tensor([[0, -1], [1, 0]], device=DEVICE),
            "topk_weights": weights,
            "output_index": torch.tensor([[1, -1], [0, 2]], device=DEVICE),
        }
        for empty in (False, True):
            with self.subTest(empty=empty):
                if empty:
                    state["output_index"] = torch.full((2, 2), -1, device=DEVICE)
                output = post_permute_deep_gemm_to_deepep_v2(
                    DeepGemmRunnerOutput(hidden[:0] if empty else hidden),
                    None,
                    MoeRunnerConfig(no_combine=True),
                    state,
                )
                expected = torch.tensor([[[5.0], [0.0]], [[3.0], [7.0]]], device=DEVICE)
                if empty:
                    expected.zero_()
                self.assertTrue(torch.equal(output.hidden_states, expected))
                self.assertIs(output.topk_weights, weights)
                self.assertEqual(output.routewise_layout, RoutewiseLayout.TOKEN_TOPK)

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

    def _production_packed_ue8m0_layout(self, counts, group_size):
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
            group_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.assertEqual(recv_x_scale.dtype, torch.int32)
        self.assertGreater(recv_x_scale.shape[1], 1, "pack dim must be indexed")
        self.assertNotEqual(recv_x_scale.stride(1), 1)
        return recv_x, recv_x_scale, psum, starts, total, hidden

    def test_fp8_packed_ue8m0_scale_from_production_quantizer(self):
        for group_size in (32, 128):
            with self.subTest(group_size=group_size):
                self._check_packed_scale(group_size)

    def _check_packed_scale(self, group_size):
        counts = [3, 1, 6, 2]
        recv_x, recv_x_scale, psum, starts, _, hidden = (
            self._production_packed_ue8m0_layout(counts, group_size)
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
        for group_size in (32, 128):
            with self.subTest(group_size=group_size):
                self._check_graph_capture(group_size)

    def _check_graph_capture(self, group_size):
        # Exercise replay with the production packed scale layout.
        counts = [3, 1, 6, 2]
        recv_x, recv_x_scale, psum, starts, _, _ = self._production_packed_ue8m0_layout(
            counts, group_size
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
            impl.combine(SimpleNamespace(routewise_layout=None))
        self.assertIsNone(impl._handle)
        self.assertFalse(impl._pad_empty_combine)

    def test_unfinalized_routes_are_rejected_and_release_handle(self):
        from sglang.srt.layers.moe.token_dispatcher.base import (
            CombineInputChecker,
            RoutewiseLayout,
        )
        from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
            DeepEPv2CombineInput,
        )

        impl = self._bare_impl()
        impl._handle = object()
        output = DeepEPv2CombineInput(
            torch.empty(2, 8), torch.ones(2), RoutewiseLayout.EXPANDED
        )
        self.assertTrue(CombineInputChecker.needs_model_route_finalization(output))
        with self.assertRaisesRegex(ValueError, "model route finalization"):
            impl.combine(output)
        self.assertIsNone(impl._handle)
        self.assertFalse(
            CombineInputChecker.needs_model_route_finalization(
                DeepEPv2CombineInput(torch.empty(2, 8), None)
            )
        )


if __name__ == "__main__":
    unittest.main()
