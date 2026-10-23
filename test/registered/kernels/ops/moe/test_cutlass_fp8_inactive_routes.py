"""CUTLASS FP8 maps and reductions skip inactive expert routes."""

import unittest
from collections import defaultdict
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.shuffle_weighted_sum import (
    _shuffle_weighted_sum_kernel,
    shuffle_weighted_sum,
)
from sglang.srt.layers.moe import cutlass_moe
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestCutlassFp8InactiveRoutes(CustomTestCase):
    def test_masked_reduction_ignores_poisoned_slots(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = torch.full((6, 257), float("nan"), device="cuda", dtype=dtype)
                inputs[0].fill_(2)
                inputs[1].fill_(4)
                mapping = torch.tensor(
                    [0, 1, -1, -1, 0, -1], device="cuda", dtype=torch.int32
                )
                weights = torch.tensor(
                    [[0.25, 0.75], [float("nan"), float("nan")], [0.5, float("nan")]],
                    device="cuda",
                    dtype=dtype,
                )
                output = torch.empty((3, 257), device="cuda", dtype=dtype)
                shuffle_weighted_sum(inputs, output, mapping, weights)
                expected = torch.tensor([3.5, 0, 1], device="cuda", dtype=dtype)[
                    :, None
                ].expand_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_reduction_reuses_kernels_across_token_counts(self):
        """New lengths reuse the reduction and preserve mixed/empty route results."""
        kernel = _shuffle_weighted_sum_kernel
        lengths = (1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65)
        for dtype in (torch.float16, torch.bfloat16):
            with (
                self.subTest(dtype=dtype),
                patch.object(
                    kernel, "device_caches", defaultdict(kernel.create_binder)
                ),
            ):
                cache = kernel.device_caches[torch.cuda.current_device()][0]
                specialization_count = None
                for strided_weights in (False, True):
                    for tokens in lengths:
                        with self.subTest(
                            tokens=tokens, strided_weights=strided_weights
                        ):
                            copies = (tokens + 2) // 3
                            inputs = torch.full(
                                (tokens * 2, 257),
                                float("nan"),
                                device="cuda",
                                dtype=dtype,
                            )
                            inputs[0].fill_(2)
                            inputs[1].fill_(4)
                            mapping = (
                                torch.tensor(
                                    [[0, 1], [1, -1], [-1, -1]],
                                    device="cuda",
                                    dtype=torch.int32,
                                )
                                .repeat(copies, 1)[:tokens]
                                .flatten()
                            )
                            weights = torch.tensor(
                                [
                                    [0.25, 0.75],
                                    [0.5, float("nan")],
                                    [float("nan"), float("nan")],
                                ],
                                device="cuda",
                                dtype=dtype,
                            ).repeat(copies, 1)[:tokens]
                            if strided_weights:
                                weights = weights.t().contiguous().t()
                            output = torch.empty(
                                (tokens, 257), device="cuda", dtype=dtype
                            )
                            shuffle_weighted_sum(inputs, output, mapping, weights)
                            expected = (
                                torch.tensor([3.5, 2, 0], device="cuda", dtype=dtype)
                                .repeat(copies)[:tokens, None]
                                .expand_as(output)
                            )
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                            if specialization_count is None:
                                specialization_count = len(cache)
                            self.assertEqual(len(cache), specialization_count)

                            mapping.fill_(-1)
                            inputs.fill_(float("nan"))
                            weights.fill_(float("nan"))
                            output.fill_(float("nan"))
                            shuffle_weighted_sum(inputs, output, mapping, weights)
                            self.assertTrue(bool(torch.isfinite(output).all()))
                            self.assertEqual(int(torch.count_nonzero(output)), 0)
                            self.assertEqual(len(cache), specialization_count)

    def test_native_fp8_padding_survives_graph_replay(self):
        if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
            self.skipTest("CUTLASS block FP8 requires SM90, SM100 or SM120")
        for mxfp8 in (False, True):
            if mxfp8 and torch.cuda.get_device_capability()[0] != 10:
                continue
            with self.subTest(mxfp8=mxfp8):
                self._check_native(mxfp8)

    def _check_native(self, mxfp8):
        experts, tokens, hidden, intermediate = 2, 16, 256, 256
        generator = torch.Generator(device="cuda").manual_seed(945)
        inputs = (
            torch.randn(
                (tokens, hidden),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 8
        )
        ids = torch.tensor(
            [[0, 1], [1, -1], [-1, -1], [1, 0]] * 4, device="cuda", dtype=torch.int32
        )
        weights = torch.tensor(
            [
                [0.25, 0.75],
                [0.6, float("nan")],
                [float("nan"), float("nan")],
                [0.8, 0.2],
            ]
            * 4,
            device="cuda",
        )
        inputs[(ids < 0).all(dim=-1)] = float("nan")
        w1 = (
            (
                torch.randn(
                    (experts, 2 * intermediate, hidden),
                    generator=generator,
                    device="cuda",
                )
                / 16
            )
            .to(torch.float8_e4m3fn)
            .transpose(1, 2)
        )
        w2 = (
            (
                torch.randn(
                    (experts, hidden, intermediate), generator=generator, device="cuda"
                )
                / 16
            )
            .to(torch.float8_e4m3fn)
            .transpose(1, 2)
        )
        if mxfp8:
            scale1 = torch.full(
                (experts, hidden // 32, 2 * intermediate),
                127,
                device="cuda",
                dtype=torch.uint8,
            )
            scale2 = torch.full(
                (experts, intermediate // 32, hidden),
                127,
                device="cuda",
                dtype=torch.uint8,
            )
        else:
            scale1 = torch.ones(
                (experts, 2 * intermediate // 128, hidden // 128), device="cuda"
            ).transpose(1, 2)
            scale2 = torch.ones(
                (experts, hidden // 128, intermediate // 128), device="cuda"
            ).transpose(1, 2)
        params = CutlassMoEParams(
            CutlassMoEType.BlockscaledFP8,
            torch.device("cuda"),
            experts,
            intermediate,
            hidden,
        )
        workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)

        def run(x, route_ids, route_weights):
            return cutlass_moe.cutlass_fused_experts_fp8(
                x,
                w1,
                w2,
                scale1,
                scale2,
                route_weights,
                route_ids,
                params.ab_strides_13,
                params.c_strides_13,
                params.ab_strides_2,
                params.c_strides_2,
                workspace,
                params.a_ptrs,
                params.b_ptrs,
                params.out_ptrs,
                params.a_scales_ptrs,
                params.b_scales_ptrs,
                params.expert_offsets,
                params.problem_sizes1,
                params.problem_sizes2,
                use_mxfp8=mxfp8,
                enable_es=(mxfp8, mxfp8),
            )

        reference = run(inputs.nan_to_num(), ids.clamp_min(0), weights.nan_to_num())
        prepare = cutlass_moe.prepare_moe_input

        def checked_prepare(*args):
            prepare(*args)
            route_ids, offsets, _, _, a_map, c_map = args[:6]
            self.assertTrue(bool(((a_map >= 0) & (a_map < tokens)).all()))
            self.assertTrue(bool((c_map[route_ids.flatten() < 0] == -1).all()))
            self.assertEqual(int(offsets[-1]), int((route_ids >= 0).sum()))

        original_empty = torch.empty

        def poisoned_empty(*args, **kwargs):
            tensor = original_empty(*args, **kwargs)
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
            elif tensor.dtype == torch.int32:
                tensor.fill_(-999)
            return tensor

        # The pre-fix map allocation fails checked_prepare before an unsafe gather.
        with (
            patch.object(torch, "empty", poisoned_empty),
            patch.object(cutlass_moe, "prepare_moe_input", checked_prepare),
        ):
            actual = run(inputs, ids, weights)
        torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.01)
        self.assertTrue(bool(torch.isfinite(actual).all()))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result = run(inputs, ids, weights)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run(inputs, ids, weights)
        graph.replay()
        torch.testing.assert_close(result, reference, rtol=0.025, atol=0.01)
        ids.fill_(-1)
        inputs.fill_(float("nan"))
        weights.fill_(float("nan"))
        graph.replay()
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(int(torch.count_nonzero(result)), 0)


if __name__ == "__main__":
    unittest.main()
