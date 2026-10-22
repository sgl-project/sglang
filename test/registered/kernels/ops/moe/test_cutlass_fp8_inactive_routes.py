"""CUTLASS FP8 maps and reductions skip inactive expert routes."""

import unittest
from collections import defaultdict
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.cutlass_prepare_moe_input import cutlass_prepare_moe_input
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
    def test_prepare_completes_maps_across_route_changes(self):
        """Every live route is gathered once; inactive and unused capacity is defined."""
        experts, tokens, topk, n, k = 3, 265, 2, 96, 160
        patterns = (
            [0, 1, 2, -1, -7, experts, 1, 2, 0, 0],
            [-1, -9, experts, 99],
            [0, 1, 2, 0, 1, 2],
        )
        states = [
            torch.tensor(
                (pattern * ((tokens * topk + len(pattern) - 1) // len(pattern)))[
                    : tokens * topk
                ],
                device="cuda",
                dtype=torch.int32,
            ).view(tokens, topk)
            for pattern in patterns
        ]
        ids = states[0].clone()
        offsets = torch.full((experts + 1,), -999, device="cuda", dtype=torch.int32)
        sizes1 = torch.full((experts, 3), -999, device="cuda", dtype=torch.int32)
        sizes2 = torch.full_like(sizes1, -999)
        a_map = torch.full((ids.numel(),), -999, device="cuda", dtype=torch.int32)
        c_map = torch.full_like(a_map, -999)
        block_offsets = torch.full_like(offsets, -999)

        def prepare(blocks):
            cutlass_prepare_moe_input(
                ids, offsets, sizes1, sizes2, a_map, c_map, experts, n, k, blocks
            )

        def check():
            flat_ids = ids.flatten()
            live_mask = (flat_ids >= 0) & (flat_ids < experts)
            counts = torch.bincount(flat_ids[live_mask], minlength=experts)
            expected_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            torch.testing.assert_close(offsets, expected_offsets.int(), rtol=0, atol=0)
            torch.testing.assert_close(
                sizes1,
                torch.stack(
                    (
                        counts,
                        counts.new_full((experts,), 2 * n),
                        counts.new_full((experts,), k),
                    ),
                    dim=1,
                ).int(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                sizes2,
                torch.stack(
                    (
                        counts,
                        counts.new_full((experts,), k),
                        counts.new_full((experts,), n),
                    ),
                    dim=1,
                ).int(),
                rtol=0,
                atol=0,
            )
            live = int(expected_offsets[-1])
            torch.testing.assert_close(
                c_map[live_mask].sort().values,
                torch.arange(live, device="cuda", dtype=torch.int32),
                rtol=0,
                atol=0,
            )
            self.assertTrue(bool((c_map[~live_mask] == -1).all()))
            routes = torch.arange(ids.numel(), device="cuda")[live_mask]
            torch.testing.assert_close(
                a_map[c_map[live_mask].long()], (routes // topk).int(), rtol=0, atol=0
            )
            self.assertTrue(bool((a_map[live:] == 0).all()))
            for expert in range(experts):
                destinations = c_map[flat_ids == expert]
                self.assertTrue(bool((destinations >= offsets[expert]).all()))
                self.assertTrue(bool((destinations < offsets[expert + 1]).all()))
            return counts

        for blocks in (None, block_offsets):
            with self.subTest(blockscale=blocks is not None):
                prepare(blocks)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    prepare(blocks)
                for state in (*states, states[0]):
                    ids.copy_(state)
                    a_map.fill_(-999)
                    c_map.fill_(-999)
                    offsets.fill_(-999)
                    sizes1.fill_(-999)
                    sizes2.fill_(-999)
                    if blocks is not None:
                        blocks.fill_(-999)
                    graph.replay()
                    counts = check()
                    if blocks is not None:
                        expected = torch.cat(
                            (
                                counts.new_zeros(1),
                                (((counts + 127) // 128) * 128).cumsum(0),
                            )
                        )
                        torch.testing.assert_close(
                            blocks, expected.int(), rtol=0, atol=0
                        )

        for shape in ((0, topk), (tokens, 0)):
            with self.subTest(shape=shape):
                empty_ids = torch.empty(shape, device="cuda", dtype=torch.int32)
                empty_map = torch.empty((0,), device="cuda", dtype=torch.int32)
                cutlass_prepare_moe_input(
                    empty_ids,
                    offsets,
                    sizes1,
                    sizes2,
                    empty_map,
                    empty_map,
                    experts,
                    n,
                    k,
                    block_offsets,
                )
                self.assertEqual(int(offsets.count_nonzero()), 0)
                self.assertEqual(int(block_offsets.count_nonzero()), 0)
                self.assertEqual(int(sizes1[:, 0].count_nonzero()), 0)
                self.assertEqual(int(sizes2[:, 0].count_nonzero()), 0)

    def test_prepare_rejects_noncontiguous_metadata(self):
        """Reject layouts whose physical order differs from the route-map contract."""
        ids = torch.tensor([[0, 1], [1, -1], [2, 0]], device="cuda", dtype=torch.int32)
        offsets = torch.empty(4, device="cuda", dtype=torch.int32)
        sizes1 = torch.empty((3, 3), device="cuda", dtype=torch.int32)
        sizes2 = torch.empty_like(sizes1)
        a_map = torch.empty(ids.numel(), device="cuda", dtype=torch.int32)
        c_map = torch.empty_like(a_map)
        block_offsets = torch.empty_like(offsets)
        args = [ids, offsets, sizes1, sizes2, a_map, c_map, 3, 96, 160, block_offsets]
        for index in (0, 1, 2, 3, 4, 5, 9):
            value = args[index]
            strided = torch.empty((*value.shape, 2), device="cuda", dtype=torch.int32)[
                ..., 0
            ]
            strided.copy_(value)
            with self.subTest(argument=index):
                invalid_args = args.copy()
                invalid_args[index] = strided
                with self.assertRaisesRegex((RuntimeError, ValueError), "contiguous"):
                    cutlass_prepare_moe_input(*invalid_args)

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
        prepare = cutlass_moe.cutlass_prepare_moe_input

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
            patch.object(cutlass_moe, "cutlass_prepare_moe_input", checked_prepare),
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

        empty_result = run(inputs[:0], ids[:0], weights[:0])
        self.assertEqual(empty_result.shape, (0, hidden))
        no_route_result = run(inputs, ids[:, :0], weights[:, :0])
        self.assertTrue(bool(torch.isfinite(no_route_result).all()))
        self.assertEqual(int(torch.count_nonzero(no_route_result)), 0)


if __name__ == "__main__":
    unittest.main()
