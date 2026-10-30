"""CUTLASS FP8 maps and reductions skip inactive expert routes."""

import unittest
from collections import defaultdict
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.cutlass_prepare_moe_input import cutlass_prepare_moe_input
from sglang.kernels.ops.moe.shuffle_weighted_sum import (
    _shuffle_weighted_sum_kernel,
    _shuffle_weighted_sum_triton,
    shuffle_weighted_sum,
)
from sglang.srt.layers.moe import cutlass_moe
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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

    def test_prepare_has_no_expert_sized_global_scratch(self):
        """A warmed producer needs no extra storage even when experts exceed routes."""
        experts, tokens, topk = 513, 2, 2
        ids = torch.tensor(
            [[0, experts - 1], [-1, experts]], device="cuda", dtype=torch.int32
        )
        offsets = torch.empty(experts + 1, device="cuda", dtype=torch.int32)
        sizes1 = torch.empty((experts, 3), device="cuda", dtype=torch.int32)
        sizes2 = torch.empty_like(sizes1)
        a_map = torch.empty(tokens * topk, device="cuda", dtype=torch.int32)
        c_map = torch.empty_like(a_map)

        def prepare():
            cutlass_prepare_moe_input(
                ids, offsets, sizes1, sizes2, a_map, c_map, experts, 96, 160
            )

        for _ in range(3):
            prepare()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        prepare()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.max_memory_allocated(), allocated)

    def test_prepare_transposed_shapes_match_native_w4a8(self):
        """W4A8 metadata and complete maps survive live/inactive graph transitions."""
        from sgl_kernel import get_cutlass_w4a8_moe_mm_data

        experts, n, k = 17, 96, 160
        patterns = (
            [[0, 16], [1, 0], [16, 1], [0, 16]],
            [[0, 16], [1, -1], [experts, -7], [16, experts + 3]],
            [[-1, experts], [-7, experts + 1], [-1, -9], [experts, experts]],
        )
        states = [
            torch.tensor(pattern, device="cuda", dtype=torch.int32)
            for pattern in patterns
        ]
        ids = states[0].clone()
        offsets = torch.full((experts + 1,), -999, device="cuda", dtype=torch.int32)
        sizes1 = torch.full((experts, 3), -999, device="cuda", dtype=torch.int32)
        sizes2 = torch.full_like(sizes1, -999)
        a_map = torch.full((ids.numel(),), -999, device="cuda", dtype=torch.int32)
        c_map = torch.full_like(a_map, -999)
        native_offsets = offsets.clone()
        native_sizes1 = sizes1.clone()
        native_sizes2 = sizes2.clone()

        def prepare():
            cutlass_prepare_moe_input(
                ids,
                offsets,
                sizes1,
                sizes2,
                a_map,
                c_map,
                experts,
                n,
                k,
                transpose_problem_sizes=True,
            )

        prepare()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare()
        for state in (*states, states[0]):
            with self.subTest(ids=state.tolist()):
                ids.copy_(state)
                for value in (offsets, sizes1, sizes2, a_map, c_map):
                    value.fill_(-999)
                graph.replay()
                # The installed native producer supplies the integer W4A8 ABI oracle.
                # It does not read or write a_map/c_map, and does not define their order.
                get_cutlass_w4a8_moe_mm_data(
                    ids,
                    native_offsets,
                    native_sizes1,
                    native_sizes2,
                    a_map,
                    c_map,
                    experts,
                    n,
                    k,
                )
                for actual, reference in (
                    (offsets, native_offsets),
                    (sizes1, native_sizes1),
                    (sizes2, native_sizes2),
                ):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                flat = ids.flatten()
                local = (flat >= 0) & (flat < experts)
                live = int(offsets[-1])
                torch.testing.assert_close(
                    c_map[local].sort().values,
                    torch.arange(live, device="cuda", dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(bool((c_map[~local] == -1).all()))
                for expert in range(experts):
                    rows = c_map[flat == expert]
                    self.assertTrue(bool((rows >= offsets[expert]).all()))
                    self.assertTrue(bool((rows < offsets[expert + 1]).all()))
                routes = torch.arange(ids.numel(), device="cuda")[local]
                torch.testing.assert_close(
                    a_map[c_map[local].long()],
                    (routes // ids.shape[1]).int(),
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(bool((a_map[live:] == 0).all()))
                torch.testing.assert_close(ids, state, rtol=0, atol=0)

        def prepare_and_consume():
            prepare()
            return offsets + 1

        compiled = torch.compile(prepare_and_consume, fullgraph=True)
        offsets.fill_(-999)
        sizes1.fill_(-999)
        sizes2.fill_(-999)
        result = compiled()
        torch.testing.assert_close(result, native_offsets + 1, rtol=0, atol=0)
        torch.testing.assert_close(sizes1, native_sizes1, rtol=0, atol=0)
        torch.testing.assert_close(sizes2, native_sizes2, rtol=0, atol=0)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_ids = torch.empty(ids.shape, device="cuda", dtype=torch.int32)
            fake_offsets = torch.empty(experts + 1, device="cuda", dtype=torch.int32)
            fake_sizes1 = torch.empty((experts, 3), device="cuda", dtype=torch.int32)
            fake_sizes2 = torch.empty_like(fake_sizes1)
            fake_a_map = torch.empty(ids.numel(), device="cuda", dtype=torch.int32)
            fake_c_map = torch.empty_like(fake_a_map)
            self.assertIsNone(
                cutlass_prepare_moe_input(
                    fake_ids,
                    fake_offsets,
                    fake_sizes1,
                    fake_sizes2,
                    fake_a_map,
                    fake_c_map,
                    experts,
                    n,
                    k,
                    transpose_problem_sizes=True,
                )
            )

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
                            _shuffle_weighted_sum_triton(
                                inputs, output, mapping, weights
                            )
                            expected = (
                                torch.tensor([3.5, 2, 0], device="cuda", dtype=dtype)
                                .repeat(copies)[:tokens, None]
                                .expand_as(output)
                            )
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                            if specialization_count is None:
                                specialization_count = len(cache)
                                self.assertGreater(specialization_count, 0)
                            self.assertEqual(len(cache), specialization_count)

                            mapping.fill_(-1)
                            inputs.fill_(float("nan"))
                            weights.fill_(float("nan"))
                            output.fill_(float("nan"))
                            _shuffle_weighted_sum_triton(
                                inputs, output, mapping, weights
                            )
                            self.assertTrue(bool(torch.isfinite(output).all()))
                            self.assertEqual(int(torch.count_nonzero(output)), 0)
                            self.assertEqual(len(cache), specialization_count)

    def test_cuda_reduction_matches_triton_for_ragged_strided_views(self):
        """Guard masked gathers and exact slot arithmetic across dtype/layout keys."""
        for dtype in (torch.float16, torch.bfloat16):
            for weight_dtype in (torch.float16, torch.bfloat16, torch.float32):
                for index_dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        dtype=dtype, weights=weight_dtype, mapping=index_dtype
                    ):
                        tokens, rows, width, topk = 5, 6, 257, 3
                        inputs_storage = torch.full(
                            (rows + 2, width + 7),
                            float("nan"),
                            device="cuda",
                            dtype=dtype,
                        )
                        inputs = inputs_storage[1 : rows + 1, 3 : width + 3]
                        inputs[:4].copy_(
                            (
                                torch.arange(4 * width, device="cuda").view(4, width)
                                % 29
                                - 14
                            ).to(dtype)
                            / 16
                        )
                        invalid = -(2**40) if index_dtype == torch.int64 else -999
                        invalid_positive = (
                            2**40 if index_dtype == torch.int64 else 2**31 - 1
                        )
                        mapping_storage = torch.full(
                            (tokens * topk + 4,),
                            invalid,
                            device="cuda",
                            dtype=index_dtype,
                        )
                        mapping = mapping_storage[2:-2].view(tokens, topk)
                        mapping.copy_(
                            torch.tensor(
                                [
                                    [0, 1, 2],
                                    [-1, 3, rows],
                                    [invalid, invalid_positive, -1],
                                    [3, 0, -1],
                                    [1, 2, 0],
                                ],
                                device="cuda",
                                dtype=index_dtype,
                            )
                        )
                        score_storage = torch.full(
                            (tokens * 2 + 1, topk * 3 + 2),
                            float("nan"),
                            device="cuda",
                            dtype=weight_dtype,
                        )
                        weights = score_storage[
                            1 : tokens * 2 + 1 : 2, 1 : topk * 3 + 1 : 3
                        ]
                        weights.copy_(
                            torch.tensor(
                                [
                                    [0.25, 0.5, -0.125],
                                    [float("nan"), 0.75, float("nan")],
                                    [float("nan")] * 3,
                                    [0.5, -0.25, float("nan")],
                                    [0.125, -0.5, 0.25],
                                ],
                                device="cuda",
                                dtype=weight_dtype,
                            )
                        )
                        out_storage = torch.full(
                            (tokens + 2, width + 9), -17, device="cuda", dtype=dtype
                        )
                        ref_storage = out_storage.clone()
                        output = out_storage[1 : tokens + 1, 4 : width + 4]
                        reference = ref_storage[1 : tokens + 1, 4 : width + 4]
                        saved = [
                            value.clone()
                            for value in (
                                inputs_storage,
                                mapping_storage,
                                score_storage,
                            )
                        ]
                        _shuffle_weighted_sum_triton(
                            inputs, reference, mapping, weights
                        )
                        shuffle_weighted_sum(inputs, output, mapping, weights)
                        torch.testing.assert_close(
                            out_storage.view(torch.int16),
                            ref_storage.view(torch.int16),
                            rtol=0,
                            atol=0,
                        )
                        for value, before in zip(
                            (inputs_storage, mapping_storage, score_storage), saved
                        ):
                            torch.testing.assert_close(
                                value, before, rtol=0, atol=0, equal_nan=True
                            )
                        self.assertEqual(int(torch.count_nonzero(output[2])), 0)

    def test_cuda_reduction_preserves_fma_denormals_and_slot_order(self):
        """FMA residuals and BF16 subnormal operands must survive the CUDA port."""
        for dtype, value in ((torch.float16, 1025.0), (torch.bfloat16, 129.0)):
            with self.subTest(dtype=dtype):
                inputs = torch.tensor(
                    [[-value], [value], [65504.0], [-65504.0], [2**-14]],
                    device="cuda",
                    dtype=dtype,
                )
                mapping = torch.tensor(
                    [[0, 1, -1], [2, 3, 4]], device="cuda", dtype=torch.int64
                )
                weights = torch.tensor(
                    [[1.0, 1.0 - 2**-23, float("nan")], [1.0, 1.0, 1.0]],
                    device="cuda",
                    dtype=torch.float32,
                )
                output = torch.empty((2, 1), device="cuda", dtype=dtype)
                reference = torch.empty_like(output)
                _shuffle_weighted_sum_triton(inputs, reference, mapping, weights)
                shuffle_weighted_sum(inputs, output, mapping, weights)
                expected = torch.tensor(
                    [[-value * 2**-23], [2**-14]], device="cuda", dtype=dtype
                )
                torch.testing.assert_close(
                    output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    output.view(torch.int16),
                    reference.view(torch.int16),
                    rtol=0,
                    atol=0,
                )
        inputs = torch.tensor(
            [[2**-133, -(2**-133), -(2**-133)]], device="cuda", dtype=torch.bfloat16
        )
        mapping = torch.tensor([[0, -1]], device="cuda", dtype=torch.int32)
        weights = torch.tensor(
            [[2**120, float("nan")]], device="cuda", dtype=torch.float32
        )
        output = torch.empty((1, 3), device="cuda", dtype=torch.bfloat16)
        shuffle_weighted_sum(inputs, output, mapping, weights)
        expected = torch.tensor(
            [[2**-13, -(2**-13), -(2**-13)]], device="cuda", dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
        weights[0, 0] = 1.0
        shuffle_weighted_sum(inputs, output, mapping, weights)
        torch.testing.assert_close(
            output.view(torch.int16), inputs.view(torch.int16), rtol=0, atol=0
        )
        weights[0, 0] = 2**-40
        shuffle_weighted_sum(inputs, output, mapping, weights)
        # FP32 underflow creates -0 in slot0; the inactive +0 FMA restores +0.
        self.assertTrue(bool((output.view(torch.int16) == 0).all()))

    def test_cuda_reduction_graph_changes_and_empty_routes(self):
        """A current-stream graph must rewrite poisoned output on every map change."""
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = (
                    torch.arange(4 * 513, device="cuda").view(4, 513).to(dtype) / 128
                )
                mapping = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64)
                weights = torch.tensor(
                    [[0.25, 0.75], [0.5, -0.25]], device="cuda", dtype=torch.float32
                )
                output = torch.empty((2, 513), device="cuda", dtype=dtype)
                reference = torch.empty_like(output)
                shuffle_weighted_sum(inputs, output, mapping, weights)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    shuffle_weighted_sum(inputs, output, mapping, weights)
                torch.cuda.current_stream().wait_stream(side)
                for state in ("partial", "zero", "recovery"):
                    inputs.copy_(
                        torch.arange(4 * 513, device="cuda").view(4, 513).to(dtype)
                        / 128
                    )
                    mapping.copy_(
                        torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64)
                    )
                    weights.copy_(
                        torch.tensor([[0.25, 0.75], [0.5, -0.25]], device="cuda")
                    )
                    if state == "partial":
                        mapping[0] = -1
                        weights[0, 0] = float("nan")
                        inputs[0].fill_(float("nan"))
                    elif state == "zero":
                        mapping.fill_(-1)
                        weights.fill_(float("nan"))
                        inputs.fill_(float("nan"))
                    output.fill_(float("nan"))
                    _shuffle_weighted_sum_triton(inputs, reference, mapping, weights)
                    graph.replay()
                    torch.testing.assert_close(
                        output.view(torch.int16),
                        reference.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(bool(torch.isfinite(output).all()))
                for input_rows, topk in ((0, 2), (4, 0)):
                    empty_inputs = torch.empty(
                        (input_rows, 513), device="cuda", dtype=dtype
                    )
                    empty_maps = torch.zeros(
                        (2 * topk,), device="cuda", dtype=torch.int32
                    )
                    empty_weights = torch.full(
                        (2, topk), float("nan"), device="cuda", dtype=torch.float32
                    )
                    output.fill_(float("nan"))
                    shuffle_weighted_sum(
                        empty_inputs, output, empty_maps, empty_weights
                    )
                    self.assertTrue(bool((output.view(torch.int16) == 0).all()))
                empty_output = output[:0]
                shuffle_weighted_sum(inputs, empty_output, mapping[:0], weights[:0])
                shuffle_weighted_sum(inputs[:, :0], output[:, :0], mapping, weights)
        # FP32 input/output retains the original Triton path and contract.
        inputs = torch.tensor([[1.0, 2.0]], device="cuda")
        output = torch.empty_like(inputs)
        shuffle_weighted_sum(
            inputs,
            output,
            torch.tensor([0], device="cuda"),
            torch.tensor([[0.5]], device="cuda"),
        )
        torch.testing.assert_close(output, inputs / 2, rtol=0, atol=0)

    def test_cuda_reduction_preserves_identity_alias_and_compiled_mutation(self):
        """Preserve safe in-place topk1 and expose the mutation to compile/fake."""
        dtype = torch.bfloat16
        inputs = torch.arange(3 * 257, device="cuda").view(3, 257).to(dtype) / 128
        saved = inputs.clone()
        mapping = torch.arange(3, device="cuda", dtype=torch.int32)
        weights = torch.full((3, 1), 0.5, device="cuda")
        with torch.no_grad():
            shuffle_weighted_sum(inputs, inputs, mapping, weights)
        torch.testing.assert_close(inputs, saved / 2, rtol=0, atol=0)
        storage = torch.cat((saved, torch.full_like(saved, float("nan"))))
        with torch.no_grad():
            shuffle_weighted_sum(storage[:3], storage[3:], mapping, weights)
        torch.testing.assert_close(storage[:3], saved, rtol=0, atol=0)
        torch.testing.assert_close(storage[3:], saved / 2, rtol=0, atol=0)

        def reduce_and_consume(values, output, route_map, scores):
            shuffle_weighted_sum(values, output, route_map, scores)
            return output + 1

        compiled = torch.compile(reduce_and_consume, fullgraph=True)
        output = torch.empty_like(inputs)
        reference = torch.empty_like(inputs)
        for active in (True, False, True):
            mapping.copy_(torch.arange(3, device="cuda", dtype=torch.int32))
            weights.fill_(0.5)
            if not active:
                mapping.fill_(-1)
                weights.fill_(float("nan"))
            _shuffle_weighted_sum_triton(inputs, reference, mapping, weights)
            output.fill_(float("nan"))
            with torch.no_grad():
                result = compiled(inputs, output, mapping, weights)
            torch.testing.assert_close(
                output.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0
            )
            torch.testing.assert_close(result, reference + 1, rtol=0, atol=0)
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_values = torch.empty((6, 257), dtype=dtype, device="cuda")
            fake_output = torch.empty((3, 257), dtype=dtype, device="cuda")
            fake_map = torch.empty(6, dtype=torch.int32, device="cuda")
            fake_scores = torch.empty((3, 2), dtype=torch.float32, device="cuda")
            self.assertIsNone(
                shuffle_weighted_sum(fake_values, fake_output, fake_map, fake_scores)
            )

    def _assert_reduction_bits_equal(self, actual, expected):
        torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
        bit_dtype = torch.int32 if expected.dtype == torch.float32 else torch.int16
        finite_or_inf = ~torch.isnan(expected)
        torch.testing.assert_close(
            actual.view(bit_dtype)[finite_or_inf],
            expected.view(bit_dtype)[finite_or_inf],
            rtol=0,
            atol=0,
        )

    def test_reduction_fuses_score_cast_to_input_dtype(self):
        """The optional cast matches the previous explicit Torch conversion chain."""
        width, topk, rows = 17, 2, 5
        for dtype, value in ((torch.float16, 1025.0), (torch.bfloat16, 129.0)):
            for weight_dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(dtype=dtype, weight_dtype=weight_dtype):
                    input_storage = torch.full(
                        (rows, width + 3), float("nan"), device="cuda", dtype=dtype
                    )
                    inputs = input_storage[:, :width]
                    inputs[0].fill_(-value)
                    inputs[1].fill_(value)
                    inputs[2].fill_(1)
                    inputs[3].fill_(2.0**120 if dtype == torch.bfloat16 else 1.0)
                    mapping = torch.tensor(
                        [[0, 1]]
                        + [[2, 2]] * 4
                        + [[2, -1]] * 2
                        + [[3, -1], [2, 2], [2, 2], [-1, rows + 5], [2, 2], [2, -1]],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    tokens = mapping.shape[0]
                    score_storage = torch.full(
                        (tokens * 2 + 1, topk * 3 + 2),
                        float("nan"),
                        device="cuda",
                        dtype=weight_dtype,
                    )
                    weights = score_storage[
                        1 : 2 * tokens + 1 : 2, 1 : 3 * topk + 1 : 3
                    ]
                    weights.copy_(
                        torch.tensor(
                            [
                                [1.0, 1.0 - 2**-23],
                                [1.0 + 2**-11, -1.0],
                                [1.0 + 3 * 2**-11, -1.0],
                                [1.0 + 2**-8, -1.0],
                                [1.0 + 3 * 2**-8, -1.0],
                                [2**-24, float("nan")],
                                [2**-25, float("nan")],
                                [2**-133, float("nan")],
                                [float("inf"), -float("inf")],
                                [float("nan"), 1.0],
                                [float("nan"), float("inf")],
                                [-0.0, 0.0],
                                [0.5, float("nan")],
                            ],
                            device="cuda",
                            dtype=weight_dtype,
                        )
                    )
                    output = torch.empty((tokens, width), device="cuda", dtype=dtype)
                    reference = torch.empty_like(output)
                    triton_output = torch.empty_like(output)
                    saved = [x.clone() for x in (input_storage, mapping, score_storage)]
                    shuffle_weighted_sum(
                        inputs, reference, mapping, weights.to(inputs.dtype)
                    )
                    shuffle_weighted_sum(
                        inputs,
                        output,
                        mapping,
                        weights,
                        cast_weights_to_input_dtype=True,
                    )
                    _shuffle_weighted_sum_triton(
                        inputs,
                        triton_output,
                        mapping,
                        weights,
                        cast_weights_to_input_dtype=True,
                    )
                    self._assert_reduction_bits_equal(output, reference)
                    self._assert_reduction_bits_equal(triton_output, reference)
                    self.assertTrue(bool((output[10].view(torch.int16) == 0).all()))
                    if weight_dtype == torch.float32:
                        uncast = torch.empty_like(output)
                        shuffle_weighted_sum(inputs, uncast, mapping, weights)
                        self.assertTrue(bool((output[0].view(torch.int16) == 0).all()))
                        self.assertTrue(bool((uncast[0] != output[0]).all()))
                        # A wider output must still round scores to the input dtype.
                        wide_output = output.float()
                        wide_reference = reference.float()
                        shuffle_weighted_sum(
                            inputs,
                            wide_reference,
                            mapping,
                            weights.to(inputs.dtype),
                        )
                        shuffle_weighted_sum(
                            inputs,
                            wide_output,
                            mapping,
                            weights,
                            cast_weights_to_input_dtype=True,
                        )
                        self._assert_reduction_bits_equal(wide_output, wide_reference)
                    for tensor, before in zip(
                        (input_storage, mapping, score_storage), saved
                    ):
                        torch.testing.assert_close(
                            tensor, before, rtol=0, atol=0, equal_nan=True
                        )
            for weight_dtype in (torch.float64, torch.int64):
                with self.subTest(dtype=dtype, weight_dtype=weight_dtype):
                    inputs = torch.ones((1, width), device="cuda", dtype=dtype)
                    mapping = torch.tensor([[0, 0]], device="cuda", dtype=torch.int32)
                    weights = torch.tensor(
                        [[257, -256]], device="cuda", dtype=weight_dtype
                    )
                    output = torch.empty_like(inputs)
                    reference = torch.empty_like(inputs)
                    shuffle_weighted_sum(
                        inputs, reference, mapping, weights.to(inputs.dtype)
                    )
                    shuffle_weighted_sum(
                        inputs,
                        output,
                        mapping,
                        weights,
                        cast_weights_to_input_dtype=True,
                    )
                    self._assert_reduction_bits_equal(output, reference)

    def test_reduction_score_cast_survives_graph_and_compile(self):
        """Captured score casts follow live metadata without changing borrowed tensors."""
        width = 17
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                value = 1025.0 if dtype == torch.float16 else 129.0
                initial = torch.tensor(
                    [[-value] * width, [value] * width, [1.0] * width, [2.0] * width],
                    device="cuda",
                    dtype=dtype,
                )
                inputs = initial.clone()
                mapping = torch.tensor(
                    [[0, 1], [2, 3]], device="cuda", dtype=torch.int64
                )
                scores = torch.full((2, 4), float("nan"), device="cuda")
                weights = scores[:, ::2]
                weights.copy_(
                    torch.tensor([[1.0, 1.0 - 2**-23], [0.25, 0.75]], device="cuda")
                )
                output = torch.empty((2, width), device="cuda", dtype=dtype)
                reference = torch.empty_like(output)

                def run(values, result, route_map, route_weights):
                    shuffle_weighted_sum(
                        values,
                        result,
                        route_map,
                        route_weights,
                        cast_weights_to_input_dtype=True,
                    )
                    return result + 1

                run(inputs, output, mapping, weights)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    shuffle_weighted_sum(
                        inputs,
                        output,
                        mapping,
                        weights,
                        cast_weights_to_input_dtype=True,
                    )
                for state in ("partial", "zero", "recovery"):
                    inputs.copy_(initial)
                    mapping.copy_(torch.tensor([[0, 1], [2, 3]], device="cuda"))
                    weights.copy_(
                        torch.tensor([[1.0, 1.0 - 2**-23], [0.25, 0.75]], device="cuda")
                    )
                    if state == "partial":
                        mapping[1, 1] = -1
                        inputs[3].fill_(float("nan"))
                        weights[1, 1] = float("nan")
                    elif state == "zero":
                        mapping.fill_(-1)
                        inputs.fill_(float("nan"))
                        weights.fill_(float("nan"))
                    saved = [x.clone() for x in (inputs, mapping, scores)]
                    shuffle_weighted_sum(
                        inputs, reference, mapping, weights.to(inputs.dtype)
                    )
                    output.fill_(float("nan"))
                    graph.replay()
                    self._assert_reduction_bits_equal(output, reference)
                    for tensor, before in zip((inputs, mapping, scores), saved):
                        torch.testing.assert_close(
                            tensor, before, rtol=0, atol=0, equal_nan=True
                        )
                if dtype == torch.bfloat16:
                    compiled = torch.compile(run, fullgraph=True)
                    with torch.no_grad():
                        result = compiled(inputs, output, mapping, weights)
                    self._assert_reduction_bits_equal(output, reference)
                    torch.testing.assert_close(result, reference + 1, rtol=0, atol=0)
                    from torch._subclasses.fake_tensor import FakeTensorMode

                    with FakeTensorMode():
                        self.assertIsNone(
                            shuffle_weighted_sum(
                                torch.empty((4, width), device="cuda", dtype=dtype),
                                torch.empty((2, width), device="cuda", dtype=dtype),
                                torch.empty((2, 2), device="cuda", dtype=torch.int64),
                                torch.empty((2, 2), device="cuda"),
                                cast_weights_to_input_dtype=True,
                            )
                        )

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

        from sgl_kernel import prepare_moe_input as legacy_prepare_moe_input

        valid_inputs = inputs.nan_to_num()
        valid_ids = ids.clamp_min(0)
        valid_weights = torch.full_like(weights, 1 / ids.size(1))
        # Both producers define every map entry when all routes are valid.
        # Keep the same native entry and GEMM arguments so only map production
        # changes; compare raw output bits rather than a numerical tolerance.
        with patch.object(
            cutlass_moe, "cutlass_prepare_moe_input", legacy_prepare_moe_input
        ):
            legacy_valid = run(valid_inputs, valid_ids, valid_weights)
            legacy_repeat = run(valid_inputs, valid_ids, valid_weights)
        torch.testing.assert_close(
            legacy_valid.view(torch.int16),
            legacy_repeat.view(torch.int16),
            rtol=0,
            atol=0,
            msg="Legacy native provider is nondeterministic; exact producer parity is unresolved",
        )
        actual_valid = run(valid_inputs, valid_ids, valid_weights)
        torch.testing.assert_close(
            actual_valid.view(torch.int16),
            legacy_valid.view(torch.int16),
            rtol=0,
            atol=0,
        )

        reference = run(valid_inputs, valid_ids, weights.nan_to_num())
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
