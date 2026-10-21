"""W4A8 routing must pack inactive routes after live experts, including at EP1."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.ep_moe_kernels import (
    prepare_cutlass_w4a8_moe_routes_and_scale,
)
from sglang.srt.layers.moe import cutlass_w4a8_moe as w4a8
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestCutlassW4A8InactiveRoutes(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_route_preparation_excludes_inactive_rows(self):
        """Prepare sentinel IDs and a live-row scale without changing strided inputs."""
        experts, rows, width = 2, 11, 257
        floor = 1e-10 / torch.finfo(torch.float8_e4m3fn).max
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            values = torch.empty((rows, width * 2), device="cuda", dtype=dtype)[:, ::2]
            values.copy_(
                torch.arange(rows * width, device="cuda")
                .reshape(rows, width)
                .remainder(53)
                .sub(26)
                .float()
                / 8
            )
            values[9, -1] = 7.25
            ids = torch.tensor(
                [
                    [5, -1, -3],
                    [2, 0, 2],
                    [1, 2, -1],
                    [-1, -1, 2],
                    [2, 2, 1],
                    [1, 0, 2],
                    [2, -1, 0],
                    [2, 2, 2],
                    [0, 2, 1],
                    [2, 1, 2],
                    [-1, 2, -1],
                ],
                device="cuda",
                dtype=torch.int64 if dtype == torch.float32 else torch.int32,
            ).repeat_interleave(2, dim=1)[:, ::2]
            active = ((ids >= 0) & (ids < experts)).any(dim=-1)
            expected = values[active].float().abs().amax() / 448.0
            scale = torch.full((1,), floor, device="cuda", dtype=torch.float32)

            def prepare(value, route_ids, expected_scale, *, rtol=0):
                source_values, source_ids = value.clone(), route_ids.clone()
                canonical = source_ids.masked_fill(source_ids == -1, experts)
                prepared_ids = torch.full(
                    route_ids.shape, -99, device="cuda", dtype=route_ids.dtype
                )
                scale.fill_(floor)
                prepare_cutlass_w4a8_moe_routes_and_scale(
                    input=value,
                    topk_ids=route_ids,
                    output_topk_ids=prepared_ids,
                    output_s=scale,
                    num_local_experts=experts,
                )
                torch.testing.assert_close(prepared_ids, canonical, rtol=0, atol=0)
                torch.testing.assert_close(route_ids, source_ids, rtol=0, atol=0)
                torch.testing.assert_close(
                    value, source_values, rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    scale,
                    torch.as_tensor(
                        expected_scale, device=scale.device, dtype=scale.dtype
                    ).reshape_as(scale),
                    rtol=rtol,
                    atol=0,
                )

            for poison in (1e4, float("inf"), float("nan")):
                with self.subTest(dtype=dtype, poison=poison):
                    values[~active] = poison
                    prepare(values, ids, expected, rtol=1e-6)
            for case in ("zero", "inactive", "empty", "zero-topk", "zero-width"):
                with self.subTest(dtype=dtype, case=case):
                    values.zero_()
                    if case in ("inactive", "zero-topk"):
                        values.fill_(float("nan"))
                    if case == "inactive":
                        ids.fill_(-1)
                    elif case == "zero-width":
                        ids.fill_(0)
                        ids[0, 0] = -1
                    route_ids = ids
                    if case == "empty":
                        route_ids = ids[:0]
                    elif case == "zero-topk":
                        route_ids = ids[:, :0]
                    value = values
                    if case == "empty":
                        value = values[:0]
                    elif case == "zero-width":
                        value = values[:, :0]
                    prepare(value, route_ids, floor)

    def test_route_preparation_graph_replay_updates_ids_and_scale(self):
        """A replay resets the scale and rewrites IDs as live routes change."""
        experts, rows, width = 2, 13, 257
        floor = 1e-10 / torch.finfo(torch.float8_e4m3fn).max
        values = torch.ones((rows, width), device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros((rows, 2), device="cuda", dtype=torch.int32)
        prepared_ids = torch.full_like(ids, -99)
        scale = torch.full((1,), floor, device="cuda", dtype=torch.float32)

        def run():
            scale.fill_(floor)
            prepare_cutlass_w4a8_moe_routes_and_scale(
                input=values,
                topk_ids=ids,
                output_topk_ids=prepared_ids,
                output_s=scale,
                num_local_experts=experts,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for case, expected in (
            ("live", 2 / 448),
            ("zero", floor),
            ("inactive", floor),
            ("partial", 3 / 448),
        ):
            with self.subTest(case=case):
                values.fill_(2.0 if case == "live" else 0.0)
                ids.fill_(0)
                if case == "inactive":
                    values.fill_(float("nan"))
                    ids.fill_(-1)
                elif case == "partial":
                    ids.fill_(experts)
                    ids[0, 0], ids[0, 1] = -1, -3
                    ids[-1, 1] = 1
                    values.fill_(float("nan"))
                    values[-1].zero_()
                    values[-1, -1] = 3.0
                source_values, source_ids = values.clone(), ids.clone()
                graph.replay()
                torch.testing.assert_close(
                    prepared_ids,
                    source_ids.masked_fill(source_ids == -1, experts),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(ids, source_ids, rtol=0, atol=0)
                torch.testing.assert_close(
                    values, source_values, rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    scale, torch.full_like(scale, expected), rtol=1e-6, atol=0
                )

    def test_ep1_and_ep2_reorder_match_dense_experts(self):
        """Real CUDA permutation/count/finalize kernels must ignore padded routes.

        Only the external CUTLASS GEMMs are replaced with dense expert GEMMs;
        this checks routing, not native INT4 GEMM numerics.
        """
        experts, hidden, intermediate = 2, 128, 128
        ids = torch.tensor(
            [[1, 0], [0, -1], [-1, -1], [1, 0]], device="cuda", dtype=torch.int32
        )
        weights = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2]], device="cuda"
        )
        inputs = (
            torch.arange(4 * hidden, device="cuda")
            .reshape(4, hidden)
            .remainder(7)
            .float()
            - 3
        ) / 16
        inputs = inputs.to(torch.bfloat16)
        eye = torch.eye(hidden, device="cuda")
        dense_w13 = [
            torch.cat((eye * ((e + 1) / 4), eye * ((e + 2) / 4)))
            for e in range(experts)
        ]
        dense_w2 = [eye * ((e + 1) / 2) for e in range(experts)]

        def dense_reference(dynamic):
            scale1 = (
                inputs[(ids >= 0).any(dim=-1)].float().abs().amax() / 448.0
                if dynamic
                else 1.0
            )
            activated = {}
            for token in range(ids.shape[0]):
                for slot in range(ids.shape[1]):
                    expert = int(ids[token, slot])
                    if expert < 0:
                        continue
                    quantized = (inputs[token].float() / scale1).to(torch.float8_e4m3fn)
                    gate_up = F.linear(
                        quantized.float() * scale1, dense_w13[expert]
                    ).to(torch.bfloat16)
                    gate, up = gate_up.float().chunk(2)
                    activated[token, slot] = F.silu(gate) * up
            scale2 = (
                torch.stack(list(activated.values())).abs().amax().clamp_min(1e-10)
                / 448.0
                if dynamic
                else 1.0
            )
            result = torch.zeros_like(inputs, dtype=torch.float32)
            for (token, slot), value in activated.items():
                expert = int(ids[token, slot])
                quantized = (value / scale2).to(torch.float8_e4m3fn)
                down = F.linear(quantized.float() * scale2, dense_w2[expert]).to(
                    torch.bfloat16
                )
                result[token] += down.float() * weights[token, slot]
            return result

        w13_q = torch.zeros(
            (experts, 2 * intermediate, hidden // 2), device="cuda", dtype=torch.int8
        )
        w2_q = torch.zeros(
            (experts, hidden, intermediate // 2), device="cuda", dtype=torch.int8
        )
        scales = torch.ones((experts, 1), device="cuda")
        strides = [
            torch.zeros((experts, 3), device="cuda", dtype=torch.int64)
            for _ in range(8)
        ]
        offsets = torch.empty(experts + 1, device="cuda", dtype=torch.int32)
        sizes1 = torch.empty((experts, 3), device="cuda", dtype=torch.int32)
        sizes2 = torch.empty_like(sizes1)

        def dense_gemm(
            out, x, packed, input_scale, _weight_scale, starts, sizes, *args
        ):
            self.assertTrue(torch.isfinite(input_scale).all().item())
            self.assertGreater(input_scale.item(), 0)
            out.fill_(float("nan"))
            matrices = dense_w13 if packed is w13_q else dense_w2
            for expert in range(experts):
                start, count = int(starts[expert]), int(sizes[expert, 1])
                if count:
                    values = x[start : start + count].float() * input_scale
                    out[start : start + count].copy_(F.linear(values, matrices[expert]))

        def run(value, *, dynamic, route_ids=ids, route_weights=weights):
            return w4a8.cutlass_w4a8_moe(
                value,
                w13_q,
                w2_q,
                scales,
                scales,
                route_weights,
                route_ids,
                *strides,
                offsets,
                sizes1,
                sizes2,
                a1_scale=None if dynamic else torch.ones(1, device="cuda"),
                a2_scale=None if dynamic else torch.ones(1, device="cuda"),
            )

        for ep_size in (1, 2):
            with (
                get_parallel().override(
                    tp_size=ep_size,
                    tp_rank=0,
                    attn_tp_size=ep_size,
                    attn_tp_rank=0,
                    moe_ep_size=ep_size,
                    moe_ep_rank=0,
                    moe_tp_size=1,
                    moe_tp_rank=0,
                ),
                patch.object(w4a8, "cutlass_w4a8_moe_mm", side_effect=dense_gemm),
            ):
                for dynamic in (False, True):
                    reference = dense_reference(dynamic)
                    for poison in (0.0, 1e6, float("inf"), float("nan")):
                        with self.subTest(
                            ep_size=ep_size, dynamic=dynamic, poison=poison
                        ):
                            value = inputs.clone()
                            value[2].fill_(poison)
                            route_weights = weights.masked_fill(ids < 0, float("nan"))
                            actual = run(
                                value, dynamic=dynamic, route_weights=route_weights
                            )
                            torch.testing.assert_close(
                                offsets,
                                torch.tensor(
                                    [0, 3, 5], device="cuda", dtype=torch.int32
                                ),
                            )
                            self.assertTrue(torch.isfinite(actual).all().item())
                            torch.testing.assert_close(
                                actual, reference.to(actual.dtype), rtol=1e-2, atol=1e-3
                            )
                            self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)
                actual = run(
                    torch.zeros_like(inputs),
                    dynamic=True,
                    route_weights=weights.masked_fill(ids < 0, float("nan")),
                )
                self.assertTrue(torch.isfinite(actual).all().item())
                self.assertEqual(torch.count_nonzero(actual).item(), 0)
                inactive = torch.full_like(ids, -1)
                actual = run(
                    torch.full_like(inputs, float("nan")),
                    dynamic=True,
                    route_ids=inactive,
                    route_weights=torch.full_like(weights, float("nan")),
                )
                self.assertTrue(torch.isfinite(actual).all().item())
                self.assertEqual(torch.count_nonzero(actual).item(), 0)
                torch.testing.assert_close(
                    ids,
                    torch.tensor(
                        [[1, 0], [0, -1], [-1, -1], [1, 0]],
                        device="cuda",
                        dtype=torch.int32,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
