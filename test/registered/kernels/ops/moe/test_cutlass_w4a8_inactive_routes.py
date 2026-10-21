"""W4A8 routing must pack inactive routes after live experts, including at EP1."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

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
