# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_fp8_rowwise,
    fp8_rowwise,
    fused_packed_swiglu_fp8_rowwise,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def reference(x, alignment=16):
    x = F.pad(x, (0, 0, 0, -x.shape[0] % alignment)).float()
    s = (x.abs().amax(1) / 448.0).clamp(min=1e-12)
    return (x / s[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn), s


@pytest.mark.parametrize(
    "shape",
    [
        (0, 256),
        (1, 256),
        (17, 33),
        (32, 3072),
        (340, 3072),
        (2720, 3072),
        (3173, 3072),
        (3173, 9216),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rowwise_exact(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    if not can_use_fp8_rowwise(x):
        pytest.skip("SM89+ required")
    if shape[0]:
        x[0].zero_()
    if shape[0] > 3:
        x[1] *= 1e-13
        x[2] *= 1e3
        x[3, ::2] = -0.0
    q, s = fp8_rowwise(x, 16)
    rq, rs = reference(x)
    assert torch.equal(s, rs)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))


def test_linear_and_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.multimodal_gen.runtime.models.dits.flux3 import Flux3Fp8RowwiseLinear

    x = torch.randn((1, 17, 256), device="cuda", dtype=torch.bfloat16)
    if not can_use_fp8_rowwise(x.reshape(17, 256)):
        pytest.skip("SM89+ required")
    w, ws = reference(torch.randn((32, 256), device="cuda", dtype=torch.bfloat16))
    linear = Flux3Fp8RowwiseLinear(w, ws, tuple_output=False)
    q, s = reference(x.reshape(17, 256))
    expected = torch._scaled_mm(
        q, w.T, s[:, None], ws[None, :], out_dtype=torch.bfloat16, use_fast_accum=True
    )[:17].reshape(1, 17, 32)
    actual = linear(x)
    assert torch.equal(actual, expected)
    # Warm up on a side stream before capture, then change input on replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            linear(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear(x)
    x.normal_()
    graph.replay()
    assert torch.equal(captured, linear(x))


def test_predicate():
    assert not can_use_fp8_rowwise(torch.empty(4, 32))


def test_fp32_rounding_boundaries():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Near E4M3 halfway points: an intermediate FP16 cast changes the answer.
    x = torch.tensor(
        [[448.0, -448.0, 68.02094, -68.02094, 8.502618, -25.005346, 200.04277, -0.0]],
        device="cuda",
    )
    if not can_use_fp8_rowwise(x):
        pytest.skip("SM89+ required")
    q, s = fp8_rowwise(x, 16)
    rq, rs = reference(x)
    assert torch.equal(s, rs)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))


class TestPackedSwiGLUFp8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not can_use_fp8_rowwise(
            torch.empty((1, 1), device="cuda", dtype=torch.bfloat16)
        ):
            pytest.skip("NVIDIA SM89+ required")

    def assert_matches_reference(self, packed, alignment=16):
        gate, up = packed.chunk(2, dim=-1)
        values = (F.silu(gate) * up).reshape(-1, gate.shape[-1])
        rq, rs = reference(values, alignment)
        q, scales = fused_packed_swiglu_fp8_rowwise(packed, alignment)
        self.assertTrue(torch.equal(scales, rs))
        self.assertTrue(torch.equal(q.view(torch.uint8), rq.view(torch.uint8)))
        if gate.shape[-1] == 9216 and values.shape[0]:
            from sglang.kernels.ops.diffusion.quantization.swiglu_fp8_rowwise_triton import (
                _swiglu_fp8_rowwise_tiled_kernel,
            )

            tq, ts = torch.empty_like(q), torch.empty_like(scales)
            _swiglu_fp8_rowwise_tiled_kernel[(q.shape[0],)](
                packed,
                tq,
                ts,
                values.shape[0],
                9216,
                packed.stride(1),
                1024,
                num_warps=16,
                enable_fp_fusion=False,
            )
            self.assertTrue(torch.equal(ts, rs))
            self.assertTrue(torch.equal(tq.view(torch.uint8), rq.view(torch.uint8)))
        return q, scales

    def test_strided_rounding_and_padding(self):
        """Guard row-strided projection slices, BF16 rounds and zero padding."""
        torch.manual_seed(42)
        for batch, rows, hidden in [
            (1, 0, 33),
            (2, 17, 257),
            (1, 32, 9216),
            (1, 340, 9216),
            (2, 170, 9216),
            (1, 3173, 9216),
        ]:
            with self.subTest(batch=batch, rows=rows, hidden=hidden):
                storage = torch.randn(
                    (batch, rows, 3 * hidden), device="cuda", dtype=torch.bfloat16
                )
                packed = storage[..., hidden:]
                if rows:
                    packed[:, 0].zero_()
                if rows > 1:
                    packed[:, 1] *= 1e-13
                self.assert_matches_reference(packed)

    def test_bf16_gate_values(self):
        """Keep SiLU/multiply rounding across every finite BF16 gate encoding."""
        bits = torch.arange(65536, device="cuda", dtype=torch.int32)
        gate = bits.to(torch.int16).view(torch.bfloat16)
        finite_gate = torch.where(torch.isfinite(gate), gate, 0)
        gate = torch.zeros(8 * 9216, device="cuda", dtype=torch.bfloat16)
        gate[:65536] = finite_gate
        gate = gate.reshape(1, 8, 9216)
        up = (
            torch.tensor([0.125, -0.25, 0.5, -1.0], device="cuda", dtype=torch.bfloat16)
            .repeat(18432)
            .reshape_as(gate)
        )
        self.assert_matches_reference(torch.cat((gate, up), dim=-1), 1)

    def test_subnormal_gate_rounding(self):
        """Large up values expose lost BF16 subnormals before quantization."""
        gate = torch.tensor(
            [1e-38, 1e-39, 1e-40, 1e-41, 1e-42], device="cuda", dtype=torch.bfloat16
        )
        gate = gate[:, None].expand(5, 9216).clone()[None]
        self.assert_matches_reference(
            torch.cat((gate, torch.full_like(gate, 1e30)), -1)
        )

    def test_graph_replay_and_gemm(self):
        """Fresh graph inputs preserve quantized operands and downstream GEMM."""
        packed = torch.randn((1, 17, 27648), device="cuda", dtype=torch.bfloat16)[
            ..., 9216:
        ]
        w, ws = reference(torch.randn((32, 9216), device="cuda", dtype=torch.bfloat16))

        def chain():
            q, scales = fused_packed_swiglu_fp8_rowwise(packed)
            return (
                q,
                scales,
                torch._scaled_mm(
                    q,
                    w.T,
                    scales[:, None],
                    ws[None, :],
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                ),
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                chain()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, scales, out = chain()
        packed.normal_()
        graph.replay()
        rq, rs = self.assert_matches_reference(packed)
        self.assertTrue(torch.equal(q.view(torch.uint8), rq.view(torch.uint8)))
        self.assertTrue(torch.equal(scales, rs))
        expected = torch._scaled_mm(
            rq,
            w.T,
            rs[:, None],
            ws[None, :],
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
