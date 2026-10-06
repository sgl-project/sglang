# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import can_use_fp8_rowwise, fp8_rowwise
from sglang.test.ci.ci_register import register_cuda_ci

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
