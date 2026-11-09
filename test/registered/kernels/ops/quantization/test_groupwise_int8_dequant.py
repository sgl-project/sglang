import sys
from functools import partial

import pytest
import torch

from sglang.kernels.ops.quantization.groupwise_int8_dequant import (
    _dequant_into_validated,
    groupwise_int8_dequant,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _run_impl(weight, scales, *, validated):
    if not validated:
        return groupwise_int8_dequant(weight, scales)
    output = torch.empty_like(weight, dtype=torch.bfloat16)
    _dequant_into_validated(weight, scales, output)
    return output


@pytest.mark.parametrize("validated", [False, True])
@pytest.mark.parametrize(
    "shape", [(3, 128), (7, 2048), (3, 130), (2, 1), (0, 128), (3, 0)]
)
@pytest.mark.parametrize("offset", [0, 1])
def test_group_boundaries_and_alignment(shape, offset, validated):
    """Partial groups restart on each row; contiguous views need not be aligned."""
    rows, width = shape
    count = rows * width
    weight = (torch.arange(count + offset, device="cuda") % 256 - 128).to(torch.int8)
    weight = weight[offset:].reshape(shape)
    scales = torch.linspace(
        0.01003, 0.93009, rows * ((width + 127) // 128), device="cuda"
    ).reshape(rows, (width + 127) // 128)
    expected = (
        weight.float() * scales.repeat_interleave(128, dim=1)[:, :width]
    ).bfloat16()
    actual = _run_impl(weight, scales, validated=validated)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("validated", [False, True])
def test_fp32_multiply_before_bf16_rounding(validated):
    """Early BF16 scale rounding or flushing FP32 subnormals changes output bits."""
    weight = torch.arange(-128, 128, device="cuda", dtype=torch.int16).to(torch.int8)
    weight = weight.repeat(8).reshape(8, 256)
    scales = (
        torch.tensor(
            [0.01003, 0.03009, 1.00390625, 1.01171875, 2**-140, 2**-130, 2**-126, 1e30],
            dtype=torch.float32,
            device="cuda",
        )[:, None]
        .expand(8, 2)
        .contiguous()
    )
    expected = (weight.float() * scales.repeat_interleave(128, dim=1)).bfloat16()
    actual = _run_impl(weight, scales, validated=validated)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "invalid",
    ["dtype", "scales", "layout", "scale_dtype", "scale_layout", "rank", "cpu"],
)
def test_invalid_contract(invalid):
    weight = torch.ones((2, 256), dtype=torch.int8, device="cuda")
    scales = torch.ones((2, 2), device="cuda")
    if invalid == "dtype":
        weight = weight.float()
    elif invalid == "scales":
        scales = scales[:, :1].contiguous()
    elif invalid == "layout":
        weight = weight.t()
    elif invalid == "scale_dtype":
        scales = scales.bfloat16()
    elif invalid == "scale_layout":
        scales = scales.t()
    elif invalid == "rank":
        weight = weight.flatten()
    else:
        weight, scales = weight.cpu(), scales.cpu()
    with pytest.raises((RuntimeError, ValueError)):
        groupwise_int8_dequant(weight, scales)


@pytest.mark.parametrize("validated", [False, True])
def test_compile_and_cuda_graph(validated):
    """The allocating facade must retain its output and current-stream dependency."""
    weight = torch.full((2, 256), 127, dtype=torch.int8, device="cuda")
    scales = torch.full((2, 2), 0.01003, device="cuda")
    run_impl = partial(_run_impl, validated=validated)
    compiled = torch.compile(run_impl, fullgraph=True)
    expected = run_impl(weight, scales)
    assert torch.equal(compiled(weight, scales), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run_impl(weight, scales)
    scales.mul_(2)
    graph.replay()
    assert torch.equal(result, run_impl(weight, scales))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
