import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import (
    cutlass_fp8_supported,
    input_to_float8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=80, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_FP8_DIFF = 5e-4
TEST_CASES = [
    pytest.param(19, 150, 80, id="misaligned_projection_shape"),
    pytest.param(512, 3072, 4096, id="flux2_added_kv_projection_shape"),
]


def _modelopt_fp8_supported() -> bool:
    return torch.cuda.is_available() and cutlass_fp8_supported()


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _dequantize_fp8_input(qinput: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    return qinput.to(torch.float32) * x_scale.to(torch.float32)


def _dequantize_fp8_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    if weight_scale.ndim == 0 or weight_scale.numel() == 1:
        scale = weight_scale.to(torch.float32)
    else:
        scale = weight_scale.to(torch.float32).reshape(-1, 1).t()
    return weight.to(torch.float32) * scale


def _build_layer(
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> tuple[torch.nn.Module, ModelOptFp8LinearMethod]:
    output_size, input_size = weight_q.shape
    method = ModelOptFp8LinearMethod(
        ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        input_size_per_partition=input_size,
        output_partition_sizes=[output_size],
        input_size=input_size,
        output_size=output_size,
        params_dtype=DTYPE,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device=DEVICE)

    layer.weight.data.copy_(weight_q)
    layer.weight_scale.data.copy_(weight_scale.reshape_as(layer.weight_scale))
    layer.input_scale.data.copy_(input_scale.reshape_as(layer.input_scale))
    method.process_weights_after_loading(layer)
    return layer, method


@pytest.mark.skipif(
    not _modelopt_fp8_supported(),
    reason="Diffusion ModelOpt FP8 scaled mm correctness requires CUDA FP8 support",
)
@pytest.mark.parametrize("m,n,k", TEST_CASES)
def test_checkpoint_processing(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260410 + m + n + k)

    weight = torch.randn((n, k), device=DEVICE, dtype=DTYPE, generator=generator)
    weight_q, weight_scale = input_to_float8(weight)
    input_scale = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)

    layer, _ = _build_layer(weight_q, weight_scale, input_scale)

    assert tuple(layer.weight.shape) == (k, n)
    assert tuple(layer.weight.stride()) == (1, k)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.input_scale.ndim == 0
    assert tuple(layer.weight_scale.shape) == (n, 1)

    expected_weight = weight_q.t().to(torch.float32) * weight_scale.to(torch.float32)
    actual_weight = _dequantize_fp8_weight(layer.weight, layer.weight_scale)
    torch.testing.assert_close(actual_weight, expected_weight, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not _modelopt_fp8_supported(),
    reason="Diffusion ModelOpt FP8 scaled mm correctness requires CUDA FP8 support",
)
@pytest.mark.parametrize("m,n,k", TEST_CASES)
def test_shape_correctness(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260410 + m + n + k)

    x = torch.randn((m, k), device=DEVICE, dtype=DTYPE, generator=generator)
    weight = torch.randn((n, k), device=DEVICE, dtype=DTYPE, generator=generator)
    weight_q, weight_scale = input_to_float8(weight)
    _, input_scale = input_to_float8(x)

    layer, method = _build_layer(weight_q, weight_scale, input_scale)

    qinput, x_scale = static_quant_fp8(
        x.contiguous(),
        layer.input_scale,
        repeat_scale=method.cutlass_fp8_supported,
    )
    expected = torch.matmul(
        _dequantize_fp8_input(qinput, x_scale),
        _dequantize_fp8_weight(layer.weight, layer.weight_scale),
    )

    actual = method.apply(layer, x)
    diff = _calc_diff(actual, expected.to(dtype=DTYPE))
    assert diff < MAX_FP8_DIFF, f"{m=}, {n=}, {k=}, {diff=:.6f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
