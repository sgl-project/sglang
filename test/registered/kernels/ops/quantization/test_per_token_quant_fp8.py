import sys

import pytest
import torch
from sgl_kernel import sgl_per_token_quant_fp8 as aot_per_token_quant_fp8

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.kernels.ops.quantization.per_token_quant_fp8 import per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="nightly", runner_config="1-gpu-large")


def _run_impl(input: torch.Tensor, *, use_jit: bool):
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = torch.empty((input.shape[0], 1), dtype=torch.float32, device="cuda")
    if use_jit:
        per_token_quant_fp8(input, output, scale)
    else:
        aot_per_token_quant_fp8(input, output, scale)
    return output, scale


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor):
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _warp_dispatch_num_tokens() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count * 16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dispatch", ["cta", "warp"])
@pytest.mark.parametrize("hidden_dim", [1076, 1368])
def test_per_token_quant_fp8_is_bit_exact(dtype, dispatch, hidden_dim):
    """The JIT migration must preserve every output and scale bit from AOT."""
    num_tokens = 39 if dispatch == "cta" else _warp_dispatch_num_tokens()
    input = torch.rand((num_tokens, hidden_dim), dtype=dtype, device="cuda")

    actual_output, actual_scale = _run_impl(input, use_jit=True)
    expected_output, expected_scale = _run_impl(input, use_jit=False)

    _assert_bitwise_equal(actual_scale, expected_scale)
    _assert_bitwise_equal(actual_output, expected_output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_per_token_quant_fp8_zero_rows_are_bit_exact(dtype, dispatch):
    """Zero-scale behavior differs by legacy dispatch and must remain unchanged."""
    num_tokens = 1 if dispatch == "cta" else _warp_dispatch_num_tokens()
    input = torch.zeros((num_tokens, 512), dtype=dtype, device="cuda")

    actual_output, actual_scale = _run_impl(input, use_jit=True)
    expected_output, expected_scale = _run_impl(input, use_jit=False)

    _assert_bitwise_equal(actual_scale, expected_scale)
    _assert_bitwise_equal(actual_output, expected_output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_per_token_quant_fp8_midpoints_are_bit_exact(dtype, dispatch):
    """FP8 rounding ties must select the same representable value as AOT."""
    num_tokens = 1 if dispatch == "cta" else _warp_dispatch_num_tokens()
    midpoint_values = torch.tensor(
        [448.0, 1.0625, 1.1875, 1.375, -1.0625, -1.1875, -1.375],
        dtype=dtype,
        device="cuda",
    )
    input = midpoint_values.repeat(num_tokens, 512 // midpoint_values.numel() + 1)[
        :, :512
    ].contiguous()

    actual_output, actual_scale = _run_impl(input, use_jit=True)
    expected_output, expected_scale = _run_impl(input, use_jit=False)

    _assert_bitwise_equal(actual_scale, expected_scale)
    _assert_bitwise_equal(actual_output, expected_output)


def test_scaled_fp8_quant_accepts_padded_outputs():
    """Dynamic per-token quantization supports the serving padding contract."""
    input = torch.rand((1, 512), dtype=torch.float16, device="cuda")

    output, scale = scaled_fp8_quant(
        input, num_token_padding=17, use_per_token_if_dynamic=True
    )
    expected_output, expected_scale = _run_impl(input, use_jit=False)

    assert output.shape == (17, 512)
    assert scale.shape == (17, 1)
    _assert_bitwise_equal(output[:1], expected_output)
    _assert_bitwise_equal(scale[:1], expected_scale)


def test_per_token_quant_fp8_preserves_padded_tail():
    input = torch.rand((1, 512), dtype=torch.float16, device="cuda")
    output = torch.full((17, 512), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.full((17, 1), 2.0, dtype=torch.float32, device="cuda")

    per_token_quant_fp8(input, output, scale)

    assert torch.all(output[1:].float() == 1.0)
    assert torch.all(scale[1:] == 2.0)


def test_per_token_quant_fp8_rejects_unsupported_dtype():
    input = torch.ones((1, 512), dtype=torch.int32, device="cuda")
    output = torch.empty((1, 512), dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.empty((1, 1), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="Unsupported dtype"):
        per_token_quant_fp8(input, output, scale)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
