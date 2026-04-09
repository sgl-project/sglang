import sys

import flashinfer
import pytest
import torch

from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.layers.quantization.modelopt_quant import pad_nvfp4_weight
from sglang.test.ci.ci_register import register_cuda_ci

# B200-only correctness coverage for diffusion NVFP4 scaled mm.
register_cuda_ci(est_time=15, suite="stage-b-kernel-unit-1-gpu-b200")

DEVICE = "cuda"
DTYPE = torch.bfloat16
BLOCK_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_VALUE_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEEPGEMM_FP4_MAX_DIFF = 0.02
TEST_CASES = [
    pytest.param(19, 150, 80, id="padding_regression"),
    pytest.param(512, 6144, 128, id="flux2_projection_shape"),
]
FLUX2_PROJECTION_SHAPE = (512, 6144, 128)


def _nvfp4_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _make_global_scale(x: torch.Tensor) -> torch.Tensor:
    max_abs = torch.amax(x.abs()).clamp_min_(1e-6)
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max_abs).to(torch.float32)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _swap_fp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return ((packed >> 4) | (packed << 4)).contiguous()


def _fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(FP4_VALUE_LUT, dtype=torch.float32, device=device)


def _unpack_fp4_bytes(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    lut = _fp4_lut(packed.device)

    def _decode(nibbles: torch.Tensor) -> torch.Tensor:
        values = lut[(nibbles & 0x7).to(torch.long)]
        return torch.where((nibbles & 0x8) != 0, -values, values)

    low = _decode(packed & 0x0F)
    high = _decode((packed & 0xF0) >> 4)
    return torch.stack((low, high), dim=-1).reshape(
        packed.shape[0], packed.shape[1] * 2
    )


def _swizzled_to_linear(
    scales_swizzled: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scales_swizzled = scales_swizzled.view(torch.float8_e4m3fn)
    row_tiles = (rows + 128 - 1) // 128
    tile_cols = BLOCK_SIZE * 4
    col_tiles = (cols + tile_cols - 1) // tile_cols
    tmp = scales_swizzled.reshape(1, row_tiles, col_tiles, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    linear = tmp.reshape(row_tiles * 128, col_tiles * tile_cols // BLOCK_SIZE)
    return linear[:rows, : cols // BLOCK_SIZE]


def _dequantize_nvfp4(
    packed: torch.Tensor,
    scales_swizzled: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    unpacked = _unpack_fp4_bytes(packed).reshape(rows, cols // BLOCK_SIZE, BLOCK_SIZE)
    scales_linear = _swizzled_to_linear(scales_swizzled, rows, cols).to(torch.float32)
    return (unpacked * (scales_linear / global_scale).unsqueeze(-1)).reshape(rows, cols)


def _quantize_weight_for_checkpoint(
    weight: torch.Tensor, weight_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp4, weight_scale_linear = flashinfer.fp4_quantize(
        weight,
        weight_global_scale,
        is_sf_swizzled_layout=False,
    )
    if weight_scale_linear.dtype == torch.uint8:
        weight_scale_linear = weight_scale_linear.view(torch.float8_e4m3fn)
    return weight_fp4, weight_scale_linear.contiguous()


def _build_layer(
    weight_fp4: torch.Tensor,
    weight_scale_linear: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> None:
    output_size, input_size_half = weight_fp4.shape
    input_size = input_size_half * 2
    method = ModelOptFp4LinearMethod(
        ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=BLOCK_SIZE)
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=input_size,
        output_partition_sizes=[output_size],
        input_size=input_size,
        output_size=output_size,
        params_dtype=DTYPE,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device=DEVICE)

    checkpoint_weight = _swap_fp4_nibbles(weight_fp4)
    layer.weight.data.copy_(checkpoint_weight)
    layer.input_scale.data.copy_(
        (1.0 / input_global_scale).reshape_as(layer.input_scale)
    )
    layer.weight_scale_2.data.copy_(
        (1.0 / weight_global_scale).reshape_as(layer.weight_scale_2)
    )
    layer.weight_scale.data.copy_(weight_scale_linear)

    method.process_weights_after_loading(layer)

    expected_weight, expected_padding_cols = pad_nvfp4_weight(weight_fp4)
    expected_scale_shape = (
        ((output_size + 128 - 1) // 128) * 128,
        (((input_size // BLOCK_SIZE) + 4 - 1) // 4) * 4,
    )

    assert torch.equal(layer.weight, expected_weight)
    assert layer.weight_scale_interleaved.shape == expected_scale_shape
    assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
    assert layer.weights_padding_cols == expected_padding_cols
    torch.testing.assert_close(
        layer.alpha,
        (1.0 / (input_global_scale * weight_global_scale)).to(torch.float32),
    )
    torch.testing.assert_close(
        layer.input_scale_inv,
        input_global_scale.to(torch.float32),
    )


def _resolve_mode(mode: str):
    if mode == "jit_cutlass":
        return scaled_fp4_quant, cutlass_scaled_fp4_mm, None
    if mode == "flashinfer2":
        return flashinfer.fp4_quantize, flashinfer.mm_fp4, "cudnn"
    raise ValueError(f"Unknown mode: {mode}")


@pytest.mark.skipif(
    not _nvfp4_supported(),
    reason="Diffusion NVFP4 scaled mm correctness requires Blackwell GPUs",
)
@pytest.mark.parametrize("m,n,k", TEST_CASES)
def test_checkpoint_processing(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260404 + m + n + k)

    weight = torch.randn((n, k), device=DEVICE, dtype=DTYPE, generator=generator)
    input_global_scale = torch.tensor(512.0, device=DEVICE, dtype=torch.float32)
    weight_global_scale = _make_global_scale(weight)
    weight_fp4, weight_scale_linear = _quantize_weight_for_checkpoint(
        weight, weight_global_scale
    )

    _build_layer(
        weight_fp4, weight_scale_linear, input_global_scale, weight_global_scale
    )


@pytest.mark.skipif(
    not _nvfp4_supported(),
    reason="Diffusion NVFP4 scaled mm correctness requires Blackwell GPUs",
)
@pytest.mark.parametrize("mode", ["jit_cutlass", "flashinfer2"])
def test_flux2_shape_correctness(mode: str) -> None:
    m, n, k = FLUX2_PROJECTION_SHAPE
    quantize_op, gemm_op, gemm_backend = _resolve_mode(mode)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260404 + m + n + k)

    x = torch.randn((m, k), device=DEVICE, dtype=DTYPE, generator=generator)
    weight = torch.randn((n, k), device=DEVICE, dtype=DTYPE, generator=generator)
    input_global_scale = _make_global_scale(x)
    weight_global_scale = _make_global_scale(weight)
    alpha = (1.0 / (input_global_scale * weight_global_scale)).to(torch.float32)

    x_fp4, x_scale_swizzled = quantize_op(x, input_global_scale)
    weight_fp4, weight_scale_swizzled = quantize_op(weight, weight_global_scale)
    if x_scale_swizzled.dtype == torch.uint8:
        x_scale_swizzled = x_scale_swizzled.view(torch.float8_e4m3fn)
    if weight_scale_swizzled.dtype == torch.uint8:
        weight_scale_swizzled = weight_scale_swizzled.view(torch.float8_e4m3fn)

    expected = torch.matmul(
        _dequantize_nvfp4(x_fp4, x_scale_swizzled, input_global_scale),
        _dequantize_nvfp4(weight_fp4, weight_scale_swizzled, weight_global_scale).t(),
    )

    if gemm_backend is None:
        actual = gemm_op(
            x_fp4,
            weight_fp4,
            x_scale_swizzled,
            weight_scale_swizzled,
            alpha,
            DTYPE,
        )
    else:
        actual = gemm_op(
            x_fp4,
            weight_fp4.t(),
            x_scale_swizzled,
            weight_scale_swizzled.t(),
            alpha,
            DTYPE,
            backend=gemm_backend,
        )

    diff = _calc_diff(actual, expected.to(dtype=DTYPE))
    assert diff < DEEPGEMM_FP4_MAX_DIFF, f"{mode=}, {m=}, {n=}, {k=}, {diff=:.6f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
