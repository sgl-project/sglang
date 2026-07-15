from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.layers import deep_gemm_wrapper
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm90_mxfp8_grouped_gemm() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SM90 MXFP8 grouped GEMM tests")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"SM90 MXFP8 grouped GEMM tests require sm_90, got sm_{major}x")
    if not deep_gemm_wrapper.supports_sm90_mxfp8_fp8_grouped_gemm():
        pytest.skip("deep_gemm does not expose SM90 MXFP8 grouped GEMM APIs")


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _cast_back_from_fp8_1d(
    x: torch.Tensor, sf: torch.Tensor, gran_k: int
) -> torch.Tensor:
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def _e8m0_from_fp32_pow2(sf: torch.Tensor) -> torch.Tensor:
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _e8m0_to_fp32(sf: torch.Tensor) -> torch.Tensor:
    return (sf.to(torch.int32) << 23).view(torch.float32)


def _pack_ue8m0_u8_to_i32(sf: torch.Tensor) -> torch.Tensor:
    padded_k = ((sf.shape[-1] + 3) // 4) * 4
    if padded_k != sf.shape[-1]:
        padded = torch.zeros(
            (*sf.shape[:-1], padded_k), device=sf.device, dtype=torch.uint8
        )
        padded[..., : sf.shape[-1]] = sf
        sf = padded
    sf_i32 = sf.to(torch.int32).reshape(*sf.shape[:-1], sf.shape[-1] // 4, 4)
    return (
        sf_i32[..., 0]
        | (sf_i32[..., 1] << 8)
        | (sf_i32[..., 2] << 16)
        | (sf_i32[..., 3] << 24)
    ).contiguous()


def _unpack_ue8m0_i32_to_u8(sf: torch.Tensor) -> torch.Tensor:
    sf_i32 = sf.to(torch.int32)
    return torch.stack(
        [
            torch.bitwise_and(torch.bitwise_right_shift(sf_i32, shift), 0xFF).to(
                torch.uint8
            )
            for shift in (0, 8, 16, 24)
        ],
        dim=-1,
    ).reshape(*sf.shape[:-1], sf.shape[-1] * 4)


def test_sm90_mxfp8_e8m0_rounding_helpers_match():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SM90 MXFP8 rounding helper test")

    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import _e8m0_fp32_to_u8
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        _cast_to_e8m0_with_rounding_up,
    )

    torch.manual_seed(0)
    random_values = torch.exp2(
        torch.empty((2, 3, 8), device="cuda", dtype=torch.float32).uniform_(-20, 20)
    )
    edge_values = torch.tensor(
        [
            0.0,
            torch.finfo(torch.float32).tiny / 2,
            torch.finfo(torch.float32).tiny,
            1.0,
            1.0 + 2**-24,
            1.0 + 2**-10,
            448.0,
            57344.0,
        ],
        device="cuda",
        dtype=torch.float32,
    ).reshape(1, 1, 8)
    sf = torch.cat([random_values, edge_values.expand(2, 1, 8)], dim=1)

    expected = _e8m0_fp32_to_u8(sf)
    packed = _cast_to_e8m0_with_rounding_up(sf)
    actual = _unpack_ue8m0_i32_to_u8(packed)[..., : sf.shape[-1]]

    assert torch.equal(actual, expected)


def test_ue8m0_scale_layout_isolated_between_sm90_and_sm100():
    _require_sm90_mxfp8_grouped_gemm()

    from sglang.kernels.ops.quantization import fp8_kernel

    sf = torch.ones((17, 8), device="cuda", dtype=torch.float32)
    kwargs = dict(num_groups=None, mn=17, k=1024, group_size=128)

    sm90_scale = fp8_kernel._format_ue8m0_scale_for_deepgemm(sf, **kwargs)
    assert sm90_scale.dtype == torch.int32
    assert sm90_scale.shape == (17, 2)

    with (
        patch.object(fp8_kernel, "_is_sm90_supported", False),
        patch.object(fp8_kernel, "_is_sm100_supported", True),
    ):
        sm100_scale = fp8_kernel._format_ue8m0_scale_for_deepgemm(sf, **kwargs)

    assert sm100_scale.dtype == torch.float32
    assert sm100_scale.shape == sf.shape


def test_sm90_mxfp8_grouped_contiguous_wrapper_accuracy():
    _require_sm90_mxfp8_grouped_gemm()

    from deep_gemm.utils.math import per_token_cast_to_fp8

    groups, m_per_group, n, k = 2, 128, 48, 512
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data, a_sf_fp32 = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=128)
    a = (a_data, _pack_ue8m0_u8_to_i32(_e8m0_from_fp32_pow2(a_sf_fp32)))
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32)
    grouped_layout = grouped_layout.repeat_interleave(m_per_group)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm_wrapper.grouped_gemm_nt_mxfp8_f8f8bf16_contig(
        a,
        (b_data, _e8m0_from_fp32_pow2(b_sf_fp32)),
        d,
        grouped_layout,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )

    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=128)
    ref = torch.empty_like(d)
    for group_id in range(groups):
        start = group_id * m_per_group
        end = start + m_per_group
        b_dequant = _cast_back_from_fp8_1d(
            b_data[group_id], b_sf_fp32[group_id], gran_k=32
        )
        ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)

    assert _calc_diff(d, ref) < 0.03


def test_sm90_mxfp8_grouped_masked_wrapper_accuracy():
    _require_sm90_mxfp8_grouped_gemm()

    from deep_gemm.utils.math import per_token_cast_to_fp8

    groups, max_m, n, k = 2, 32, 48, 512
    masked_m = torch.tensor([7, 19], device="cuda", dtype=torch.int32)
    a_ref = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf_fp32 = torch.empty(
        (groups, max_m, k // 128), device="cuda", dtype=torch.float32
    )
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        a_data[group_id], a_sf_fp32[group_id] = per_token_cast_to_fp8(
            a_ref[group_id], use_ue8m0=True, gran_k=128
        )
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    d = torch.empty((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm_wrapper.grouped_gemm_nt_mxfp8_f8f8bf16_masked(
        (a_data, _pack_ue8m0_u8_to_i32(_e8m0_from_fp32_pow2(a_sf_fp32))),
        (b_data, _e8m0_from_fp32_pow2(b_sf_fp32)),
        d,
        masked_m,
        expected_m=max_m,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )

    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=128)
    ref = torch.zeros_like(d)
    for group_id, valid_m in enumerate(masked_m.tolist()):
        b_dequant = _cast_back_from_fp8_1d(
            b_data[group_id], b_sf_fp32[group_id], gran_k=32
        )
        ref[group_id, :valid_m] = (a_dequant[group_id, :valid_m] @ b_dequant.t()).to(
            torch.bfloat16
        )

    diff = max(
        _calc_diff(d[group_id, :valid_m], ref[group_id, :valid_m])
        for group_id, valid_m in enumerate(masked_m.tolist())
    )
    assert diff < 0.03


@pytest.mark.parametrize("m", [1, 7, 32, 181, 128])
def test_sm90_mxfp8_dense_linear_uses_grouped_kernel_accuracy(m):
    _require_sm90_mxfp8_grouped_gemm()

    from deep_gemm.utils.math import per_token_cast_to_fp8

    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize

    n, k = 48, 128
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w_ref = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    w_data, w_sf_fp32 = per_token_cast_to_fp8(w_ref, use_ue8m0=True, gran_k=32)

    layer = nn.Module()
    layer.weight = nn.Parameter(w_data, requires_grad=False)
    layer.weight_scale_inv = nn.Parameter(
        _e8m0_from_fp32_pow2(w_sf_fp32), requires_grad=False
    )

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[1, 32],
        use_mxfp8=True,
    )
    quant_method = Fp8LinearMethod(config)
    assert quant_method.use_sm90_mxfp8_deepgemm_linear

    out = quant_method.apply(layer, x)

    x_data, x_sf_u8 = mxfp8_group_quantize(x.contiguous())
    x_dequant = _cast_back_from_fp8_1d(x_data, _e8m0_to_fp32(x_sf_u8), gran_k=32)
    w_dequant = _cast_back_from_fp8_1d(w_data, w_sf_fp32, gran_k=32)
    ref = (x_dequant @ w_dequant.t()).to(torch.bfloat16)

    assert _calc_diff(out, ref) < 0.03


@pytest.mark.parametrize("m,n,k", [(65, 256, 7168), (181, 128, 7168)])
def test_sm90_mxfp8_dense_linear_contig_tail_large_k_accuracy(m, n, k):
    _require_sm90_mxfp8_grouped_gemm()

    from deep_gemm.utils.math import per_token_cast_to_fp8

    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize

    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w_ref = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    w_data, w_sf_fp32 = per_token_cast_to_fp8(w_ref, use_ue8m0=True, gran_k=32)

    layer = nn.Module()
    layer.weight = nn.Parameter(w_data, requires_grad=False)
    layer.weight_scale_inv = nn.Parameter(
        _e8m0_from_fp32_pow2(w_sf_fp32), requires_grad=False
    )

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[1, 32],
        use_mxfp8=True,
    )
    quant_method = Fp8LinearMethod(config)
    assert quant_method.use_sm90_mxfp8_deepgemm_linear
    assert m % 128 != 0

    out = quant_method.apply(layer, x)

    x_data, x_sf_u8 = mxfp8_group_quantize(x.contiguous())
    x_dequant = _cast_back_from_fp8_1d(x_data, _e8m0_to_fp32(x_sf_u8), gran_k=32)
    w_dequant = _cast_back_from_fp8_1d(w_data, w_sf_fp32, gran_k=32)
    ref = (x_dequant @ w_dequant.t()).to(torch.bfloat16)

    assert _calc_diff(out, ref) < 0.03


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
