import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit

_NUM_EXPERTS = 128


def _require_sm12x() -> None:
    if not current_arch().is_sm(12):
        raise RuntimeError("inkling2_moe_gemm_lowm_sm120 requires NVIDIA SM12x")


@cache_once
def _gemm_module():
    _require_sm12x()
    return load_jit(
        "inkling2_moe_gemm_lowm_sm120",
        cuda_files=["inkling2/moe_lowm_sm120.cuh"],
        cuda_wrappers=[
            (
                "nvfp4_gemm_lowm_scaled",
                "inkling2_moe_lowm_sm120::nvfp4_gemm_lowm_scaled",
            ),
            ("mxfp8_gemm_lowm", "inkling2_moe_lowm_sm120::mxfp8_gemm_lowm"),
        ],
    )


def nvfp4_expert_matmul_lowm_scaled(
    values, scales, weight, weight_scales, offsets, x_amax, w_amax, w_denom
):
    if offsets.numel() != _NUM_EXPERTS + 1:
        raise ValueError(
            f"nvfp4_expert_matmul_lowm_scaled requires {_NUM_EXPERTS} experts, "
            f"got {offsets.numel() - 1}"
        )
    out = torch.empty(
        (values.shape[0], weight.shape[1]), dtype=torch.bfloat16, device=values.device
    )
    if values.shape[0]:
        _gemm_module().nvfp4_gemm_lowm_scaled(
            values,
            weight,
            scales,
            weight_scales.view(torch.uint8),
            out,
            offsets,
            x_amax,
            w_amax,
            w_denom,
            current_arch().num_sms,
        )
    return out


def mxfp8_expert_matmul_lowm(values, scales_cute_t, weight, weight_scales_raw, offsets):
    if offsets.numel() != _NUM_EXPERTS + 1:
        raise ValueError(
            f"mxfp8_expert_matmul_lowm requires {_NUM_EXPERTS} experts, "
            f"got {offsets.numel() - 1}"
        )
    out = torch.empty(
        (values.shape[0], weight.shape[1]), dtype=torch.bfloat16, device=values.device
    )
    if values.shape[0]:
        scales_cute = scales_cute_t.t().contiguous()
        _gemm_module().mxfp8_gemm_lowm(
            values.view(torch.uint8),
            weight.view(torch.uint8),
            scales_cute,
            weight_scales_raw.view(torch.uint8),
            out,
            offsets,
            current_arch().num_sms,
        )
    return out
