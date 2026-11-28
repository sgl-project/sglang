import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit

_NUM_EXPERTS = 128


def _require_sm12x() -> None:
    if not current_arch().is_sm(12):
        raise RuntimeError("inkling2_swiglu_quant_lowm_sm120 requires NVIDIA SM12x")


@cache_once
def _swiglu_quant_module(intermediate_size: int):
    _require_sm12x()
    return load_jit(
        "inkling2_swiglu_quant_lowm_sm120",
        str(intermediate_size),
        cuda_files=["inkling2/moe_nvfp4_quant_lowm_sm120.cuh"],
        cuda_wrappers=[
            (
                "nvfp4_swiglu_quant_bf16_lowm",
                f"inkling2_moe_nvfp4_quant_sm120::nvfp4_swiglu_quant_bf16_lowm<{intermediate_size}>",
            ),
        ],
    )


def nvfp4_swiglu_quant_bf16_lowm(w13_bf16, gammas, offsets, gate_cap, up_cap):
    experts = offsets.numel() - 1
    if experts != _NUM_EXPERTS:
        raise ValueError(
            f"nvfp4_swiglu_quant_bf16_lowm requires {_NUM_EXPERTS} experts, "
            f"got {experts}"
        )
    rows, two_n = w13_bf16.shape
    width = two_n // 2
    scale_rows = ((rows + 127 * experts + 127) // 128) * 128
    values = torch.empty((rows, width // 2), dtype=torch.uint8, device=w13_bf16.device)
    scales = torch.empty(
        (scale_rows, width // 16), dtype=torch.uint8, device=w13_bf16.device
    )
    amax = torch.empty(rows, dtype=torch.float32, device=w13_bf16.device)
    if rows:
        _swiglu_quant_module(width).nvfp4_swiglu_quant_bf16_lowm(
            w13_bf16,
            gammas,
            offsets,
            values,
            scales,
            amax,
            float(gate_cap),
            float(up_cap),
            current_arch().num_sms,
        )
    return values, scales, amax
