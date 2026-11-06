"""DeepSeek-V4.1 MegaAttention ``wo_a`` (``bhr,hdr->bhd``) on FlashInfer's SM107 batched MXFP8 GEMM.

FlashMLA's prefill writes its output scales in FlashInfer's 128x4 layout (``out_sf_flashinfer_layout``)
and ``flashinfer.bmm_mxfp8(backend="cute-dsl")`` runs the head groups in one launch, writing the FP8
result and its 128x4 scales straight into ``wo_b``'s input. The output bytes equal DeepGEMM's
``fp8_einsum`` with a FlashInfer-layout output pair.
"""

import functools
import inspect
from typing import Tuple, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.mxfp8_input import deepgemm_pair_takes_flashinfer
from sglang.srt.utils import get_device_sm


@functools.cache
def _available() -> bool:
    try:
        import flashinfer

        from sglang.kernels.ops.attention.flash_mla_interface import (
            MEGA_PREFILL_FLASHINFER_SF,
        )
    except ImportError:
        return False
    return (
        MEGA_PREFILL_FLASHINFER_SF
        and "out_scale" in inspect.signature(flashinfer.bmm_mxfp8).parameters
        and get_device_sm() == 107
    )


def flashinfer_wo_a_enabled(rows: int, device: torch.device) -> bool:
    """Whether a MegaAttention prefill of ``rows`` tokens should hand FlashInfer-layout scales to wo_a."""
    return (
        envs.SGLANG_OPT_DSV41_FLASHINFER_WO_A.get()
        and device.type == "cuda"
        and rows > 0
        and rows % 128 == 0
        and deepgemm_pair_takes_flashinfer()
        and _available()
    )


def scales_in_flashinfer_layout(o_s: torch.Tensor) -> bool:
    # FlashMLA returns FlashInfer-layout scales as a [s_q, groups, words] view with stride(0) != 1;
    # its default (DeepGEMM) layout is contiguous along s_q.
    return o_s.stride(0) != 1


def wo_a_scale_to_flashinfer(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """DeepGEMM-packed UE8M0 scales of the [groups, d, r] wo_a weight -> FlashInfer 128x4 bytes."""
    groups, d, r = weight.shape
    codes = weight_scale.contiguous().view(torch.uint8).reshape(groups * d, r // 32)
    return (
        codes.reshape(groups * d // 128, 4, 32, r // 128, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(-1)
    )


def flashinfer_wo_a_einsum(
    o_fp8: torch.Tensor,
    o_s: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_flashinfer: torch.Tensor,
    out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """``out`` is a bf16 [rows, groups, d] tensor or an (fp8 [rows, groups * d], FlashInfer 128x4 scale) pair."""
    import flashinfer

    rows, groups, _ = o_fp8.shape
    d = weight.shape[1]
    a = o_fp8.transpose(0, 1)
    b = weight.transpose(1, 2)
    a_scale = torch.as_strided(o_s, (o_s.numel(),), (1,)).view(torch.uint8)
    if isinstance(out, tuple):
        y, y_sf = out
        flashinfer.bmm_mxfp8(
            a,
            b,
            a_scale,
            weight_scale_flashinfer,
            torch.float8_e4m3fn,
            out=y.view(rows, groups, d).transpose(0, 1),
            backend="cute-dsl",
            out_scale=y_sf.view(torch.uint8).view(-1),
        )
    else:
        flashinfer.bmm_mxfp8(
            a,
            b,
            a_scale,
            weight_scale_flashinfer,
            out.dtype,
            out=out.transpose(0, 1),
            backend="cute-dsl",
        )
