from typing import Tuple, List

import deep_gemm.utils.layout
import torch
from tqdm import trange

from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant


def requant_weight_ue8m0_inplace(weight, weight_scale_inv, weight_block_size):
    assert isinstance(weight, torch.nn.Parameter)
    assert isinstance(weight_scale_inv, torch.nn.Parameter)
    weight.data, weight_scale_inv.data = _requant_weight_ue8m0(
        weight, weight_scale_inv, weight_block_size
    )


def _requant_weight_ue8m0(
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]

    *_, n, k = weight.shape

    weight_dequant = block_quant_dequant(
        weight,
        weight_scale_inv,
        weight_block_size,
        torch.bfloat16,
    )

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat)

    out_w = out_w_flat.view(weight.shape)
    out_s = out_s_flat.view(weight_scale_inv.shape)

    out_s = _transform_scale(out_s, mn=out_w.shape[-2])

    return out_w, out_s


def _transform_scale(sf, mn: int):
    # NOTE copy and modified from DeepGEMM
    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = deep_gemm.utils.layout.get_col_major_tma_aligned_packed_tensor(sf)
    return sf


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    # TODO: stronger tests
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y
