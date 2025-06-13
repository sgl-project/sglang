from typing import Tuple, List

import deep_gemm.utils.layout
import torch
from tqdm import trange

from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant



def _transform_scale(sf, mn: int):
    # NOTE copy and modified from DeepGEMM
    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = deep_gemm.utils.layout.get_col_major_tma_aligned_packed_tensor(sf)
    return sf


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    # TODO: stronger tests
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


# COPIED FROM DeepGEMM
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


# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# COPIED FROM DeepGEMM
def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y
