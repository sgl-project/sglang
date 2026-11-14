"""Decode/verify mHC post + next combine + RMSNorm, optionally emitting MXFP8.

One launch replaces ``mhc_post_split_h`` + ``hc_combine_norm`` (+ ``mxfp8_quantize``).
The post matches ``mhc_post_split_h`` bitwise; the MXFP8 pair uses FlashInfer's
128x4 scale swizzle with the UE8M0 conversion of ``mxfp8_epilogue``.
"""

from typing import Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils import get_device_core_count
from sglang.srt.utils.custom_op import register_custom_op

MAX_ROWS = 1024
SPLIT4_MAX_ROWS = 128
SPLIT2_MAX_ROWS = 640


@cache_once
def _module(threads: int, split: int = 1):
    return load_jit(
        "mhc_post_combine_norm_decode",
        str(threads),
        str(split),
        cuda_files=["deepseek_v4/mhc_post_combine_norm_decode.cuh"],
        cuda_wrappers=[
            ("run", f"MhcPostCombineNormDecode<{threads}, {split}>::run"),
            (
                "max_active_ctas",
                f"MhcPostCombineNormDecode<{threads}, {split}>::max_active_ctas",
            ),
        ],
    )


_num_sms = cache_once(get_device_core_count)


@cache_once
def _rows_per_wave(device_index: int, threads: int, quant: bool) -> int:
    with torch.cuda.device(device_index):
        ctas = int(_module(threads).max_active_ctas(quant))
    return _num_sms(device_index) * ctas


def _pick_config(m: int, device: torch.device, quant: bool) -> Tuple[int, int]:
    """(threads, CTAs per row): split rows over a cluster while the grid is small,
    else the thread count that needs the fewest waves."""
    if m <= SPLIT4_MAX_ROWS:
        return (128, 5) if quant else (160, 4)
    if _num_sms(device.index) < m <= SPLIT2_MAX_ROWS:
        return 160, 2
    threads = min(
        (640, 320, 160),
        key=lambda threads: -(-m // _rows_per_wave(device.index, threads, quant)),
    )
    return threads, 1


@register_custom_op(
    mutates_args=[
        "updated",
        "normalized",
        "q",
        "sf",
        "moe_x",
        "moe_sf",
        "moe_shared_sf",
    ]
)
def _mhc_post_combine_norm_decode(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    updated: torch.Tensor,
    normalized: torch.Tensor,
    q: torch.Tensor,
    sf: torch.Tensor,
    eps: float,
    quant: bool,
    threads: int,
    split: int,
    moe_x: Optional[torch.Tensor] = None,
    moe_sf: Optional[torch.Tensor] = None,
    moe_shared_sf: Optional[torch.Tensor] = None,
    moe_shared_block_m: int = 0,
) -> None:
    _module(threads, split).run(
        x,
        residual,
        post,
        comb,
        pre,
        weight,
        updated,
        normalized,
        q,
        sf,
        eps,
        quant,
        moe_x,
        moe_sf,
        moe_shared_sf,
        moe_shared_block_m,
    )


def mhc_post_combine_norm_decode(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    quant: bool = False,
    moe_input: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(updated_residual, normed_rows, q, sf)``; ``q``/``sf`` are None
    unless ``quant``. ``moe_input`` is ``MegaMoeInput.views()``: the normed rows
    are also quantized into the MegaMoE buffer, as mega_moe_pre_dispatch would."""
    m = x.shape[0]
    updated, normalized = torch.empty_like(residual), torch.empty_like(x)
    if quant:
        q = torch.empty((m, 5120), dtype=torch.float8_e4m3fn, device=x.device)
        sf = torch.empty(
            (m + 127) // 128 * 128 * 160, dtype=torch.uint8, device=x.device
        )
    else:
        q = sf = x.new_empty(0)
    _mhc_post_combine_norm_decode(
        x,
        residual,
        post,
        comb,
        pre,
        weight,
        updated,
        normalized,
        q,
        sf,
        eps,
        quant,
        *_pick_config(m, x.device, quant),
        *(() if moe_input is None else moe_input),
    )
    if not quant:
        return updated, normalized, None, None
    return updated, normalized, q, sf
