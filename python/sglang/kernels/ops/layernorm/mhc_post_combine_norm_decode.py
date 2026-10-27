"""Decode/verify mHC post + next combine + RMSNorm, optionally emitting MXFP8.

One launch replaces ``mhc_post_split_h`` + ``hc_combine_norm`` (+ ``mxfp8_quantize``).
The post matches ``mhc_post_split_h`` bitwise; the MXFP8 pair uses FlashInfer's
128x4 scale swizzle with the UE8M0 conversion of ``mxfp8_epilogue``.
"""

from typing import Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

MAX_ROWS = 1024


@cache_once
def _module(threads: int):
    return load_jit(
        "mhc_post_combine_norm_decode",
        str(threads),
        cuda_files=["deepseek_v4/mhc_post_combine_norm_decode.cuh"],
        cuda_wrappers=[("run", f"MhcPostCombineNormDecode<{threads}>::run")],
    )


@cache_once
def _rows_per_wave(device_index: int, threads: int) -> int:
    props = torch.cuda.get_device_properties(device_index)
    return props.multi_processor_count * (
        props.max_threads_per_multi_processor // threads
    )


def _pick_threads(m: int, device: torch.device) -> int:
    for threads in (640, 320):
        if m <= _rows_per_wave(device.index, threads):
            return threads
    return 160


@register_custom_op(mutates_args=["updated", "normalized", "q", "sf"])
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
) -> None:
    _module(threads).run(
        x, residual, post, comb, pre, weight, updated, normalized, q, sf, eps, quant
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
    threads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(updated_residual, normed_rows, q, sf)``; ``q``/``sf`` are None
    unless ``quant``."""
    m = x.shape[0]
    if threads is None:
        threads = _pick_threads(m, x.device)
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
        threads,
    )
    if not quant:
        return updated, normalized, None, None
    return updated, normalized, q, sf
