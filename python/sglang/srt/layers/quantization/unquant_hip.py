"""gfx950 BF16 GEMM routing under AITER for UnquantizedLinearMethod and the GEMMs that
bypass it (Kimi-K3's, the DCP q-replicate q_b): the shapes hipBLASLt takes from AITER's
tuned table, and AITER's tuned FlyDSL kernel straight into caller storage."""

import functools
from typing import Optional

import torch

from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_gfx95 = _is_hip and is_gfx95_supported()
# (N, K) of the Kimi-K3.1 TP8 dense GEMMs hipBLASLt runs faster than AITER's tuned table at
# 256-512 rows on MI350X (MoE front, KDA in_proj, latent up_proj, o_proj); level at 128.
# Past the measured range (prefill batches) they keep the table.
_HIPBLASLT_MIN_M = 129
_HIPBLASLT_MAX_M = 512
_HIPBLASLT_SHAPES = frozenset({(6016, 7168), (6144, 7168), (7168, 3584), (7168, 1536)})


@functools.lru_cache(maxsize=4096)
def a16w16_is_tuned(m: int, n: int, k: int, has_bias: bool) -> bool:
    """Whether AITER's bf16 GEMM table holds a tuned kernel for this shape."""
    from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config

    return (
        get_GEMM_A16W16_tuned_config(
            m, n, k, has_bias, str(torch.bfloat16), str(torch.bfloat16)
        )
        is not None
    )


@functools.lru_cache(maxsize=4096)
def _a16w16_flydsl_launch(m: int, n: int, k: int) -> Optional[dict]:
    """The FlyDSL tile AITER's bf16 table tuned for this shape, if it picked FlyDSL.

    Where the table picked another library at this M, the FlyDSL tile tuned at the
    nearest M takes it (a tile runs at any M); measured within ~0.2 us at decode M.
    """
    from aiter.ops.flydsl import gemm_kernels
    from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config

    keys = ("block_m", "block_n", "block_k", "stages", "split_k")
    keys += ("m_waves", "n_waves", "k_waves", "group_m", "use_half_tile_interleaved")
    for cand in (m, m + 8, m - 8, 2 * m, m // 2, m + 16, m - 16):
        if cand <= 0:
            continue
        config = get_GEMM_A16W16_tuned_config(
            cand, n, k, False, str(torch.bfloat16), str(torch.bfloat16)
        )
        if config is None or config["libtype"] != "flydsl":
            continue
        p = gemm_kernels.get_flydsl_hgemm_kernel_params(config["kernelName"])
        if p is not None:
            return {key: p[key] for key in keys}
    return None


def _dma_aligned(t: torch.Tensor) -> bool:
    # FlyDSL's gfx950 GEMM loads A and B rows by 16-byte DMA, at any such row stride
    return (
        t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and t.stride(0) * t.element_size() % 16 == 0
    )


def prefer_hipblaslt(m: int, n: int, k: int) -> bool:
    """Whether this bf16 GEMM goes to hipBLASLt (torch.mm) rather than AITER's tuned table."""
    return (
        _is_gfx95
        and _HIPBLASLT_MIN_M <= m <= _HIPBLASLT_MAX_M
        and (n, k) in _HIPBLASLT_SHAPES
    )


def a16w16_into(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> bool:
    """Run AITER's tuned FlyDSL bf16 GEMM for this shape straight into ``out``.

    tgemm.mm cannot write caller storage or an fp32 output, and copies a
    row-strided input, so such GEMMs otherwise fall back to hipBLASLt. ``x``
    may be a row slice of a wider buffer; ``out`` is contiguous bf16 or fp32
    (the tile tuned for bf16 output). Returns False when the tuned kernel is
    not FlyDSL or the operands do not fit it.
    """
    if not (
        _use_aiter
        and _is_gfx95
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and out.dtype in (torch.bfloat16, torch.float32)
        and out.is_contiguous()
        and out.shape == (x.shape[0], weight.shape[0])
        and x.shape[1] == weight.shape[1]
        and x.shape[0] > 0
        and _dma_aligned(x)
        and _dma_aligned(weight)
        and not prefer_hipblaslt(x.shape[0], weight.shape[0], weight.shape[1])
    ):
        return False
    launch = _a16w16_flydsl_launch(x.shape[0], weight.shape[0], weight.shape[1])
    if launch is None:
        return False
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16

    gemm_a16w16(
        x,
        weight.t(),
        out=out,
        user_kwargs=launch,
        stream=torch.cuda.current_stream(x.device),
        layout="nt",
        out_dtype=out.dtype,
    )
    return True


def a16w16_tuned(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    """``x @ weight.T`` through AITER's tuned FlyDSL kernel, else None."""
    out = x.new_empty((x.shape[0], weight.shape[0]))
    return out if a16w16_into(x, weight, out) else None


def a16w16_tgemm(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    """``x @ weight.T`` through AITER's tuned bf16 table (tgemm) when it holds the shape
    and hipBLASLt is not faster there, else None."""
    if not (
        _use_aiter
        and x.dim() == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and a16w16_is_tuned(x.shape[0], weight.shape[0], x.shape[1], False)
        and not prefer_hipblaslt(x.shape[0], weight.shape[0], x.shape[1])
    ):
        return None
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(x, weight, otype=x.dtype)


def a16w16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight)`` for a bf16 weight outside any linear method, e.g. the DCP
    q-replicate q_b (a16w16_tgemm is ~2x F.linear there on MI350X at M=8)."""
    out = a16w16_tgemm(x, weight)
    return torch.nn.functional.linear(x, weight) if out is None else out
