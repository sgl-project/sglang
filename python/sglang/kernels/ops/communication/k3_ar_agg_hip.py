"""K3 ROCm TP8: a TP collective fused with the attention-residual aggregation
that consumes it, in one launch (see ``jit/csrc/kimi_k3/comm/ar_agg_hip.cuh``).

``ar_agg``  o_proj all-reduce (AITER 2-stage summation order) + the MLP-side
            aggregation.
``ag_agg``  MoE up_proj column all-gather + 3-way add (AITER
            all_gather_lastdim_add math) + the next layer's attention-side
            aggregation and bank snapshot.

The collective's row and the bank snapshot are bit-identical to the unfused
path; the normalized output matches up to rare 1-ulp bf16 flips.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

WORLD_SIZE = 8
MAX_T = 80
MAX_NVB = 8
THREADS = 256
REGION_BYTES = 1 << 20

_PARITY = [0]


def _split(hidden_size: int) -> int:
    """Consumer blocks per token: each owns <= THREADS 8-wide column vectors."""
    return -(-(hidden_size // 8) // THREADS)


def _next_parity() -> int:
    p = _PARITY[0]
    _PARITY[0] = p ^ 1
    return p


def _region_off(ca) -> int:
    # tail of AITER's 2 x max_size tmp area (its kernels use the head)
    assert ca.max_size >= 32 << 20, "ar_agg needs AITER max_size >= 32 MiB"
    return 2 * ca.max_size - REGION_BYTES


@cache_once
def _module():
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "k3_ar_agg_hip",
        cuda_files=["kimi_k3/comm/ar_agg_hip.cuh"],
        cuda_wrappers=[("run", "k3_ar_agg::ArAgg::run")],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def supported(num_tokens: int, hidden_size: int, nvb: int) -> bool:
    return (
        1 <= num_tokens <= MAX_T
        and 1 <= nvb <= MAX_NVB
        and hidden_size % (8 * WORLD_SIZE) == 0
        and hidden_size // 8 // WORLD_SIZE <= THREADS
        and _split(hidden_size) * WORLD_SIZE <= THREADS
    )


def usable(ca, num_tokens: int, hidden_size: int, nvb: int) -> bool:
    """Whether the fused kernels can run now on this AITER communicator (TP8,
    not disabled, not in the collective-free graph warm-up pass)."""
    return (
        ca is not None
        and not getattr(ca, "disabled", True)
        and hasattr(ca, "_pool")
        and getattr(ca, "world_size", 0) == WORLD_SIZE
        and getattr(ca, "max_size", 0) >= 32 << 20
        and supported(num_tokens, hidden_size, nvb)
        and not (ca._IS_CAPTURING and not torch.cuda.is_current_stream_capturing())
    )


def _flags(a, c, write_bank, ow, out_fp8, stream_out) -> int:
    return (
        (1 if a is not None else 0)
        | (2 if c is not None else 0)
        | (4 if write_bank else 0)
        | (8 if ow is not None else 0)
        | (16 if out_fp8 is not None else 0)
        | (32 if stream_out is not None else 0)
    )


def ar_agg(
    ca,
    partial: torch.Tensor,
    prefix_sum: Optional[torch.Tensor],
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    nvb: int,
    score_eps: float,
    out_eps: float,
    *,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    out_fp8: Optional[torch.Tensor] = None,
    write_bank: bool = False,
    stream_out: Optional[torch.Tensor] = None,
) -> None:
    """prefix_out = AR(partial) [+ prefix_sum] (bf16, AITER 2-stage sums);
    out = attn_res_smallm_hip(prefix_out, bank, ...) (normed when ow given).
    Graph capture (inside AITER's capture()) registers ``partial``; eager
    calls stage it through AITER's input pool like AITER's all_reduce."""
    T, H = partial.shape
    assert supported(T, H, nvb), (T, H, nvb)
    assert partial.is_contiguous() and cw.dtype == torch.float32
    if torch.cuda.is_current_stream_capturing() and getattr(
        ca, "enable_register_for_capturing", True
    ):
        assert ca._IS_CAPTURING
        reg_ptr, reg_bytes = 0, 0
    else:
        reg_ptr, reg_bytes = ca._pool["input"].data_ptr, ca._pool["input"].max_size
    _module().run(
        ca._ptr,
        0,
        partial,
        prefix_sum if prefix_sum is not None else partial,
        partial,
        partial,
        prefix_out,
        bank,
        cw,
        ow if ow is not None else partial,
        out,
        out_fp8.view(torch.uint8) if out_fp8 is not None else out,
        stream_out if stream_out is not None else out,
        nvb,
        _flags(prefix_sum, None, write_bank, ow, out_fp8, stream_out),
        float(score_eps),
        float(out_eps),
        reg_ptr,
        reg_bytes,
        _region_off(ca),
        _next_parity(),
        _split(H),
    )


def ag_agg(
    ca,
    y: torch.Tensor,
    add_b: torch.Tensor,
    add_c: Optional[torch.Tensor],
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    nvb: int,
    score_eps: float,
    out_eps: float,
    *,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    out_fp8: Optional[torch.Tensor] = None,
    write_bank: bool = False,
    stream_out: Optional[torch.Tensor] = None,
) -> None:
    """prefix_out = bf16(bf16(all_gather(y, -1) + add_b) [+ add_c]);
    out = attn_res_smallm_hip(prefix_out, bank, ...). Each rank reads only its
    own y (the exchange goes through the private tmp region): no input
    registration, eager or graph."""
    T, ld = y.shape
    H = ld * WORLD_SIZE
    assert supported(T, H, nvb), (T, H, nvb)
    assert y.is_contiguous() and cw.dtype == torch.float32
    _module().run(
        ca._ptr,
        1,
        y,
        add_b,
        add_b,
        add_c if add_c is not None else add_b,
        prefix_out,
        bank,
        cw,
        ow if ow is not None else add_b,
        out,
        out_fp8.view(torch.uint8) if out_fp8 is not None else out,
        stream_out if stream_out is not None else out,
        nvb,
        _flags(None, add_c, write_bank, ow, out_fp8, stream_out),
        float(score_eps),
        float(out_eps),
        0,
        0,
        _region_off(ca),
        _next_parity(),
        _split(H),
    )
