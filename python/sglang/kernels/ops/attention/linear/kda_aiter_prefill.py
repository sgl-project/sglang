# SPDX-License-Identifier: Apache-2.0
"""AITER Gluon Kimi Delta Attention chunked prefill for gfx950 (MI350X / MI355X).

``TritonKDAKernel.extend`` tries :func:`extend` on ROCm after the HIP kernel in
``kda_gfx950_prefill``, which only takes an fp32 state pool. It returns None for
anything it does not cover, and the caller keeps the Triton ``chunk_kda`` path.
Needs ``aiter.ops.triton.attention.chunk_kda`` (ROCm/aiter#5866); without it the
Triton path runs. ``SGLANG_OPT_HIP_KDA_AITER_PREFILL=0`` turns it off.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from sglang.kernels.ops.attention.fla.utils import tensor_cache
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_HEAD_DIM = 128


@lru_cache(maxsize=1)
def _aiter_chunk_kda_module():
    if not envs.SGLANG_OPT_HIP_KDA_AITER_PREFILL.get():
        return None
    if not torch.version.hip or not torch.cuda.is_available():
        return None
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        return None
    try:
        from aiter.ops.triton.attention import chunk_kda
    except ImportError as e:
        logger.info("KDA prefill: AITER Gluon chunk_kda unavailable (%s)", e)
        return None
    logger.info("KDA prefill: using AITER Gluon chunk_kda")
    return chunk_kda


@tensor_cache
def _chunk_metadata(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Host sync; cached per cu_seqlens tensor so it runs once per forward, not per layer.
    return _aiter_chunk_kda_module().prepare_chunk_kda_metadata(cu_seqlens)


def _dense_heads(x: torch.Tensor) -> bool:
    # aiter chunk_kda needs dense [H, 128] rows at a 16-byte token stride.
    return (
        x.ndim == 4
        and x.shape[0] == 1
        and x.shape[-1] == _HEAD_DIM
        and x.dtype == torch.bfloat16
        and x.stride()[-2:] == (_HEAD_DIM, 1)
        and x.stride(1) % 8 == 0
    )


def _covered(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    beta_is_raw: bool,
    return_intermediate_states: bool,
    **kwargs,
) -> bool:
    num_heads = q.shape[2] if q.ndim == 4 else 0
    return (
        # the kernel writes only the final state per sequence
        not return_intermediate_states
        and kwargs.get("track_state") is None
        # breakable prefill CUDA graphs pass padded chunk tables; the walk syncs
        and kwargs.get("chunk_indices") is None
        # the kernel fuses the safe gate and the beta sigmoid
        and beta_is_raw
        and A_log is not None
        and dt_bias is not None
        and lower_bound is not None
        # the prepare's 2^-G stays finite only down to -5.5
        and -5.5 <= lower_bound < 0
        and query_start_loc.dtype in (torch.int32, torch.int64)
        and q.shape == k.shape == v.shape == g.shape
        and all(_dense_heads(x) for x in (q, k, v, g))
        and beta.shape == (1, q.shape[1], num_heads)
        and beta.stride(-1) == 1
        and A_log.numel() == num_heads
        and dt_bias.numel() == num_heads * _HEAD_DIM
        and ssm_states.ndim == 4
        and ssm_states.shape[1:] == (num_heads, _HEAD_DIM, _HEAD_DIM)
    )


def extend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    beta_is_raw: bool = False,
    return_intermediate_states: bool = False,
    **kwargs,
) -> torch.Tensor | None:
    """``TritonKDAKernel.extend`` on AITER Gluon chunk_kda, or None when not covered.

    Gathers the ``cache_indices`` rows of ``ssm_states`` (bf16 or fp32) into the
    kernel's fp32 initial state and scatters the final state back in place.
    """
    aiter_chunk_kda = _aiter_chunk_kda_module()
    if aiter_chunk_kda is None or not _covered(
        q,
        k,
        v,
        g,
        beta,
        ssm_states=ssm_states,
        query_start_loc=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        return_intermediate_states=return_intermediate_states,
        **kwargs,
    ):
        return None
    # A negative slot is padding: it starts from zeros and leaves the pool unchanged;
    # its write-back lands on slot 0, which the mamba pool never hands out.
    valid = (cache_indices >= 0).view(-1, 1, 1, 1)
    slots = cache_indices.clamp(min=0).long()
    current = ssm_states.index_select(0, slots)
    initial = torch.where(valid, current.float(), 0.0)
    chunk_indices, chunk_offsets = _chunk_metadata(query_start_loc)
    out, final = aiter_chunk_kda.chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log.reshape(-1).float().contiguous(),
        dt_bias.reshape(-1).float().contiguous(),
        float(lower_bound),
        query_start_loc.contiguous(),
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_state=initial,
        output_final_state=True,
    )
    ssm_states.index_copy_(
        0, slots, torch.where(valid, final.to(ssm_states.dtype), current)
    )
    return out
