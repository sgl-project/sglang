from typing import Optional

import torch

from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported

CHUNK_SIZE = 64


def _get_aiter_chunk_kda():
    try:
        from aiter.ops.triton.attention.chunk_kda import chunk_kda

        return chunk_kda
    except (ImportError, ModuleNotFoundError):
        return None


def can_use_aiter_chunk_kda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float],
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    output_intermediate_states: bool,
    beta_is_raw: bool,
) -> bool:
    if cu_seqlens is None:
        return False
    num_sequences = len(cu_seqlens) - 1
    measured_cell = q.shape[1] == 16384 and (
        (q.shape[-2] == 16 and num_sequences == 3)
        or (q.shape[-2] == 8 and num_sequences in (2, 3, 4))
    )
    return (
        envs.SGLANG_OPT_KDA_FUSED_PREFILL.get()
        and is_gfx95_supported()
        and _get_aiter_chunk_kda() is not None
        and measured_cell
        and use_qk_l2norm_in_kernel
        and not output_intermediate_states
        and beta_is_raw
        and A_log is not None
        and A_log.is_contiguous()
        and dt_bias is not None
        and dt_bias.is_contiguous()
        and lower_bound is not None
        and -5.5 <= lower_bound < 0
        and initial_state is not None
        and initial_state.dtype in (torch.bfloat16, torch.float32)
        and initial_state_indices is not None
        and initial_state_indices.dtype in (torch.int32, torch.int64)
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and q.shape[0] == 1
        and q.shape[-2] in (8, 16)
        and q.shape[-1] == 128
        and k.shape == q.shape
        and g.shape == q.shape
        and v.shape == q.shape
        and q.stride()[2:] == (128, 1)
        and k.stride()[2:] == (128, 1)
        and v.stride()[2:] == (128, 1)
        and g.stride()[2:] == (128, 1)
        and beta.shape == q.shape[:-1]
        and beta.stride(-1) == 1
        and A_log.numel() == q.shape[-2]
        and dt_bias.numel() == q.shape[-2] * 128
    )


def aiter_chunk_kda_fwd(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    chunk_kda = _get_aiter_chunk_kda()
    assert chunk_kda is not None
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
    output, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        out=v,
        state_cache=initial_state,
        state_indices=initial_state_indices,
        has_initial_state=initial_state_indices >= 0,
    )
    assert final_state is None
    return output
