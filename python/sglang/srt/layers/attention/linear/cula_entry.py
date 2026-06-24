import torch

try:
    from cula.lightning import (
        lightning_attn_fwd_varlen,
        linear_attention_decode,
        linear_attention_state_update_kvbuffer,
        linear_attention_state_update_kvbuffer_fused,
        linear_attention_verify_kvbuffer,
    )

    CULA_AVAILABLE = True
except ImportError:
    CULA_AVAILABLE = False


def _check_cula():
    if not CULA_AVAILABLE:
        raise ImportError(
            "cuLA is required for linear_backend='cula'. "
            "Install it from https://github.com/inclusionAI/cuLA:\n"
            "  git clone https://github.com/inclusionAI/cuLA.git\n"
            "  pip install -e cuLA --no-build-isolation\n"
            "Requirements: Python 3.12+, CUDA 12.9+, PyTorch 2.9.1+."
        )


def _as_bf16(t: torch.Tensor) -> torch.Tensor:
    """Cast to contiguous bf16. Unused on the live cula path now (the kernels
    compute in fp32/TF32 internally and accept fp32 q/k/v directly — see
    cula_decode/cula_verify); kept for cula_prefill and as a helper.
    """
    if t.dtype != torch.bfloat16:
        return t.to(torch.bfloat16).contiguous()
    return t.contiguous()


def _as_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Ensure contiguous without dtype cast. cuLA's decode/verify kernels compute
    in fp32/TF32 (SMEM is fp32, MMA is TF32) and accept fp32 q/k/v natively, so we
    pass Ling's fp32 q/k/v straight through — no fp32->bf16 cast, which would add 3
    cast kernels/layer/step that dominated the graph-replay cost.
    """
    return t.contiguous()


# Kept as the entry point for a future all-cuLA prefill path. The current
# linear_backend="cula" routes prefill through the seg_la kernel family with
# [v,k] state layout (see lightning_backend.py) so only decode/verify/commit
# use cuLA; this wrapper is therefore not on the live call path today.
def cula_prefill(q, k, v, temporal, cache_indices, cu_seqlens, decay, scale):
    """Prefill via cuLA varlen Lightning Attention.

    Args:
        q, k, v: [total_tokens, H, D] bf16 packed
        temporal: [pool_size, H, V, K] fp32 V-major state pool (per-layer slice)
        cache_indices: [N] int32 indices into pool
        cu_seqlens: [N+1] int32 cumulative seq lens
        decay: [H, 1, 1] fp32 positive slopes
        scale: float softmax scale
    Returns:
        o: [total_tokens, H, D] bf16
    """
    _check_cula()
    q, k, v = _as_bf16(q), _as_bf16(k), _as_bf16(v)
    total_tokens = q.shape[0]
    o, _ = lightning_attn_fwd_varlen(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        decay.view(-1),
        cu_seqlens,
        scale=scale,
        state_pool=temporal,
        initial_state_indices=cache_indices.to(torch.int32),
    )
    return o.squeeze(0)


def cula_decode(q, k, v, temporal, cache_indices, decay, scale, out):
    """Single-token decode via cuLA.

    Args:
        q, k, v: [B, H, D] bf16
        temporal: [pool_size, HV, V, K] fp32 V-major state pool (per-layer slice)
        cache_indices: [B] int32 indices into pool
        decay: [H, 1, 1] fp32 positive slopes
        scale: float softmax scale
        out: [B, H, D] bf16 preallocated output
    Returns:
        out: [B, H, D] bf16
    """
    _check_cula()
    # cuLA decode computes in fp32 (element-wise recurrence); pass q/k/v through
    # in their native dtype (fp32 at runtime) — casting to bf16 would only add
    # cast kernels that the kernel re-converts to fp32 internally.
    q, k, v = _as_contiguous(q), _as_contiguous(k), _as_contiguous(v)
    # Caller must pass a bf16 output buffer; a dtype-dependent re-allocation here
    # would silently swap the buffer baked into a captured CUDA graph.
    assert out.dtype == torch.bfloat16, "cula_decode out must be bf16"
    HEAD_DIM = q.shape[-1]
    V_DIM = v.shape[-1]
    pool_size = temporal.shape[0]
    HV = temporal.shape[1]
    temporal_3d = temporal.view(pool_size * HV, temporal.shape[2], temporal.shape[3])
    linear_attention_decode(
        q,
        k,
        v,
        temporal_3d,
        out,
        softmax_scale=scale,
        stride_q=q.stride(0),
        stride_k=k.stride(0),
        stride_v=v.stride(0),
        stride_s=temporal_3d.stride(0),
        stride_o=out.stride(0),
        s_offsets=cache_indices.to(torch.int32),
        decay_scales=decay.view(-1),
        HEAD_DIM=HEAD_DIM,
        K_SPLIT_DIM=HEAD_DIM,
        V_SPLIT_DIM=V_DIM,
    )
    return out


def cula_verify(
    q, k, v, temporal, cache_indices, decay, scale, T, out, k_buf=None, v_buf=None
):
    """Parallel verify via cuLA KVBuffer.

    Args:
        q, k, v: [B*T, H, D] fp32 packed (uniform T per request)
        temporal: [pool_size, HV, V, K] fp32 V-major state pool (per-layer slice)
        cache_indices: [B] int32 indices into pool
        decay: [H, 1, 1] fp32 positive slopes
        scale: float softmax scale
        T: int draft_token_num
        out: [B*T, HV, V] bf16 preallocated output
        k_buf, v_buf: [pool_size, T, H, K] / [pool_size, T, HV, V] fp32 draft
            buffers. When provided, the kernel writes k/v into them (fused,
            write_kv=True) so the caller does NOT need separate scatter kernels.
    Returns:
        out reshaped to [B*T, HV, V]
    """
    _check_cula()
    # cuLA verify kernel loads q/k into fp32 registers (autovec_copy requires
    # matching bit-width). At runtime Ling feeds fp32 (fp32 rotary) -> no cast.
    # For non-fp32 inputs (test kit bf16), cast to fp32 (defensive, not hot path).
    if q.dtype != torch.float32:
        q = q.to(torch.float32).contiguous()
        k = k.to(torch.float32).contiguous()
        v = v.to(torch.float32).contiguous()
    else:
        q, k, v = _as_contiguous(q), _as_contiguous(k), _as_contiguous(v)
    assert out.dtype == torch.bfloat16, "cula_verify out must be bf16"
    B = cache_indices.shape[0]
    H = q.shape[1]
    K = q.shape[2]
    HV = v.shape[1]
    V = v.shape[2]
    q4 = q.view(B, T, H, K)
    k4 = k.view(B, T, H, K)
    v4 = v.view(B, T, HV, V)
    out4 = out.view(B, T, HV, V)
    linear_attention_verify_kvbuffer(
        q4,
        k4,
        v4,
        temporal,
        out4,
        decay.view(-1),
        cache_indices.to(torch.int32),
        scale,
        T,
        k_buf=k_buf,
        v_buf=v_buf,
    )
    return out


def cula_commit(draft_k, draft_v, temporal, cache_indices, accepted_len, decay, T):
    """Commit accepted state via cuLA KVBuffer state_update.

    Args:
        draft_k: [B, T, H, K] bf16
        draft_v: [B, T, HV, V] bf16
        temporal: [pool_size, HV, V, K] fp32 state pool
        cache_indices: [B] int32 indices into pool
        accepted_len: [B] int32 in [0, T]
        decay: [H, 1, 1] fp32 positive slopes
        T: int draft_token_num
    """
    _check_cula()
    linear_attention_state_update_kvbuffer(
        draft_k,
        draft_v,
        temporal,
        decay.view(-1),
        cache_indices.to(torch.int32),
        accepted_len.to(torch.int32),
        T,
    )


def cula_commit_fused(
    draft_k, draft_v, temporal, cache_indices, accepted_len, decay_all, T
):
    """Layer-fused commit: advance ALL mamba layers' state in one kernel launch.

    Args:
        draft_k: [L, pool_size, T, H, K] fp32
        draft_v: [L, pool_size, T, HV, V] fp32
        temporal: [L, pool_size, HV, V, K] fp32 V-major state pool (written in place)
        cache_indices: [B] int32 indices into pool
        accepted_len: [B] int32 in [0, T]
        decay_all: [L, H] fp32 per-layer positive slopes
        T: int draft_token_num
    """
    _check_cula()
    linear_attention_state_update_kvbuffer_fused(
        draft_k,
        draft_v,
        temporal,
        decay_all,
        cache_indices.to(torch.int32),
        accepted_len.to(torch.int32),
        T,
    )
