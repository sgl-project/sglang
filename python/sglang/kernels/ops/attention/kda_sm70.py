"""SM70 KDA recurrence.

Fp16 activations, fp32 state, K = V = 128. Covers decode (one token per
sequence), prefill, and chain verify. The bf16 Triton and packed-decode
kernels do not run on Volta.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _kda_sm70_module():
    return load_jit(
        "kda_sm70",
        cuda_files=["attention/kda_sm70.cuh"],
        cuda_wrappers=[("run", "KdaSm70Kernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def _fp16_token(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float16:
        raise ValueError(f"SM70 KDA {name} must be fp16, got {tensor.dtype}")
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"SM70 KDA {name} batch axis must be 1, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"SM70 KDA {name} must be [T, ..., D], got {tuple(tensor.shape)}")
    return _dense_per_token(tensor)


def _dense_per_token(tensor: torch.Tensor) -> torch.Tensor:
    # The kernel takes a token stride, so a slice of the fused projection
    # output needs no copy as long as each token's row is dense.
    if tensor.stride(-1) == 1 and (tensor.ndim < 3 or tensor.stride(-2) == tensor.shape[-1]):
        return tensor
    return tensor.contiguous()


def _int32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        tensor = tensor.to(torch.int32)
    return tensor.contiguous()


def kda_sm70_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = None,
    raw_beta: bool = True,
    commit_state: bool = True,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    intermediate_states: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: int = 0,
) -> torch.Tensor:
    """Run the SM70 KDA recurrence.

    ``q``/``k``/``v`` are ``[1, T, heads, dim]`` or ``[T, heads, dim]`` fp16.
    ``a`` is the raw per-channel gate and ``b`` is the raw beta logit when
    ``raw_beta`` is set. State is fp32 ``[slots, HV, 128, 128]`` and is updated
    in place when ``commit_state`` is set. Returns ``[1, T, HV, 128]`` fp16.
    """
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("kda_sm70_recurrent requires an SM70 CUDA device")

    q = _fp16_token("q", q)
    k = _fp16_token("k", k)
    v = _fp16_token("v", v)
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must cover the same number of tokens")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("SM70 KDA is specialized for head dim 128")

    tokens = q.shape[0]
    q_heads = q.shape[1]
    v_heads = v.shape[1]
    if k.shape[1] != q_heads:
        raise ValueError("k heads must match q heads")
    if a.numel() != tokens * v_heads * 128:
        raise ValueError(
            f"a has {a.numel()} values, expected {tokens}*{v_heads}*128 from shape {tuple(a.shape)}"
        )
    if b.numel() != tokens * v_heads:
        raise ValueError(
            f"b has {b.numel()} values, expected {tokens}*{v_heads} from shape {tuple(b.shape)}"
        )
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("SM70 KDA gate inputs must be fp16")
    a = _dense_per_token(a.reshape(tokens, v_heads, 128))
    b = _dense_per_token(b.reshape(tokens, v_heads))

    while ssm_states.ndim > 4:
        if ssm_states.shape[0] != 1:
            raise ValueError(f"ssm_states must be [slots, HV, V, K], got {tuple(ssm_states.shape)}")
        ssm_states = ssm_states[0]
    if ssm_states.ndim != 4 or ssm_states.dtype != torch.float32:
        raise ValueError("ssm_states must be fp32 [slots, HV, V, K]")
    if tuple(ssm_states.shape[1:]) != (v_heads, 128, 128):
        raise ValueError(f"ssm_states trailing shape must be {(v_heads, 128, 128)}, got {tuple(ssm_states.shape)}")

    A_log = A_log.reshape(-1).contiguous()
    if A_log.numel() != v_heads or A_log.dtype != torch.float32:
        raise ValueError("A_log must be fp32 with one value per value head")
    dt_bias = dt_bias.reshape(v_heads, 128).contiguous()
    if dt_bias.dtype != torch.float32:
        raise ValueError("dt_bias must be fp32")

    cache_indices = _int32(cache_indices)
    cu_seqlens = _int32(cu_seqlens)
    if cu_seqlens.numel() != cache_indices.numel() + 1:
        raise ValueError("cu_seqlens must have length len(cache_indices) + 1")

    out = torch.empty(tokens, v_heads, 128, dtype=torch.float16, device=q.device)
    track = None
    track_chunk = None
    if track_state is not None:
        if not track_state.is_contiguous():
            raise ValueError("KDA track_state must be contiguous so the snapshot is in place")
        track = track_state
        if track_chunk_idx is None:
            raise ValueError("track_chunk_idx is required with track_state")
        track_chunk = _int32(track_chunk_idx)

    inter = None
    inter_idx = None
    inter_steps = 0
    if intermediate_states is not None:
        if not intermediate_states.is_contiguous():
            raise ValueError("intermediate KDA state must be contiguous")
        if intermediate_states.dtype != torch.float32:
            raise ValueError("intermediate KDA state must be fp32")
        inner = v_heads * 128 * 128
        if intermediate_states.stride(0) % inner != 0:
            raise ValueError("intermediate state slot stride is not a multiple of HV*V*K")
        # Slot pitch in steps. Matches the Triton verify kernel, which reads
        # this from the buffer rather than from the caller's cache_steps.
        inter_steps = int(intermediate_states.stride(0) // inner)
        del cache_steps
        inter = intermediate_states.view(-1)
        if intermediate_state_indices is None:
            raise ValueError("intermediate_state_indices is required")
        inter_idx = _int32(intermediate_state_indices)

    if scale is None:
        scale = 128**-0.5

    _kda_sm70_module().run(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        out,
        track,
        track_chunk,
        inter,
        inter_idx,
        int(inter_steps),
        float(scale),
        float(lower_bound) if lower_bound is not None else 0.0,
        lower_bound is not None,
        bool(raw_beta),
        bool(commit_state),
    )
    return out.view(1, tokens, v_heads, 128)
