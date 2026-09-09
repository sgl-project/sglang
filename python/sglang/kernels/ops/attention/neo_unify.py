# SPDX-License-Identifier: Apache-2.0
"""NEO-Unify block-causal prefill and non-causal image denoising.

Q/K/V use [batch, sequence, heads, head_dim]. Batches are unpadded;
Q is the suffix of KV for causal attention. Image boundaries are exclusive
logical KV positions, not rotary positions or generation-expert indicators.
"""

import inspect
from functools import lru_cache

import torch


def build_image_token_end(block_ids: torch.Tensor, prefix_len: int = 0) -> torch.Tensor:
    """Build O(S) metadata once per prefix, outside the decoder-layer loop.

    Consecutive equal block IDs denote a bidirectional image span. Singleton
    IDs denote causal text. Cached prefixes must end at a complete block boundary.
    """
    if block_ids.ndim not in (1, 2) or block_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("block_ids must be a 1D or 2D integer tensor")
    if prefix_len < 0 or prefix_len + block_ids.shape[-1] >= 2**31:
        raise ValueError("prefix and sequence length must fit nonnegative int32")
    if torch.any(block_ids[..., 1:] < block_ids[..., :-1]).item():
        raise ValueError(
            "block_ids must be non-decreasing; disjoint image blocks need distinct IDs"
        )
    size = block_ids.shape[-1]
    if size == 0:
        return torch.empty_like(block_ids, dtype=torch.int32)
    boundary = torch.ones_like(block_ids, dtype=torch.bool)
    boundary[..., :-1] = block_ids[..., :-1] != block_ids[..., 1:]
    positions = torch.arange(1, size + 1, device=block_ids.device, dtype=torch.int32)
    ends = torch.where(boundary, positions + prefix_len, size + prefix_len)
    ends = ends.flip(-1).cummin(-1).values.flip(-1)
    repeated = ~boundary
    repeated[..., 1:] |= ~boundary[..., :-1]
    return torch.where(repeated, ends, 0).contiguous()


@lru_cache(maxsize=1)
def _neo_fa3():
    # Optional build: WANDY666/flash-attention, support_neo at e2077ee6e5.
    # Older image_token_tag builds have different semantics and are not usable.
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        return None
    if "image_token_end" not in inspect.signature(flash_attn_with_kvcache).parameters:
        return None
    return flash_attn_with_kvcache


def resolve_neo_backend(q: torch.Tensor, *, image_aware: bool, backend: str) -> str:
    """Resolve before benchmarking so fallback cannot be reported as FA3."""
    if backend not in ("auto", "torch", "triton", "fa3"):
        raise ValueError(f"Unknown NEO attention backend: {backend}")
    if backend == "torch":
        return backend
    supported = (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (32, 64, 128, 256)
    )
    if not supported:
        if backend == "auto":
            return "torch"
        raise RuntimeError(
            "NEO GPU attention requires CUDA FP16/BF16 and head_dim 32/64/128/256"
        )
    major, _ = torch.cuda.get_device_capability(q.device)
    if backend == "triton":
        return backend
    neo_fa3_available = major in (8, 9) and _neo_fa3() is not None
    if backend == "fa3":
        if image_aware and neo_fa3_available:
            return backend
        if not image_aware and (major == 9 or neo_fa3_available):
            return backend
        raise RuntimeError(
            "NEO image-aware FA3 requires the support_neo build on SM80-SM90; "
            "ordinary FA3 requires Hopper or that build on SM80-SM89"
        )
    if image_aware:
        # Keep SM80-SM89 opt-in until it has target-device performance data.
        fa3_available = major == 9 and neo_fa3_available
    else:
        # Restrict auto selection to Hopper, the initial validation target.
        fa3_available = major == 9
    if fa3_available:
        return "fa3"
    return "triton"


@lru_cache(maxsize=32)
def _fa3_lengths(batch: int, qlen: int, klen: int, device: torch.device):
    # Shape-only immutable metadata, shared across layers and denoising steps.
    return (
        torch.arange(batch + 1, device=device, dtype=torch.int32) * qlen,
        torch.full((batch,), klen, device=device, dtype=torch.int32),
    )


def neo_unify_attention(
    q, k, v, *, image_token_end=None, causal=False, softmax_scale=None, backend="auto"
):
    """Inference-only attention; reads KV without appending or mutating it.

    image_token_end is int32 [S_q] (shared by the batch) or [B, S_q], with
    values in [0, S_k]. Build it with build_image_token_end before the layer
    loop. Arbitrary padding, sliding windows and partial cached image blocks
    are not supported by this interface.
    """
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("Q/K/V must use [B, S, H, D] layout")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K/V batch, KV shape and head dimensions must match")
    if k.shape[2] == 0 or q.shape[2] == 0 or q.shape[2] % k.shape[2]:
        raise ValueError("Query heads must be a positive multiple of KV heads")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ValueError("Q/K/V must have the same device and dtype")
    if not q.is_floating_point():
        raise ValueError("Q/K/V must be floating point")
    batch, qlen, heads, dim = q.shape
    klen = k.shape[1]
    if causal and qlen > klen:
        raise ValueError("Causal Q must be a suffix of KV")
    if image_token_end is not None:
        if not causal:
            raise ValueError("image_token_end requires causal=True")
        if (
            image_token_end.shape not in ((qlen,), (batch, qlen))
            or image_token_end.dtype != torch.int32
            or image_token_end.device != q.device
        ):
            raise ValueError(
                "image_token_end must be int32 [S_q] or [B, S_q] on the Q device"
            )
        image_token_end = image_token_end.expand(batch, qlen)
    resolved = resolve_neo_backend(
        q, image_aware=image_token_end is not None, backend=backend
    )
    if batch == 0 or qlen == 0:
        return torch.empty_like(q)
    if klen == 0:
        raise ValueError("Nonempty Q requires nonempty KV")
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    if resolved == "torch":
        # Deliberately simple FP32 oracle/fallback; no GPU extension imports.
        qh = q.transpose(1, 2).float()
        kh = k.transpose(1, 2).float().repeat_interleave(heads // k.shape[2], 1)
        vh = v.transpose(1, 2).float().repeat_interleave(heads // v.shape[2], 1)
        scores = (qh @ kh.transpose(-1, -2)) * scale
        if causal:
            kp = torch.arange(klen, device=q.device)
            qp = torch.arange(qlen, device=q.device) + klen - qlen
            allow = (kp <= qp[:, None]).expand(batch, qlen, klen)
            if image_token_end is not None:
                allow = allow | (kp < image_token_end[:, :, None])
            scores = scores.masked_fill(~allow[:, None], float("-inf"))
        return (scores.softmax(-1) @ vh).transpose(1, 2).to(q.dtype).contiguous()
    if resolved == "triton":
        from sglang.kernels.ops.attention.neo_unify_triton import (
            neo_unify_attention_triton,
        )

        return neo_unify_attention_triton(q, k, v, image_token_end, causal, scale)
    cu_q, kv_lengths = _fa3_lengths(batch, qlen, klen, q.device)
    if image_token_end is not None:
        fn = _neo_fa3()
        extra = {"image_token_end": image_token_end.reshape(-1).contiguous()}
    elif torch.cuda.get_device_capability(q.device)[0] == 8:
        fn = _neo_fa3()
        extra = {}
    else:
        from sglang.kernels.ops.attention.flash_attention import (
            flash_attn_with_kvcache as fn,
        )

        extra = {"ver": 3}
    out = fn(
        q=q.reshape(batch * qlen, heads, dim).contiguous(),
        k_cache=k.contiguous(),
        v_cache=v.contiguous(),
        cu_seqlens_q=cu_q,
        cache_seqlens=kv_lengths,
        max_seqlen_q=qlen,
        causal=causal,
        softmax_scale=scale,
        **extra,
    )
    return out.reshape(batch, qlen, heads, dim)
