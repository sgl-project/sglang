"""Eager QSA prefill using CANN sparse attention's MLA layout.

This adapter keeps the indexer's physical token slots unchanged. For already
rotated Q/K, [Q, 0] @ [K, V].T = Q @ K.T; the V half of the MLA output is
therefore sparse GQA with the original scale. The extra RoPE inputs are zero.

Targeted at Ascend 910C with CANN 9.0 / torch-npu 2.10, BF16 D256.
Packing has a host synchronization and temporary storage proportional to the
largest referenced physical slot, so this is deliberately eager-prefill only.
"""

from typing import Optional

import torch
import torch.nn.functional as F

# Bound the packed cache to 512 MiB (BF16, two KV heads, K+V of width 512).
# Packing and the CANN workspace require additional temporary memory.
_MAX_CACHE_EXTENT = 262144
_HEAD_SHAPES = {(16, 2), (24, 2), (12, 1), (6, 1), (3, 1)}


def try_qsa_native_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Return None for unsupported inputs, preserving the caller's fallback.

    Slots are physical cache indices followed by -1 padding, with causality
    and request isolation already resolved by the existing indexer/expansion/
    mapping path. CANN must not apply a second causal mask based on these
    physical addresses. Rows with interior padding use the caller's fallback.
    """
    if (
        q.device.type != "npu"
        or q.ndim != 3
        or k_cache.ndim != 3
        or v_cache.shape != k_cache.shape
        or q.shape[-1] != 256
        or k_cache.shape[-1] != 256
        or (q.shape[1], k_cache.shape[1]) not in _HEAD_SHAPES
        or any(x.dtype != torch.bfloat16 for x in (q, k_cache, v_cache))
        or any(x.device != q.device for x in (k_cache, v_cache, token_slots))
        or token_slots.ndim != 2
        or token_slots.shape[0] != q.shape[0]
        or token_slots.dtype != torch.int32
        or token_slots.shape[1] > 2051
    ):
        return None

    import torch_npu

    if (
        not hasattr(torch_npu, "npu_sparse_flash_attention")
        or not torch.npu.get_device_name(q.device).startswith("Ascend910_93")
        or torch.npu.is_current_stream_capturing()
    ):
        return None
    rows, heads, dim = q.shape
    if rows == 0 or token_slots.shape[1] == 0:
        return torch.zeros_like(q)

    # The pool can be much larger than the live context. Packing it in full
    # caused OOM in the prototype. This scalar read is intentional and must
    # stay outside graph capture, decode, and speculative verification.
    # CANN treats -1 as the end of a row, not as an arbitrary masked column.
    # Check for invalid-to-valid transitions while obtaining the cache extent;
    # copy only these two scalars to the host in one synchronization.
    has_holes = ((token_slots[:, :-1] < 0) & (token_slots[:, 1:] >= 0)).any()
    max_slot, has_holes = (
        torch.stack((token_slots.max(), has_holes.to(token_slots.dtype))).cpu().tolist()
    )
    if has_holes:
        return None
    extent = max(1, max_slot + 1)
    if extent > min(k_cache.shape[0], _MAX_CACHE_EXTENT):
        return None

    kv_heads = k_cache.shape[1]
    groups = heads // kv_heads
    # CANN requires a power-of-two query/KV head ratio. Qwen3.8-Flash-Next
    # uses 12 (or 6/3 after attention-TP sharding). Heads are independent, so
    # zero-padding this axis and discarding the extra outputs is exact.
    padded_groups = 1 << (groups - 1).bit_length()
    latent = (
        torch.cat((k_cache[:extent], v_cache[:extent]), dim=-1)
        .permute(1, 0, 2)
        .unsqueeze(2)
        .contiguous()
    )
    query = q.reshape(rows, kv_heads, groups, dim).permute(1, 0, 2, 3)
    query = F.pad(query, (0, dim, 0, padded_groups - groups)).contiguous()
    query_rope = q.new_zeros((kv_heads, rows, padded_groups, 64))
    key_rope = q.new_zeros((kv_heads, extent, 1, 64))
    slots = F.pad(token_slots, (0, -token_slots.shape[1] % 64), value=-1)
    indices = slots[None, :, None, :].expand(kv_heads, -1, 1, -1).contiguous()
    result = torch_npu.npu_sparse_flash_attention(
        query,
        latent,
        latent,
        indices,
        dim**-0.5 if softmax_scale is None else softmax_scale,
        sparse_block_size=1,
        sparse_mode=0,
        attention_mode=2,
        query_rope=query_rope,
        key_rope=key_rope,
    )
    output = result[0] if isinstance(result, tuple) else result
    output = output[..., :groups, dim:].permute(1, 0, 2, 3).reshape(rows, heads, dim)
    # Define fully masked rows even if the native softmax returns NaN there.
    return torch.where((token_slots >= 0).any(dim=1)[:, None, None], output, 0)
