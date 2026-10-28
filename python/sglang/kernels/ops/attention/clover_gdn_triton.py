"""Triton kernel for GDN local memory in diffusion."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _check_bd_gdn_softmax_scalars(
    logit_scale: torch.Tensor | None,
    history_bias: torch.Tensor | None,
    num_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logit_scale is None or history_bias is None:
        raise ValueError(
            "Clover requires per-head bd_softmax_logit_scale and "
            "bd_softmax_history_bias tensors"
        )
    if (
        logit_scale.numel() != num_value_heads
        or history_bias.numel() != num_value_heads
    ):
        raise ValueError(
            "BD-GDN softmax scalars must have one entry per local value head: "
            f"expected {num_value_heads}, got {logit_scale.numel()} and "
            f"{history_bias.numel()}"
        )
    # The kernel converts these scalars to FP32 on load, so BF16 views pass unconverted.
    return logit_scale.reshape(-1), history_bias.reshape(-1)


def _check_bd_gdn_softmax_blocksize_scalars(
    history_c: torch.Tensor | None,
    reference_block_size: float,
    num_value_heads: int,
) -> torch.Tensor:
    if history_c is None:
        raise ValueError(
            "BD-GDN style 'softmax_blocksize' requires per-head bd_softmax_history_c"
        )
    if history_c.numel() != num_value_heads:
        raise ValueError(
            "BD-GDN block-size history scalar must have one entry per local "
            f"value head: expected {num_value_heads}, got {history_c.numel()}"
        )
    if not reference_block_size > 0.0:
        raise ValueError(
            "BD-GDN softmax reference block size must be positive, got "
            f"{reference_block_size!r}"
        )
    return history_c.reshape(-1)


@triton.jit
def _clover_local_memory_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    g_ptr,
    clean_state_ptr,
    cache_indices_ptr,
    active_valid_ptr,
    softmax_logit_scale_ptr,
    softmax_history_bias_ptr,
    softmax_history_c_ptr,
    output_ptr,
    query_stride_t,
    query_stride_h,
    query_stride_k,
    key_stride_t,
    key_stride_h,
    key_stride_k,
    value_stride_t,
    value_stride_h,
    value_stride_v,
    g_stride_t,
    g_stride_h,
    state_stride_slot,
    state_stride_h,
    state_stride_v,
    state_stride_k,
    output_stride_t,
    output_stride_h,
    output_stride_v,
    scale,
    softmax_reference_block_size,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Compute the read-only BD-GDN preview output for one fixed-size block.

    Block-local softmax includes the gate-decayed clean readout as one
    history bucket. Its logit is history_bias plus
    history_c * log(valid_block_size / reference_block_size).

    One program handles one request, one value head, and one value-dimension
    tile. Persistent state is read directly in SGLang's [slot, H, V, K]
    layout and is never written.
    """
    value_tile = tl.program_id(0)
    query_program = tl.program_id(1)
    request_idx = query_program // NUM_QUERY_TILES
    query_tile = query_program % NUM_QUERY_TILES
    value_head = tl.program_id(2)

    token_offsets = query_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    local_key_offsets = tl.arange(0, BLOCK_T)
    key_offsets = tl.arange(0, BLOCK_K)
    value_offsets = value_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    token_mask = token_offsets < BLOCK_SIZE
    local_key_mask = local_key_offsets < BLOCK_SIZE
    key_mask = key_offsets < KEY_DIM
    value_mask = value_offsets < VALUE_DIM
    token_indices = request_idx * BLOCK_SIZE + token_offsets

    cache_index = tl.load(cache_indices_ptr + request_idx)
    request_active = cache_index >= 0
    safe_cache_index = tl.maximum(cache_index, 0)
    active = (
        tl.load(active_valid_ptr + token_indices, mask=token_mask, other=0) != 0
    ) & request_active

    query_head = value_head // (NUM_V_HEADS // NUM_Q_HEADS)
    key_head = value_head // (NUM_V_HEADS // NUM_K_HEADS)

    query_offsets = (
        token_indices[:, None] * query_stride_t
        + query_head * query_stride_h
        + key_offsets[None, :] * query_stride_k
    )
    key_offsets_transposed = (
        (request_idx * BLOCK_SIZE + local_key_offsets)[None, :] * key_stride_t
        + key_head * key_stride_h
        + key_offsets[:, None] * key_stride_k
    )
    value_matrix_offsets = (
        (request_idx * BLOCK_SIZE + local_key_offsets)[:, None] * value_stride_t
        + value_head * value_stride_h
        + value_offsets[None, :] * value_stride_v
    )

    query = tl.load(
        query_ptr + query_offsets,
        mask=token_mask[:, None] & key_mask[None, :] & request_active,
        other=0.0,
    )
    key_transposed = tl.load(
        key_ptr + key_offsets_transposed,
        mask=key_mask[:, None] & local_key_mask[None, :] & request_active,
        other=0.0,
    )
    values = tl.load(
        value_ptr + value_matrix_offsets,
        mask=local_key_mask[:, None] & value_mask[None, :] & request_active,
        other=0.0,
    )

    # The 1e-6 epsilon matches the GDN L2 normalization.
    query_f32 = query.to(tl.float32)
    query_norm = tl.rsqrt(tl.sum(query_f32 * query_f32, axis=1) + 1.0e-6)
    query = (query_f32 * query_norm[:, None]).to(query.dtype)
    key_f32 = key_transposed.to(tl.float32)
    key_norm = tl.rsqrt(tl.sum(key_f32 * key_f32, axis=0) + 1.0e-6)
    key_transposed = (key_f32 * key_norm[None, :]).to(key_transposed.dtype)

    state_offsets = (
        safe_cache_index * state_stride_slot
        + value_head * state_stride_h
        + value_offsets[None, :] * state_stride_v
        + key_offsets[:, None] * state_stride_k
    )
    clean_state = tl.load(
        clean_state_ptr + state_offsets,
        mask=key_mask[:, None] & value_mask[None, :] & request_active,
        other=0.0,
    )

    scalar_offsets = token_indices * g_stride_t + value_head * g_stride_h
    g = tl.load(g_ptr + scalar_offsets, mask=token_mask & request_active, other=0.0).to(
        tl.float32
    )
    local_active = (
        tl.load(
            active_valid_ptr + request_idx * BLOCK_SIZE + local_key_offsets,
            mask=local_key_mask,
            other=0,
        )
        != 0
    ) & request_active

    history = tl.dot(query, clean_state.to(query.dtype))
    scores = tl.dot(query, key_transposed)
    logit_scale = tl.load(softmax_logit_scale_ptr + value_head).to(tl.float32)
    history_bias = tl.load(softmax_history_bias_ptr + value_head).to(tl.float32)
    history_c = tl.load(softmax_history_c_ptr + value_head).to(tl.float32)
    valid_block_size = tl.maximum(tl.sum(local_active, axis=0), 1.0)
    history_bias += history_c * tl.log(valid_block_size / softmax_reference_block_size)
    logits = scores * logit_scale
    logits = tl.where(local_active[None, :], logits, float("-inf"))
    row_max = tl.maximum(tl.max(logits, axis=1), history_bias)
    exp_local = tl.exp(logits - row_max[:, None])
    exp_history = tl.exp(history_bias - row_max)
    denominator = tl.sum(exp_local, axis=1) + exp_history
    local_weights = exp_local / denominator[:, None]
    history_weight = exp_history / denominator
    local = tl.dot(local_weights.to(values.dtype), values)
    output = history * (scale * tl.exp(g) * history_weight)[:, None] + local
    output = tl.where(active[:, None], output, 0.0)

    output_offsets = (
        token_indices[:, None] * output_stride_t
        + value_head * output_stride_h
        + value_offsets[None, :] * output_stride_v
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=token_mask[:, None] & value_mask[None, :],
    )


def supports_clover_local_memory_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
) -> bool:
    """Return whether the specialized production kernel supports this shape."""
    return (
        query.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[0] == key.shape[0] == value.shape[0] == 1
        and block_size in (1, 2, 4, 8, 16, 32)
        and query.shape[-1] == key.shape[-1] == 128
        and value.shape[-1] == 128
        and value.shape[2] % query.shape[2] == 0
        and value.shape[2] % key.shape[2] == 0
        and torch.cuda.get_device_capability(query.device)[0] in (9, 10)
    )


def clover_local_memory_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    *,
    clean_ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    active_valid_mask: torch.Tensor,
    scale: float,
    softmax_logit_scale: torch.Tensor | None = None,
    softmax_history_bias: torch.Tensor | None = None,
    softmax_history_c: torch.Tensor | None = None,
    softmax_reference_block_size: float = 32.0,
) -> torch.Tensor:
    """Run the non-mutating fixed-block BD-GDN preview kernel.

    Uses Clover's softmax_blocksize rule with learned per-head history scalars.
    """
    batch_size = cache_indices.numel()
    if batch_size <= 0:
        raise ValueError("BD-GDN Triton preview requires a non-empty batch")
    num_tokens = query.shape[1]
    if num_tokens % batch_size:
        raise ValueError(
            f"BD-GDN token count {num_tokens} is not divisible by batch {batch_size}"
        )
    block_size = num_tokens // batch_size
    if not supports_clover_local_memory_triton(
        query, key, value, block_size=block_size
    ):
        raise NotImplementedError(
            "BD-GDN Triton preview supports CUDA BF16/FP16, block_size in "
            "{1, 2, 4, 8, 16, 32}, "
            "and key/value dimensions 128"
        )
    if active_valid_mask.numel() != num_tokens:
        raise ValueError(f"{active_valid_mask.numel()=} does not match {num_tokens=}")

    num_q_heads = query.shape[2]
    num_k_heads = key.shape[2]
    num_v_heads = value.shape[2]
    key_dim = query.shape[-1]
    value_dim = value.shape[-1]
    if clean_ssm_states.ndim != 4 or clean_ssm_states.shape[1:] != (
        num_v_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError(
            "BD-GDN clean state must use SGLang [slot, H, V, K] layout, got "
            f"{tuple(clean_ssm_states.shape)}"
        )
    if clean_ssm_states.device != value.device:
        raise ValueError("BD-GDN clean state and preview tensors must share a device")
    logit_scale, history_bias = _check_bd_gdn_softmax_scalars(
        softmax_logit_scale, softmax_history_bias, num_v_heads
    )
    if logit_scale.device != value.device or history_bias.device != value.device:
        raise ValueError(
            "BD-GDN softmax scalars and preview tensors must share a device"
        )
    logit_scale = logit_scale.contiguous()
    history_bias = history_bias.contiguous()
    history_c = _check_bd_gdn_softmax_blocksize_scalars(
        softmax_history_c,
        softmax_reference_block_size,
        num_v_heads,
    )
    if history_c.device != value.device:
        raise ValueError(
            "BD-GDN block-size history scalar and preview tensors must share a device"
        )
    history_c = history_c.contiguous()
    output = torch.empty(
        (1, num_tokens, num_v_heads, value_dim),
        dtype=value.dtype,
        device=value.device,
    )

    # tl.dot needs K of at least 16, and token_mask zeroes the padded rows.
    block_t = max(16, triton.next_power_of_2(block_size))
    block_q = block_t
    num_query_tiles = triton.cdiv(block_size, block_q)
    block_k = triton.next_power_of_2(key_dim)
    block_v = 128
    grid = (
        triton.cdiv(value_dim, block_v),
        batch_size * num_query_tiles,
        num_v_heads,
    )
    _clover_local_memory_kernel[grid](
        query,
        key,
        value,
        g,
        clean_ssm_states,
        cache_indices,
        active_valid_mask,
        logit_scale,
        history_bias,
        history_c,
        output,
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        g.stride(1),
        g.stride(2),
        clean_ssm_states.stride(0),
        clean_ssm_states.stride(1),
        clean_ssm_states.stride(2),
        clean_ssm_states.stride(3),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        float(scale),
        float(softmax_reference_block_size),
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        BLOCK_SIZE=block_size,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_T=block_t,
        BLOCK_Q=block_q,
        NUM_QUERY_TILES=num_query_tiles,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=8,
        num_stages=3,
    )
    return output
