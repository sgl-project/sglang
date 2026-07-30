"""Shared helpers for int2 quantized KV prefill.

These utilities were originally inline methods on :class:`TritonAttnBackend`.
They're factored out here so other attention backends (notably FA3) can reuse
the same rotation (Hadamard for MHA-int2, Oscar for unified HP+int2) and
HP+int2 aware dequantization pipeline, keeping a single source of truth for
int2 prefill semantics.

Callers are expected to drive the higher level flow themselves:

  1. Call :func:`prepare_quantized_extend_qkv` before writing KV to the pool.
     Pass the returned ``pre_rotated_k`` / ``pre_rotated_v`` to
     ``set_kv_buffer(..., already_hadamard_transformed=True)`` so the pool does
     not rotate again.
  2. Call :func:`dequantize_prefix_kv` to materialize contiguous per-token K/V
     for the cached prefix (any mix of HP and int2 tiers is handled).
  3. Concatenate the dequantized prefix with the newly rotated extend K/V and
     run ``flash_attn_varlen_func`` (or equivalent).
  4. Call :func:`apply_inverse_v_rotation` on the attention output if the V
     path was rotated (Hadamard self-inverse, or ``result @ R_v.T`` for
     Oscar).

The module is intentionally framework-light: it only depends on torch +
the JIT Hadamard kernel (``sglang.jit_kernel.hadamard``, with an optional
``fast_hadamard_transform`` fast path) and on the pool surface
(``get_raw_key_buffer`` / ``get_key_scales_zeros`` / ...) that both
``MHATokenToKVPool`` and ``UnifiedInt2HPKVPool`` already implement.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    from sglang.jit_kernel.hadamard import hadamard_transform

from sglang.srt.mem_cache.kv_quant_kernels import (
    _get_num_scale_groups,
    dequantize_kv_int2_triton,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


def apply_segmented_hadamard_transform(
    tensor: torch.Tensor,
    hadamard_order: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Segmented (blockwise) FWHT along the last dim, matching the fused int2
    write / decode kernels.

    ``hadamard_order`` defaults to ``envs.SGLANG_HADAMARD_ORDER.get()`` when
    unspecified. ``out_dtype``, when set, casts the input before transforming
    (used by callers that want to upcast bf16 → fp32 internally before the
    rotation, then cast back). The returned tensor has the same shape as
    ``tensor`` and a dtype determined by ``hadamard_transform``'s rules
    (typically the input dtype after the optional cast). Self-inverse with
    the ``1/sqrt(order)`` pre-normalization.

    Single canonical implementation; previously duplicated in three places
    (memory_pool.py, this file, and inline in triton_backend.py).
    """
    if hadamard_order is None:
        hadamard_order = envs.SGLANG_HADAMARD_ORDER.get()
    if out_dtype is not None:
        tensor = tensor.to(out_dtype)
    return hadamard_transform(
        tensor.view(
            *tensor.shape[:-1],
            tensor.shape[-1] // hadamard_order,
            hadamard_order,
        )
        / math.sqrt(hadamard_order)
    ).view_as(tensor)


# Backward-compatible alias for module-internal callers.
_apply_segmented_hadamard_transform = apply_segmented_hadamard_transform


def _pool_uses_oscar_rotation(kv_pool) -> bool:
    """True for the mixed HP+int2 unified pool that loads per-layer Oscar
    rotation matrices. Non-oscar int2 pools (MHA without mixed HP+int2
    storage) implicitly use the legacy segmented Hadamard rotation.
    """
    return getattr(kv_pool, "_R_k", None) is not None


def _apply_oscar_rotation(tensor: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply ``tensor @ R`` along the last dim. Returns a contiguous tensor in
    ``R.dtype``. Works for shapes ``[*, head_dim]``.
    """
    return (tensor.to(R.dtype) @ R).contiguous()


def prepare_quantized_extend_qkv(
    kv_pool,
    layer: RadixAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_already_hadamard_transformed: bool = False,
    kv_already_hadamard_transformed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Apply the pool's active rotation to Q/K/V for int2 extend.

    Returns the (possibly) rotated ``q, k, v`` tensors and a flag indicating
    whether the attention output must be inverse-rotated afterwards. For
    non-int2 pools this is a no-op.
    """
    need_v_inverse = False
    kv_dtype = kv_pool.dtype
    if kv_dtype != "int2":
        return q, k, v, need_v_inverse

    if _pool_uses_oscar_rotation(kv_pool):
        layer_idx = layer.layer_id - kv_pool.start_layer
        R_k = kv_pool._R_k[layer_idx]
        R_v = kv_pool._R_v[layer_idx]
        v_rotation_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))
        if not q_already_hadamard_transformed:
            q = _apply_oscar_rotation(q, R_k)
        if not kv_already_hadamard_transformed:
            k = _apply_oscar_rotation(k, R_k)
            if v_rotation_absorbed:
                v = v.to(R_v.dtype).contiguous()
            else:
                v = _apply_oscar_rotation(v, R_v)
        need_v_inverse = True
        return q, k, v, need_v_inverse

    if not q_already_hadamard_transformed:
        q = _apply_segmented_hadamard_transform(q)
    if not kv_already_hadamard_transformed:
        k = _apply_segmented_hadamard_transform(k)
        v = _apply_segmented_hadamard_transform(v)
    need_v_inverse = True
    return q, k, v, need_v_inverse


@triton.jit
def _mixed_prefix_dequant_kernel(
    prefix_indices_ptr,
    quant_ptr,
    scales_zeros_ptr,
    hp_ptr,
    out_ptr,
    num_tokens,
    num_heads,
    head_dim: tl.constexpr,
    quant_stride_token: tl.constexpr,
    quant_stride_head: tl.constexpr,
    quant_stride_dim: tl.constexpr,
    sz_stride_token: tl.constexpr,
    sz_stride_head: tl.constexpr,
    sz_stride_dim: tl.constexpr,
    hp_stride_token: tl.constexpr,
    hp_stride_head: tl.constexpr,
    hp_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_DIM)
    dim_mask = offs < head_dim

    slot = tl.load(prefix_indices_ptr + token_idx)
    is_hp = slot >= HP_OFFSET
    quarter_dim = head_dim // 4

    byte_offsets = offs % quarter_dim
    packed = tl.load(
        quant_ptr
        + slot * quant_stride_token
        + head_idx * quant_stride_head
        + byte_offsets * quant_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=0,
    )
    shift = (offs // quarter_dim) * 2
    q = ((packed >> shift) & 0x03).to(tl.float32)

    group_ids = offs // GROUP_SIZE
    scale = tl.load(
        scales_zeros_ptr
        + slot * sz_stride_token
        + head_idx * sz_stride_head
        + (group_ids * 2) * sz_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=1.0,
    ).to(tl.float32)
    zero = tl.load(
        scales_zeros_ptr
        + slot * sz_stride_token
        + head_idx * sz_stride_head
        + (group_ids * 2 + 1) * sz_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=0.0,
    ).to(tl.float32)
    quant_val = (q - zero) * scale

    hp_slot = slot - HP_OFFSET
    hp_val = tl.load(
        hp_ptr
        + hp_slot * hp_stride_token
        + head_idx * hp_stride_head
        + offs * hp_stride_dim,
        mask=is_hp & dim_mask,
        other=0.0,
    )
    out_val = tl.where(is_hp, hp_val, quant_val)
    tl.store(
        out_ptr
        + token_idx * out_stride_token
        + head_idx * out_stride_head
        + offs * out_stride_dim,
        out_val,
        mask=(token_idx < num_tokens) & (head_idx < num_heads) & dim_mask,
    )


def _mixed_prefix_dequantize_tensor(
    prefix_indices: torch.Tensor,
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    hp: torch.Tensor,
    hp_offset: int,
    head_dim: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    num_tokens = prefix_indices.shape[0]
    num_heads = quantized.shape[1]
    out = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=model_dtype,
        device=prefix_indices.device,
    )
    if num_tokens == 0:
        return out
    num_groups = _get_num_scale_groups(scales_zeros)
    group_size = head_dim // num_groups
    grid = (num_tokens, num_heads)
    _mixed_prefix_dequant_kernel[grid](
        prefix_indices,
        quantized,
        scales_zeros,
        hp,
        out,
        num_tokens,
        num_heads,
        head_dim,
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        scales_zeros.stride(0),
        scales_zeros.stride(1),
        scales_zeros.stride(2),
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HP_OFFSET=int(hp_offset),
        GROUP_SIZE=group_size,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )
    return out


def dequantize_prefix_kv(
    kv_pool,
    layer_id: int,
    prefix_indices: torch.Tensor,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the prefix slots referenced by ``prefix_indices`` into dense
    ``[num_tokens, head_num, head_dim]`` tensors in ``model_dtype``.

    Supports both ``MHATokenToKVPool`` (all slots are int2) and
    ``UnifiedInt2HPKVPool`` (some slots are HP, others are int2; classified
    by ``slot >= hp_global_offset`` -- HP slot ids start at exactly
    ``hp_global_offset``).

    Grouped scales (``scales.shape[-1] > 2``) are handled by
    ``dequantize_kv_int2_triton`` internally.
    """
    device = prefix_indices.device
    if prefix_indices.numel() == 0:
        return (
            torch.empty(
                (0, kv_pool.head_num, kv_pool.head_dim),
                dtype=model_dtype,
                device=device,
            ),
            torch.empty(
                (0, kv_pool.head_num, kv_pool.v_head_dim),
                dtype=model_dtype,
                device=device,
            ),
        )

    prefix_indices = prefix_indices.to(torch.int64)
    if (
        getattr(kv_pool, "mixed_kv_enabled", None) is not None
        and kv_pool.mixed_kv_enabled()
    ):
        assert (
            kv_pool.dtype == "int2"
        ), f"Unsupported quantized KV dtype: {kv_pool.dtype}"
        return (
            _mixed_prefix_dequantize_tensor(
                prefix_indices,
                kv_pool.get_raw_key_buffer(layer_id),
                kv_pool.get_key_scales_zeros(layer_id),
                kv_pool.get_hp_key_buffer(layer_id),
                kv_pool.hp_global_offset,
                kv_pool.head_dim,
                model_dtype,
            ),
            _mixed_prefix_dequantize_tensor(
                prefix_indices,
                kv_pool.get_raw_value_buffer(layer_id),
                kv_pool.get_value_scales_zeros(layer_id),
                kv_pool.get_hp_value_buffer(layer_id),
                kv_pool.hp_global_offset,
                kv_pool.v_head_dim,
                model_dtype,
            ),
        )

    raw_k = kv_pool.get_raw_key_buffer(layer_id)[prefix_indices]
    raw_v = kv_pool.get_raw_value_buffer(layer_id)[prefix_indices]
    scales_k = kv_pool.get_key_scales_zeros(layer_id)[prefix_indices]
    scales_v = kv_pool.get_value_scales_zeros(layer_id)[prefix_indices]
    assert kv_pool.dtype == "int2", f"Unsupported quantized KV dtype: {kv_pool.dtype}"
    return (
        dequantize_kv_int2_triton(raw_k, scales_k, kv_pool.head_dim, model_dtype),
        dequantize_kv_int2_triton(raw_v, scales_v, kv_pool.v_head_dim, model_dtype),
    )


def apply_inverse_v_rotation(
    result: torch.Tensor,
    kv_pool,
    layer: RadixAttention,
    need_v_inverse: bool,
) -> torch.Tensor:
    """Apply the inverse V rotation on an attention output tensor, when
    required. Hadamard is self-inverse (segmented FWHT), Oscar inverts as
    ``result @ R_v.T`` in ``R_v``'s dtype.

    ``result`` must have shape ``[..., v_head_dim]``; callers should reshape
    beforehand if their output is stored flattened.
    """
    if not need_v_inverse or kv_pool.dtype != "int2":
        return result
    if _pool_uses_oscar_rotation(kv_pool):
        layer_idx = layer.layer_id - kv_pool.start_layer
        R_v = kv_pool._R_v[layer_idx]
        return (result.to(R_v.dtype) @ R_v.T).contiguous()
    return _apply_segmented_hadamard_transform(result)


@triton.jit
def _build_prefix_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    prefix_indptr_ptr,
    out_ptr,
    req_to_token_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    prefix_len = tl.load(prefix_lens_ptr + req_idx)
    out_start = tl.load(prefix_indptr_ptr + req_idx)
    mask = offs < prefix_len
    slots = tl.load(
        req_to_token_ptr + req_pool_idx * req_to_token_stride + offs,
        mask=mask,
        other=0,
    ).to(tl.int64)
    tl.store(out_ptr + out_start + offs, slots, mask=mask)


def _cpu_int_list(values) -> Optional[list[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def build_prefix_indices_from_req_to_token(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_seqlens_cpu=None,
) -> torch.Tensor:
    """Gather valid prefix slot ids per request as a flat 1-D tensor by
    reading ``req_to_token`` directly (no page striding / page_size division).

    Required by the unified mixed HP+int2 prefill path. HP slot ids in that
    pool are encoded ``HP_OFFSET + hp_page_id`` and adjacent positions can
    map to arbitrary HP page ids, so the page-table-based reconstruction
    used by :func:`build_prefix_indices_from_page_table` returns garbled HP
    slot ids and reads the wrong rows out of ``hp_k_buffer`` / ``hp_v_buffer``.

    Inputs:
        req_to_token       : int32 [max_req_slots, max_context_len]
        req_pool_indices   : int64 [bs] -- per-request row index
        cache_seqlens      : int32 or int64 [bs] -- valid prefix length

    Returns:
        flat int64 1-D tensor of slot ids, in (request, position) order
        consistent with what the FA varlen kernel expects when concatenated
        with the freshly written extend slots.
    """
    device = req_to_token.device
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    cache_seqlens_cpu = _cpu_int_list(cache_seqlens_cpu)
    if cache_seqlens_cpu is None:
        raise ValueError(
            "build_prefix_indices_from_req_to_token requires CPU prefix lengths "
            "to avoid CUDA boolean-index synchronization"
        )
    total_prefix = sum(cache_seqlens_cpu)
    out = torch.empty((total_prefix,), dtype=torch.int64, device=device)
    if total_prefix == 0:
        return out
    prefix_lens_cpu = torch.tensor(cache_seqlens_cpu, dtype=torch.int32)
    prefix_indptr_cpu = torch.empty((bs + 1,), dtype=torch.int32)
    prefix_indptr_cpu[0] = 0
    prefix_indptr_cpu[1:] = torch.cumsum(prefix_lens_cpu, dim=0)
    prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
    prefix_indptr = prefix_indptr_cpu.to(device, non_blocking=True)
    max_prefix_len = max(cache_seqlens_cpu)
    _build_prefix_indices_kernel[(bs,)](
        req_to_token,
        req_pool_indices.to(torch.int64),
        prefix_lens,
        prefix_indptr,
        out,
        req_to_token.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(max_prefix_len),
        num_warps=8,
        num_stages=1,
    )
    return out
