"""Fused Triton kernel for Quest decode bounds update.

Replaces the per-step per-layer Python loop in
``QuestAlgorithm.update_decode_representations`` (called once per layer per
decode step from both ``flashinfer_quest`` and ``flashinfer_hisparse``
paths) with a single kernel that processes ALL Quest layers and ALL active
requests at once.

For each (layer, active_req) pair the kernel:
  1. Reads the just-decoded K row from that layer's K buffer at
     ``device_locs[req]``.
  2. Element-wise min/max with ``running_k_min/max[layer, req_idx]``.
  3. Writes the updated min/max back.

The expensive part of the previous implementation wasn't the math (it's
small) — it was the per-layer launch overhead × num_layers on the
backup/main stream, every decode step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_decode_bounds_kernel(
    k_data_ptrs,          # [num_total_layers] uint64 (pointer per layer)
    device_locs_ptr,      # [num_active_reqs] int (32 or 64)
    req_indices_ptr,      # [num_active_reqs] int64
    running_k_min_ptr,    # [num_quest_layers, max_reqs, kv_heads, head_dim] bf16
    running_k_max_ptr,    # ditto
    layer_offset_start,
    MAX_REQS: tl.constexpr,
    KVHD: tl.constexpr,
    BLOCK_KVHD: tl.constexpr,
):
    """One block per (layer, active_req, kvhd_tile)."""
    pid_layer = tl.program_id(0)
    pid_req = tl.program_id(1)
    pid_kvhd = tl.program_id(2)

    # Load this layer's K buffer pointer.
    k_ptr_int = tl.load(k_data_ptrs + layer_offset_start + pid_layer)
    k_buf_ptr = k_ptr_int.to(tl.pointer_type(tl.bfloat16))

    # Per-active-req metadata.
    device_loc = tl.load(device_locs_ptr + pid_req).to(tl.int64)
    req_idx = tl.load(req_indices_ptr + pid_req).to(tl.int64)

    # KVHD tile.
    d_off_local = tl.arange(0, BLOCK_KVHD)
    d_off = pid_kvhd * BLOCK_KVHD + d_off_local

    # Read K[device_loc, kvhd_tile] from this layer's buffer.
    k_offsets = device_loc * KVHD + d_off
    new_k = tl.load(k_buf_ptr + k_offsets)  # [BLOCK_KVHD] bf16

    # Read running min/max for this (layer, req, kvhd_tile).
    out_base = pid_layer * MAX_REQS * KVHD + req_idx * KVHD + d_off
    cur_min = tl.load(running_k_min_ptr + out_base)
    cur_max = tl.load(running_k_max_ptr + out_base)

    # Element-wise min/max in fp32 to be safe with bf16 NaN handling
    # (running buffers are seeded with bf16 ±max which equal each other in bf16
    #  but compare correctly in fp32).
    new_k_f32 = new_k.to(tl.float32)
    cur_min_f32 = cur_min.to(tl.float32)
    cur_max_f32 = cur_max.to(tl.float32)
    new_min = tl.minimum(cur_min_f32, new_k_f32).to(tl.bfloat16)
    new_max = tl.maximum(cur_max_f32, new_k_f32).to(tl.bfloat16)

    tl.store(running_k_min_ptr + out_base, new_min)
    tl.store(running_k_max_ptr + out_base, new_max)


def quest_decode_bounds(
    k_data_ptrs: torch.Tensor,
    device_locs: torch.Tensor,
    req_indices: torch.Tensor,
    running_k_min: torch.Tensor,
    running_k_max: torch.Tensor,
    layer_offset_start: int,
) -> None:
    """Fused all-layer Quest decode-bounds update.

    Replaces the per-layer Python loop:

        for layer in range(start_layer, end_layer):
            k_buf = pool.get_key_buffer(layer)
            quest.update_decode_representations(layer, req_indices, k_buf, device_locs)

    Args:
      k_data_ptrs: ``[num_total_layers]`` uint64 — pointer to each layer's K
        buffer (from ``MHATokenToKVPool.k_data_ptrs``).
      device_locs: ``[num_active_reqs]`` int — the just-decoded token's
        physical location in the K buffer for each active req.
      req_indices: ``[num_active_reqs]`` int64 — request slots being
        updated.
      running_k_min: ``[num_quest_layers, max_reqs, kv_heads, head_dim]``
        bf16 contiguous.
      running_k_max: same shape.
      layer_offset_start: int — index into ``k_data_ptrs`` of the first
        Quest layer (i.e., ``quest.start_layer``).
    """
    num_active = int(device_locs.shape[0])
    if num_active == 0:
        return

    assert running_k_min.is_contiguous(), "running_k_min must be contiguous"
    assert running_k_max.is_contiguous(), "running_k_max must be contiguous"
    assert running_k_min.dtype == torch.bfloat16
    assert running_k_max.dtype == torch.bfloat16

    num_quest_layers, max_reqs, kv_heads, head_dim = running_k_min.shape
    kvhd = kv_heads * head_dim

    if kvhd >= 256 and kvhd % 128 == 0:
        block_kvhd = 128
    elif kvhd >= 128 and kvhd % 64 == 0:
        block_kvhd = 64
    else:
        block_kvhd = kvhd
    kvhd_tiles = kvhd // block_kvhd

    grid = (num_quest_layers, num_active, kvhd_tiles)

    _quest_decode_bounds_kernel[grid](
        k_data_ptrs,
        device_locs,
        req_indices,
        running_k_min,
        running_k_max,
        layer_offset_start,
        MAX_REQS=max_reqs,
        KVHD=kvhd,
        BLOCK_KVHD=block_kvhd,
        num_warps=2,
    )
