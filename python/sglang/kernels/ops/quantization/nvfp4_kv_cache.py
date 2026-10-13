# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA Graph-safe NVFP4 KV-cache workspace helpers."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_device_core_count


@triton.jit
def _e2m1_to_float(value):
    magnitude = value & 0x7
    decoded = tl.where(
        magnitude < 5,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
    )
    return tl.where((value & 0x8) != 0, -decoded, decoded)


@triton.jit
def _dequantize_nvfp4_kv_for_speculative_extend_kernel(
    k_fp4_ptr,
    v_fp4_ptr,
    k_block_scales_ptr,
    v_block_scales_ptr,
    k_global_scale_ptr,
    v_global_scale_ptr,
    k_current_ptr,
    v_current_ptr,
    current_locs_ptr,
    dq_k_ptr,
    dq_v_ptr,
    req_to_token_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    num_reqs,
    num_current_tokens_per_req,
    prefix_len_delta,
    req_to_token_stride,
    k_current_row_stride,
    v_current_row_stride,
    ROW_ELEMENTS: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    """Dequantize cached prefixes and copy speculative-token KV by physical slot."""
    worker_id = tl.program_id(0)
    num_workers = tl.num_programs(0)
    element_offsets = tl.arange(0, BLOCK_ELEMENTS)
    element_mask = element_offsets < ROW_ELEMENTS

    packed_row_elements: tl.constexpr = ROW_ELEMENTS // 2
    scale_row_elements: tl.constexpr = ROW_ELEMENTS // 16

    k_global_scale = tl.load(k_global_scale_ptr).to(tl.float32)
    v_global_scale = tl.load(v_global_scale_ptr).to(tl.float32)

    for request_offset in tl.range(0, num_reqs):
        req_idx = tl.load(req_pool_indices_ptr + request_offset)
        prefix_len = (
            tl.load(prefix_lens_ptr + request_offset).to(tl.int32) - prefix_len_delta
        )
        prefix_len = tl.maximum(prefix_len, 0)
        req_row = req_idx.to(tl.int64) * req_to_token_stride
        current_row_base = request_offset * num_current_tokens_per_req

        # CUDA Graph padding uses request row 0 but reserves KV slot 0 as its
        # write sink. Suppress the padded request entirely so it neither repeats
        # row 0's prefix work nor races a live request's workspace writes.
        first_current_slot = tl.load(current_locs_ptr + current_row_base)
        prefix_len = tl.where(first_current_slot > 0, prefix_len, 0)

        # The worker grid is capture-stable. Runtime prefix lengths stay on the
        # GPU, so replay can select different physical KV slots without baking
        # capture-time host values into the graph.
        for token_offset in tl.range(worker_id, prefix_len, num_workers):
            kv_slot = tl.load(req_to_token_ptr + req_row + token_offset).to(tl.int64)
            packed_row = kv_slot * packed_row_elements
            packed_offsets = packed_row + element_offsets // 2

            k_packed = tl.load(k_fp4_ptr + packed_offsets, mask=element_mask, other=0)
            v_packed = tl.load(v_fp4_ptr + packed_offsets, mask=element_mask, other=0)
            use_high_nibble = (element_offsets & 1) != 0
            k_e2m1 = tl.where(use_high_nibble, (k_packed >> 4) & 0xF, k_packed & 0xF)
            v_e2m1 = tl.where(use_high_nibble, (v_packed >> 4) & 0xF, v_packed & 0xF)

            scale_row = kv_slot * scale_row_elements
            scale_offsets = scale_row + element_offsets // 16
            k_block_scale = tl.load(
                k_block_scales_ptr + scale_offsets,
                mask=element_mask,
                other=0.0,
            ).to(tl.float32)
            v_block_scale = tl.load(
                v_block_scales_ptr + scale_offsets,
                mask=element_mask,
                other=0.0,
            ).to(tl.float32)

            output_offsets = kv_slot * ROW_ELEMENTS + element_offsets
            k_dequant = (_e2m1_to_float(k_e2m1) * k_block_scale * k_global_scale).to(
                tl.bfloat16
            )
            v_dequant = (_e2m1_to_float(v_e2m1) * v_block_scale * v_global_scale).to(
                tl.bfloat16
            )
            tl.store(dq_k_ptr + output_offsets, k_dequant, mask=element_mask)
            tl.store(dq_v_ptr + output_offsets, v_dequant, mask=element_mask)

        # Current speculative tokens have not been committed to the quantized
        # cache yet. Keep the existing prefill behavior and expose their
        # unquantized K/V through the FP8 workspace directly.
        for current_offset in tl.range(
            worker_id, num_current_tokens_per_req, num_workers
        ):
            current_row = current_row_base + current_offset
            # Graph padding points current_locs at the reserved slot 0. Reading
            # current locations directly prevents padded req_pool_indices (also
            # zero) from aliasing a live request's req_to_token row.
            kv_slot = tl.load(current_locs_ptr + current_row).to(tl.int64)
            current_mask = element_mask & (kv_slot > 0)
            k_current_offsets = current_row * k_current_row_stride + element_offsets
            v_current_offsets = current_row * v_current_row_stride + element_offsets
            output_offsets = kv_slot * ROW_ELEMENTS + element_offsets
            current_k = tl.load(
                k_current_ptr + k_current_offsets, mask=current_mask, other=0.0
            )
            current_v = tl.load(
                v_current_ptr + v_current_offsets, mask=current_mask, other=0.0
            )
            tl.store(dq_k_ptr + output_offsets, current_k, mask=current_mask)
            tl.store(dq_v_ptr + output_offsets, current_v, mask=current_mask)


_cached_num_workers = None


def dequantize_nvfp4_kv_for_speculative_extend(
    k_fp4: torch.Tensor,
    v_fp4: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    dq_k: torch.Tensor,
    dq_v: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    current_locs: torch.Tensor,
    num_current_tokens_per_req: int,
    prefix_len_delta: int,
) -> None:
    """Populate the physical-slot FP8 workspace for a speculative extend.

    This wrapper performs no allocation or device-to-host synchronization. The
    Triton launch shape depends only on static KV-row metadata, while request
    indices and prefix lengths remain device tensors read by the kernel. It is
    therefore safe to record once and replay with different requests/lengths.

    ``prefix_lens`` may contain either committed prefix lengths (target verify)
    or full post-write sequence lengths (draft extend). ``prefix_len_delta`` is
    zero for the former and the fixed current-token width for the latter.

    Each workspace row is one tensor-parallel rank's flattened KV head row:
    ``per_rank_kv_heads * head_dim`` elements.
    """
    if req_pool_indices.numel() == 0:
        return

    if num_current_tokens_per_req <= 0:
        raise ValueError(
            "num_current_tokens_per_req must be positive for speculative NVFP4 "
            f"workspace preparation, got {num_current_tokens_per_req}."
        )
    if prefix_len_delta < 0:
        raise ValueError(
            f"prefix_len_delta must be non-negative, got {prefix_len_delta}."
        )

    if not (
        k_fp4.is_contiguous()
        and v_fp4.is_contiguous()
        and k_block_scales.is_contiguous()
        and v_block_scales.is_contiguous()
        and dq_k.is_contiguous()
        and dq_v.is_contiguous()
    ):
        raise ValueError("NVFP4 speculative workspace tensors must be contiguous.")

    if prefix_lens.numel() != req_pool_indices.numel():
        raise ValueError(
            "NVFP4 speculative prefix lengths must match the request batch size."
        )

    row_elements = dq_k[0].numel()
    if row_elements % 16 != 0:
        raise ValueError(
            f"NVFP4 KV row size must be divisible by 16, got {row_elements}."
        )
    if dq_v[0].numel() != row_elements:
        raise ValueError("NVFP4 K/V workspace rows must have the same size.")
    if k_current[0].numel() != row_elements or v_current[0].numel() != row_elements:
        raise ValueError("Current K/V rows must match the dequant workspace row size.")
    if k_current.shape[0] != req_pool_indices.numel() * num_current_tokens_per_req:
        raise ValueError(
            "Current speculative K/V rows must equal "
            "batch_size * num_current_tokens_per_req."
        )
    if current_locs.numel() != k_current.shape[0]:
        raise ValueError(
            "Speculative KV write locations must match the current K/V rows."
        )

    global _cached_num_workers
    if _cached_num_workers is None:
        _cached_num_workers = max(1, get_device_core_count())

    block_elements = triton.next_power_of_2(row_elements)
    _dequantize_nvfp4_kv_for_speculative_extend_kernel[(_cached_num_workers,)](
        k_fp4,
        v_fp4,
        k_block_scales,
        v_block_scales,
        k_global_scale,
        v_global_scale,
        k_current,
        v_current,
        current_locs,
        dq_k,
        dq_v,
        req_to_token,
        req_pool_indices,
        prefix_lens,
        req_pool_indices.numel(),
        num_current_tokens_per_req,
        prefix_len_delta,
        req_to_token.stride(0),
        k_current.stride(0),
        v_current.stride(0),
        ROW_ELEMENTS=row_elements,
        BLOCK_ELEMENTS=block_elements,
        num_warps=8 if block_elements >= 1024 else 4,
    )
