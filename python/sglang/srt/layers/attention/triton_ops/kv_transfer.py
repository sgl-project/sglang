# Copyright 2023-2024 SGLang Team
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
"""
Triton kernels for gathering/scattering KV cache data between GPU and pinned CPU memory.

These kernels enable efficient KV cache transfers for disaggregated inference by:
1. Gathering scattered KV data from GPU into pinned CPU buffer (device -> host)
2. Scattering KV data from pinned CPU buffer to GPU KV cache (host -> device)

Primary API:
- gather_kv_to_pinned_all_layers(): Gather KV from GPU to pinned CPU (single kernel)
- scatter_kv_with_staging_all_layers(): Scatter KV from pinned CPU to GPU (single kernel)

Both kernels achieve ~100% of PCIe bandwidth with O(1) extra GPU memory overhead.
They process all layers in a single kernel launch using pointer tensors.

Data layout: HEAD-FIRST [num_heads, num_layers, 2, num_tokens, head_dim]
This layout allows easy head slicing for mixed-TP transfers.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kv_all_layers_kernel(
    # Pointers to all K buffers (tensor of uint64 pointers)
    k_data_ptrs,
    # Pointers to all V buffers (tensor of uint64 pointers)
    v_data_ptrs,
    # Slot indices to gather
    slot_indices_ptr,
    # Output pinned CPU buffer
    output_ptr,
    # Dimensions
    num_layers,
    num_tokens,
    head_dim: tl.constexpr,
    # Head slicing params
    head_start: tl.constexpr,
    num_heads_to_gather: tl.constexpr,
    # Source strides (in elements) - same for all layers
    src_slot_stride,
    src_head_stride,
    # Output layout strides (HEAD-FIRST)
    out_head_stride,  # = num_layers * 2 * num_tokens * head_dim
    out_layer_stride,  # = 2 * num_tokens * head_dim
    out_kv_stride,  # = num_tokens * head_dim
    out_token_stride,  # = head_dim
    # Block sizes
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    # Element size: 1 for fp8, 2 for fp16/bf16
    ELEM_BYTES: tl.constexpr = 2,
):
    """
    Gather KV data from ALL layers in a single kernel launch.

    Reads scattered from GPU KV cache, writes contiguous to pinned CPU.
    O(1) extra GPU memory. Dtype-agnostic (copies raw bytes).

    Grid: (num_heads_to_gather, num_layers * 2) - one program per (head, layer_kv)
    - program_id(0): head index (0 to num_heads_to_gather-1)
    - program_id(1): layer_kv index (0 to num_layers*2-1), where even=K, odd=V
    """
    head_id = tl.program_id(0)
    layer_kv_id = tl.program_id(1)

    # Decode layer and K/V from combined index
    layer_id = layer_kv_id // 2
    is_v = layer_kv_id % 2  # 0 = K, 1 = V

    # Load the data pointer for this layer's K or V buffer
    # Use int8 or int16 based on element size for dtype-agnostic byte copying
    if ELEM_BYTES == 1:
        if is_v == 0:
            src_base_ptr = tl.load(k_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int8)
            )
        else:
            src_base_ptr = tl.load(v_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int8)
            )
    else:  # ELEM_BYTES == 2
        if is_v == 0:
            src_base_ptr = tl.load(k_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int16)
            )
        else:
            src_base_ptr = tl.load(v_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int16)
            )

    # Source head index (absolute in source KV cache)
    src_head = head_start + head_id

    # Cast strides to int64
    src_slot_stride_i64 = src_slot_stride.to(tl.int64)
    src_head_stride_i64 = src_head_stride.to(tl.int64)
    out_head_stride_i64 = out_head_stride.to(tl.int64)
    out_layer_stride_i64 = out_layer_stride.to(tl.int64)
    out_kv_stride_i64 = out_kv_stride.to(tl.int64)
    out_token_stride_i64 = out_token_stride.to(tl.int64)
    src_head_i64 = src_head.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)
    layer_id_i64 = layer_id.to(tl.int64)
    is_v_i64 = is_v.to(tl.int64)

    # Base output offset for this (head, layer, kv)
    out_base = (
        head_id_i64 * out_head_stride_i64
        + layer_id_i64 * out_layer_stride_i64
        + is_v_i64 * out_kv_stride_i64
    )

    # Process tokens in blocks
    for token_block_start in range(0, num_tokens, BLOCK_TOKENS):
        token_offsets = token_block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < num_tokens
        token_offsets_i64 = token_offsets.to(tl.int64)

        # Load slot indices for these tokens
        slot_ids = tl.load(slot_indices_ptr + token_offsets, mask=token_mask, other=0)
        slot_ids_i64 = slot_ids.to(tl.int64)

        # Process head_dim in blocks
        for dim_start in range(0, head_dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < head_dim
            dim_offsets_i64 = dim_offsets.to(tl.int64)

            # Compute source addresses in GPU KV cache
            src_offsets = (
                slot_ids_i64[:, None] * src_slot_stride_i64
                + src_head_i64 * src_head_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Load from GPU source
            mask = token_mask[:, None] & dim_mask[None, :]
            data = tl.load(src_base_ptr + src_offsets, mask=mask, other=0.0)

            # Compute output addresses
            out_offsets = (
                out_base
                + token_offsets_i64[:, None] * out_token_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Store to output buffer
            tl.store(output_ptr + out_offsets, data, mask=mask)


def gather_kv_to_pinned_all_layers(
    k_data_ptrs: torch.Tensor,  # [num_layers] uint64 tensor of K buffer pointers
    v_data_ptrs: torch.Tensor,  # [num_layers] uint64 tensor of V buffer pointers
    slot_indices: torch.Tensor,  # [num_tokens] on GPU
    pinned_output: torch.Tensor,  # pinned CPU buffer
    head_start: int,
    num_heads_to_gather: int,
    num_layers: int,
    head_dim: int,
    src_slot_stride: int,  # stride between slots in source (num_heads * head_dim)
    src_head_stride: int,  # stride between heads in source (head_dim)
    kv_elem_bytes: int = None,  # element size of KV cache (1 for fp8, 2 for fp16/bf16)
) -> None:
    """
    Gather KV data from ALL layers using a SINGLE kernel launch.

    O(1) extra GPU memory - writes directly to pinned CPU memory.

    Args:
        k_data_ptrs: Tensor of uint64 pointers to each layer's K buffer
        v_data_ptrs: Tensor of uint64 pointers to each layer's V buffer
        slot_indices: Tensor of slot indices to gather
        pinned_output: Pinned CPU buffer to write to
        head_start: First head index to gather
        num_heads_to_gather: Number of heads to gather
        num_layers: Number of layers
        head_dim: Dimension of each head
        src_slot_stride: Stride between slots in source buffers (num_heads * head_dim)
        src_head_stride: Stride between heads in source buffers (head_dim)
        kv_elem_bytes: Element size of KV cache in bytes. Must match pinned_output.element_size().

    Output layout: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim] (HEAD-FIRST)
    """
    assert pinned_output.is_pinned(), "Output buffer must be pinned CPU memory"
    assert slot_indices.is_cuda, "slot_indices must be on GPU"
    assert k_data_ptrs.dtype == torch.uint64, "k_data_ptrs must be uint64"
    assert v_data_ptrs.dtype == torch.uint64, "v_data_ptrs must be uint64"

    # Validate element size consistency between KV cache and pinned buffer
    pinned_elem_bytes = pinned_output.element_size()
    if kv_elem_bytes is not None:
        assert pinned_elem_bytes == kv_elem_bytes, (
            f"KV cache element size ({kv_elem_bytes} bytes) does not match "
            f"pinned buffer element size ({pinned_elem_bytes} bytes). "
            f"The pinned buffer dtype must match the KV cache dtype."
        )

    num_tokens = slot_indices.shape[0]

    # Block sizes
    BLOCK_TOKENS = 64
    BLOCK_DIM = min(64, triton.next_power_of_2(head_dim))

    # HEAD-FIRST output layout strides (in elements)
    out_head_stride = num_layers * 2 * num_tokens * head_dim
    out_layer_stride = 2 * num_tokens * head_dim
    out_kv_stride = num_tokens * head_dim
    out_token_stride = head_dim

    # Grid: (num_heads_to_gather, num_layers * 2)
    grid = (num_heads_to_gather, num_layers * 2)

    # Element size in bytes (1 for fp8, 2 for fp16/bf16)
    elem_bytes = pinned_output.element_size()

    # View as int8/int16 for dtype-agnostic byte copying
    # This ensures Triton pointer arithmetic matches our load/store types
    if elem_bytes == 1:
        output_view = pinned_output.view(torch.int8)
    else:
        output_view = pinned_output.view(torch.int16)

    _gather_kv_all_layers_kernel[grid](
        k_data_ptrs,
        v_data_ptrs,
        slot_indices,
        output_view,
        num_layers=num_layers,
        num_tokens=num_tokens,
        head_dim=head_dim,
        head_start=head_start,
        num_heads_to_gather=num_heads_to_gather,
        src_slot_stride=src_slot_stride,
        src_head_stride=src_head_stride,
        out_head_stride=out_head_stride,
        out_layer_stride=out_layer_stride,
        out_kv_stride=out_kv_stride,
        out_token_stride=out_token_stride,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_DIM=BLOCK_DIM,
        ELEM_BYTES=elem_bytes,
    )

    torch.cuda.synchronize()


@triton.jit
def _scatter_kv_all_layers_from_pinned_kernel(
    # Pointers to all K buffers (tensor of uint64 pointers)
    k_data_ptrs,
    # Pointers to all V buffers (tensor of uint64 pointers)
    v_data_ptrs,
    # Slot indices to scatter to
    slot_indices_ptr,
    # Input pinned CPU buffer (HEAD-FIRST layout)
    input_ptr,
    # Dimensions
    num_layers,
    num_tokens,
    head_dim: tl.constexpr,
    # Head slicing params
    head_start: tl.constexpr,
    num_heads_to_scatter: tl.constexpr,
    # Destination strides (in elements) - same for all layers
    dst_slot_stride,
    dst_head_stride,
    # Input layout strides (HEAD-FIRST: [num_heads, num_layers, 2, num_tokens, head_dim])
    in_head_stride,  # = num_layers * 2 * num_tokens * head_dim
    in_layer_stride,  # = 2 * num_tokens * head_dim
    in_kv_stride,  # = num_tokens * head_dim
    in_token_stride,  # = head_dim
    # Block sizes
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    # Element size: 1 for fp8, 2 for fp16/bf16
    ELEM_BYTES: tl.constexpr = 2,
):
    """
    Scatter KV data from pinned CPU to ALL GPU layers in a single kernel launch.

    Reads contiguous from pinned CPU, writes scattered to GPU KV cache.
    O(1) extra GPU memory. Dtype-agnostic (copies raw bytes).

    Grid: (num_heads_to_scatter, num_layers * 2) - one program per (head, layer_kv)
    - program_id(0): head index (0 to num_heads_to_scatter-1)
    - program_id(1): layer_kv index (0 to num_layers*2-1), where even=K, odd=V
    """
    head_id = tl.program_id(0)
    layer_kv_id = tl.program_id(1)

    # Decode layer and K/V from combined index
    layer_id = layer_kv_id // 2
    is_v = layer_kv_id % 2  # 0 = K, 1 = V

    # Load the data pointer for this layer's K or V buffer
    # Use int8 or int16 based on element size for dtype-agnostic byte copying
    if ELEM_BYTES == 1:
        if is_v == 0:
            dst_base_ptr = tl.load(k_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int8)
            )
        else:
            dst_base_ptr = tl.load(v_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int8)
            )
    else:  # ELEM_BYTES == 2
        if is_v == 0:
            dst_base_ptr = tl.load(k_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int16)
            )
        else:
            dst_base_ptr = tl.load(v_data_ptrs + layer_id).to(
                tl.pointer_type(tl.int16)
            )

    # Destination head index (absolute in destination KV cache)
    dst_head = head_start + head_id

    # Cast strides to int64
    dst_slot_stride_i64 = dst_slot_stride.to(tl.int64)
    dst_head_stride_i64 = dst_head_stride.to(tl.int64)
    in_head_stride_i64 = in_head_stride.to(tl.int64)
    in_layer_stride_i64 = in_layer_stride.to(tl.int64)
    in_kv_stride_i64 = in_kv_stride.to(tl.int64)
    in_token_stride_i64 = in_token_stride.to(tl.int64)
    dst_head_i64 = dst_head.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)
    layer_id_i64 = layer_id.to(tl.int64)
    is_v_i64 = is_v.to(tl.int64)

    # Base input offset for this (head, layer, kv) in HEAD-FIRST layout
    in_base = (
        head_id_i64 * in_head_stride_i64
        + layer_id_i64 * in_layer_stride_i64
        + is_v_i64 * in_kv_stride_i64
    )

    # Process tokens in blocks
    for token_block_start in range(0, num_tokens, BLOCK_TOKENS):
        token_offsets = token_block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < num_tokens
        token_offsets_i64 = token_offsets.to(tl.int64)

        # Load slot indices for these tokens
        slot_ids = tl.load(slot_indices_ptr + token_offsets, mask=token_mask, other=0)
        slot_ids_i64 = slot_ids.to(tl.int64)

        # Process head_dim in blocks
        for dim_start in range(0, head_dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < head_dim
            dim_offsets_i64 = dim_offsets.to(tl.int64)

            # Compute input addresses in pinned CPU buffer (HEAD-FIRST layout)
            in_offsets = (
                in_base
                + token_offsets_i64[:, None] * in_token_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Load from pinned CPU buffer (zero-copy read over PCIe)
            mask = token_mask[:, None] & dim_mask[None, :]
            data = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)

            # Compute destination addresses in GPU KV cache (scattered writes)
            dst_offsets = (
                slot_ids_i64[:, None] * dst_slot_stride_i64
                + dst_head_i64 * dst_head_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Store to GPU KV cache
            tl.store(dst_base_ptr + dst_offsets, data, mask=mask)


def scatter_kv_with_staging_all_layers(
    pinned_input: torch.Tensor,
    k_data_ptrs: torch.Tensor,  # [num_layers] uint64 tensor of K buffer pointers
    v_data_ptrs: torch.Tensor,  # [num_layers] uint64 tensor of V buffer pointers
    slot_indices: torch.Tensor,
    head_start: int,
    num_heads_to_scatter: int,
    num_layers: int,
    head_dim: int,
    dst_slot_stride: int,  # stride between slots in dest (num_heads * head_dim)
    dst_head_stride: int,  # stride between heads in dest (head_dim)
    kv_elem_bytes: int = None,  # element size of KV cache (1 for fp8, 2 for fp16/bf16)
) -> None:
    """
    Scatter KV data to ALL layers using a SINGLE kernel launch.

    O(1) extra GPU memory - reads directly from pinned CPU memory.

    Args:
        pinned_input: Pinned CPU buffer in HEAD-FIRST layout
        k_data_ptrs: Tensor of uint64 pointers to each layer's K buffer
        v_data_ptrs: Tensor of uint64 pointers to each layer's V buffer
        slot_indices: Tensor of slot indices to scatter to
        head_start: First head index to scatter to
        num_heads_to_scatter: Number of heads to scatter
        num_layers: Number of layers
        head_dim: Dimension of each head
        dst_slot_stride: Stride between slots in dest buffers
        dst_head_stride: Stride between heads in dest buffers
        kv_elem_bytes: Element size of KV cache in bytes. Must match pinned_input.element_size().

    Input layout: [num_heads_to_scatter, num_layers, 2, num_tokens, head_dim] (HEAD-FIRST)
    """
    assert pinned_input.is_pinned(), "Input buffer must be pinned CPU memory"
    assert slot_indices.is_cuda, "slot_indices must be on GPU"
    assert k_data_ptrs.dtype == torch.uint64, "k_data_ptrs must be uint64"
    assert v_data_ptrs.dtype == torch.uint64, "v_data_ptrs must be uint64"

    # Validate element size consistency between KV cache and pinned buffer
    pinned_elem_bytes = pinned_input.element_size()
    if kv_elem_bytes is not None:
        assert pinned_elem_bytes == kv_elem_bytes, (
            f"KV cache element size ({kv_elem_bytes} bytes) does not match "
            f"pinned buffer element size ({pinned_elem_bytes} bytes). "
            f"The pinned buffer dtype must match the KV cache dtype."
        )

    num_tokens = slot_indices.shape[0]

    # Block sizes
    BLOCK_TOKENS = 64
    BLOCK_DIM = min(64, triton.next_power_of_2(head_dim))

    # HEAD-FIRST input layout: [num_heads_to_scatter, num_layers, 2, num_tokens, head_dim]
    # Strides for this layout (in elements):
    in_head_stride = num_layers * 2 * num_tokens * head_dim
    in_layer_stride = 2 * num_tokens * head_dim
    in_kv_stride = num_tokens * head_dim
    in_token_stride = head_dim

    # Grid: (num_heads_to_scatter, num_layers * 2)
    grid = (num_heads_to_scatter, num_layers * 2)

    # Element size in bytes (1 for fp8, 2 for fp16/bf16)
    elem_bytes = pinned_input.element_size()

    # View as int8/int16 for dtype-agnostic byte copying
    # This ensures Triton pointer arithmetic matches our load/store types
    if elem_bytes == 1:
        input_view = pinned_input.view(torch.int8)
    else:
        input_view = pinned_input.view(torch.int16)

    _scatter_kv_all_layers_from_pinned_kernel[grid](
        k_data_ptrs,
        v_data_ptrs,
        slot_indices,
        input_view,
        num_layers=num_layers,
        num_tokens=num_tokens,
        head_dim=head_dim,
        head_start=head_start,
        num_heads_to_scatter=num_heads_to_scatter,
        dst_slot_stride=dst_slot_stride,
        dst_head_stride=dst_head_stride,
        in_head_stride=in_head_stride,
        in_layer_stride=in_layer_stride,
        in_kv_stride=in_kv_stride,
        in_token_stride=in_token_stride,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_DIM=BLOCK_DIM,
        ELEM_BYTES=elem_bytes,
    )

    torch.cuda.synchronize()
