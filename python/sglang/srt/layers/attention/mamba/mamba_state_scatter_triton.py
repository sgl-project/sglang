"""
Fused Triton kernel for Mamba state scatter operations.

This kernel replaces the expensive advanced indexing operations in
`update_mamba_state_after_mtp_verify` with a single fused gather-scatter kernel,
avoiding multiple `index_elementwise_kernel` launches.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mamba_state_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    step_indices_raw_ptr,  # [total_requests] - accepted_steps or mamba_steps_to_track
    # Total number of requests
    total_requests,
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused gather-scatter kernel with built-in masking.

    This kernel fuses the index_select operations by:
    1. Iterating over all requests (pid_req from 0 to total_requests-1)
    2. Checking if step_indices_raw[pid_req] >= 0 (valid mask)
    3. If valid, performing the scatter:
       dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :]

    Grid: (total_requests, num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    # Load step index to check validity (step >= 0 means valid)
    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)

    # Early exit if this request is not valid (step < 0)
    if step_idx < 0:
        return

    # Load destination index
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)

    # Source index is just the request index itself
    src_idx = pid_req

    # Bounds check to avoid illegal memory access
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    # Compute base offsets
    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    # Compute element range for this block
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_per_entry

    # Load from source and store to destination
    data = tl.load(src_ptr + src_offset + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw indices (e.g., state_indices_tensor)
    step_indices_raw: torch.Tensor,  # [total_requests] - raw step indices (step >= 0 means valid)
):
    """
    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

    Args:
        dst: Destination tensor [num_layers, cache_size, *state_shape]
        src: Source tensor [num_layers, spec_size, draft_tokens, *state_shape]
        dst_indices_raw: Raw destination indices for all requests [total_requests]
        step_indices_raw: Raw step indices; entry >= 0 means valid [total_requests]
    """
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return

    if dst.device != src.device:
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )
    if not dst.is_cuda or not src.is_cuda:
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA tensors."
        )
    if dst.ndim < 2 or src.ndim < 3:
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")
    if dst.shape[0] != src.shape[0]:
        raise ValueError(
            f"Layer dimension mismatch: {dst.shape[0]=} vs {src.shape[0]=}"
        )
    if dst.shape[2:] != src.shape[3:]:
        raise ValueError(
            f"Trailing dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}"
        )
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )

    num_layers = dst.shape[0]
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]

    # Flatten trailing dimensions: number of elements per (layer, cache_line) entry.
    elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])

    # Get strides (in elements, not bytes)
    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)

    # Ensure indices are int32 and contiguous
    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()

    # Ensure tensors are contiguous
    if not dst.is_contiguous():
        raise ValueError("dst tensor must be contiguous")
    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")

    # Block size for copying elements
    BLOCK_SIZE = 1024

    # Grid over all requests - invalid ones will early-exit in the kernel
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))

    _fused_mamba_state_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
        total_requests,
        elem_per_entry,
        src_layer_stride,
        src_req_stride,
        src_step_stride,
        dst_layer_stride,
        dst_req_stride,
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# todo: move to sgl_kernel_npu
from sgl_kernel_npu.utils.triton_utils import get_device_properties

MAX_ROWS_PER_ITER = 64


@triton.jit(do_not_specialize=["total_rows", "rows_per_vec"])
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    total_rows,
    rows_per_vec,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_ROW_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    BA_OUT_ROW_STRIDE: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    """
    Fused kernel to split and reshape mixed QKVZ and BA tensors.
    This kernel performs the following transformations:
    - Input mixed_qkvz: [num_tokens, num_heads_qk * (Q + K + V + Z)] where each
      head block contains [Q(HEAD_QK), K(HEAD_QK), V(V_DIM_PER_QK), Z(V_DIM_PER_QK)]
    - Input mixed_ba: [num_tokens, num_heads_qk * (B + A)] where each head block
      contains [B(V_HEADS_PER_QK), A(V_HEADS_PER_QK)]
    - Output mixed_qkv: [num_tokens, Q_all | K_all | V_all] concatenated by type
    - Output z: [num_tokens, num_heads_v, head_v]
    - Output b, a: [num_tokens, num_heads_v]
    """
    # Each vector core processes a contiguous chunk of rows
    vec_id = tl.program_id(0)

    V_HEADS_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    V_DIM_PER_QK: tl.constexpr = V_HEADS_PER_QK * HEAD_V
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_DIM_PER_QK * 2
    BA_DIM_T: tl.constexpr = V_HEADS_PER_QK * 2

    Q_TOTAL: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    K_TOTAL: tl.constexpr = NUM_HEADS_QK * HEAD_QK

    row_start = vec_id * rows_per_vec
    row_end = min(row_start + rows_per_vec, total_rows)

    row_offset = row_start

    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    # ========== Main Iteration Loop ==========
    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        # ========== Head Iteration Loop ==========
        # Iterate over each Q/K head group to extract and rearrange data
        for head_id in tl.static_range(NUM_HEADS_QK):
            # Byte offset to the current head's data block in mixed_qkvz
            src_head_offset = head_id * QKVZ_DIM_T

            # ----- Q (Query) Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + 0:HEAD_QK]
            # Dest layout: mixed_qkv[row, head_id * HEAD_QK : (head_id+1) * HEAD_QK]
            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * QKV_ROW_STRIDE + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])

            # ----- K (Key) Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + HEAD_QK : +HEAD_QK]
            # Dest layout: mixed_qkv[row, Q_TOTAL + head_id * HEAD_QK : ...]
            # K is stored after Q in the source; in dest, K starts after all Q heads
            k_src = row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + HEAD_QK + q_range[None, :]
            k_dst = row_indices[:, None] * QKV_ROW_STRIDE + Q_TOTAL + head_id * HEAD_QK + q_range[None, :]
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])

            # ----- V (Value) Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + HEAD_QK*2 : +V_DIM_PER_QK]
            # Dest layout: mixed_qkv[row, Q_TOTAL + K_TOTAL + head_id * V_DIM_PER_QK : ...]
            # V follows Q and K in source; in dest, V starts after all Q and K heads
            v_range = tl.arange(0, V_DIM_PER_QK)
            v_src = row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = (
                row_indices[:, None] * QKV_ROW_STRIDE + Q_TOTAL + K_TOTAL + head_id * V_DIM_PER_QK + v_range[None, :]
            )
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])

            # ----- Z Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + HEAD_QK*2 + V_DIM_PER_QK : ...]
            # Dest layout: z[row, head_id * V_DIM_PER_QK : (head_id+1) * V_DIM_PER_QK]
            # Z follows V in source; output z is reshaped to [batch, num_heads_v, head_v]
            z_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + HEAD_QK * 2 + V_DIM_PER_QK + v_range[None, :]
            )
            z_dst = row_indices[:, None] * Z_ROW_STRIDE + head_id * V_DIM_PER_QK + v_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])

            # ----- B Extraction -----
            # Source layout: mixed_ba[row, head_id * BA_DIM_T : +V_HEADS_PER_QK]
            # Dest layout: b[row, head_id * V_HEADS_PER_QK : (head_id+1) * V_HEADS_PER_QK]
            b_range = tl.arange(0, V_HEADS_PER_QK)
            ba_head_offset = head_id * BA_DIM_T
            b_src = row_indices[:, None] * BA_ROW_STRIDE + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * BA_OUT_ROW_STRIDE + head_id * V_HEADS_PER_QK + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])

            # ----- A Extraction -----
            # Source layout: mixed_ba[row, head_id * BA_DIM_T + V_HEADS_PER_QK : ...]
            # Dest layout: a[row, head_id * V_HEADS_PER_QK : ...] (same as b_dst)
            # A follows B in source; output layout is same as B
            a_src = row_indices[:, None] * BA_ROW_STRIDE + ba_head_offset + V_HEADS_PER_QK + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


def fused_qkvzba_split_reshape_cat_npu(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvz.shape[0], 1
    total_rows = batch * seq_len

    v_heads_per_qk = num_heads_v // num_heads_qk
    v_dim_per_qk = v_heads_per_qk * head_v
    qkvz_dim_t = head_qk * 2 + v_dim_per_qk * 2
    ba_dim_t = v_heads_per_qk * 2

    # row stride
    qkvz_row_stride = num_heads_qk * qkvz_dim_t
    ba_row_stride = num_heads_qk * ba_dim_t
    qkv_row_stride = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    z_row_stride = num_heads_v * head_v
    ba_out_row_stride = num_heads_v
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )

    num_vectorcore = get_device_properties()[0]

    grid_size = min(num_vectorcore, total_rows)
    grid_size = max(1, grid_size)

    rows_per_vec = triton.cdiv(total_rows, grid_size)

    ub_size = 85 * 1024 // mixed_qkvz.element_size()

    elements_per_row = qkvz_row_stride + ba_row_stride + qkv_row_stride + z_row_stride + ba_out_row_stride * 2

    rows_per_iter = max(1, ub_size // elements_per_row)
    rows_per_iter = triton.next_power_of_2(rows_per_iter)
    rows_per_iter = min(rows_per_iter, rows_per_vec, MAX_ROWS_PER_ITER)

    grid = (grid_size, 1)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        total_rows,
        rows_per_vec,
        qkvz_row_stride,
        ba_row_stride,
        qkv_row_stride,
        z_row_stride,
        ba_out_row_stride,
        rows_per_iter,
    )
    return mixed_qkv, z, b, a


@triton.jit
def move_cache_dynamic_last_kernel_h_block(
    dst_cache_ptr,
    src_cache_ptr,
    valid_indices_ptr,
    last_steps_ptr,
    layer_stride,
    size_stride,
    draft_stride,
    dst_layer_stride,
    dst_size_stride,
    h_dim,
    dim_v,
    dim_k,
    num_layers,
    H_BLOCK_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,  # dim_v 的块大小
    BLOCK_K: tl.constexpr,  # dim_k 的块大小
):
    valid_id = tl.program_id(0)

    # 加载真实的索引值
    valid_idx_val = tl.load(valid_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return
    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    v_offsets = tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, BLOCK_K)

    # 循环处理 layer
    for l in range(num_layers):
        src_base_addr = src_cache_ptr + tl.cast(l, tl.int64) * layer_stride + tl.cast(valid_idx_val,
                                                                                      tl.int64) * size_stride
        dst_base_addr = dst_cache_ptr + tl.cast(l, tl.int64) * dst_layer_stride + tl.cast(valid_idx_val,
                                                                                          tl.int64) * dst_size_stride
        src_addr = src_base_addr + tl.cast(last_step_val, tl.int64) * draft_stride
        # --- 循环处理 h 维度 ---
        for h_start in range(0, h_dim, H_BLOCK_SIZE):
            # if l == 0:
            #     tl.device_print("h_start", h_start)
            h_real = h_start + h_offsets  # [h_start, h_start+1, ...]
            h_mask = h_real < h_dim  # 1D mask

            v_mask = v_offsets < dim_v
            k_mask = k_offsets < dim_k

            mask = h_mask[:, None, None] & v_mask[None, :, None] & k_mask[None, None, :]

            # linear_offset = h * dim_v * dim_k + v * dim_k + k
            linear_offset = h_real[:, None, None] * dim_v * dim_k + \
                            v_offsets[None, :, None] * dim_k + \
                            k_offsets[None, None, :]

            src_block = tl.load(src_addr + linear_offset, mask=mask, other=0)

            tl.store(dst_base_addr + linear_offset, src_block, mask=mask)


def move_intermediate_cache_dynamic_h_block(ssm_states, intermediate_state_cache, valid_tensor, last_steps_tensor,
                                            h_block_size=2):
    L, S, D, H, V, K = intermediate_state_cache.shape

    strides = intermediate_state_cache.stride()
    layer_stride, size_stride, draft_stride = int(strides[0]), int(strides[1]), int(strides[2])
    dst_layer_stride, dst_size_stride = int(ssm_states.stride()[0]), int(ssm_states.stride()[1])
    assert len(valid_tensor) == len(last_steps_tensor), "长度必须一致"

    # Grid[0]: 每个 valid_idx 一个线程
    grid = (len(valid_tensor),)

    move_cache_dynamic_last_kernel_h_block[grid](
        dst_cache_ptr=ssm_states,
        src_cache_ptr=intermediate_state_cache,
        valid_indices_ptr=valid_tensor,
        last_steps_ptr=last_steps_tensor,
        layer_stride=layer_stride,
        size_stride=size_stride,
        draft_stride=draft_stride,
        dst_layer_stride=dst_layer_stride,
        dst_size_stride=dst_size_stride,
        h_dim=H,
        dim_v=V,
        dim_k=K,
        num_layers=L,
        H_BLOCK_SIZE=h_block_size,  # 每次处理 2 个 h
        BLOCK_V=triton.next_power_of_2(V),  # v 维度的块大小
        BLOCK_K=triton.next_power_of_2(K),  # k 维度的块大小
    )

    return intermediate_state_cache


@triton.jit
def move_cache_dynamic_last_kernel_h_block_v1(
    cache_ptr,
    valid_indices_ptr,
    last_steps_ptr,
    layer_stride,
    size_stride,
    draft_stride,
    h_dim,
    dim_v,
    dim_k,
    num_layers,
    H_BLOCK_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,  # dim_v 的块大小
    BLOCK_K: tl.constexpr,  # dim_k 的块大小
):
    """
    使用清晰的 3D 偏移计算方式。
    """
    valid_id = tl.program_id(0)

    # 加载真实的索引值
    valid_idx_val = tl.load(valid_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return
    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    v_offsets = tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, BLOCK_K)

    # 循环处理 layer
    for l in range(num_layers):
        base_addr = cache_ptr + tl.cast(l, tl.int64) * layer_stride + tl.cast(valid_idx_val, tl.int64) * size_stride
        src_addr = base_addr + tl.cast(last_step_val, tl.int64) * draft_stride
        # --- 循环处理 h 维度 ---
        for h_start in range(0, h_dim, H_BLOCK_SIZE):
            # if l == 0:
            #     tl.device_print("h_start", h_start)
            h_real = h_start + h_offsets  # [h_start, h_start+1, ...]
            h_mask = h_real < h_dim  # 1D mask

            v_mask = v_offsets < dim_v
            k_mask = k_offsets < dim_k

            # --- 组合 3D 掩码 ---
            mask = h_mask[:, None, None] & v_mask[None, :, None] & k_mask[None, None, :]

            # --- 计算线性偏移 (核心公式) ---
            # linear_offset = h * dim_v * dim_k + v * dim_k + k
            linear_offset = h_real[:, None, None] * dim_v * dim_k + \
                            v_offsets[None, :, None] * dim_k + \
                            k_offsets[None, None, :]

            src_block = tl.load(src_addr + linear_offset, mask=mask, other=0)

            # 写入地址 (目标: 0)
            dst_addr = base_addr + 0 * draft_stride
            tl.store(dst_addr + linear_offset, src_block, mask=mask)


def move_intermediate_cache_dynamic_h_block_v1(intermediate_state_cache, valid_tensor, last_steps_tensor,
                                               h_block_size=2):
    """
    封装函数：支持 H 维度分块的版本。
    """
    L, S, D, H, V, K = intermediate_state_cache.shape

    # 1. 获取 Stride
    strides = intermediate_state_cache.stride()
    layer_stride, size_stride, draft_stride = int(strides[0]), int(strides[1]), int(strides[2])
    assert len(valid_tensor) == len(last_steps_tensor), "长度必须一致"

    # 3. 配置 Grid
    # Grid[0]: 每个 valid_idx 一个线程
    grid = (len(valid_tensor),)
    # print(f"{intermediate_state_cache.shape}, {valid_tensor=}, {last_steps_tensor=}")
    # 4. 启动 Kernel
    # 注意：这里我们不再需要 pid_h 和 pid_v 作为 program_id
    # 所有的分块逻辑都在 Kernel 内部通过循环实现
    move_cache_dynamic_last_kernel_h_block_v1[grid](
        cache_ptr=intermediate_state_cache,
        valid_indices_ptr=valid_tensor,
        last_steps_ptr=last_steps_tensor,
        layer_stride=layer_stride,
        size_stride=size_stride,
        draft_stride=draft_stride,
        h_dim=H,
        dim_v=V,
        dim_k=K,
        num_layers=L,
        H_BLOCK_SIZE=h_block_size,  # 每次处理 2 个 h
        BLOCK_V=triton.next_power_of_2(V),  # v 维度的块大小 (这里假设 V 不大，一次性处理)
        BLOCK_K=triton.next_power_of_2(K),  # k 维度的块大小
    )

    return intermediate_state_cache


@triton.jit
def _conv_state_rollback_kernel(
    conv_states_ptr,
    state_indices_ptr,
    step_indices_ptr,
    draft_token_num,
    num_layers,
    num_dims: tl.constexpr,
    conv_window_size: tl.constexpr,
    layer_stride: tl.constexpr,
    req_stride: tl.constexpr,
    window_stride: tl.constexpr,
    dim_stride: tl.constexpr,
):
    """
    Triton kernel for rolling back conv states after MTP verification.

    Args:
        conv_states_ptr: Pointer to conv states tensor [num_layers, pool_size, conv_window_size, num_dims]
        state_indices_ptr: Pointer to state indices [num_requests]
        step_indices_ptr: Pointer to step indices (accepted steps) [num_requests]
        draft_token_num: Number of draft tokens
        num_layers: Number of layers
        num_dims: Number of dimensions
        conv_window_size: Convolution window size
        layer_stride: Stride for layer dimension
        req_stride: Stride for request dimension
        window_stride: Stride for window dimension
        dim_stride: Stride for dimension dimension
    """
    pid_req = tl.program_id(0)

    # Load state index and step index
    state_idx = tl.load(state_indices_ptr + pid_req).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + pid_req).to(tl.int64)

    if step_idx < 0:
        return
    # Calculate shift
    shift = (draft_token_num - 1) - step_idx

    # Early exit if no rollback needed
    if shift <= 0:
        return

    # Generate dim offsets once (0 to num_dims-1)
    dim_offsets = tl.arange(0, num_dims)

    # Process each layer
    for layer in range(num_layers):
        # Calculate base offset for this request and layer
        base_offset = state_idx * req_stride + layer * layer_stride

        # Process each window position that needs to be moved
        # Move data from [0, conv_window_size-shift) to [shift, conv_window_size)
        for window_idx1 in range(0, conv_window_size - shift):
            window_idx = conv_window_size - shift - 1 - window_idx1
            # if layer == 1:
            #     tl.device_print("window_idx", window_idx)
            #     tl.device_print("shift", shift)
            # Calculate source pointer (beginning part)
            src_offset = base_offset + window_idx * window_stride + dim_offsets * dim_stride
            src_ptr = conv_states_ptr + src_offset

            # Calculate destination pointer (shifted part)
            dst_offset = base_offset + (window_idx + shift) * window_stride + dim_offsets * dim_stride
            dst_ptr = conv_states_ptr + dst_offset

            # Load all dims at once (dim is innermost, so it's continuous)
            data = tl.load(src_ptr)
            # Store all dims at once
            tl.store(dst_ptr, data)


def conv_state_rollback(
    conv_states: torch.Tensor,  # [num_layers, pool_size, conv_window_size, num_dims]
    state_indices: torch.Tensor,  # [num_requests]
    step_indices: torch.Tensor,  # [num_requests]
    draft_token_num: int,
):
    """
    Roll back conv states after MTP verification using Triton kernel.

    Args:
        conv_states: Conv states tensor [num_layers, pool_size, conv_window_size, num_dims]
        state_indices: State indices for each request [num_requests]
        step_indices: Accepted steps for each request [num_requests]
        draft_token_num: Number of draft tokens
    """
    num_requests = state_indices.shape[0]
    if num_requests == 0:
        return

    if conv_states.ndim != 4:
        raise ValueError(f"conv_states must be 4D, got {conv_states.ndim}D")
    if state_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError("state_indices and step_indices must be 1D")
    if state_indices.shape[0] != step_indices.shape[0]:
        raise ValueError("state_indices and step_indices must have the same length")

    num_layers = conv_states.shape[0]
    pool_size = conv_states.shape[1]
    conv_window_size = conv_states.shape[2]
    num_dims = conv_states.shape[3]

    # Get strides (in elements, not bytes)
    layer_stride = conv_states.stride(0)  # Stride for layer dimension
    req_stride = conv_states.stride(1)  # Stride for request/pool_size dimension
    window_stride = conv_states.stride(2)  # Stride for window dimension
    dim_stride = conv_states.stride(3)  # Stride for dimension dimension

    # Ensure indices are int32 and contiguous
    state_indices = state_indices.to(torch.int32).contiguous()
    step_indices = step_indices.to(torch.int32).contiguous()

    # Ensure conv_states is contiguous
    if not conv_states.is_contiguous():
        # print("Warning: conv_states tensor is not contiguous, making it contiguous...")
        conv_states = conv_states.contiguous()

    # Grid over all requests
    grid = (num_requests,)

    _conv_state_rollback_kernel[grid](
        conv_states,
        state_indices,
        step_indices,
        draft_token_num,
        num_layers,
        num_dims,
        conv_window_size,
        layer_stride,
        req_stride,
        window_stride,
        dim_stride,
    )

    return conv_states
