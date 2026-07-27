"""
Complete the following functions:
    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

follow gpu kernel: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _speculative_state_scatter_kernel(
    dst_ptr,
    src_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    step_indices_ptr,
    dst_layer_stride,
    dst_slot_stride,
    src_layer_stride,
    src_slot_stride,
    src_step_stride,
    dst_tail_stride_0,
    dst_tail_stride_1,
    dst_tail_stride_2,
    src_tail_stride_0,
    src_tail_stride_1,
    src_tail_stride_2,
    tail_numel,
    TAIL_DIM_1: tl.constexpr,
    TAIL_DIM_2: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    GRID_Y: tl.constexpr,
    GRID_Z: tl.constexpr,
    GRID_BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    for iii in range(pid, GRID, GRID_BLOCK):
        tmp0 = iii
        pid_tail = tmp0 % GRID_Z
        tmp0 //= GRID_Z
        pid_layer = tmp0 % GRID_Y
        tmp0 //= GRID_Y
        pid_req = tmp0

        # pid_req = tl.program_id(0)
        # pid_layer = tl.program_id(1)
        # pid_tail = tl.program_id(2)

        dst_idx = tl.load(dst_indices_ptr + pid_req).to(tl.int64)
        src_idx = tl.load(src_indices_ptr + pid_req).to(tl.int64)
        step_idx = tl.load(step_indices_ptr + pid_req).to(tl.int64)
        offsets = pid_tail * BLOCK + tl.arange(0, BLOCK)
        valid = (dst_idx >= 0) & (src_idx >= 0) & (step_idx >= 0)
        mask = valid & (offsets < tail_numel)
        tail_0 = offsets // (TAIL_DIM_1 * TAIL_DIM_2)
        tail_rem = offsets % (TAIL_DIM_1 * TAIL_DIM_2)
        tail_1 = tail_rem // TAIL_DIM_2
        tail_2 = tail_rem % TAIL_DIM_2

        src_offsets = (
            pid_layer * src_layer_stride
            + src_idx * src_slot_stride
            + step_idx * src_step_stride
            + tail_0 * src_tail_stride_0
            + tail_1 * src_tail_stride_1
            + tail_2 * src_tail_stride_2
        )
        dst_offsets = (
            pid_layer * dst_layer_stride
            + dst_idx * dst_slot_stride
            + tail_0 * dst_tail_stride_0
            + tail_1 * dst_tail_stride_1
            + tail_2 * dst_tail_stride_2
        )
        values = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + dst_offsets, values, mask=mask)


def speculative_state_scatter_npu(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> torch.Tensor:
    """Commit one speculative snapshot per request into a persistent cache.

    ``dst`` has shape ``[layers, dst_slots, *tail]`` and ``src`` has shape
    ``[layers, src_slots, steps, *tail]``. A negative value in any index
    tensor masks that request without a host synchronization.
    """
    if dst.ndim < 3:
        raise ValueError(f"dst must have at least 3 dimensions, got {dst.ndim}")
    if src.ndim != dst.ndim + 1:
        raise ValueError(
            f"src must have exactly one more dimension than dst, got "
            f"dst.ndim={dst.ndim}, src.ndim={src.ndim}"
        )
    if dst.shape[0] != src.shape[0] or tuple(dst.shape[2:]) != tuple(src.shape[3:]):
        raise ValueError(
            "dst and src must have matching layer and state-tail dimensions, "
            f"got dst={tuple(dst.shape)}, src={tuple(src.shape)}"
        )
    if dst.dtype != src.dtype:
        raise ValueError(
            f"dst and src dtypes must match, got {dst.dtype} and {src.dtype}"
        )
    if dst.device != src.device:
        raise ValueError("dst and src must be on the same device")
    tail_shape = tuple(dst.shape[2:])
    if not 1 <= len(tail_shape) <= 3:
        raise ValueError(
            f"state tail must have between 1 and 3 dimensions, got {tail_shape}"
        )
    if dst_indices.ndim != 1 or src_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError("all index tensors must be 1D")
    num_requests = dst_indices.numel()
    if src_indices.numel() != num_requests or step_indices.numel() != num_requests:
        raise ValueError("all index tensors must have the same length")
    if num_requests == 0:
        return dst
    if (
        dst_indices.device != dst.device
        or src_indices.device != dst.device
        or step_indices.device != dst.device
    ):
        raise ValueError("all index tensors must be on the cache device")

    dst_indices = dst_indices.to(torch.int32).contiguous()
    src_indices = src_indices.to(torch.int32).contiguous()
    step_indices = step_indices.to(torch.int32).contiguous()

    tail_numel = dst[0, 0].numel()
    padded_tail_shape = (1,) * (3 - len(tail_shape)) + tail_shape
    dst_tail_strides = (0,) * (3 - len(tail_shape)) + dst.stride()[2:]
    src_tail_strides = (0,) * (3 - len(tail_shape)) + src.stride()[3:]
    block = min(1024, triton.next_power_of_2(tail_numel))
    # grid = (num_requests, dst.shape[0], triton.cdiv(tail_numel, block))
    
    grid_x = num_requests
    grid_y = dst.shape[0]
    grid_z = triton.cdiv(tail_numel, block)
    total_grid = grid_x * grid_y * grid_z
    launch_grid = 48
    _speculative_state_scatter_kernel[(launch_grid,)](
        dst,
        src,
        dst_indices,
        src_indices,
        step_indices,
        dst.stride(0),
        dst.stride(1),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        *dst_tail_strides,
        *src_tail_strides,
        tail_numel,
        TAIL_DIM_1=padded_tail_shape[1],
        TAIL_DIM_2=padded_tail_shape[2],
        BLOCK=block,
        GRID=total_grid,
        GRID_Y=grid_y,
        GRID_Z=grid_z,
        GRID_BLOCK=launch_grid
    )
    return dst


@triton.jit
def move_cache_dynamic_last_kernel_h_block(
    dst_cache_ptr,
    src_cache_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    last_steps_ptr,
    layer_stride,
    size_stride,
    draft_stride,
    dst_layer_stride,
    dst_size_stride,
    dst_h_stride,
    dst_v_stride,
    dst_k_stride,
    h_dim,
    dim_v,
    dim_k,
    num_layers,
    H_BLOCK_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,  # Block size for dim_v
    BLOCK_K: tl.constexpr,  # Block size for dim_k
):
    valid_id = tl.program_id(0)

    # Load actual indices
    dst_idx_val = tl.load(dst_indices_ptr + valid_id)
    src_idx_val = tl.load(src_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return
    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_K)

    # Process each layer
    for l in range(num_layers):
        src_base_addr = (
            src_cache_ptr
            + tl.cast(l, tl.int64) * layer_stride
            + tl.cast(src_idx_val, tl.int64) * size_stride
        )
        dst_base_addr = (
            dst_cache_ptr
            + tl.cast(l, tl.int64) * dst_layer_stride
            + tl.cast(dst_idx_val, tl.int64) * dst_size_stride
        )
        src_addr = src_base_addr + tl.cast(last_step_val, tl.int64) * draft_stride

        # Process h dimension in blocks
        for h_start in range(0, h_dim, H_BLOCK_SIZE):
            h_real = h_start + h_offsets
            h_mask = h_real < h_dim
            k_mask = k_offsets < dim_k

            # Split the V dimension into BLOCK_V-sized chunks (e.g. 64) to keep
            # the on-chip tile (H_BLOCK_SIZE * BLOCK_V * BLOCK_K) within budget.
            for v_start in range(0, dim_v, BLOCK_V):
                v_offsets = v_start + tl.arange(0, BLOCK_V)
                v_mask = v_offsets < dim_v

                mask = (
                    h_mask[:, None, None]
                    & v_mask[None, :, None]
                    & k_mask[None, None, :]
                )

                # src is contiguous in (H, V, K) -> flat offset.
                src_linear_offset = (
                    h_real[:, None, None] * dim_v * dim_k
                    + v_offsets[None, :, None] * dim_k
                    + k_offsets[None, None, :]
                )
                # dst uses its real per-element strides so a transposed
                # (e.g. NPU transpose(-1, -2)) layout is handled correctly.
                dst_linear_offset = (
                    h_real[:, None, None] * dst_h_stride
                    + v_offsets[None, :, None] * dst_v_stride
                    + k_offsets[None, None, :] * dst_k_stride
                )

                src_block = tl.load(src_addr + src_linear_offset, mask=mask, other=0)
                tl.store(
                    dst_base_addr + dst_linear_offset, src_block, mask=mask
                )


def move_intermediate_cache(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    """
    Move intermediate cache to SSM states using Triton kernel.

    Args:
        ssm_states: Destination SSM states tensor
        intermediate_state_cache: Source intermediate state cache
        dst_indices_tensor: Valid destination indices tensor
        src_indices_tensor: Valid source indices tensor
        last_steps_tensor: Last steps tensor
        h_block_size: Block size for h dimension processing
    """
    L, S, D, H, V, K = intermediate_state_cache.shape

    strides = intermediate_state_cache.stride()
    layer_stride, size_stride, draft_stride = (
        int(strides[0]),
        int(strides[1]),
        int(strides[2]),
    )
    dst_strides = ssm_states.stride()
    dst_layer_stride, dst_size_stride = int(dst_strides[0]), int(dst_strides[1])
    # Per-element strides for the trailing (H, V, K) dims. On NPU the temporal
    # state is transposed (-1, -2), so the dst layout differs from the src and
    # must be indexed through its real strides instead of a flat (V*K, K, 1).
    dst_h_stride, dst_v_stride, dst_k_stride = (
        int(dst_strides[2]),
        int(dst_strides[3]),
        int(dst_strides[4]),
    )
    assert len(dst_indices_tensor) == len(
        last_steps_tensor
    ), "Destination indices lengths must match"
    assert len(src_indices_tensor) == len(
        last_steps_tensor
    ), "Source indices lengths must match"

    # Grid: one thread per valid index
    grid = (len(dst_indices_tensor),)

    move_cache_dynamic_last_kernel_h_block[grid](
        dst_cache_ptr=ssm_states,
        src_cache_ptr=intermediate_state_cache,
        dst_indices_ptr=dst_indices_tensor,
        src_indices_ptr=src_indices_tensor,
        last_steps_ptr=last_steps_tensor,
        layer_stride=layer_stride,
        size_stride=size_stride,
        draft_stride=draft_stride,
        dst_layer_stride=dst_layer_stride,
        dst_size_stride=dst_size_stride,
        dst_h_stride=dst_h_stride,
        dst_v_stride=dst_v_stride,
        dst_k_stride=dst_k_stride,
        h_dim=H,
        dim_v=V,
        dim_k=K,
        num_layers=L,
        H_BLOCK_SIZE=h_block_size,  # Process 2 h elements per block
        BLOCK_V=64,  # Split dim_v into 64-wide chunks to fit on-chip memory
        BLOCK_K=triton.next_power_of_2(K),  # Block size for dim_k
    )

    return ssm_states


@triton.jit
def _conv_state_rollback_kernel_v2(
    conv_states_ptr,
    state_indices_ptr,
    step_indices_ptr,
    draft_token_num,
    num_dims,
    layer_stride,
    req_stride,
    window_stride,
    dim_stride,
    conv_window_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_dim = tl.program_id(2)

    state_idx = tl.load(state_indices_ptr + pid_req).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + pid_req).to(tl.int64)
    shift = (draft_token_num - 1) - step_idx
    offsets = pid_dim * BLOCK + tl.arange(0, BLOCK)
    valid_request = (state_idx >= 0) & (step_idx >= 0) & (shift > 0)
    mask_dim = valid_request & (offsets < num_dims)
    base = state_idx * req_stride + pid_layer * layer_stride

    # Copy right-to-left so source and destination may alias safely.
    for reverse_idx in tl.static_range(0, conv_window_size):
        src_window = conv_window_size - shift - 1 - reverse_idx
        active = (reverse_idx < conv_window_size - shift) & (src_window >= 0)
        src_offsets = base + src_window * window_stride + offsets * dim_stride
        dst_offsets = (
            base + (src_window + shift) * window_stride + offsets * dim_stride
        )
        values = tl.load(
            conv_states_ptr + src_offsets, mask=mask_dim & active, other=0.0
        )
        tl.store(conv_states_ptr + dst_offsets, values, mask=mask_dim & active)


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

    # Load state and step indices
    state_idx = tl.load(state_indices_ptr + pid_req).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + pid_req).to(tl.int64)

    if step_idx < 0:
        return

    # Calculate rollback shift
    shift = (draft_token_num - 1) - step_idx

    # Early exit if no rollback needed
    if shift <= 0:
        return

    # Generate dimension offsets once
    dim_offsets = tl.arange(0, num_dims)

    # Process each layer
    for layer in range(num_layers):
        # Calculate base offset for this request and layer
        base_offset = state_idx * req_stride + layer * layer_stride

        # Process each window position that needs to be moved
        # Move data from [0, conv_window_size-shift) to [shift, conv_window_size)
        for window_idx1 in range(0, conv_window_size - shift):
            window_idx = conv_window_size - shift - 1 - window_idx1

            # Calculate source and destination pointers
            src_offset = (
                base_offset + window_idx * window_stride + dim_offsets * dim_stride
            )
            src_ptr = conv_states_ptr + src_offset

            dst_offset = (
                base_offset
                + (window_idx + shift) * window_stride
                + dim_offsets * dim_stride
            )
            dst_ptr = conv_states_ptr + dst_offset

            # Load and store all dimensions at once
            data = tl.load(src_ptr)
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
    conv_window_size = conv_states.shape[2]
    num_dims = conv_states.shape[3]

    # Get strides (in elements, not bytes)
    layer_stride = conv_states.stride(0)
    req_stride = conv_states.stride(1)
    window_stride = conv_states.stride(2)
    dim_stride = conv_states.stride(3)

    # Ensure indices are int32 and contiguous
    state_indices = state_indices.to(torch.int32).contiguous()
    step_indices = step_indices.to(torch.int32).contiguous()

    if not conv_states.is_contiguous():
        raise ValueError("conv_states must be contiguous")

    block = min(1024, triton.next_power_of_2(num_dims))
    grid = (num_requests, num_layers, triton.cdiv(num_dims, block))

    _conv_state_rollback_kernel_v2[grid](
        conv_states,
        state_indices,
        step_indices,
        draft_token_num,
        num_dims,
        layer_stride,
        req_stride,
        window_stride,
        dim_stride,
        conv_window_size=conv_window_size,
        BLOCK=block,
    )

    return conv_states



device="npu"
import random
@torch.no_grad
def test_move_intermediate_cache(
    L: int,
    S: int,
    D: int,
    H: int,
    V: int,
    K: int,
    num_valid: int,
    dtype: torch.dtype,
    dst_transposed: bool = False,
):
    """Verify move_intermediate_cache and speculative_state_scatter_npu.

    Args:
        dst_transposed: If True, allocate dst contiguous then apply
            ``transpose(-1, -2)`` to mimic the NPU temporal_state layout
            (memory_pool.py does this for ``_is_npu``). This makes dst
            non-contiguous in (H, V, K) and is the case that exposed the
            ``move_intermediate_cache`` precision bug.
    """
    torch.manual_seed(42)
    # prepare input data
    dst_cache = torch.randn(L, S, H, V, K, device=device, dtype=dtype)
    if dst_transposed:
        dst_cache = dst_cache.transpose(-1, -2)  # logical (L, S, H, K, V)
    dst_cache_clone = dst_cache.clone()
    src_cache = torch.randn(L, S, D, H, V, K, device=device, dtype=dtype)

    # prepare input data
    population = range(S)
    valid_indices = random.sample(population, num_valid)
    last_step_pos = [random.randint(0, D - 1) for _ in range(num_valid)]
    dst_indices_tensor = torch.tensor(valid_indices, device=device, dtype=torch.int32)
    src_indices_tensor = torch.arange(
        dst_indices_tensor.shape[0], device=device, dtype=torch.int32
    )
    last_steps_tensor = torch.tensor(last_step_pos, device=device, dtype=torch.int32)

    valid_mask = last_steps_tensor >= 0
    dst_state_indices = dst_indices_tensor[valid_mask].to(torch.int64)
    src_state_indices = src_indices_tensor[valid_mask].to(torch.int64)
    valid_last_steps = last_steps_tensor[valid_mask].to(torch.int64)
    # prepare output verify (PyTorch ground truth, layout-agnostic)
    dst_cache[:, dst_state_indices, :] = src_cache[
        :, src_state_indices, valid_last_steps
    ]

    # Verify move_intermediate_cache against the reference on an isolated clone
    dst_move = dst_cache_clone.clone()
    move_intermediate_cache(
        dst_move,
        src_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
    )
    torch.testing.assert_close(dst_cache, dst_move, atol=1e-3, rtol=1e-3)

    # Verify speculative_state_scatter_npu against the reference on an isolated clone
    dst_scatter = dst_cache_clone.clone()
    speculative_state_scatter_npu(
        dst_scatter,
        src_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
    )
    torch.testing.assert_close(dst_cache, dst_scatter, atol=1e-3, rtol=1e-3)



# "69,17,8,6,128,128;1;1;1"
if __name__ == "__main__":
    # Contiguous dst layout (non-NPU): both kernels must pass.
    test_move_intermediate_cache(69, 17, 8, 6, 128, 128, 17, torch.bfloat16)
    # NPU transposed dst layout: reproduces the move_intermediate_cache bug
    # where dst stride was hard-coded as (V*K, K, 1) instead of read from the
    # transposed tensor.
    test_move_intermediate_cache(
        69, 17, 8, 6, 128, 128, 17, torch.bfloat16, dst_transposed=True
    )
    