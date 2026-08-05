"""NPU speculative SSM-state commit with destination-stride support.

The persistent K3 SSM state is a transposed view, so its trailing dimensions
must be addressed with their real strides rather than as a contiguous H/V/K
tensor.  Keep this source-local implementation until the same fix is available
from the installed ``sgl_kernel_npu`` package.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _move_intermediate_cache_kernel(
    dst_cache_ptr,
    src_cache_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    last_steps_ptr,
    src_layer_stride,
    src_size_stride,
    src_draft_stride,
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
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    valid_id = tl.program_id(0)
    dst_idx = tl.load(dst_indices_ptr + valid_id)
    src_idx = tl.load(src_indices_ptr + valid_id)
    last_step = tl.load(last_steps_ptr + valid_id)
    if last_step < 0:
        return

    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_K)

    for layer in range(num_layers):
        src_base = (
            src_cache_ptr
            + tl.cast(layer, tl.int64) * src_layer_stride
            + tl.cast(src_idx, tl.int64) * src_size_stride
            + tl.cast(last_step, tl.int64) * src_draft_stride
        )
        dst_base = (
            dst_cache_ptr
            + tl.cast(layer, tl.int64) * dst_layer_stride
            + tl.cast(dst_idx, tl.int64) * dst_size_stride
        )

        for h_start in range(0, h_dim, H_BLOCK_SIZE):
            h_real = h_start + h_offsets
            h_mask = h_real < h_dim
            k_mask = k_offsets < dim_k

            for v_start in range(0, dim_v, BLOCK_V):
                v_offsets = v_start + tl.arange(0, BLOCK_V)
                v_mask = v_offsets < dim_v
                mask = (
                    h_mask[:, None, None]
                    & v_mask[None, :, None]
                    & k_mask[None, None, :]
                )

                src_offset = (
                    h_real[:, None, None] * dim_v * dim_k
                    + v_offsets[None, :, None] * dim_k
                    + k_offsets[None, None, :]
                )
                dst_offset = (
                    h_real[:, None, None] * dst_h_stride
                    + v_offsets[None, :, None] * dst_v_stride
                    + k_offsets[None, None, :] * dst_k_stride
                )
                values = tl.load(src_base + src_offset, mask=mask, other=0)
                tl.store(dst_base + dst_offset, values, mask=mask)


def move_intermediate_cache(
    ssm_states: torch.Tensor,
    intermediate_state_cache: torch.Tensor,
    dst_indices_tensor: torch.Tensor,
    src_indices_tensor: torch.Tensor,
    last_steps_tensor: torch.Tensor,
    h_block_size: int = 1,
) -> torch.Tensor:
    """Commit selected speculative SSM snapshots to the persistent cache."""
    num_layers, _, _, h_dim, dim_v, dim_k = intermediate_state_cache.shape
    src_strides = intermediate_state_cache.stride()
    dst_strides = ssm_states.stride()

    assert len(dst_indices_tensor) == len(last_steps_tensor)
    assert len(src_indices_tensor) == len(last_steps_tensor)

    _move_intermediate_cache_kernel[(len(dst_indices_tensor),)](
        dst_cache_ptr=ssm_states,
        src_cache_ptr=intermediate_state_cache,
        dst_indices_ptr=dst_indices_tensor,
        src_indices_ptr=src_indices_tensor,
        last_steps_ptr=last_steps_tensor,
        src_layer_stride=int(src_strides[0]),
        src_size_stride=int(src_strides[1]),
        src_draft_stride=int(src_strides[2]),
        dst_layer_stride=int(dst_strides[0]),
        dst_size_stride=int(dst_strides[1]),
        dst_h_stride=int(dst_strides[2]),
        dst_v_stride=int(dst_strides[3]),
        dst_k_stride=int(dst_strides[4]),
        h_dim=h_dim,
        dim_v=dim_v,
        dim_k=dim_k,
        num_layers=num_layers,
        H_BLOCK_SIZE=h_block_size,
        BLOCK_V=64,
        BLOCK_K=triton.next_power_of_2(dim_k),
    )
    return ssm_states
