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
    LOGICAL_GRID: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    TAIL_GRID: tl.constexpr,
    BLOCK: tl.constexpr,
    PHYSICAL_GRID: tl.constexpr,
):
    physical_pid = tl.program_id(0)
    for logical_pid in range(physical_pid, LOGICAL_GRID, PHYSICAL_GRID):
        tail_pid = logical_pid % TAIL_GRID
        layer_pid = (logical_pid // TAIL_GRID) % NUM_LAYERS
        request_pid = logical_pid // (TAIL_GRID * NUM_LAYERS)

        dst_idx = tl.load(dst_indices_ptr + request_pid).to(tl.int64)
        src_idx = tl.load(src_indices_ptr + request_pid).to(tl.int64)
        step_idx = tl.load(step_indices_ptr + request_pid).to(tl.int64)
        offsets = tail_pid * BLOCK + tl.arange(0, BLOCK)
        valid = (dst_idx >= 0) & (src_idx >= 0) & (step_idx >= 0)
        mask = valid & (offsets < tail_numel)

        tail_0 = offsets // (TAIL_DIM_1 * TAIL_DIM_2)
        tail_remainder = offsets % (TAIL_DIM_1 * TAIL_DIM_2)
        tail_1 = tail_remainder // TAIL_DIM_2
        tail_2 = tail_remainder % TAIL_DIM_2
        src_offsets = (
            layer_pid * src_layer_stride
            + src_idx * src_slot_stride
            + step_idx * src_step_stride
            + tail_0 * src_tail_stride_0
            + tail_1 * src_tail_stride_1
            + tail_2 * src_tail_stride_2
        )
        dst_offsets = (
            layer_pid * dst_layer_stride
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
    """Commit one speculative snapshot per request into an Ascend cache.

    ``dst`` is ``[layers, slots, *state]`` and ``src`` is
    ``[layers, scratch, steps, *state]``. Negative indices are masked on
    device, which keeps the operation graph-capture safe.
    """
    if dst.ndim < 3 or src.ndim != dst.ndim + 1:
        raise ValueError(
            f"invalid state ranks: dst.ndim={dst.ndim}, src.ndim={src.ndim}"
        )
    if dst.shape[0] != src.shape[0] or tuple(dst.shape[2:]) != tuple(src.shape[3:]):
        raise ValueError(
            "persistent and speculative state shapes do not match: "
            f"dst={tuple(dst.shape)}, src={tuple(src.shape)}"
        )
    if dst.dtype != src.dtype or dst.device != src.device:
        raise ValueError("persistent and speculative states must share dtype/device")
    if any(index.ndim != 1 for index in (dst_indices, src_indices, step_indices)):
        raise ValueError("state index tensors must be one-dimensional")

    num_requests = dst_indices.numel()
    if src_indices.numel() != num_requests or step_indices.numel() != num_requests:
        raise ValueError("state index tensors must have equal lengths")
    if num_requests == 0:
        return dst
    if any(
        index.device != dst.device
        for index in (dst_indices, src_indices, step_indices)
    ):
        raise ValueError("state indices must be on the cache device")

    tail_shape = tuple(dst.shape[2:])
    if not 1 <= len(tail_shape) <= 3:
        raise ValueError(f"state tail rank must be in [1, 3], got {tail_shape}")
    padded_tail_shape = (1,) * (3 - len(tail_shape)) + tail_shape
    dst_tail_strides = (0,) * (3 - len(tail_shape)) + dst.stride()[2:]
    src_tail_strides = (0,) * (3 - len(tail_shape)) + src.stride()[3:]
    tail_numel = dst[0, 0].numel()
    block = min(1024, triton.next_power_of_2(tail_numel))
    tail_grid = triton.cdiv(tail_numel, block)
    logical_grid = num_requests * dst.shape[0] * tail_grid
    physical_grid = min(48, logical_grid)

    _speculative_state_scatter_kernel[(physical_grid,)](
        dst,
        src,
        dst_indices.to(torch.int32).contiguous(),
        src_indices.to(torch.int32).contiguous(),
        step_indices.to(torch.int32).contiguous(),
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
        LOGICAL_GRID=logical_grid,
        NUM_LAYERS=dst.shape[0],
        TAIL_GRID=tail_grid,
        BLOCK=block,
        PHYSICAL_GRID=physical_grid,
    )
    return dst
