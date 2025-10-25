## pytorch sdpa version of block sparse ##
import torch
import triton
import triton.language as tl


@triton.jit
def topk_index_to_map_kernel(
    map_ptr,
    index_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    topk,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = (
        index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    )
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    for i in tl.static_range(topk):
        index = tl.load(index_ptr_base + i * index_kv_stride)
        tl.store(map_ptr_base + index * map_kv_stride, 1.0)


@triton.jit
def map_to_index_kernel(
    map_ptr,
    index_ptr,
    index_num_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    index_num_bs_stride,
    index_num_h_stride,
    index_num_q_stride,
    num_kv_blocks,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = (
        index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    )
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    num = 0
    for i in tl.range(num_kv_blocks):
        map_entry = tl.load(map_ptr_base + i * map_kv_stride)
        if map_entry:
            tl.store(index_ptr_base + num * index_kv_stride, i)
            num += 1

    tl.store(
        index_num_ptr
        + b * index_num_bs_stride
        + h * index_num_h_stride
        + q * index_num_q_stride,
        num,
    )


def topk_index_to_map(
    index: torch.Tensor, num_kv_blocks: int, transpose_map: bool = False
):
    """
    Convert topk indices to a map.

    Args:
        index: [bs, h, num_q_blocks, topk]
            The topk indices tensor.
        num_kv_blocks: int
            The number of key-value blocks in the block_map returned
        transpose_map: bool
            If True, the block_map will be transposed on the final two dimensions.

    Returns:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            A binary map where 1 indicates that the q block attends to the kv block.
    """
    bs, h, num_q_blocks, topk = index.shape

    if transpose_map is False:
        block_map = torch.zeros(
            (bs, h, num_q_blocks, num_kv_blocks), dtype=torch.bool, device=index.device
        )
    else:
        block_map = torch.zeros(
            (bs, h, num_kv_blocks, num_q_blocks), dtype=torch.bool, device=index.device
        )
        block_map = block_map.transpose(2, 3)

    grid = (bs, h, num_q_blocks)
    topk_index_to_map_kernel[grid](
        block_map,
        index,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        topk=topk,
    )

    return block_map


def map_to_index(block_map: torch.Tensor):
    """
    Convert a block map to indices and counts.

    Args:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            The block map tensor.

    Returns:
        index: [bs, h, num_q_blocks, num_kv_blocks]
            The indices of the blocks.
        index_num: [bs, h, num_q_blocks]
            The number of blocks for each q block.
    """
    bs, h, num_q_blocks, num_kv_blocks = block_map.shape

    index = torch.full(
        (block_map.shape), -1, dtype=torch.int32, device=block_map.device
    )
    index_num = torch.empty(
        (bs, h, num_q_blocks), dtype=torch.int32, device=block_map.device
    )

    grid = (bs, h, num_q_blocks)
    map_to_index_kernel[grid](
        block_map,
        index,
        index_num,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        index_num.stride(0),
        index_num.stride(1),
        index_num.stride(2),
        num_kv_blocks=num_kv_blocks,
    )

    return index, index_num
