import itertools
from typing import Tuple

import pytest
import torch

LAYERS = 4
DTYPE = torch.float16
CUDA_SIZE = 512 * 1024
HOST_SIZE = 2 * CUDA_SIZE
IDTYPES = [torch.int32, torch.int64]


def make_ptrs(tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [tensor[i].data_ptr() for i in range(LAYERS)],
        dtype=torch.uint64,
        device="cuda",
    )


CUDA_KV = torch.empty((2, LAYERS, CUDA_SIZE, 1024), dtype=DTYPE, device="cuda")
HOST_KV = torch.empty((2, LAYERS, HOST_SIZE, 1024), dtype=DTYPE, pin_memory=True)
CUDA_KV_PTRS = make_ptrs(CUDA_KV[0]), make_ptrs(CUDA_KV[1])
HOST_KV_PTRS = make_ptrs(HOST_KV[0]), make_ptrs(HOST_KV[1])
STRIDE_BYTES = CUDA_KV.stride(-2) * DTYPE.itemsize

BS_LIST = [2**n for n in range(5, 13)]  # 32 to 4096
BS_LIST += [x + i + 1 for i, x in enumerate(BS_LIST)]  # some unaligned sizes
BLOCK_QUOTA = [1, 2]
WORLD_SIZE = [1, 2, 4, 8, 16]


def generate_indices(bs: int, size: int, dtype: torch.dtype) -> torch.Tensor:
    result = torch.randperm(size, dtype=dtype, device="cuda")[:bs].sort().values
    assert torch.all(0 <= result)
    assert torch.all(result < size)
    return result


def _test_hicache_correctness_one_layer(
    host_kv: torch.Tensor,
    cuda_kv: torch.Tensor,
    bs: int,
    element_dim: int,
    indices_dtype: torch.dtype,
    block_quota: int,
) -> None:
    from sglang.jit_kernel.hicache import transfer_hicache_one_layer

    host_indices = generate_indices(bs=bs, size=HOST_SIZE, dtype=indices_dtype)
    cuda_indices = generate_indices(bs=bs, size=CUDA_SIZE, dtype=indices_dtype)

    init_value = torch.randn((2, LAYERS, bs, element_dim), dtype=DTYPE)
    host_kv[:, :, host_indices.cpu(), :] = init_value
    for i in range(LAYERS):
        transfer_hicache_one_layer(
            k_cache_dst=cuda_kv[0, i],
            v_cache_dst=cuda_kv[1, i],
            indices_dst=cuda_indices,
            k_cache_src=host_kv[0, i],
            v_cache_src=host_kv[1, i],
            indices_src=host_indices,
            block_quota=block_quota,
        )
    post_value = cuda_kv[:, :, cuda_indices, :]
    assert torch.all(post_value == init_value.cuda())

    init_value = torch.randn((2, LAYERS, bs, element_dim), dtype=DTYPE)
    cuda_kv[:, :, cuda_indices, :] = init_value.cuda()
    for i in range(LAYERS):
        transfer_hicache_one_layer(
            k_cache_dst=host_kv[0, i],
            v_cache_dst=host_kv[1, i],
            indices_dst=host_indices,
            k_cache_src=cuda_kv[0, i],
            v_cache_src=cuda_kv[1, i],
            indices_src=cuda_indices,
            block_quota=block_quota,
        )
    post_value = host_kv[:, :, host_indices.cpu(), :]
    assert torch.all(post_value == init_value.cpu())


def _test_hicache_correctness_all_layer(
    host_kv: torch.Tensor,
    cuda_kv: torch.Tensor,
    host_kv_ptrs: Tuple[torch.Tensor, torch.Tensor],
    cuda_kv_ptrs: Tuple[torch.Tensor, torch.Tensor],
    bs: int,
    element_dim: int,
    indices_dtype: torch.dtype,
    block_quota: int,
) -> None:
    from sglang.jit_kernel.hicache import transfer_hicache_all_layer

    host_indices = generate_indices(bs=bs, size=HOST_SIZE, dtype=indices_dtype)
    cuda_indices = generate_indices(bs=bs, size=CUDA_SIZE, dtype=indices_dtype)

    init_value = torch.randn((2, LAYERS, bs, element_dim), dtype=DTYPE)
    host_kv[:, :, host_indices.cpu(), :] = init_value
    transfer_hicache_all_layer(
        k_ptr_dst=cuda_kv_ptrs[0],
        v_ptr_dst=cuda_kv_ptrs[1],
        indices_dst=cuda_indices,
        k_ptr_src=host_kv_ptrs[0],
        v_ptr_src=host_kv_ptrs[1],
        indices_src=host_indices,
        kv_cache_src_stride_bytes=STRIDE_BYTES,
        kv_cache_dst_stride_bytes=STRIDE_BYTES,
        element_size=element_dim * DTYPE.itemsize,
        block_quota=block_quota,
    )
    post_value = cuda_kv[:, :, cuda_indices, :]
    assert torch.all(post_value == init_value.cuda())

    init_value = torch.randn((2, LAYERS, bs, element_dim), dtype=DTYPE)
    cuda_kv[:, :, cuda_indices, :] = init_value.cuda()
    transfer_hicache_all_layer(
        k_ptr_dst=host_kv_ptrs[0],
        v_ptr_dst=host_kv_ptrs[1],
        indices_dst=host_indices,
        k_ptr_src=cuda_kv_ptrs[0],
        v_ptr_src=cuda_kv_ptrs[1],
        indices_src=cuda_indices,
        kv_cache_src_stride_bytes=STRIDE_BYTES,
        kv_cache_dst_stride_bytes=STRIDE_BYTES,
        element_size=element_dim * DTYPE.itemsize,
        block_quota=block_quota,
    )
    post_value = host_kv[:, :, host_indices.cpu(), :]
    assert torch.all(post_value == init_value.cpu())


@pytest.mark.parametrize(
    "batch_size, block_quota, world_size, dtype",
    list(itertools.product(BS_LIST, BLOCK_QUOTA, WORLD_SIZE, IDTYPES)),
)
def test_hicache(
    batch_size: int, block_quota: int, world_size: int, dtype: torch.dtype
):
    stride = CUDA_KV.stride(-2)
    assert stride == HOST_KV.stride(-2)
    element_dim = 1024 // world_size
    _test_hicache_correctness_one_layer(
        host_kv=HOST_KV[:, :, :, :element_dim],
        cuda_kv=CUDA_KV[:, :, :, :element_dim],
        bs=batch_size,
        element_dim=element_dim,
        indices_dtype=dtype,
        block_quota=block_quota,
    )
    _test_hicache_correctness_all_layer(
        host_kv=HOST_KV[:, :, :, :element_dim],
        cuda_kv=CUDA_KV[:, :, :, :element_dim],
        host_kv_ptrs=HOST_KV_PTRS,
        cuda_kv_ptrs=CUDA_KV_PTRS,
        bs=batch_size,
        element_dim=element_dim,
        indices_dtype=dtype,
        block_quota=block_quota,
    )


if __name__ == "__main__":
    pytest.main([__file__])
