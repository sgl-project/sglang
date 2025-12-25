from typing import Tuple

import torch

LAYERS = 4
DTYPE = torch.float16
CUDA_SIZE = 512 * 1024
HOST_SIZE = 2 * CUDA_SIZE


def generate_indices(bs: int, size: int) -> torch.Tensor:
    result = torch.randperm(size, dtype=torch.int64, device="cuda")[:bs].sort().values
    assert torch.all(0 <= result)
    assert torch.all(result < size)
    return result


def test_hicache_correctness_one_layer(
    host_kv: torch.Tensor,
    cuda_kv: torch.Tensor,
    bs: int,
) -> None:
    from sglang.jit_kernel.hicache import transfer_hicache_one_layer

    host_indices = generate_indices(bs=bs, size=HOST_SIZE)
    cuda_indices = generate_indices(bs=bs, size=CUDA_SIZE)

    init_value = torch.randn((2, LAYERS, bs, ELEM_SIZE), dtype=DTYPE)
    host_kv[:, :, host_indices.cpu(), :] = init_value
    for i in range(LAYERS):
        transfer_hicache_one_layer(
            k_cache_dst=cuda_kv[0, i],
            v_cache_dst=cuda_kv[1, i],
            indices_dst=cuda_indices,
            k_cache_src=host_kv[0, i],
            v_cache_src=host_kv[1, i],
            indices_src=host_indices,
        )
    post_value = cuda_kv[:, :, cuda_indices, :]
    assert torch.all(post_value == init_value.cuda())

    init_value = torch.randn((2, LAYERS, bs, ELEM_SIZE), dtype=DTYPE)
    cuda_kv[:, :, cuda_indices, :] = init_value.cuda()
    for i in range(LAYERS):
        transfer_hicache_one_layer(
            k_cache_dst=host_kv[0, i],
            v_cache_dst=host_kv[1, i],
            indices_dst=host_indices,
            k_cache_src=cuda_kv[0, i],
            v_cache_src=cuda_kv[1, i],
            indices_src=cuda_indices,
        )
    post_value = host_kv[:, :, host_indices.cpu(), :]
    assert torch.all(post_value == init_value.cpu())


def test_hicache_correctness_all_layer(
    host_kv: torch.Tensor,
    cuda_kv: torch.Tensor,
    host_kv_ptrs: Tuple[torch.Tensor, torch.Tensor],
    cuda_kv_ptrs: Tuple[torch.Tensor, torch.Tensor],
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    element_size: int,
    bs: int,
) -> None:
    from sglang.jit_kernel.hicache import transfer_hicache_all_layer

    host_indices = generate_indices(bs=bs, size=HOST_SIZE)
    cuda_indices = generate_indices(bs=bs, size=CUDA_SIZE)

    init_value = torch.randn((2, LAYERS, bs, ELEM_SIZE), dtype=DTYPE)
    host_kv[:, :, host_indices.cpu(), :] = init_value
    transfer_hicache_all_layer(
        k_ptr_dst=cuda_kv_ptrs[0],
        v_ptr_dst=cuda_kv_ptrs[1],
        indices_dst=cuda_indices,
        k_ptr_src=host_kv_ptrs[0],
        v_ptr_src=host_kv_ptrs[1],
        indices_src=host_indices,
        kv_cache_src_stride_bytes=kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes=kv_cache_dst_stride_bytes,
        element_size=element_size,
    )
    post_value = cuda_kv[:, :, cuda_indices, :]
    assert torch.all(post_value == init_value.cuda())

    init_value = torch.randn((2, LAYERS, bs, ELEM_SIZE), dtype=DTYPE)
    cuda_kv[:, :, cuda_indices, :] = init_value.cuda()
    transfer_hicache_all_layer(
        k_ptr_dst=host_kv_ptrs[0],
        v_ptr_dst=host_kv_ptrs[1],
        indices_dst=host_indices,
        k_ptr_src=cuda_kv_ptrs[0],
        v_ptr_src=cuda_kv_ptrs[1],
        indices_src=cuda_indices,
        kv_cache_src_stride_bytes=kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes=kv_cache_dst_stride_bytes,
        element_size=element_size,
    )
    post_value = host_kv[:, :, host_indices.cpu(), :]
    assert torch.all(post_value == init_value.cpu())


def make_ptrs(tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [tensor[i].data_ptr() for i in range(LAYERS)],
        dtype=torch.uint64,
        device="cuda",
    )


def main() -> None:
    global ELEM_SIZE
    cuda_kv = torch.empty((2, LAYERS, CUDA_SIZE, 1024), dtype=DTYPE, device="cuda")
    host_kv = torch.empty((2, LAYERS, HOST_SIZE, 1024), dtype=DTYPE, pin_memory=True)
    cuda_kv_ptrs = make_ptrs(cuda_kv[0]), make_ptrs(cuda_kv[1])
    host_kv_ptrs = make_ptrs(host_kv[0]), make_ptrs(host_kv[1])
    stride = cuda_kv.stride(-2)
    assert stride == host_kv.stride(-2)
    stride_bytes = stride * DTYPE.itemsize

    for N_HEAD in [1, 2, 4, 8]:
        ELEM_SIZE = 128 * N_HEAD
        assert ELEM_SIZE <= 1024
        for BS in [2**n for n in range(6, 14)]:  # 64 to 8192
            test_hicache_correctness_one_layer(
                host_kv=host_kv[:, :, :, :ELEM_SIZE],
                cuda_kv=cuda_kv[:, :, :, :ELEM_SIZE],
                bs=BS,
            )
            test_hicache_correctness_all_layer(
                host_kv=host_kv[:, :, :, :ELEM_SIZE],
                cuda_kv=cuda_kv[:, :, :, :ELEM_SIZE],
                host_kv_ptrs=host_kv_ptrs,
                cuda_kv_ptrs=cuda_kv_ptrs,
                kv_cache_src_stride_bytes=stride_bytes,
                kv_cache_dst_stride_bytes=stride_bytes,
                element_size=ELEM_SIZE * DTYPE.itemsize,
                bs=BS,
            )
        print(f"HiCache correctness test passed for ELEM_SIZE={ELEM_SIZE}")
    print("All HiCache correctness tests passed.")


if __name__ == "__main__":
    main()
