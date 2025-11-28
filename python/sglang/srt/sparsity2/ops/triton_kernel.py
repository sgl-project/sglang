from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def nsa_sparse_diff_triton_kernel(
    prev_top_k_result_ptr,
    curr_top_k_result_ptr,
    prev_device_indices_ptr,
    curr_device_indices_ptr,
    bitmap_ptr,
    host_start_indices_ptr,
    full_host_indices_ptr,
    should_load_device_indices_ptr,
    should_load_host_indices_ptr,
    out_cache_loc_ptr,
    seq_lens_ptr,
    req_pool_indices_ptr,
    prev_top_k_result_stride_0: tl.constexpr,
    prev_top_k_result_stride_1: tl.constexpr,
    curr_top_k_result_stride: tl.constexpr,
    prev_device_indices_stride_0: tl.constexpr,
    prev_device_indices_stride_1: tl.constexpr,
    curr_device_indices_stride: tl.constexpr,
    bitmap_stride: tl.constexpr,
    should_load_device_indices_stride: tl.constexpr,
    should_load_host_indices_stride: tl.constexpr,
    layer_id: tl.constexpr,
    TOPK: tl.constexpr,
):
    bid = tl.program_id(0)
    offset = tl.arange(0, TOPK)
    req_pool_index = tl.load(req_pool_indices_ptr + bid)
    # start address of current block
    prev_top_k_result_start_ptr = (
        prev_top_k_result_ptr
        + req_pool_index * prev_top_k_result_stride_0
        + layer_id * prev_top_k_result_stride_1
    )
    prev_device_indices_start_ptr = (
        prev_device_indices_ptr
        + req_pool_index * prev_device_indices_stride_0
        + layer_id * prev_device_indices_stride_1
    )

    tmp_curr_top_k_result = tl.load(
        curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
    )

    prev_top_k_result = tl.load(prev_top_k_result_start_ptr + offset)
    max_val = tl.max(prev_top_k_result)
    seq_len = tl.load(seq_lens_ptr + bid)

    if max_val == -1:
        # After prefilling the first round, the entire cache needs to be loaded.
        no_exist_top_k_result = tl.load(
            curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
        )
        mask = no_exist_top_k_result < seq_len
        start_index = tl.load(host_start_indices_ptr + bid)
        no_exist_host_indices = tl.load(
            full_host_indices_ptr + start_index + no_exist_top_k_result, mask=mask
        )
        tl.store(
            should_load_host_indices_ptr
            + should_load_host_indices_stride * bid
            + offset,
            no_exist_host_indices,
            mask=mask,
        )
    else:
        tl.store(bitmap_ptr + bitmap_stride * bid + prev_top_k_result, offset)

        prev_out_cache_loc_index = tl.load(
            bitmap_ptr + bitmap_stride * bid + seq_len - 1
        )
        # The `out_cache_loc` from the previous step is no longer valid.
        if prev_out_cache_loc_index != -1:
            tl.store(bitmap_ptr + bitmap_stride * bid + seq_len - 1, -1)
            tl.store(prev_device_indices_start_ptr + prev_out_cache_loc_index, -1)

        curr_top_k_result = tl.load(
            curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
        )
        exist_indices = tl.load(bitmap_ptr + bitmap_stride * bid + curr_top_k_result)

        mask = exist_indices >= 0
        exist_prev_device_indices = tl.load(
            prev_device_indices_start_ptr + exist_indices,
            mask=mask,
        )
        tl.store(
            curr_device_indices_ptr + curr_device_indices_stride * bid + offset,
            exist_prev_device_indices,
            mask=mask,
        )

        tl.store(
            prev_device_indices_start_ptr + exist_indices,
            -1,
            mask=mask,
        )
        tl.store(
            curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset,
            -1,
            mask=mask,
        )
        tl.store(bitmap_ptr + bitmap_stride * bid + prev_top_k_result, -1)

        no_exist_top_k_result = tl.load(
            curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
        )

        start_index = tl.load(host_start_indices_ptr + bid)
        # Excluding out_cache_loc, because it hasn't been offloaded to the host yet.
        mask1 = no_exist_top_k_result < seq_len
        host_mask = ~mask & mask1
        no_exist_host_indices = tl.load(
            full_host_indices_ptr + start_index + no_exist_top_k_result, mask=host_mask
        )
        tl.store(
            should_load_host_indices_ptr
            + should_load_host_indices_stride * bid
            + offset,
            no_exist_host_indices,
            mask=host_mask,
        )

    out_cache_loc = tl.load(out_cache_loc_ptr + bid)
    curr_top_k_result = tl.load(
        curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
    )
    out_cache_loc_mask = curr_top_k_result == seq_len
    tl.store(
        curr_device_indices_ptr + curr_device_indices_stride * bid + offset,
        out_cache_loc,
        mask=out_cache_loc_mask,
    )

    device_count = 0
    host_count = 0
    for idx in range(TOPK + 1):
        device_val = tl.load(prev_device_indices_start_ptr + idx)
        if device_val != -1:
            tl.store(
                should_load_device_indices_ptr
                + should_load_device_indices_stride * bid
                + device_count,
                device_val,
            )
            device_count += 1
        if idx < TOPK:  # The length of host_indices is only TOPK.
            host_val = tl.load(
                should_load_host_indices_ptr
                + should_load_host_indices_stride * bid
                + idx
            )
            if host_val != -1:
                tl.store(
                    should_load_host_indices_ptr
                    + should_load_host_indices_stride * bid
                    + host_count,
                    host_val,
                )
                host_count += 1

    replace_ptr = 0
    for idx in range(TOPK + 1):
        curr_val = tl.load(
            curr_device_indices_ptr + curr_device_indices_stride * bid + idx
        )
        if curr_val == -1:
            new_val = tl.load(
                should_load_device_indices_ptr
                + should_load_device_indices_stride * bid
                + replace_ptr
            )
            tl.store(
                curr_device_indices_ptr + curr_device_indices_stride * bid + idx,
                new_val,
            )
            replace_ptr += 1

    mask = offset >= host_count
    tl.store(
        should_load_device_indices_ptr
        + should_load_device_indices_stride * bid
        + offset,
        -1,
        mask=mask,
    )
    tl.store(
        should_load_host_indices_ptr + should_load_host_indices_stride * bid + offset,
        -1,
        mask=mask,
    )

    # update prev_top_k_result and curr_device_indices in req states
    tl.store(prev_top_k_result_start_ptr + offset, tmp_curr_top_k_result)
    curr_device_indices = tl.load(
        curr_device_indices_ptr + curr_device_indices_stride * bid + offset
    )
    tl.store(prev_device_indices_start_ptr + offset, curr_device_indices)
    last_index = tl.load(
        curr_device_indices_ptr + curr_device_indices_stride * bid + TOPK
    )
    tl.store(prev_device_indices_start_ptr + TOPK, last_index)


def invoke_nsa_sparse_diff_kernel(
    prev_top_k_result_pool: torch.Tensor,
    curr_top_k_result: torch.Tensor,
    prev_device_indices_pool: torch.Tensor,
    curr_device_indices: torch.Tensor,
    bitmap: torch.Tensor,
    full_host_indices: List[torch.Tensor],
    should_load_device_indices: torch.Tensor,
    should_load_host_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    layer_id: int,
):
    bs = curr_top_k_result.shape[0]
    top_k = curr_top_k_result.shape[1]

    host_start_indices = [0] + [
        len(full_host_indices[idx]) for idx in range(len(full_host_indices))
    ][:-1]
    host_start_indices = torch.tensor(
        host_start_indices,
        dtype=torch.int64,
        device=bitmap.device,
    )
    host_start_indices = torch.cumsum(host_start_indices, dim=-1)
    full_host_indices = torch.cat(full_host_indices)

    grid = (bs,)
    nsa_sparse_diff_triton_kernel[grid](
        prev_top_k_result_pool,
        curr_top_k_result,
        prev_device_indices_pool,
        curr_device_indices,
        bitmap,
        host_start_indices,
        full_host_indices,
        should_load_device_indices,
        should_load_host_indices,
        out_cache_loc,
        seq_lens,
        req_pool_indices,
        prev_top_k_result_pool.stride(0),
        prev_top_k_result_pool.stride(1),
        curr_top_k_result.stride(0),
        prev_device_indices_pool.stride(0),
        prev_device_indices_pool.stride(1),
        curr_device_indices.stride(0),
        bitmap.stride(0),
        should_load_device_indices.stride(0),
        should_load_host_indices.stride(0),
        layer_id,
        top_k,
    )


if __name__ == "__main__":
    bs = 2
    num_layers = 2
    layer_id = 1
    top_k = 8
    bitmap = torch.full(
        (8, 64),
        -1,
        dtype=torch.int16,
        device="cuda",
    )

    prev_top_k_result_pool = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [3, 5, 6, 7, 9, 1, 16, 11]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    curr_top_k_result = torch.tensor(
        [[5, 6, 7, 9, 1, 13, 15, 17], [2, 3, 5, 6, 7, 9, 12, 10]],
        dtype=torch.int64,
        device="cuda",
    )

    prev_device_indices_pool = torch.tensor(
        [
            [
                [9001, 9002, 9003, 9004, 9005, 9006, 42121, 9008, 9009],
                [9001, 9002, 9003, 9004, 9005, 9006, 42121, 9008, 9009],
            ],
            [
                [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, -1],
                [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, -1],
            ],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    full_host_indices = [
        torch.tensor(
            [
                1000,
                1001,
                1002,
                1003,
                1004,
                1005,
                1006,
                1007,
                1008,
                1009,
                1010,
                1011,
                1012,
                1013,
                1014,
                1015,
                1016,
            ],
            dtype=torch.int64,
            device="cuda",
        ),
        torch.tensor(
            [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011],
            dtype=torch.int64,
            device="cuda",
        ),
    ]

    seq_lens = torch.tensor([17, 12], dtype=torch.int64, device="cuda")
    out_cache_loc = torch.tensor([9017, 8012], dtype=torch.int64, device="cuda")
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    curr_device_indices = torch.full(
        (bs, top_k + 1), -1, dtype=torch.int64, device="cuda"
    )

    should_load_device_indices = torch.full(
        (bs, top_k), -1, dtype=torch.int64, device="cuda"
    )
    should_load_host_indices = torch.full(
        (bs, top_k), -1, dtype=torch.int64, device="cuda"
    )

    invoke_nsa_sparse_diff_kernel(
        prev_top_k_result_pool,
        curr_top_k_result,
        prev_device_indices_pool,
        curr_device_indices,
        bitmap,
        full_host_indices,
        should_load_device_indices,
        should_load_host_indices,
        out_cache_loc,
        seq_lens,
        req_pool_indices,
        layer_id,
    )

    # print(bitmap.tolist())

    ref_curr_device_indices = torch.tensor(
        [
            [9002, 9003, 9004, 9005, 9006, 9001, 9008, 9017, 9009],
            [8001, 8002, 8003, 8004, 8005, 8006, 8012, 8007, 8008],
        ],
        device="cuda:0",
    )
    ref_should_load_device_indices = torch.tensor(
        [
            [9001, 9008, -1, -1, -1, -1, -1, -1],
            [8001, 8002, 8003, 8004, 8005, 8006, 8007, -1],
        ],
        device="cuda:0",
    )
    ref_should_load_host_indices = torch.tensor(
        [
            [1013, 1015, -1, -1, -1, -1, -1, -1],
            [2002, 2003, 2005, 2006, 2007, 2009, 2010, -1],
        ],
        device="cuda:0",
    )
    ref_prev_device_indices_pool = torch.tensor(
        [
            [
                [9001, 9002, 9003, 9004, 9005, 9006, 42121, 9008, 9009],
                [9002, 9003, 9004, 9005, 9006, 9001, 9008, 9017, 9009],
            ],
            [
                [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, -1],
                [8001, 8002, 8003, 8004, 8005, 8006, 8012, 8007, 8008],
            ],
        ],
        device="cuda:0",
    )
    ref_prev_top_k_result_pool = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [5, 6, 7, 9, 1, 13, 15, 17]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [2, 3, 5, 6, 7, 9, 12, 10]],
        ],
        device="cuda:0",
    )

    assert torch.all(curr_device_indices == ref_curr_device_indices)
    assert torch.all(should_load_device_indices == ref_should_load_device_indices)
    assert torch.all(should_load_host_indices == ref_should_load_host_indices)
    assert torch.all(prev_device_indices_pool == ref_prev_device_indices_pool)
    assert torch.all(prev_top_k_result_pool == ref_prev_top_k_result_pool)
    assert torch.max(bitmap) == -1
