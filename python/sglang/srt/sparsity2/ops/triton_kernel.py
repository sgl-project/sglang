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
    full_host_indices_ptr,
    should_load_device_indices_ptr,
    should_load_host_indices_ptr,
    out_cache_loc_ptr,
    seq_lens_ptr,
    req_pool_indices_ptr,
    sparse_mask_ptr,
    page_table_ptr,
    prev_top_k_result_stride_0: tl.constexpr,
    prev_top_k_result_stride_1: tl.constexpr,
    curr_top_k_result_stride: tl.constexpr,
    prev_device_indices_stride_0: tl.constexpr,
    prev_device_indices_stride_1: tl.constexpr,
    curr_device_indices_stride: tl.constexpr,
    bitmap_stride: tl.constexpr,
    full_host_indices_stride: tl.constexpr,
    should_load_device_indices_stride: tl.constexpr,
    should_load_host_indices_stride: tl.constexpr,
    page_table_stride: tl.constexpr,
    layer_id,
    TOPK: tl.constexpr,
):
    """Optimized version with vectorized operations instead of serial loops."""
    bid = tl.program_id(0)
    offset = tl.arange(0, TOPK)
    req_pool_index = tl.load(req_pool_indices_ptr + bid)
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

    # Refill -1
    tl.store(curr_device_indices_ptr + curr_device_indices_stride * bid + offset, -1)
    tl.store(curr_device_indices_ptr + curr_device_indices_stride * bid + TOPK, -1)
    tl.store(
        should_load_device_indices_ptr
        + should_load_device_indices_stride * bid
        + offset,
        -1,
    )
    tl.store(
        should_load_host_indices_ptr + should_load_host_indices_stride * bid + offset,
        -1,
    )
    
    sparse_mask_val = tl.load(sparse_mask_ptr + bid)
    
    if sparse_mask_val == 0:
        page_table_ptr = page_table_ptr + page_table_stride * bid
        topk_indices_ptr = curr_top_k_result_ptr + curr_top_k_result_stride * bid
        result_ptr = curr_device_indices_ptr + curr_device_indices_stride * bid

        loaded_topk_indices = tl.load(topk_indices_ptr + offset)
        mask = loaded_topk_indices >= 0
        loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
        tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
        # tl.store(result_ptr + offset, -1, mask=~mask)
        return

    # Load current top-k (save for later update)
    tmp_curr_top_k_result = tl.load(
        curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
    )

    prev_top_k_result = tl.load(prev_top_k_result_start_ptr + offset)
    max_val = tl.max(prev_top_k_result)
    seq_len = tl.load(seq_lens_ptr + bid)

    if max_val == -1:
        no_exist_top_k_result = tl.load(
            curr_top_k_result_ptr + curr_top_k_result_stride * bid + offset
        )
        mask = no_exist_top_k_result < seq_len
        no_exist_host_indices = tl.load(
            full_host_indices_ptr
            + req_pool_index * full_host_indices_stride
            + no_exist_top_k_result,
            mask=mask,
        )
        tl.store(
            should_load_host_indices_ptr
            + should_load_host_indices_stride * bid
            + offset,
            no_exist_host_indices,
            mask=mask,
        )
    else:
        # Use atomic_max to ensure deterministic result when prev_top_k_result has duplicates
        tl.atomic_max(
            bitmap_ptr + bitmap_stride * bid + prev_top_k_result, offset.to(tl.int32)
        )

        prev_out_cache_loc_index = tl.load(
            bitmap_ptr + bitmap_stride * bid + seq_len - 1
        )
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

        mask1 = no_exist_top_k_result < seq_len
        host_mask = ~mask & mask1
        no_exist_host_indices = tl.load(
            full_host_indices_ptr
            + req_pool_index * full_host_indices_stride
            + no_exist_top_k_result,
            mask=host_mask,
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

    # Vectorized compaction (replaces serial loop 1)
    device_vals_topk = tl.load(prev_device_indices_start_ptr + offset)
    device_valid_topk = device_vals_topk != -1
    device_valid_topk_int = device_valid_topk.to(tl.int32)
    device_cumsum_topk = tl.cumsum(device_valid_topk_int, axis=0)
    device_write_pos_topk = device_cumsum_topk - device_valid_topk_int

    device_val_last = tl.load(prev_device_indices_start_ptr + TOPK)
    device_count_topk = tl.sum(device_valid_topk_int)

    tl.store(
        should_load_device_indices_ptr
        + should_load_device_indices_stride * bid
        + device_write_pos_topk,
        device_vals_topk,
        mask=device_valid_topk,
    )

    device_count = device_count_topk
    if device_val_last != -1:
        tl.store(
            should_load_device_indices_ptr
            + should_load_device_indices_stride * bid
            + device_count_topk,
            device_val_last,
        )
        device_count = device_count_topk + 1

    host_vals_all = tl.load(
        should_load_host_indices_ptr + should_load_host_indices_stride * bid + offset
    )
    host_valid = host_vals_all != -1
    host_valid_int = host_valid.to(tl.int32)
    host_cumsum = tl.cumsum(host_valid_int, axis=0)
    host_write_pos = host_cumsum - host_valid_int

    tl.store(
        should_load_host_indices_ptr
        + should_load_host_indices_stride * bid
        + host_write_pos,
        host_vals_all,
        mask=host_valid,
    )
    host_count = tl.sum(host_valid_int)

    # Vectorized fill (replaces serial loop 2)
    curr_vals_topk = tl.load(
        curr_device_indices_ptr + curr_device_indices_stride * bid + offset
    )
    empty_mask_topk = curr_vals_topk == -1
    empty_mask_topk_int = empty_mask_topk.to(tl.int32)
    empty_cumsum_topk = tl.cumsum(empty_mask_topk_int, axis=0)
    fill_positions_topk = empty_cumsum_topk - empty_mask_topk_int

    fill_vals_topk = tl.load(
        should_load_device_indices_ptr
        + should_load_device_indices_stride * bid
        + fill_positions_topk,
        mask=empty_mask_topk,
        other=-1,
    )
    result_topk = tl.where(empty_mask_topk, fill_vals_topk, curr_vals_topk)
    tl.store(
        curr_device_indices_ptr + curr_device_indices_stride * bid + offset, result_topk
    )

    curr_val_last = tl.load(
        curr_device_indices_ptr + curr_device_indices_stride * bid + TOPK
    )
    if curr_val_last == -1:
        empty_count_topk = tl.sum(empty_mask_topk_int)
        fill_val_last = tl.load(
            should_load_device_indices_ptr
            + should_load_device_indices_stride * bid
            + empty_count_topk
        )
        tl.store(
            curr_device_indices_ptr + curr_device_indices_stride * bid + TOPK,
            fill_val_last,
        )

    # Clear invalid entries
    clear_mask = offset >= host_count
    tl.store(
        should_load_device_indices_ptr
        + should_load_device_indices_stride * bid
        + offset,
        -1,
        mask=clear_mask,
    )
    tl.store(
        should_load_host_indices_ptr + should_load_host_indices_stride * bid + offset,
        -1,
        mask=clear_mask,
    )

    # Update state
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
    full_host_indices: torch.Tensor,
    should_load_device_indices: torch.Tensor,
    should_load_host_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    page_table: torch.Tensor,
    layer_id: int,
    page_size: int,
):
    bs = curr_top_k_result.shape[0]
    top_k = curr_top_k_result.shape[1]
    grid = (bs,)
    assert page_size == 1
    nsa_sparse_diff_triton_kernel[grid](
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
        sparse_mask,
        page_table,
        prev_top_k_result_pool.stride(0),
        prev_top_k_result_pool.stride(1),
        curr_top_k_result.stride(0),
        prev_device_indices_pool.stride(0),
        prev_device_indices_pool.stride(1),
        curr_device_indices.stride(0),
        bitmap.stride(0),
        full_host_indices.stride(0),
        should_load_device_indices.stride(0),
        should_load_host_indices.stride(0),
        page_table.stride(0),
        layer_id,
        top_k,
    )


if __name__ == "__main__":
    bs = 3
    num_layers = 2
    layer_id = 1
    top_k = 8
    max_seqlen_k = 17
    bitmap = torch.full(
        (8, 64),
        -1,
        dtype=torch.int32,
        device="cuda",
    )

    prev_top_k_result_pool = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [3, 5, 6, 7, 9, 1, 16, 11]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    curr_top_k_result = torch.tensor(
        [
            [5, 6, 7, 9, 1, 13, 15, 17],
            [2, 3, 5, 6, 7, 9, 12, 10],
            [0, 1, 2, 3, -1, -1, -1, -1],
        ],
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
            [
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, -1],
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, -1],
            ],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    full_host_indices = torch.tensor(
        [
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
                -1,
                -1,
                -1,
            ],
            [
                2000,
                2001,
                2002,
                2003,
                2004,
                2005,
                2006,
                2007,
                2008,
                2009,
                2010,
                2011,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            [
                3000,
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    seq_lens = torch.tensor([17, 12, 15], dtype=torch.int64, device="cuda")
    out_cache_loc = torch.tensor([9017, 8012, 7015], dtype=torch.int64, device="cuda")
    req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")

    # Initialize sparse_mask: 0 uses page_table direct lookup, non-zero uses normal flow
    # First two batches use normal flow (sparse_mask=1), third batch uses page_table path (sparse_mask=0)
    sparse_mask = torch.tensor([1, 1, 0], dtype=torch.int32, device="cuda")

    # Initialize page_table: shape (bs, max_seqlen_k), maps logical positions to physical KV cache positions
    # page_table[i][j] represents the physical KV cache position for logical position j in batch i
    page_table = torch.full(
        (bs, max_seqlen_k), -1, dtype=torch.int64, device="cuda"
    )
    # Set page_table for batch 0: logical position i maps to physical position 9000+i
    for i in range(max_seqlen_k):
        page_table[0, i] = 9000 + i
    # Set page_table for batch 1: logical position i maps to physical position 8000+i
    for i in range(max_seqlen_k):
        page_table[1, i] = 8000 + i
    # Set page_table for batch 2: logical position i maps to physical position 7000+i
    for i in range(max_seqlen_k):
        page_table[2, i] = 7000 + i

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
        sparse_mask,
        page_table,
        layer_id,
        1,  # page_size
        max_seqlen_k,
    )

    # print(bitmap.tolist())

    ref_curr_device_indices = torch.tensor(
        [
            [9002, 9003, 9004, 9005, 9006, 9001, 9008, 9017, 9009],
            [8001, 8002, 8003, 8004, 8005, 8006, 8012, 8007, 8008],
            [7000, 7001, 7002, 7003, -1, -1, -1, -1, -1],
        ],
        device="cuda:0",
    )
    ref_should_load_device_indices = torch.tensor(
        [
            [9001, 9008, -1, -1, -1, -1, -1, -1],
            [8001, 8002, 8003, 8004, 8005, 8006, 8007, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        device="cuda:0",
    )
    ref_should_load_host_indices = torch.tensor(
        [
            [1013, 1015, -1, -1, -1, -1, -1, -1],
            [2002, 2003, 2005, 2006, 2007, 2009, 2010, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
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
            [
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, -1],
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, -1],
            ],
        ],
        device="cuda:0",
    )
    ref_prev_top_k_result_pool = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [5, 6, 7, 9, 1, 13, 15, 17]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [2, 3, 5, 6, 7, 9, 12, 10]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        device="cuda:0",
    )
    print(curr_device_indices)
    assert torch.all(curr_device_indices == ref_curr_device_indices)
    assert torch.all(should_load_device_indices == ref_should_load_device_indices)
    assert torch.all(should_load_host_indices == ref_should_load_host_indices)
    assert torch.all(prev_device_indices_pool == ref_prev_device_indices_pool)
    assert torch.all(prev_top_k_result_pool == ref_prev_top_k_result_pool)
    assert torch.max(bitmap) == -1

    print("✓ Original kernel test passed!")

    # ========== Large-scale test (TOPK=2048) ==========
    print("\n" + "=" * 60)
    print("Large-Scale Test (TOPK=2048)")
    print("=" * 60)

    import time

    def benchmark_kernel(kernel_func, name, num_warmup=10, num_iters=100):
        # Warmup
        for _ in range(num_warmup):
            # Reset states
            bitmap_bench = torch.full((bs, 64), -1, dtype=torch.int32, device="cuda")
            prev_top_k_result_bench = prev_top_k_result_pool.clone()
            curr_top_k_result_bench = curr_top_k_result.clone()
            prev_device_indices_bench = prev_device_indices_pool.clone()
            curr_device_indices_bench = torch.full(
                (bs, top_k + 1), -1, dtype=torch.int64, device="cuda"
            )
            should_load_device_bench = torch.full(
                (bs, top_k), -1, dtype=torch.int64, device="cuda"
            )
            should_load_host_bench = torch.full(
                (bs, top_k), -1, dtype=torch.int64, device="cuda"
            )
            sparse_mask_bench = sparse_mask.clone()
            page_table_bench = page_table.clone()

            kernel_func(
                prev_top_k_result_bench,
                curr_top_k_result_bench,
                prev_device_indices_bench,
                curr_device_indices_bench,
                bitmap_bench,
                full_host_indices,
                should_load_device_bench,
                should_load_host_bench,
                out_cache_loc,
                seq_lens,
                req_pool_indices,
                sparse_mask_bench,
                page_table_bench,
                layer_id,
                1,  # page_size
                max_seqlen_k,
            )

        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            bitmap_bench = torch.full((bs, 64), -1, dtype=torch.int32, device="cuda")
            prev_top_k_result_bench = prev_top_k_result_pool.clone()
            curr_top_k_result_bench = curr_top_k_result.clone()
            prev_device_indices_bench = prev_device_indices_pool.clone()
            curr_device_indices_bench = torch.full(
                (bs, top_k + 1), -1, dtype=torch.int64, device="cuda"
            )
            should_load_device_bench = torch.full(
                (bs, top_k), -1, dtype=torch.int64, device="cuda"
            )
            should_load_host_bench = torch.full(
                (bs, top_k), -1, dtype=torch.int64, device="cuda"
            )
            sparse_mask_bench = sparse_mask.clone()
            page_table_bench = page_table.clone()

            kernel_func(
                prev_top_k_result_bench,
                curr_top_k_result_bench,
                prev_device_indices_bench,
                curr_device_indices_bench,
                bitmap_bench,
                full_host_indices,
                should_load_device_bench,
                should_load_host_bench,
                out_cache_loc,
                seq_lens,
                req_pool_indices,
                sparse_mask_bench,
                page_table_bench,
                layer_id,
                1,  # page_size
                max_seqlen_k,
            )

        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / num_iters * 1000
        print(f"{name:30s}: {avg_time_ms:.3f} ms")
        return avg_time_ms

    bs_large = 4
    top_k_large = 2048
    seq_len_large = 4096
    max_seqlen_k_large = seq_len_large

    bitmap_large = torch.full(
        (bs_large, seq_len_large), -1, dtype=torch.int32, device="cuda"
    )

    prev_top_k_result_pool_large = torch.full(
        (bs_large, 2, top_k_large), -1, dtype=torch.int64, device="cuda"
    )
    # Set layer 1 to have some values
    prev_top_k_result_pool_large[:, 1, :] = (
        torch.arange(top_k_large, device="cuda").unsqueeze(0).expand(bs_large, -1)
    )

    curr_top_k_result_large = torch.randint(
        0, seq_len_large, (bs_large, top_k_large), dtype=torch.int64, device="cuda"
    )

    prev_device_indices_pool_large = torch.randint(
        1000, 50000, (bs_large, 2, top_k_large + 1), dtype=torch.int64, device="cuda"
    )

    full_host_indices_large = [
        torch.arange(
            100000 + i * 10000,
            100000 + i * 10000 + seq_len_large,
            dtype=torch.int64,
            device="cuda",
        )
        for i in range(bs_large)
    ]

    full_host_indices_large = torch.cat(full_host_indices_large, dim=0)

    seq_lens_large = torch.full(
        (bs_large,), seq_len_large - 1, dtype=torch.int64, device="cuda"
    )
    out_cache_loc_large = torch.randint(
        50000, 60000, (bs_large,), dtype=torch.int64, device="cuda"
    )
    req_pool_indices_large = torch.arange(bs_large, dtype=torch.int64, device="cuda")

    # Initialize sparse_mask and page_table for large-scale testing
    sparse_mask_large = torch.ones(bs_large, dtype=torch.int32, device="cuda")
    page_table_large = torch.full(
        (bs_large, max_seqlen_k_large), -1, dtype=torch.int64, device="cuda"
    )
    # Set page_table for each batch: logical position i maps to physical position 10000*batch_id + i
    for i in range(bs_large):
        for j in range(max_seqlen_k_large):
            page_table_large[i, j] = 10000 * i + j

    # Test original kernel
    curr_device_indices_large_orig = torch.full(
        (bs_large, top_k_large + 1), -1, dtype=torch.int64, device="cuda"
    )
    should_load_device_indices_large_orig = torch.full(
        (bs_large, top_k_large), -1, dtype=torch.int64, device="cuda"
    )
    should_load_host_indices_large_orig = torch.full(
        (bs_large, top_k_large), -1, dtype=torch.int64, device="cuda"
    )

    time_large_orig = benchmark_kernel(
        lambda *args: invoke_nsa_sparse_diff_kernel(
            prev_top_k_result_pool_large.clone(),
            curr_top_k_result_large.clone(),
            prev_device_indices_pool_large.clone(),
            curr_device_indices_large_orig,
            bitmap_large.clone(),
            full_host_indices_large,
            should_load_device_indices_large_orig,
            should_load_host_indices_large_orig,
            out_cache_loc_large,
            seq_lens_large,
            req_pool_indices_large,
            sparse_mask_large.clone(),
            page_table_large.clone(),
            1,  # layer_id
            1,  # page_size
            max_seqlen_k_large,
        ),
        "Original (TOPK=2048)",
        num_warmup=5,
        num_iters=20,
    )

    print(f"Time taken: {time_large_orig:.3f} ms")
    print("All tests passed! ✓")
    print("=" * 60)
