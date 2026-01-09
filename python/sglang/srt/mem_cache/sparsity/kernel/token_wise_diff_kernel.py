import torch
import triton
import triton.language as tl


@triton.jit
def sparse_diff_triton_kernel(
    prev_topk_ptr,
    curr_topk_ptr,
    prev_dev_idx_ptr,
    curr_dev_idx_ptr,
    diff_map_ptr,
    host_idx_ptr,
    load_dev_idx_ptr,
    load_host_idx_ptr,
    out_cache_loc_ptr,
    seq_len_ptr,
    req_idx_ptr,
    sparse_mask_ptr,
    page_table_ptr,
    prev_topk_s0: tl.constexpr,
    prev_topk_s1: tl.constexpr,
    curr_topk_s: tl.constexpr,
    prev_dev_s0: tl.constexpr,
    prev_dev_s1: tl.constexpr,
    curr_dev_s: tl.constexpr,
    diff_map_s: tl.constexpr,
    host_idx_s: tl.constexpr,
    load_dev_s: tl.constexpr,
    load_host_s: tl.constexpr,
    page_table_s: tl.constexpr,
    layer_id,
    TOPK: tl.constexpr,
    LRU_LEN: tl.constexpr,
):
    """Optimized version with vectorized operations instead of serial loops."""
    bid = tl.program_id(0)
    offset = tl.arange(0, TOPK)
    offset_lru = tl.arange(0, LRU_LEN)
    req_idx = tl.load(req_idx_ptr + bid)
    seq_len = tl.load(seq_len_ptr + bid) - 1
    out_cache_loc = tl.load(out_cache_loc_ptr + bid)

    # ---- Pointer Setup ----
    prev_topk_base = prev_topk_ptr + req_idx * prev_topk_s0 + layer_id * prev_topk_s1
    prev_dev_base = prev_dev_idx_ptr + req_idx * prev_dev_s0 + layer_id * prev_dev_s1
    curr_topk_base = curr_topk_ptr + bid * curr_topk_s
    curr_dev_base = curr_dev_idx_ptr + curr_dev_s * bid
    load_dev_base = load_dev_idx_ptr + load_dev_s * bid
    load_host_base = load_host_idx_ptr + load_host_s * bid

    # ----  Refill -1 ----
    tl.store(curr_dev_base + offset_lru, -1)
    tl.store(load_dev_base + offset_lru, -1)
    tl.store(load_host_base + offset_lru, -1)

    # Handling reqs where seq_len < min_sparse_len
    sparse_mask_val = tl.load(sparse_mask_ptr + bid)
    if (sparse_mask_val == 0) | (seq_len <= 0):
        loaded_topk_indices = tl.load(curr_topk_base + offset)
        mask = loaded_topk_indices >= 0
        loaded_kv_indices = tl.load(
            page_table_ptr + page_table_s * req_idx + loaded_topk_indices, mask=mask
        )
        tl.store(curr_dev_base + offset, loaded_kv_indices, mask=mask)
        return

    # ----- Calculate intersection of prev_topk and curr_topk -----
    prev_topk = tl.load(prev_topk_base + offset_lru)
    curr_topk_origin = tl.load(curr_topk_base + offset)
    tl.store(diff_map_ptr + diff_map_s * bid + prev_topk, offset_lru)
    tl.debug_barrier()

    # 1. remove previous step's out_cache_loc
    prev_cache_idx = tl.load(diff_map_ptr + diff_map_s * bid + seq_len - 1)
    if prev_cache_idx != -1:
        tl.store(diff_map_ptr + diff_map_s * bid + seq_len - 1, -1)
        tl.store(prev_dev_base + prev_cache_idx, -1)

    # 2. get intersection and store
    exist_topk_idx = tl.load(diff_map_ptr + diff_map_s * bid + curr_topk_origin)
    mask = exist_topk_idx >= 0
    existing_dev_idx = tl.load(prev_dev_base + exist_topk_idx, mask=mask)
    tl.store(curr_dev_base + offset, existing_dev_idx, mask=mask)

    # 3. clear existence slots
    tl.store(prev_dev_base + exist_topk_idx, -1, mask=mask)
    tl.store(curr_topk_base + offset, -1, mask=mask)
    tl.store(diff_map_ptr + diff_map_s * bid + prev_topk, -1)  # reset diff map

    # 4. get should load host slots
    no_exist_topk = tl.load(curr_topk_base + offset)
    mask1 = no_exist_topk < seq_len
    host_mask = ~mask & mask1
    no_exist_host_indices = tl.load(
        host_idx_ptr + req_idx * host_idx_s + no_exist_topk,
        mask=host_mask,
    )
    tl.store(load_host_base + offset, no_exist_host_indices, mask=host_mask)

    # 5. set out_cache_loc
    out_cache_loc_mask = curr_topk_origin == seq_len
    tl.store(curr_dev_base + offset, out_cache_loc, mask=out_cache_loc_mask)

    # 6. get empty slots in curr_dev
    curr_dev = tl.load(curr_dev_base + offset_lru)
    curr_topk = tl.load(curr_topk_base + offset_lru)
    empty_slots = curr_dev == -1
    empty_slots_int = empty_slots.to(tl.int32)
    fill_cumsum = tl.cumsum(empty_slots_int, axis=0)
    fill_pos = fill_cumsum - empty_slots_int

    empty_slots_topk = tl.where(offset_lru < TOPK, empty_slots_int, 0)
    fill_count = tl.sum(empty_slots_topk)

    # 7. get non-empty slots in prev_dev
    dev_vals = tl.load(prev_dev_base + offset_lru)
    dev_topk = tl.load(prev_topk_base + offset_lru)
    dev_valid = dev_vals != -1
    dev_valid_int = dev_valid.to(tl.int32)
    dev_valid_count = tl.sum(dev_valid_int)
    dev_cumsum = tl.cumsum(dev_valid_int, axis=0)
    dev_pos = dev_cumsum - dev_valid_int
    move_count = dev_valid_count - fill_count
    fill_slots = dev_pos >= move_count
    dev_pos = tl.where(fill_slots, dev_pos - move_count, dev_pos + fill_count)

    # 8. Store the slots that need to be loaded and left-aligned.
    tl.store(load_dev_base + dev_pos, dev_vals, mask=dev_valid)
    tl.store(prev_topk_base + dev_pos, dev_topk, mask=dev_valid)

    # 9. merge slots
    fill_vals = tl.load(load_dev_base + fill_pos, mask=empty_slots, other=-1)
    fill_topk = tl.load(prev_topk_base + fill_pos, mask=empty_slots, other=-1)
    final_dev = tl.where(empty_slots, fill_vals, curr_dev)
    final_topk = tl.where(empty_slots, fill_topk, curr_topk)
    tl.store(curr_dev_base + offset_lru, final_dev)
    tl.store(prev_dev_base + offset_lru, final_dev)
    tl.store(prev_topk_base + offset_lru, final_topk)
    tl.store(prev_topk_base + offset, curr_topk_origin)

    tl.store(load_dev_base + offset_lru, -1, mask=offset_lru >= fill_count)
    host_vals_all = tl.load(load_host_base + offset_lru)
    tl.store(load_host_base + fill_pos, host_vals_all, mask=empty_slots)
    tl.store(load_host_base + offset_lru, -1, mask=offset_lru >= fill_count)


def invoke_sparse_diff_kernel(
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
    lru_len: int,
):
    bs = curr_top_k_result.shape[0]
    top_k = curr_top_k_result.shape[1]
    grid = (bs,)
    assert page_size == 1
    sparse_diff_triton_kernel[grid](
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
        lru_len,
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
            [[-1, -1, -1, -1, -1, -1, -1, -1], [0, 1, 2, 3, 4, 5, 6, 7]],
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

    seq_lens = torch.tensor([18, 13, 16], dtype=torch.int64, device="cuda")
    out_cache_loc = torch.tensor([9017, 8012, 7015], dtype=torch.int64, device="cuda")

    prev_device_indices_pool = torch.tensor(
        [
            [
                [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008],
                [9001, 9002, 9003, 9004, 9005, 9006, 90070, 9008],
            ],
            [
                [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008],
                [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008],
            ],
            [
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008],
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008],
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

    req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")

    # Initialize sparse_mask: 0 uses page_table direct lookup, non-zero uses normal flow
    # First two batches use normal flow (sparse_mask=1), third batch uses page_table path (sparse_mask=0)
    sparse_mask = torch.tensor([1, 1, 0], dtype=torch.int32, device="cuda")

    # Initialize page_table: shape (bs, max_seqlen_k), maps logical positions to physical KV cache positions
    # page_table[i][j] represents the physical KV cache position for logical position j in batch i
    page_table = torch.full((bs, max_seqlen_k), -1, dtype=torch.int64, device="cuda")
    # Set page_table for batch 0: logical position i maps to physical position 9000+i
    for i in range(max_seqlen_k):
        page_table[0, i] = 9000 + i
    # Set page_table for batch 1: logical position i maps to physical position 8000+i
    for i in range(max_seqlen_k):
        page_table[1, i] = 8000 + i
    # Set page_table for batch 2: logical position i maps to physical position 7000+i
    for i in range(max_seqlen_k):
        page_table[2, i] = 7000 + i

    curr_device_indices = torch.full((bs, top_k), -1, dtype=torch.int64, device="cuda")

    should_load_device_indices = torch.full(
        (bs, top_k), -1, dtype=torch.int64, device="cuda"
    )
    should_load_host_indices = torch.full(
        (bs, top_k), -1, dtype=torch.int64, device="cuda"
    )

    invoke_sparse_diff_kernel(
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
    )

    # print(bitmap.tolist())

    ref_curr_device_indices = torch.tensor(
        [
            [9002, 9003, 9004, 9005, 9006, 9001, 9008, 9017],
            [8003, 8004, 8006, 8007, 8008, 8001, 8012, 8002],
            [7000, 7001, 7002, 7003, -1, -1, -1, -1],
        ],
        device="cuda:0",
    )
    ref_should_load_device_indices = torch.tensor(
        [
            [9001, 9008, -1, -1, -1, -1, -1, -1],
            [8001, 8002, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        device="cuda:0",
    )
    ref_should_load_host_indices = torch.tensor(
        [
            [1013, 1015, -1, -1, -1, -1, -1, -1],
            [2009, 2010, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        device="cuda:0",
    )
    ref_prev_device_indices_pool = torch.tensor(
        [
            [
                [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008],
                [9002, 9003, 9004, 9005, 9006, 9001, 9008, 9017],
            ],
            [
                [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008],
                [8003, 8004, 8006, 8007, 8008, 8001, 8012, 8002],
            ],
            [
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008],
                [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008],
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
        lambda *args: invoke_sparse_diff_kernel(
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
        ),
        "Original (TOPK=2048)",
        num_warmup=5,
        num_iters=20,
    )

    print(f"Time taken: {time_large_orig:.3f} ms")
    print("All tests passed! ✓")
    print("=" * 60)
