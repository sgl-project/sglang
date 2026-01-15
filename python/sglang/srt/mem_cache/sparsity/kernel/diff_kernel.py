import torch
import triton
import triton.language as tl


@triton.jit
def page_wise_diff_triton_kernel(
    last_top_k_idx,
    top_k_idx,
    last_page_ids,
    page_ids,
    diff_map,
    req_to_tokens_host,
    load_tokens,
    load_tokens_host,
    seq_lens,
    req_pool_indices,
    sparse_mask,
    page_table,
    last_top_k_s0: tl.constexpr,
    last_top_k_s1: tl.constexpr,
    top_k_s: tl.constexpr,
    last_page_ids_s0: tl.constexpr,
    last_page_ids_s1: tl.constexpr,
    page_ids_s: tl.constexpr,
    diff_map_s: tl.constexpr,
    req_to_tokens_host_s: tl.constexpr,
    load_tokens_s: tl.constexpr,
    load_tokens_host_s: tl.constexpr,
    page_table_s: tl.constexpr,
    layer_id,
    top_k: tl.constexpr,
    top_k_page: tl.constexpr,
    hot_buffer_len: tl.constexpr,
    hot_buffer_page: tl.constexpr,
    page_size: tl.constexpr,
):
    bid = tl.program_id(0)
    offset_page = tl.arange(0, top_k_page)
    offset_top_k = tl.arange(0, top_k)
    offset_lru = tl.arange(0, hot_buffer_page)
    seq_len = tl.load(seq_lens + bid) - 1
    req_idx = tl.load(req_pool_indices + bid)
    sparse_mask_val = tl.load(sparse_mask + bid)

    last_top_k_base = (
        last_top_k_idx + req_idx * last_top_k_s0 + layer_id * last_top_k_s1
    )
    last_page_ids_base = (
        last_page_ids + req_idx * last_page_ids_s0 + layer_id * last_page_ids_s1
    )
    top_k_base = top_k_idx + bid * top_k_s
    page_ids_base = page_ids + bid * page_ids_s
    load_tokens_base = load_tokens + bid * load_tokens_s
    load_tokens_host_base = load_tokens_host + bid * load_tokens_host_s
    tokens_host_base = req_to_tokens_host + req_idx * req_to_tokens_host_s

    # Refill -1
    tl.store(page_ids_base + offset_page, -1)
    tl.store(load_tokens_base + offset_top_k, -1)
    tl.store(load_tokens_host_base + offset_top_k, -1)

    if (sparse_mask_val == 0) | (seq_len <= 0):
        top_k_vals = tl.load(top_k_base + offset_page)
        mask = top_k_vals >= 0
        loaded_page_start = tl.load(
            page_table + page_table_s * req_idx + top_k_vals, mask=mask
        )
        tl.store(page_ids_base + offset_page, loaded_page_start / page_size, mask=mask)
        return

    last_top_k = tl.load(last_top_k_base + offset_lru)
    top_k_origin = tl.load(top_k_base + offset_page)

    last_max_top_k = tl.max(last_top_k)
    curr_max_top_k = tl.max(top_k_origin)
    #
    if curr_max_top_k != last_max_top_k:
        last_top_k = tl.where(last_top_k < last_max_top_k, last_top_k, curr_max_top_k)

    tl.store(diff_map + diff_map_s * bid + last_top_k, offset_lru)
    tl.debug_barrier()

    # 2. get intersection and store
    exist_top_k_idx = tl.load(diff_map + diff_map_s * bid + top_k_origin)
    mask = exist_top_k_idx >= 0
    exist_page = tl.load(last_page_ids_base + exist_top_k_idx, mask=mask)
    tl.store(page_ids_base + offset_page, exist_page, mask=mask)

    # 3. clear existence slots
    tl.store(last_page_ids_base + exist_top_k_idx, -1, mask=mask)
    tl.store(top_k_base + offset_page, -1, mask=mask)
    tl.store(diff_map + diff_map_s * bid + last_top_k, -1)

    # 4. get should load host slots
    no_exist_top_k = tl.load(top_k_base + offset_page)
    need_from_host_mask = no_exist_top_k >= 0
    tl.store(
        load_tokens_host_base + offset_page, no_exist_top_k, mask=need_from_host_mask
    )

    # 7. get empty slots in curr_dev
    mask_topk = offset_lru < top_k_page
    curr_page = tl.load(page_ids_base + offset_lru, mask=mask_topk, other=-1)
    curr_top_k = tl.load(top_k_base + offset_lru, mask=mask_topk, other=-1)
    empty_slots = curr_page == -1
    empty_slots_int = empty_slots.to(tl.int32)
    fill_cumsum = tl.cumsum(empty_slots_int, axis=0)
    fill_pos = fill_cumsum - empty_slots_int

    empty_slots_topk = tl.where(mask_topk, empty_slots_int, 0)
    fill_count = tl.sum(empty_slots_topk)

    # 7. get non-empty slots in prev_dev
    last_page_vals = tl.load(last_page_ids_base + offset_lru)
    last_top_k = tl.load(last_top_k_base + offset_lru)
    page_valid = last_page_vals != -1
    page_valid_int = page_valid.to(tl.int32)
    page_valid_count = tl.sum(page_valid_int)
    page_cumsum = tl.cumsum(page_valid_int, axis=0)
    page_pos = page_cumsum - page_valid_int
    move_count = page_valid_count - fill_count
    fill_slots = page_pos >= move_count
    page_pos = tl.where(fill_slots, page_pos - move_count, page_pos + fill_count)

    # 8. Store the slots that need to be loaded and left-aligned.
    tl.store(load_tokens_base + page_pos, last_page_vals, mask=page_valid)
    tl.store(last_top_k_base + page_pos, last_top_k, mask=page_valid)

    # 9. merge slots
    fill_page = tl.load(load_tokens_base + fill_pos, mask=empty_slots, other=-1)
    fill_top_k = tl.load(last_top_k_base + fill_pos, mask=empty_slots, other=-1)
    final_page = tl.where(empty_slots, fill_page, curr_page)
    final_top_k = tl.where(empty_slots, fill_top_k, curr_top_k)

    tl.store(last_page_ids_base + offset_lru, final_page)
    tl.store(page_ids_base + offset_lru, final_page, mask=mask_topk)
    tl.store(last_top_k_base + offset_lru, final_top_k)
    tl.store(last_top_k_base + offset_page, top_k_origin)

    tl.store(load_tokens_base + offset_lru, -1, mask=offset_lru >= fill_count)

    host_top_k_vals = tl.load(load_tokens_host_base + offset_lru)
    tl.store(load_tokens_host_base + fill_pos, host_top_k_vals, mask=empty_slots)
    tl.store(load_tokens_host_base + offset_lru, -1, mask=offset_lru >= fill_count)

    # page ids -> token ids
    page_idx = offset_top_k // page_size
    token_offset = offset_top_k % page_size
    load_page_mask = page_idx < fill_count
    # device
    page_id = tl.load(load_tokens_base + page_idx, mask=load_page_mask, other=0)
    token_slot = page_id * page_size + token_offset
    tl.store(load_tokens_base + offset_top_k, token_slot, mask=load_page_mask)
    # host
    page_id_host = tl.load(
        load_tokens_host_base + page_idx, mask=load_page_mask, other=0
    )
    token_idx_host = page_id_host * page_size + token_offset
    token_slot_host = tl.load(tokens_host_base + token_idx_host)
    tl.store(load_tokens_host_base + offset_top_k, token_slot_host, mask=load_page_mask)


def invoke_sparse_diff_kernel(
    last_top_k_idx: torch.Tensor,
    top_k_idx: torch.Tensor,
    last_page_ids: torch.Tensor,
    page_ids: torch.Tensor,
    diff_map: torch.Tensor,
    req_to_tokens_host: torch.Tensor,
    load_tokens: torch.Tensor,
    load_tokens_host: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    page_table: torch.Tensor,
    layer_id: int,
    top_k: int,
    hot_buffer_len: int,
    page_size: int,
):
    batch_size = top_k_idx.shape[0]
    grid = (batch_size,)

    page_wise_diff_triton_kernel[grid](
        last_top_k_idx,
        top_k_idx,
        last_page_ids,
        page_ids,
        diff_map,
        req_to_tokens_host,
        load_tokens,
        load_tokens_host,
        seq_lens,
        req_pool_indices,
        sparse_mask,
        page_table,
        last_top_k_idx.stride(0),
        last_top_k_idx.stride(1),
        top_k_idx.stride(0),
        last_page_ids.stride(0),
        last_page_ids.stride(1),
        page_ids.stride(0),
        diff_map.stride(0),
        req_to_tokens_host.stride(0),
        load_tokens.stride(0),
        load_tokens_host.stride(0),
        page_table.stride(0),
        layer_id,
        top_k,
        top_k // page_size,
        hot_buffer_len,
        hot_buffer_len // page_size,
        page_size,
    )


if __name__ == "__main__":
    print("Running original test...")
    page_size = 4
    top_k = 16
    hot_buffer_len = 32

    last_top_k_idx = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [3, 5, 6, 7, 9, 1, 2, 4]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [0, 1, 2, 3, 4, 5, 6, 7]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [2, 3, 6, 7, 0, 1, 4, 5]],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    last_page_ids = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [13, 15, 16, 17, 19, 11, 12, 14]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [10, 11, 12, 13, 14, 15, 16, 17]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [12, 13, 16, 17, 10, 11, 14, 15]],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    top_k_idx = torch.tensor(
        [
            [5, 6, 8, 9],
            [1, 3, 5, 7],
            [2, 3, 5, 8],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    diff_map = torch.full(
        (3, 16),
        -1,
        dtype=torch.int32,
        device="cuda",
    )

    # [[100 ～ 163], [200 ～ 263]] host tokens
    req_to_tokens_host = torch.tensor(
        [[i + j * 100 for i in range(64)] for j in range(1, 4)],
        dtype=torch.int64,
        device="cuda",
    )
    seq_lens = torch.tensor([35, 21, 32], dtype=torch.int64, device="cuda")
    sparse_mask = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")
    req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    page_table = torch.full((3, 64), -1, dtype=torch.int64, device="cuda")
    page_ids = torch.full((3, 4), -1, dtype=torch.int64, device="cuda")
    load_tokens = torch.full((3, 16), -1, dtype=torch.int64, device="cuda")
    load_tokens_host = torch.full((3, 16), -1, dtype=torch.int64, device="cuda")

    ref_page_ids = torch.tensor(
        [
            [15, 16, 14, 19],
            [11, 13, 15, 17],
            [12, 13, 15, 17],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ref_last_top_k_idx = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [5, 6, 8, 9, 3, 7, 1, 2]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [1, 3, 5, 7, 0, 2, 4, 6]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [2, 3, 5, 8, 6, 0, 1, 4]],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ref_last_page_ids = torch.tensor(
        [
            [[-1, -1, -1, -1, -1, -1, -1, -1], [15, 16, 14, 19, 13, 17, 11, 12]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [11, 13, 15, 17, 10, 12, 14, 16]],
            [[-1, -1, -1, -1, -1, -1, -1, -1], [12, 13, 15, 17, 16, 10, 11, 14]],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ref_load_tokens = torch.tensor(
        [
            [56, 57, 58, 59, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ref_load_tokens_host = torch.tensor(
        [
            [132, 133, 134, 135, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    invoke_sparse_diff_kernel(
        last_top_k_idx,
        top_k_idx,
        last_page_ids,
        page_ids,
        diff_map,
        req_to_tokens_host,
        load_tokens,
        load_tokens_host,
        seq_lens,
        req_pool_indices,
        sparse_mask,
        page_table,
        layer_id=1,
        top_k=top_k,
        hot_buffer_len=hot_buffer_len,
        page_size=page_size,
    )

    assert torch.all(page_ids == ref_page_ids)
    assert torch.all(last_top_k_idx == ref_last_top_k_idx)
    assert torch.all(last_page_ids == ref_last_page_ids)
    assert torch.all(load_tokens == ref_load_tokens)
    assert torch.all(load_tokens_host == ref_load_tokens_host)
    print("Original test: success!")
