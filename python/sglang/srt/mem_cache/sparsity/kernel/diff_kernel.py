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
    offset_lru = tl.arange(0, hot_buffer_page)
    seq_len = tl.load(seq_lens + bid) - 1
    req_idx = tl.load(req_pool_indices + bid)
    sparse_mask_val = tl.load(sparse_mask + bid)

    last_top_k_base = last_top_k_idx + req_idx * last_top_k_s0 + layer_id * last_top_k_s1
    last_page_ids_base = last_page_ids + req_idx * last_page_ids_s0 + layer_id * last_page_ids_s1
    top_k_base = top_k_idx + bid * top_k_s
    page_ids_base = page_ids + bid * page_ids_s
    load_tokens_base = load_tokens + bid * load_tokens_s
    load_tokens_host_base = load_tokens_host + bid * load_tokens_host_s
    tokens_host_base = req_to_tokens_host + req_idx * req_to_tokens_host_s

    # Refill -1 for current batch
    tl.store(page_ids_base + offset_page, -1)
    offset_hot_buffer_tokens = tl.arange(0, hot_buffer_len)
    tl.store(load_tokens_base + offset_hot_buffer_tokens, -1)
    tl.store(load_tokens_host_base + offset_hot_buffer_tokens, -1)

    if (sparse_mask_val == 0) | (seq_len <= 0):
        top_k_vals = tl.load(top_k_base + offset_page)
        mask = top_k_vals >= 0
        loaded_page_start = tl.load(
            page_table + page_table_s * req_idx + top_k_vals, mask=mask
        )
        tl.store(page_ids_base + offset_page, loaded_page_start / page_size, mask=mask)

    last_top_k = tl.load(last_top_k_base + offset_lru)
    top_k_origin = tl.load(top_k_base + offset_page)

    # CRITICAL DESIGN: Handle physical memory reuse during decode growth
    # In decode, the "current last page" (most recent generated page) reuses a FIXED physical memory slot.
    # When sequence grows from page N to page N+1:
    #   - Page N+1 OVERWRITES page N's physical memory (same page_id)
    #   - If page N is still needed (in new topk), it MUST be loaded from host
    # 
    # This replacement logic forces page N to be treated as "not in intersection":
    #   Example: last=[0,1], curr=[1,2], curr_max=2 > last_max=1
    #     - Replace: last=[0,1] -> [0,2]  (force page 1 out)
    #     - diff_map[0]=slot0, diff_map[2]=slot1
    #     - Search topk=[1,2]:
    #         page 1: NOT found -> host load (correct! its memory was overwritten)
    #         page 2: found at slot1 -> reuses page_id from page 1 (correct! same physical memory)
    last_max_top_k = tl.max(last_top_k)
    curr_max_top_k = tl.max(top_k_origin)
    if curr_max_top_k != last_max_top_k:
        # Replace all pages >= last_max with curr_max
        last_top_k = tl.where(last_top_k < last_max_top_k, last_top_k, curr_max_top_k)

    # Only write valid last_top_k positions into diff_map
    valid_last_mask = last_top_k >= 0
    tl.store(diff_map + diff_map_s * bid + last_top_k, offset_lru, mask=valid_last_mask)
    tl.debug_barrier()

    # 2. get intersection and store
    exist_top_k_idx = tl.load(diff_map + diff_map_s * bid + top_k_origin)
    mask = exist_top_k_idx >= 0
    exist_page = tl.load(last_page_ids_base + exist_top_k_idx, mask=mask)
    tl.store(page_ids_base + offset_page, exist_page, mask=mask)

    # 3. clear existence slots
    tl.store(last_page_ids_base + exist_top_k_idx, -1, mask=mask)
    tl.store(top_k_base + offset_page, -1, mask=mask)
    tl.store(diff_map + diff_map_s * bid + last_top_k, -1, mask=valid_last_mask)

    # 4. mark pages that need to be loaded from host (non-intersection, valid pages)
    #    Note: we keep the original behavior; the newest page may still require host load
    #    if its device slot was overwritten by the rolling buffer in previous steps.
    no_exist_top_k = tl.load(top_k_base + offset_page)
    need_from_host_mask = no_exist_top_k >= 0
    tl.store(load_tokens_host_base + offset_page, no_exist_top_k, mask=need_from_host_mask)

    # 6. Check empty slots in page_ids (first top_k_page positions)
    mask_topk = offset_lru < top_k_page
    curr_page = tl.load(page_ids_base + offset_lru, mask=mask_topk, other=-1)
    curr_top_k = tl.load(top_k_base + offset_lru, mask=mask_topk, other=-1)
    empty_slots = (curr_page == -1) & mask_topk
    empty_slots_int = empty_slots.to(tl.int32)
    fill_cumsum = tl.cumsum(empty_slots_int, axis=0)
    fill_pos = fill_cumsum - empty_slots_int
    fill_count = tl.sum(empty_slots_int)

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

    # 9. merge slots: fill empty positions with evicted pages from LRU
    fill_page = tl.load(load_tokens_base + fill_pos, mask=empty_slots, other=-1)
    fill_top_k = tl.load(last_top_k_base + fill_pos, mask=empty_slots, other=-1)
    final_page = tl.where(empty_slots, fill_page, curr_page)
    final_top_k = tl.where(empty_slots, fill_top_k, curr_top_k)
    
    # Update page_ids output (first hot_buffer_page positions)
    tl.store(page_ids_base + offset_lru, final_page, mask=mask_topk)
    
    # Update last_page_ids and last_top_k: sync from updated page_ids
    # Reload from page_ids to ensure consistency
    updated_page_ids = tl.load(page_ids_base + offset_page)
    tl.store(last_page_ids_base + offset_page, updated_page_ids)
    tl.store(last_top_k_base + offset_page, top_k_origin)

    # Clean up load_tokens after fill_count
    tl.store(load_tokens_base + offset_lru, -1, mask=offset_lru >= fill_count)
    
    # Left-align load_tokens_host: compress valid entries to front
    host_page_pos = tl.load(load_tokens_host_base + offset_page)
    host_valid = host_page_pos >= 0
    host_valid_int = host_valid.to(tl.int32)
    host_cumsum = tl.cumsum(host_valid_int, axis=0)
    host_compact_pos = host_cumsum - host_valid_int
    host_count = tl.sum(host_valid_int)
    
    # Compact: move valid host page positions to front
    tl.store(load_tokens_host_base + host_compact_pos, host_page_pos, mask=host_valid)
    # Clean page-level data beyond host_count
    tl.store(load_tokens_host_base + offset_page, -1, mask=offset_page >= host_count)

    # Page IDs -> Token slots expansion
    # Use constexpr range: hot_buffer_page * page_size
    offset_all_tokens = tl.arange(0, hot_buffer_page * page_size)
    page_idx_dev = offset_all_tokens // page_size
    page_idx_host = offset_all_tokens // page_size
    token_offset = offset_all_tokens % page_size
    
    # Device: expand page IDs to token slots
    expand_mask_dev = page_idx_dev < fill_count
    page_id = tl.load(load_tokens_base + page_idx_dev, mask=expand_mask_dev, other=0)
    token_slot = page_id * page_size + token_offset
    tl.store(load_tokens_base + offset_all_tokens, token_slot, mask=expand_mask_dev)

    # Host: expand page positions to token host indices  
    expand_mask_host = page_idx_host < host_count
    page_pos = tl.load(load_tokens_host_base + page_idx_host, mask=expand_mask_host, other=-1)
    token_idx = page_pos * page_size + token_offset
    # Only load if page_pos is valid (>= 0)
    host_load_mask = expand_mask_host & (page_pos >= 0)
    token_slot_host = tl.load(tokens_host_base + token_idx, mask=host_load_mask, other=-1)
    tl.store(load_tokens_host_base + offset_all_tokens, token_slot_host, mask=expand_mask_host)



def page_wise_diff_triton(
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
        last_top_k_idx.shape[1],
        last_top_k_idx.shape[2],
        top_k_idx.shape[1],
        last_page_ids.shape[1],
        last_page_ids.shape[2],
        page_ids.shape[1],
        diff_map.shape[1],
        req_to_tokens_host.shape[1],
        load_tokens.shape[1],
        load_tokens_host.shape[1],
        page_table.shape[1],
        layer_id,
        top_k,
        top_k // page_size,
        hot_buffer_len,
        hot_buffer_len // page_size,
        page_size
    )

def verify_page_wise_diff(
    last_top_k_idx, last_page_ids, curr_top_k_page_pos,
    page_ids, load_tokens, load_tokens_host,
    req_to_tokens_host, layer_id, batch_idx, page_size, top_k, seq_len
):
    """Verify the correctness of page-wise diff kernel output, considering replacement logic"""
    errors = []
    
    # Build prev pages set: only pages with valid page_ids in GPU
    prev_pages_with_pageids = set()
    prev_page_map = {}  # page_pos -> page_id
    prev_page_list = []
    for i, page_pos in enumerate(last_top_k_idx[batch_idx, layer_id].tolist()):
        if page_pos >= 0:
            page_id = last_page_ids[batch_idx, layer_id, i].item()
            if page_id >= 0:
                prev_pages_with_pageids.add(page_pos)
                prev_page_map[page_pos] = page_id
                prev_page_list.append(page_pos)
    
    # Current requested pages
    curr_pages_list = [p for p in curr_top_k_page_pos[batch_idx].tolist() if p >= 0]
    curr_pages_set = set(curr_pages_list)
    
    # Detect replacement logic trigger (curr_max > last_max)
    last_max = max(prev_page_list) if prev_page_list else -1
    curr_max = max(curr_pages_list) if curr_pages_list else -1
    replacement_triggered = (curr_max > last_max)

    # Compute expected results considering replacement logic:
    # - curr_max（新生成的最后一页）复用 last_max 的物理槽位，不需要 host load
    # - 如果 last_max 仍在 curr 集合里，它被覆盖，需要从 host load
    # - 其他新页需要 host load
    new_pages = curr_pages_set - prev_pages_with_pageids
    forced_evicted = {last_max} if (replacement_triggered and last_max in curr_pages_set) else set()
    if replacement_triggered:
        need_from_host = (new_pages - {curr_max}) | forced_evicted
    else:
        need_from_host = new_pages
    intersection = (prev_pages_with_pageids & curr_pages_set) - forced_evicted
    need_from_prev = prev_pages_with_pageids - curr_pages_set
    
    # Check 1: Pages that are NOT forced-evicted should reuse page_ids
    output_page_map = {}
    for i in range(top_k // page_size):
        page_pos = curr_top_k_page_pos[batch_idx, i].item()
        page_id = page_ids[batch_idx, i].item()
        if page_id >= 0:
            output_page_map[page_pos] = page_id
    
    for page_pos in intersection:
        if page_pos in output_page_map and page_pos in prev_page_map:
            if output_page_map[page_pos] != prev_page_map[page_pos]:
                errors.append(f"Intersection page {page_pos}: expected page_id {prev_page_map[page_pos]}, got {output_page_map[page_pos]}")
    
    # Check 2: load_tokens should be multiple of page_size
    load_token_slots = load_tokens[batch_idx][load_tokens[batch_idx] >= 0].tolist()
    if len(load_token_slots) > 0:
        if len(load_token_slots) % page_size != 0:
            errors.append(f"load_tokens count {len(load_token_slots)} not multiple of page_size {page_size}")
    
    # Check 3: load_tokens_host should match need_from_host
    load_host_slots = load_tokens_host[batch_idx][load_tokens_host[batch_idx] >= 0].tolist()
    expected_host_count = len(need_from_host) * page_size
    
    if len(load_host_slots) != expected_host_count:
        errors.append(f"Expected {expected_host_count} host slots for {len(need_from_host)} pages (considering replacement), got {len(load_host_slots)}")
    
    # Check 4: Verify host slots are correct
    if len(load_host_slots) > 0:
        # Should be multiple of page_size
        if len(load_host_slots) % page_size != 0:
            errors.append(f"load_tokens_host count {len(load_host_slots)} not multiple of page_size {page_size}")
        
        # Verify that the loaded pages are indeed from need_from_host
        loaded_pages_set = set()
        for page_pos in need_from_host:
            expected_host_slots = req_to_tokens_host[batch_idx, page_pos*page_size:(page_pos+1)*page_size].tolist()
            if all(slot in load_host_slots for slot in expected_host_slots):
                loaded_pages_set.add(page_pos)
        
        if loaded_pages_set != need_from_host:
            missing = need_from_host - loaded_pages_set
            errors.append(f"Missing host loads for pages: {missing}")
    
    return errors, intersection, need_from_prev, need_from_host


def simple_test():
    """Simple controlled test with manual verification"""
    print("=" * 80)
    print("Simple Test: Page-wise Diff Kernel")
    print("=" * 80)
    
    page_size = 64
    top_k = 128  # 2 pages
    hot_buffer_len = 256  # 4 pages  
    bs = 1
    layer_id = 0
    max_seq_len = 1024  # 16 pages
    
    top_k_pages = top_k // page_size  # 2
    hot_buffer_pages = hot_buffer_len // page_size  # 4
    
    print(f"Config: page_size={page_size}, top_k={top_k} ({top_k_pages} pages), hot_buffer={hot_buffer_len} ({hot_buffer_pages} pages)")
    
    # Initialize
    last_top_k_idx = torch.full((bs, 1, hot_buffer_pages), -1, dtype=torch.int64, device="cuda")
    last_page_ids = torch.full((bs, 1, hot_buffer_pages), -1, dtype=torch.int64, device="cuda")
    req_to_tokens_host = torch.arange(max_seq_len, dtype=torch.int64, device="cuda").unsqueeze(0) + 10000
    
    # Round 1: Decode at seq_len=127 (last token of page 1)
    # last topk = [0, 1], curr topk = [0, 1] (no change)
    print("\n--- Round 1: seq_len=127, topk=[0,1] -> [0,1] (at page boundary) ---")
    last_top_k_idx[0, 0, :2] = torch.tensor([0, 1], device="cuda")
    last_page_ids[0, 0, :2] = torch.tensor([100, 101], device="cuda")
    
    curr_top_k_page_pos = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
    
    diff_map = torch.full((bs, max_seq_len), -1, dtype=torch.int32, device="cuda")
    page_ids = torch.full((bs, hot_buffer_pages), -1, dtype=torch.int64, device="cuda")
    load_tokens = torch.full((bs, hot_buffer_len), -1, dtype=torch.int64, device="cuda")
    load_tokens_host = torch.full((bs, hot_buffer_len), -1, dtype=torch.int64, device="cuda")
    
    page_wise_diff_triton(
        last_top_k_idx, curr_top_k_page_pos, last_page_ids, page_ids,
        diff_map, req_to_tokens_host, load_tokens, load_tokens_host,
        torch.tensor([127], device="cuda"), torch.tensor([0], device="cuda"),
        torch.ones(bs, dtype=torch.int32, device="cuda"),
        torch.full((bs, max_seq_len), -1, dtype=torch.int64, device="cuda"),
        layer_id, top_k, hot_buffer_len, page_size
    )
    
    print(f"Input: last=[0,1] with page_ids=[100,101], curr=[0,1], seq_len=127")
    print(f"Expected: full intersection, no replacement (max unchanged)")
    print(f"Output:")
    print(f"  page_ids (first {hot_buffer_pages}): {page_ids[0].tolist()}")
    print(f"  last_page_ids: {last_page_ids[0, 0].tolist()}")
    print(f"  last_top_k_idx: {last_top_k_idx[0, 0].tolist()}")
    
    # Build mapping from page_ids and last_top_k_idx
    output_map = {}  # page_pos -> page_id
    for i in range(hot_buffer_pages):
        pos = last_top_k_idx[0, 0, i].item()
        pid = last_page_ids[0, 0, i].item()
        if pos >= 0 and pid >= 0:
            output_map[pos] = pid
    print(f"  Output mapping (pos->page_id): {output_map}")
    
    # Verify: full intersection, both pages should be present
    assert 0 in output_map and output_map[0] == 100, "Position 0 should have page_id 100"
    assert 1 in output_map and output_map[1] == 101, "Position 1 should have page_id 101"
    
    load_host_valid = load_tokens_host[0][load_tokens_host[0] >= 0]
    assert len(load_host_valid) == 0, f"Should not load from host (full intersection), got {len(load_host_valid)}"
    
    # Verify last_top_k_idx unchanged
    assert set(last_top_k_idx[0, 0, :top_k_pages].tolist()) == {0, 1}, f"last_top_k_idx should be [0, 1], got {last_top_k_idx[0, 0, :top_k_pages].tolist()}"
    
    print("✓ Round 1 passed!")
    
    # Round 2: Decode grows to seq_len=128 (first token of page 2)
    # Physical memory: page 2 now OVERWRITES page 1's physical location (reuses page_id 101)
    # Logical topk changes: [0, 1] -> [1, 2]
    # KEY: page 1 is STILL in topk, but its physical memory was overwritten by page 2!
    # So page 1 MUST be loaded from host
    print("\n--- Round 2: seq_len=128, topk=[0,1] -> [1,2] (page 2 overwrites page 1's memory) ---")
    curr_top_k_page_pos = torch.tensor([[1, 2]], dtype=torch.int64, device="cuda")
    
    diff_map.fill_(-1)
    page_ids.fill_(-1)
    load_tokens.fill_(-1)
    load_tokens_host.fill_(-1)
    
    page_wise_diff_triton(
        last_top_k_idx, curr_top_k_page_pos, last_page_ids, page_ids,
        diff_map, req_to_tokens_host, load_tokens, load_tokens_host,
        torch.tensor([128], device="cuda"), torch.tensor([0], device="cuda"),
        torch.ones(bs, dtype=torch.int32, device="cuda"),
        torch.full((bs, max_seq_len), -1, dtype=torch.int64, device="cuda"),
        layer_id, top_k, hot_buffer_len, page_size
    )
    
    print(f"Expected: page 2 reuses page 1's slot/page_id, page 1 needs host load, page 0 evicted")
    print(f"page_ids: {page_ids[0].tolist()}")
    print(f"last_page_ids: {last_page_ids[0, 0].tolist()}")
    print(f"last_top_k_idx: {last_top_k_idx[0, 0].tolist()}")
    
    # Build mapping
    output_map = {}
    for i in range(hot_buffer_pages):
        pos = last_top_k_idx[0, 0, i].item()
        pid = last_page_ids[0, 0, i].item()
        if pos >= 0 and pid >= 0:
            output_map[pos] = pid
    print(f"  Output mapping (pos->page_id): {output_map}")
    
    # Verify: page 2 should reuse page 1's old page_id (101)
    # The replacement logic makes page 2 occupy page 1's slot
    assert 2 in output_map, "Position 2 should be in output"
    assert output_map[2] == 101, f"Position 2 should reuse page_id 101 (from page 1), got {output_map.get(2)}"
    
    # Verify: page 1 should be loaded from host (because its physical memory was overwritten)
    load_host_valid = load_tokens_host[0][load_tokens_host[0] >= 0]
    assert len(load_host_valid) == 64, f"Should load page 1 from host (64 tokens), got {len(load_host_valid)}"
    expected_host_slots = req_to_tokens_host[0, 1*page_size:2*page_size].tolist()
    assert load_host_valid.tolist() == expected_host_slots, "Host slots should be page 1's tokens"
    
    # Verify last_top_k_idx updated to [1, 2]
    assert set(last_top_k_idx[0, 0, :top_k_pages].tolist()) == {1, 2}, f"last_top_k_idx should be [1, 2], got {last_top_k_idx[0, 0, :top_k_pages].tolist()}"
    
    print("✓ Round 2 passed!")
    print("\n" + "=" * 80)
    print("✅ Simple test PASSED!")
    print("=" * 80)


def production_like_test():
    """Deterministic production-like test that mirrors the diff semantics."""
    page_size = 64
    top_k = 2048
    hot_buffer_len = 4096
    bs = 2
    num_layers = 1
    layer_id = 0
    max_seq_len = hot_buffer_len + page_size  # enough for this test

    top_k_pages = top_k // page_size  # 32
    hot_buffer_pages = hot_buffer_len // page_size  # 64

    def alloc_state():
        last_top_k_idx = torch.full((bs, num_layers, hot_buffer_pages), -1, dtype=torch.int64, device="cuda")
        last_page_ids = torch.full((bs, num_layers, hot_buffer_pages), -1, dtype=torch.int64, device="cuda")
        return last_top_k_idx, last_page_ids

    req_to_tokens_host = torch.zeros((bs, max_seq_len), dtype=torch.int64, device="cuda")
    for i in range(bs):
        req_to_tokens_host[i] = torch.arange(i * 100000, i * 100000 + max_seq_len, device="cuda", dtype=torch.int64)

    sparse_mask = torch.ones(bs, dtype=torch.int32, device="cuda")
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device="cuda")
    page_table = torch.full((bs, max_seq_len), -1, dtype=torch.int64, device="cuda")

    def run_round(seq_lens, last_top_k_idx, last_page_ids, curr_topk_pages, expected_host_pages):
        diff_map = torch.full((bs, max_seq_len), -1, dtype=torch.int32, device="cuda")
        page_ids = torch.full((bs, hot_buffer_pages), -1, dtype=torch.int64, device="cuda")
        load_tokens = torch.full((bs, hot_buffer_len), -1, dtype=torch.int64, device="cuda")
        load_tokens_host = torch.full((bs, hot_buffer_len), -1, dtype=torch.int64, device="cuda")

        page_wise_diff_triton(
            last_top_k_idx,
            curr_topk_pages,
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
            layer_id=layer_id,
            top_k=top_k,
            hot_buffer_len=hot_buffer_len,
            page_size=page_size,
        )

        # Verify host loads
        for b in range(bs):
            host_tokens = load_tokens_host[b][load_tokens_host[b] >= 0]
            expected_pages = expected_host_pages[b]
            expected_tokens = []
            for p in expected_pages:
                expected_tokens.extend(req_to_tokens_host[b, p * page_size : (p + 1) * page_size].tolist())
            assert len(host_tokens) == len(expected_tokens), (
                f"batch {b}: expected {len(expected_tokens)} host tokens for pages {expected_pages}, "
                f"got {len(host_tokens)}"
            )
            if expected_tokens:
                assert sorted(host_tokens.tolist()) == sorted(expected_tokens), (
                    f"batch {b}: host tokens mismatch for pages {expected_pages}"
                )

        return last_top_k_idx, last_page_ids

    print("\n" + "=" * 80)
    print("Production-like Test (Deterministic)")
    print("=" * 80)

    # Round 1: prev = [0..31], [10..41]; curr = same; expect host load = 0
    last_top_k_idx, last_page_ids = alloc_state()
    last_top_k_idx[0, 0, :top_k_pages] = torch.arange(0, top_k_pages, device="cuda")
    last_page_ids[0, 0, :top_k_pages] = torch.arange(1000, 1000 + top_k_pages, device="cuda")
    last_top_k_idx[1, 0, :top_k_pages] = torch.arange(10, 10 + top_k_pages, device="cuda")
    last_page_ids[1, 0, :top_k_pages] = torch.arange(2000, 2000 + top_k_pages, device="cuda")

    seq_lens = torch.tensor([top_k_pages * page_size, (10 + top_k_pages) * page_size], device="cuda")
    curr_topk = torch.full((bs, top_k_pages), -1, dtype=torch.int64, device="cuda")
    curr_topk[0] = torch.arange(0, top_k_pages, device="cuda")
    curr_topk[1] = torch.arange(10, 10 + top_k_pages, device="cuda")

    last_top_k_idx, last_page_ids = run_round(
        seq_lens,
        last_top_k_idx,
        last_page_ids,
        curr_topk,
        expected_host_pages=[set(), set()],
    )
    print("✓ Round 1 passed (no host load)")

    # Round 2: shift window +1; curr_max > last_max; last_max forced to host load
    seq_lens = seq_lens + page_size  # grow by one page
    curr_topk = torch.full((bs, top_k_pages), -1, dtype=torch.int64, device="cuda")
    curr_topk[0] = torch.arange(1, top_k_pages + 1, device="cuda")   # [1..32], expect host page 31
    curr_topk[1] = torch.arange(11, 11 + top_k_pages, device="cuda") # [11..42], expect host page 41

    last_top_k_idx, last_page_ids = run_round(
        seq_lens,
        last_top_k_idx,
        last_page_ids,
        curr_topk,
        expected_host_pages=[{31}, {41}],
    )
    print("✓ Round 2 passed (forced host load for prev last_max)")

    print("\n" + "=" * 80)
    print("✅ Production-like test PASSED")
    print("=" * 80)


if __name__ == "__main__":
    # Run simple test first
    try:
        simple_test()
    except AssertionError as e:
        print(f"\n❌ Simple test FAILED: {e}")
        import sys
        sys.exit(1)
    print("\n\n")
    production_like_test()