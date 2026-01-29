import torch

from sglang.jit_kernel.sparse import load_cache_to_device_buffer_mla


def _expected_lru(
    lru_slots: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    top_k_tokens: torch.Tensor,
    num_top_k: int,
    hot_buffer_size: int,
) -> torch.Tensor:
    top_k_set = set(top_k_tokens.tolist())
    hit_slots: list[int] = []
    evictable_slots: list[int] = []
    for slot in lru_slots.tolist():
        token = int(device_buffer_tokens[slot].item())
        if token in top_k_set:
            hit_slots.append(int(slot))
        else:
            evictable_slots.append(int(slot))

    total_hits = len(hit_slots)
    assert total_hits <= num_top_k
    num_misses = num_top_k - total_hits

    lru_out = [None] * hot_buffer_size

    # Front region: older evictable slots that are not used for current misses.
    front_size = hot_buffer_size - num_top_k
    front_slots = evictable_slots[num_misses:]
    assert len(front_slots) == front_size
    for i, slot in enumerate(front_slots):
        lru_out[i] = slot

    # Middle region: slots used for current misses.
    middle_base = hot_buffer_size - num_top_k
    for i, slot in enumerate(evictable_slots[:num_misses]):
        lru_out[middle_base + i] = slot

    # End region: hits (most recent).
    end_base = hot_buffer_size - total_hits
    for i, slot in enumerate(hit_slots):
        lru_out[end_base + i] = slot

    assert all(v is not None for v in lru_out), "LRU output has unfilled slots"
    return torch.tensor(lru_out, dtype=torch.int16, device=lru_slots.device)


def _round_trip_lru_test() -> None:
    device = "cuda"
    torch.set_default_dtype(torch.bfloat16)

    # Small, deterministic sizes for LRU validation.
    num_top_k = 8
    hot_buffer_size = 16
    item_size = 64
    token_stride_size = item_size * torch.bfloat16.itemsize
    total_items_in_pool = 64

    # Single-batch, single-layer inputs.
    batch = 1
    layer_id = 0

    host_cache_k = torch.randn(total_items_in_pool * 2, item_size).pin_memory()
    device_buffer_k = torch.randn(total_items_in_pool, item_size, device=device)

    sparse_mask = torch.tensor([True], device=device)
    page_table = torch.zeros(batch, 64, dtype=torch.int32, device=device)
    diff_map = torch.full((batch, 128), -1, dtype=torch.int16, device=device)
    seq_lens = torch.tensor([64], dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)

    tasks_per_block = num_top_k * 1
    stride_per_block = tasks_per_block + 1
    max_transfer_tasks = batch * stride_per_block
    transfer_tasks_src = torch.full(
        (max_transfer_tasks,), -1, dtype=torch.int64, device="cuda"
    )
    transfer_tasks_dst = torch.full(
        (max_transfer_tasks,), -1, dtype=torch.int64, device="cuda"
    )
    # LRU slots in LRU order (0 is oldest).
    lru_slots = torch.arange(hot_buffer_size, dtype=torch.int16, device=device).reshape(1, 1, -1)

    # Buffer tokens: mostly 0..15, but replace 7 with 16 so token 7 will miss.
    buffer_tokens = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 16, 8, 9, 10, 11, 12, 13, 14, 15],
        dtype=torch.int32,
        device=device,
    ).reshape(1, 1, -1)
    buffer_locs = torch.arange(
        hot_buffer_size, dtype=torch.int32, device=device
    ).reshape(1, 1, -1)

    # host_cache_locs needs to cover miss tokens (page_size = 1).
    host_cache_locs = torch.arange(128, dtype=torch.int64, device=device).reshape(1, -1)

    top_k_device_locs = torch.zeros(batch, num_top_k, dtype=torch.int32, device=device)

    # Two rounds with different top_k selections.
    rounds = [
        torch.tensor([0, 1, 2, 3, 4, 5, 7, 16], dtype=torch.int32, device=device),
        torch.tensor([8, 9, 10, 11, 12, 13, 6, 16], dtype=torch.int32, device=device),
        torch.tensor([1, 3, 5, 7, 9, 11, 13, 16], dtype=torch.int32, device=device),
    ]

    for round_idx, top_k in enumerate(rounds):
        top_k_tokens = top_k.reshape(1, -1)

        # Snapshot pre-call state for reference.
        pre_lru = lru_slots[0, 0].detach()
        pre_tokens = buffer_tokens[0, 0].detach()
        expected_lru = _expected_lru(
            pre_lru, pre_tokens, top_k, num_top_k, hot_buffer_size
        )

        load_cache_to_device_buffer_mla(
            top_k_tokens,
            buffer_tokens,
            host_cache_locs,
            buffer_locs,
            host_cache_k,
            device_buffer_k,
            top_k_device_locs,
            page_table,
            diff_map,
            req_pool_indices,
            sparse_mask,
            seq_lens,
            lru_slots,
            transfer_tasks_src,
            transfer_tasks_dst,
            1,  # page_size
            layer_id,
            token_stride_size,
            num_top_k=num_top_k,
            hot_buffer_size=hot_buffer_size,
        )
        torch.cuda.synchronize()

        got_lru = lru_slots[0, 0]
        assert sorted(got_lru.tolist()) == sorted(
            pre_lru.tolist()
        ), f"LRU permutation broken at round {round_idx}"
        assert torch.equal(
            got_lru, expected_lru
        ), f"LRU mismatch at round {round_idx}: got={got_lru.tolist()} expected={expected_lru.tolist()}"


if __name__ == "__main__":
    _round_trip_lru_test()