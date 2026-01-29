import torch

from sglang.jit_kernel.sparse import load_cache_to_device_buffer
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost


def _expected_lru(
    lru_slots: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    top_k_tokens: torch.Tensor,
    num_top_k: int,
    hot_buffer_size: int,
    newest_token: int,
) -> torch.Tensor:
    lru_size = hot_buffer_size - 1
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
    newest_hit = 1 if newest_token in top_k_set else 0
    num_misses = num_top_k - total_hits - newest_hit
    assert num_misses >= 0

    total_evictable = lru_size - total_hits
    assert num_misses <= total_evictable

    lru_out = [None] * lru_size
    front_size = total_evictable - num_misses
    front_slots = evictable_slots[num_misses:]
    assert len(front_slots) == front_size

    for i, slot in enumerate(front_slots):
        lru_out[i] = slot

    middle_base = front_size
    miss_slots = list(reversed(evictable_slots[:num_misses]))
    for i, slot in enumerate(miss_slots):
        lru_out[middle_base + i] = slot

    end_base = lru_size - total_hits
    for i, slot in enumerate(hit_slots):
        lru_out[end_base + i] = slot

    assert all(v is not None for v in lru_out), "LRU output has unfilled slots"
    return torch.tensor(lru_out, dtype=torch.int16, device=lru_slots.device)


def _build_transfer_tasks(batch: int, num_top_k: int, page_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    tasks_per_block = num_top_k * page_size
    stride_per_block = tasks_per_block + 1
    max_transfer_tasks = batch * stride_per_block
    transfer_tasks_src = torch.full(
        (max_transfer_tasks,), -1, dtype=torch.int64, device="cuda"
    )
    transfer_tasks_dst = torch.full(
        (max_transfer_tasks,), -1, dtype=torch.int64, device="cuda"
    )
    return transfer_tasks_src, transfer_tasks_dst


def test_sparse_lru_end_to_end() -> None:
    device = "cuda"
    torch.manual_seed(0)

    num_top_k = 6
    hot_buffer_size = 8
    page_size = 1
    head_num = 8
    head_dim = 16
    layer_id = 0
    batch = 1
    pool_size = 1

    item_size_bytes = head_num * head_dim * torch.float16.itemsize

    # Allocate real pools (matches SparseKVCacheManager usage).
    device_pool_size = pool_size * hot_buffer_size * page_size
    mem_pool_device = MHATokenToKVPool(
        size=device_pool_size,
        page_size=page_size,
        dtype=torch.float16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
    )
    mem_pool_host = MHATokenToKVPoolHost(
        device_pool=mem_pool_device,
        host_to_device_ratio=4.0,
        host_size=0,
        page_size=page_size,
        layout="layer_first",
        pin_memory=True,
        device="cpu",
    )

    # Host mapping and buffer mappings.
    host_token_count = mem_pool_host.size
    host_pages = host_token_count // page_size
    assert host_pages > 24, "Host pool too small for test top-k values"
    host_cache_locs = torch.arange(
        host_token_count, dtype=torch.int64, device=device
    ).reshape(pool_size, -1)
    device_buffer_locs = torch.arange(
        hot_buffer_size, dtype=torch.int32, device=device
    ).reshape(pool_size, 1, -1)
    lru_slots = torch.arange(
        hot_buffer_size - 1, dtype=torch.int16, device=device
    ).reshape(pool_size, 1, -1)

    # Seed host pool with deterministic data.
    token_ids = torch.arange(host_token_count, dtype=torch.float16, device="cpu")
    token_ids = token_ids.view(-1, 1, 1).repeat(1, head_num, head_dim)
    mem_pool_host.k_buffer[layer_id][:host_token_count].copy_(token_ids)
    mem_pool_host.v_buffer[layer_id][:host_token_count].copy_(token_ids + 1.0)

    # Initialize device buffer tokens and preload corresponding K/V.
    device_buffer_tokens = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 0],
        dtype=torch.int32,
        device=device,
    ).reshape(pool_size, 1, -1)
    for slot in range(hot_buffer_size):
        token_id = int(device_buffer_tokens[0, 0, slot].item())
        for page_offset in range(page_size):
            host_loc = int(host_cache_locs[0, token_id * page_size + page_offset].item())
            device_loc = int(device_buffer_locs[0, 0, slot].item()) * page_size + page_offset
            mem_pool_device.k_buffer[layer_id][device_loc].copy_(
                mem_pool_host.k_buffer[layer_id][host_loc]
            )
            mem_pool_device.v_buffer[layer_id][device_loc].copy_(
                mem_pool_host.v_buffer[layer_id][host_loc]
            )

    top_k_device_locs = torch.full(
        (batch, num_top_k), -1, dtype=torch.int32, device=device
    )
    page_table = torch.zeros(batch, host_token_count, dtype=torch.int32, device=device)
    diff_map = torch.full((batch, host_token_count), -1, dtype=torch.int16, device=device)
    sparse_mask = torch.tensor([True], dtype=torch.bool, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)
    transfer_tasks_src, transfer_tasks_dst = _build_transfer_tasks(batch, num_top_k, page_size)

    # Multi-round LRU test with changing seq_len/newest token.
    rounds = [
        {"seq_len": 15, "top_k": [1, 3, 6, 9, 10, 11]},
        {"seq_len": 17, "top_k": [2, 3, 6, 10, 11, 12]},
        {"seq_len": 19, "top_k": [0, 4, 5, 9, 11, 13]},
        # max topk equals seq_len - 1
        {"seq_len": 21, "top_k": [1, 2, 4, 7, 8, 20]},
        # max topk smaller than previous round's max
        {"seq_len": 23, "top_k": [0, 1, 2, 3, 4, 5]},
        # mix of hits/misses with smaller max
        {"seq_len": 25, "top_k": [6, 7, 8, 9, 10, 24]},
    ]

    for round_idx, round_cfg in enumerate(rounds):
        seq_lens = torch.tensor([round_cfg["seq_len"]], dtype=torch.int64, device=device)
        newest_token = int((seq_lens[0].item() - 1) // page_size)
        newest_slot = hot_buffer_size - 1
        top_k_list = list(round_cfg["top_k"])
        if newest_token not in top_k_list:
            top_k_list[-1] = newest_token
        top_k_tokens = torch.tensor(
            top_k_list, dtype=torch.int32, device=device
        ).reshape(batch, -1)

        # Simulate the latest token written to newest_slot on device.
        device_buffer_tokens[0, 0, newest_slot] = newest_token
        for page_offset in range(page_size):
            host_loc = int(host_cache_locs[0, newest_token * page_size + page_offset].item())
            device_loc = int(device_buffer_locs[0, 0, newest_slot].item()) * page_size + page_offset
            mem_pool_device.k_buffer[layer_id][device_loc].copy_(
                mem_pool_host.k_buffer[layer_id][host_loc]
            )
            mem_pool_device.v_buffer[layer_id][device_loc].copy_(
                mem_pool_host.v_buffer[layer_id][host_loc]
            )

        pre_lru = lru_slots[0, 0].detach().clone()
        pre_tokens = device_buffer_tokens[0, 0].detach().clone()
        expected_lru = _expected_lru(
            pre_lru, pre_tokens, top_k_tokens[0], num_top_k, hot_buffer_size, newest_token
        )

        load_cache_to_device_buffer(
            top_k_tokens,
            device_buffer_tokens,
            host_cache_locs,
            device_buffer_locs,
            mem_pool_host.k_buffer[layer_id],
            mem_pool_host.v_buffer[layer_id],
            mem_pool_device.k_buffer[layer_id],
            mem_pool_device.v_buffer[layer_id],
            top_k_device_locs,
            page_table,
            diff_map,
            req_pool_indices,
            sparse_mask,
            seq_lens,
            lru_slots,
            transfer_tasks_src,
            transfer_tasks_dst,
            page_size,
            layer_id,
            item_size_bytes,
            num_top_k=num_top_k,
            hot_buffer_size=hot_buffer_size,
        )
        torch.cuda.synchronize()

        # 1) LRU eviction logic
        got_lru = lru_slots[0, 0]
        if not torch.equal(got_lru, expected_lru):
            print(
                "[lru mismatch] round",
                round_idx,
                "seq_len",
                int(seq_lens[0].item()),
                "newest_token",
                newest_token,
                "top_k",
                top_k_list,
            )
            print("[lru mismatch] pre_lru", pre_lru.tolist())
            print("[lru mismatch] pre_tokens", pre_tokens.tolist())
            print("[lru mismatch] expected", expected_lru.tolist())
            print("[lru mismatch] got", got_lru.tolist())
        assert torch.equal(
            got_lru, expected_lru
        ), f"LRU mismatch round {round_idx}"

        # 2) max topk logic (newest token binds to newest_slot)
        newest_idx = int((top_k_tokens[0] == newest_token).nonzero(as_tuple=False)[0].item())
        expected_newest_loc = int(device_buffer_locs[0, 0, newest_slot].item())
        assert int(top_k_device_locs[0, newest_idx].item()) == expected_newest_loc

        # 3) data transfer correctness: verify device cache matches host for top-k
        assert int(top_k_device_locs.min().item()) >= 0, "top_k_device_locs contains -1"
        for i in range(num_top_k):
            token_id = int(top_k_tokens[0, i].item())
            device_page = int(top_k_device_locs[0, i].item())
            for page_offset in range(page_size):
                host_loc = int(host_cache_locs[0, token_id * page_size + page_offset].item())
                device_loc = device_page * page_size + page_offset
                assert torch.equal(
                    mem_pool_device.k_buffer[layer_id][device_loc].cpu(),
                    mem_pool_host.k_buffer[layer_id][host_loc],
                )
                assert torch.equal(
                    mem_pool_device.v_buffer[layer_id][device_loc].cpu(),
                    mem_pool_host.v_buffer[layer_id][host_loc],
                )


if __name__ == "__main__":
    test_sparse_lru_end_to_end()