import torch

from sglang.jit_kernel.sparse import _jit_sparse_module, load_cache_to_device_buffer_mla
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost


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
    if num_misses < 0:
        num_misses = 0

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


def _expected_misses(
    lru_slots: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    top_k_tokens: torch.Tensor,
    num_top_k: int,
    newest_token: int,
) -> int:
    top_k_set = set(top_k_tokens.tolist())
    total_hits = 0
    for slot in lru_slots.tolist():
        token = int(device_buffer_tokens[slot].item())
        if token in top_k_set:
            total_hits += 1
    newest_hit = 1 if newest_token in top_k_set else 0
    num_misses = num_top_k - total_hits - newest_hit
    if num_misses < 0:
        num_misses = 0
    return num_misses


def _build_transfer_tasks(
    batch: int, num_top_k: int, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
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


def test_sparse_lru_end_to_end(
    *,
    batch_size: int = 1,
    diff_ratio: float = 0.2,
    num_top_k: int = 2048,
    hot_buffer_size: int = 4096,
    rounds: int = 6,
    warmup: int = 4,
    block_size: int = 512,
) -> None:
    device = "cuda" 
    torch.manual_seed(0)
    diff_target = int(num_top_k * diff_ratio)
    assert diff_target > 0, "diff_ratio too small for current top_k"

    page_size = 1
    layer_id = 0
    max_seq_len = max(
        num_top_k * 4,
        num_top_k + hot_buffer_size + diff_target + 1024,
    )

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    use_nsa = True
    dtype = torch.float16

    device_pool_size = batch_size * hot_buffer_size * page_size
    mem_pool_device = MLATokenToKVPool(
        size=device_pool_size,
        page_size=page_size,
        dtype=dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
    )
    mem_pool_host = MLATokenToKVPoolHost(
        device_pool=mem_pool_device,
        host_to_device_ratio=10.0,
        host_size=0,
        page_size=page_size,
        layout="layer_first",
        pin_memory=True,
        device="cpu",
    )

    host_token_count = max_seq_len
    assert mem_pool_host.size >= batch_size * host_token_count, "Host pool too small"
    assert host_token_count >= num_top_k + 1, "Host pool too small for top-k"
    assert host_token_count >= hot_buffer_size, "Host pool too small for hot buffer"
    host_cache_locs = torch.empty(
        (batch_size, host_token_count), dtype=torch.int64, device=device
    )
    device_buffer_locs = torch.empty(
        (batch_size, 1, hot_buffer_size), dtype=torch.int32, device=device
    )
    lru_slots = (
        torch.arange(hot_buffer_size - 1, dtype=torch.int16, device=device)
        .reshape(1, 1, -1)
        .repeat(batch_size, 1, 1)
    )

    token_ids = torch.arange(mem_pool_host.size, dtype=dtype, device="cpu")
    feature_dim = mem_pool_host.kv_buffer[layer_id].shape[-1]
    token_ids = token_ids.view(-1, 1).repeat(1, feature_dim)
    mem_pool_host.kv_buffer[layer_id][: mem_pool_host.size, 0].copy_(token_ids)

    device_buffer_tokens = torch.empty(
        (batch_size, 1, hot_buffer_size), dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        host_base = b * host_token_count
        host_cache_locs[b] = torch.arange(
            host_base, host_base + host_token_count, dtype=torch.int64, device=device
        )
        device_base = b * hot_buffer_size
        device_buffer_locs[b, 0] = torch.arange(
            device_base, device_base + hot_buffer_size, dtype=torch.int32, device=device
        )
        device_buffer_tokens[b, 0] = torch.arange(
            hot_buffer_size, dtype=torch.int32, device=device
        )
        for slot in range(hot_buffer_size):
            token_id = int(device_buffer_tokens[b, 0, slot].item())
            host_loc = int(host_cache_locs[b, token_id].item())
            device_loc = int(device_buffer_locs[b, 0, slot].item())
            mem_pool_device.kv_buffer[layer_id][device_loc, 0].copy_(
                mem_pool_host.kv_buffer[layer_id][host_loc, 0]
            )

    item_size_bytes = feature_dim * mem_pool_host.dtype.itemsize
    top_k_device_locs = torch.full(
        (batch_size, num_top_k), -1, dtype=torch.int32, device=device
    )
    page_table = torch.zeros(
        batch_size, host_token_count, dtype=torch.int32, device=device
    )
    diff_map = torch.full(
        (batch_size, host_token_count), -1, dtype=torch.int16, device=device
    )
    sparse_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    transfer_tasks_src, transfer_tasks_dst = _build_transfer_tasks(
        batch_size, num_top_k, page_size
    )

    _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )

    timings_ms = []
    graph = None
    transfer_tasks_per_round: list[int] = []
    per_round_ms: list[float] = []
    prev_top_k_cpu: list[list[int] | None] = [None for _ in range(batch_size)]
    total_rounds = rounds + warmup
    seq_lens = torch.empty(batch_size, dtype=torch.int64, device=device)
    top_k_tokens = torch.empty(
        (batch_size, num_top_k), dtype=torch.int32, device=device
    )
    for round_idx in range(total_rounds):
        newest_tokens = []
        expected_lrus = []
        pre_lrus = []
        pre_tokens_list = []

        for b in range(batch_size):
            min_seq_len = num_top_k + 100
            if prev_top_k_cpu[b] is not None:
                min_seq_len = max(min_seq_len, max(prev_top_k_cpu[b]) + 2)
            max_seq_len = host_token_count - 1
            # Ensure enough candidate tokens for diff generation.
            required_seq_len = num_top_k + (hot_buffer_size - 1) + diff_target + 1
            desired_min = max(min_seq_len, required_seq_len)
            if desired_min >= max_seq_len:
                seq_len = max_seq_len
            else:
                seq_len = int(torch.randint(desired_min, max_seq_len, (1,)).item())
            seq_lens[b] = seq_len
            newest_token = seq_len - 1
            newest_tokens.append(newest_token)

            if prev_top_k_cpu[b] is None:
                curr_tokens = torch.randperm(seq_len, device=device, dtype=torch.int32)[
                    :num_top_k
                ]
                if newest_token not in curr_tokens:
                    curr_tokens[-1] = newest_token
            else:
                prev_top_k = torch.tensor(prev_top_k_cpu[b], dtype=torch.int32)
                prev_set = set(prev_top_k_cpu[b])

                if newest_token in prev_set:
                    keep_count = num_top_k - diff_target - 1
                    keep_perm = torch.randperm(num_top_k - 1)[:keep_count]
                    keep_candidates = prev_top_k[
                        torch.arange(num_top_k) != prev_top_k_cpu[b].index(newest_token)
                    ]
                    keep_tokens = keep_candidates[keep_perm]
                    keep_tokens = torch.cat(
                        [keep_tokens, torch.tensor([newest_token], dtype=torch.int32)]
                    )
                    new_count = diff_target
                    include_newest_in_new = False
                else:
                    keep_count = num_top_k - diff_target
                    keep_perm = torch.randperm(num_top_k)[:keep_count]
                    keep_tokens = prev_top_k[keep_perm]
                    new_count = diff_target - 1
                    include_newest_in_new = True

                forbidden = set(
                    device_buffer_tokens[b, 0, : hot_buffer_size - 1].tolist()
                )
                candidates = [
                    t
                    for t in range(seq_len)
                    if t not in prev_set and t not in forbidden and t != newest_token
                ]
                if len(candidates) < new_count:
                    # Fallback: allow tokens already in LRU region (hits are OK),
                    # keep uniqueness against prev_set and newest_token.
                    candidates = [
                        t
                        for t in range(seq_len)
                        if t not in prev_set and t != newest_token
                    ]
                assert len(candidates) >= new_count, (
                    "Not enough candidate tokens: "
                    f"need={new_count} have={len(candidates)} "
                    f"seq_len={seq_len} host_tokens={host_token_count} "
                    f"prev_set={len(prev_set)} forbidden={len(forbidden)}"
                )
                new_indices = torch.randperm(len(candidates))[:new_count].tolist()
                new_tokens = torch.tensor(
                    [candidates[i] for i in new_indices], dtype=torch.int32
                )
                if include_newest_in_new:
                    new_tokens = torch.cat(
                        [new_tokens, torch.tensor([newest_token], dtype=torch.int32)]
                    )
                curr_tokens = torch.cat([keep_tokens, new_tokens], dim=0)
                perm = torch.randperm(num_top_k)
                curr_tokens = curr_tokens[perm]

            top_k_tokens[b] = curr_tokens

            newest_slot = hot_buffer_size - 1
            device_buffer_tokens[b, 0, newest_slot] = newest_token
            host_loc = int(host_cache_locs[b, newest_token].item())
            device_loc = int(device_buffer_locs[b, 0, newest_slot].item())
            mem_pool_device.kv_buffer[layer_id][device_loc, 0].copy_(
                mem_pool_host.kv_buffer[layer_id][host_loc, 0]
            )

            # Ensure newest_token is not duplicated in the LRU region.
            lru_region = device_buffer_tokens[b, 0, :newest_slot]
            dup_mask = lru_region == newest_token
            if dup_mask.any():
                lru_tokens = set(lru_region.tolist())
                topk_set = set(curr_tokens.tolist())
                replacement = (newest_token + 1) % host_token_count
                while (
                    replacement in lru_tokens
                    or replacement in topk_set
                    or replacement == newest_token
                ):
                    replacement = (replacement + 1) % host_token_count
                lru_region[dup_mask] = replacement
                rep_host_loc = int(host_cache_locs[b, replacement].item())
                dup_indices = dup_mask.nonzero(as_tuple=False).view(-1).tolist()
                for slot in dup_indices:
                    rep_device_loc = int(device_buffer_locs[b, 0, slot].item())
                    mem_pool_device.kv_buffer[layer_id][rep_device_loc, 0].copy_(
                        mem_pool_host.kv_buffer[layer_id][rep_host_loc, 0]
                    )

            pre_lru = lru_slots[b, 0].detach().clone()
            pre_tokens = device_buffer_tokens[b, 0].detach().clone()
            expected_lru = _expected_lru(
                pre_lru,
                pre_tokens,
                top_k_tokens[b],
                num_top_k,
                hot_buffer_size,
                newest_token,
            )
            pre_lrus.append(pre_lru)
            pre_tokens_list.append(pre_tokens)
            expected_lrus.append(expected_lru)

        top_k_device_locs.fill_(-1)

        if graph is None:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                load_cache_to_device_buffer_mla(
                    top_k_tokens,
                    device_buffer_tokens,
                    host_cache_locs,
                    device_buffer_locs,
                    mem_pool_host.kv_buffer[layer_id],
                    mem_pool_device.kv_buffer[layer_id],
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
                    block_size=block_size,
                    num_top_k=num_top_k,
                    hot_buffer_size=hot_buffer_size,
                )
            torch.cuda.synchronize()
            elapsed_ms = 0.0
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)

        if round_idx >= warmup:
            timings_ms.append(elapsed_ms)
            per_round_ms.append(elapsed_ms)

        tasks_per_block = num_top_k * page_size
        stride_per_block = tasks_per_block + 1
        round_transfer_tasks = 0
        for b in range(batch_size):
            if round_idx >= warmup:
                got_lru = lru_slots[b, 0]
                assert torch.equal(
                    got_lru, expected_lrus[b]
                ), f"LRU mismatch round {round_idx} batch {b}"

                token_to_slot = {
                    int(device_buffer_tokens[b, 0, i].item()): i
                    for i in range(hot_buffer_size)
                }
                for i in range(num_top_k):
                    token_id = int(top_k_tokens[b, i].item())
                    device_page = int(top_k_device_locs[b, i].item())
                    assert device_page >= 0, "top_k_device_locs contains -1"
                    if token_id in token_to_slot:
                        expected_loc = int(
                            device_buffer_locs[b, 0, token_to_slot[token_id]].item()
                        )
                        assert device_page == expected_loc

                if prev_top_k_cpu[b] is not None:
                    curr_top_k_cpu = top_k_tokens[b].cpu().tolist()
                    prev_set = set(prev_top_k_cpu[b])
                    curr_set = set(curr_top_k_cpu)
                    diff_count = len(curr_set - prev_set)
                    if diff_count != diff_target:
                        print(
                            "[topk diff mismatch]",
                            "round",
                            round_idx,
                            "batch",
                            b,
                            "diff",
                            diff_count,
                            "target",
                            diff_target,
                            "seq_len",
                            int(seq_lens[b].item()),
                        )
                    assert (
                        diff_count == diff_target
                    ), f"topk diff {diff_count} != {diff_target}"

                    expected_misses = _expected_misses(
                        pre_lrus[b],
                        pre_tokens_list[b],
                        top_k_tokens[b],
                        num_top_k,
                        newest_tokens[b],
                    )
                    count_idx = b * stride_per_block + tasks_per_block
                    task_count = int(transfer_tasks_src[count_idx].item())
                    round_transfer_tasks += task_count
                    assert (
                        task_count == expected_misses * page_size
                    ), f"transfer cnt {task_count} != {expected_misses}"

                for i in range(num_top_k):
                    token_id = int(top_k_tokens[b, i].item())
                    device_page = int(top_k_device_locs[b, i].item())
                    host_loc = int(host_cache_locs[b, token_id].item())
                    device_loc = device_page
                    assert torch.equal(
                        mem_pool_device.kv_buffer[layer_id][device_loc, 0].cpu(),
                        mem_pool_host.kv_buffer[layer_id][host_loc, 0],
                    )

            prev_top_k_cpu[b] = top_k_tokens[b].cpu().tolist()

        if round_idx >= warmup:
            transfer_tasks_per_round.append(round_transfer_tasks)
            print(
                f"[round] idx={round_idx - warmup} "
                f"time_ms={elapsed_ms:.3f} transfer_tasks={round_transfer_tasks}"
            )

    total = sum(timings_ms)
    avg = total / len(timings_ms) if timings_ms else 0.0
    print(
        f"[timing] batch={batch_size} diff_ratio={diff_ratio:.2f} "
        f"rounds={rounds} avg_ms={avg:.3f} total_ms={total:.3f}"
    )
    print(f"[transfer tasks] per_round={transfer_tasks_per_round}")
    print(f"[timing] per_round_ms={per_round_ms}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sparse LRU end-to-end test (MLA)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--diff_ratio", type=float, default=0.2)
    parser.add_argument("--num_top_k", type=int, default=2048)
    parser.add_argument("--hot_buffer_size", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=512)
    args = parser.parse_args()

    test_sparse_lru_end_to_end(
        batch_size=args.batch_size,
        diff_ratio=args.diff_ratio,
        num_top_k=args.num_top_k,
        hot_buffer_size=args.hot_buffer_size,
        rounds=args.rounds,
        warmup=args.warmup,
        block_size=args.block_size,
    )
