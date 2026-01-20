import pytest
import torch

WARP_SIZE = 32


def _make_host_cache(num_tokens: int, item_size_bytes: int) -> torch.Tensor:
    host_cache = torch.zeros(
        (num_tokens, item_size_bytes), dtype=torch.uint8, pin_memory=True
    )
    token_ids = torch.arange(num_tokens, dtype=torch.int32)
    header = torch.stack(
        [
            (token_ids & 0xFF),
            ((token_ids >> 8) & 0xFF),
            ((token_ids >> 16) & 0xFF),
            ((token_ids >> 24) & 0xFF),
        ],
        dim=1,
    ).to(torch.uint8)
    host_cache[:, :4] = header
    return host_cache


def _eviction_order(hot_buffer_size: int, last_evicted_slot: int) -> list[int]:
    num_buffer_chunks = (hot_buffer_size + WARP_SIZE - 1) // WARP_SIZE
    physical_chunk_offset = (last_evicted_slot + WARP_SIZE - 1) // WARP_SIZE
    order: list[int] = []
    for chunk_idx in range(num_buffer_chunks):
        physical_chunk_idx = (chunk_idx + physical_chunk_offset) % num_buffer_chunks
        base = physical_chunk_idx * WARP_SIZE
        for lane_id in range(WARP_SIZE):
            slot = base + lane_id
            if slot < hot_buffer_size:
                order.append(slot)
    return order


def _simulate_cache_update(
    top_k_tokens: list[int],
    device_buffer_tokens: list[int],
    hot_buffer_size: int,
    last_evicted_slot: int,
    req_length: int,
) -> tuple[list[int], list[int], int]:
    topk_index_by_token = {token: idx for idx, token in enumerate(top_k_tokens)}
    topk_slots = [0] * (len(top_k_tokens) + 1)
    hit_mask = [False] * len(top_k_tokens)
    evictable_slots: list[int] = []

    for slot in _eviction_order(hot_buffer_size, last_evicted_slot):
        token = device_buffer_tokens[slot]
        match_idx = topk_index_by_token.get(token)
        if match_idx is None:
            evictable_slots.append(slot)
        else:
            topk_slots[match_idx] = slot
            hit_mask[match_idx] = True

    miss_offset = 0
    for idx, token in enumerate(top_k_tokens):
        if not hit_mask[idx]:
            slot = evictable_slots[miss_offset]
            device_buffer_tokens[slot] = token
            topk_slots[idx] = slot
            miss_offset += 1

    extra_slot = evictable_slots[miss_offset]
    device_buffer_tokens[extra_slot] = req_length
    topk_slots[len(top_k_tokens)] = extra_slot
    last_evicted_slot = extra_slot
    return device_buffer_tokens, topk_slots, last_evicted_slot


def _build_buffer_tokens(
    top_k: int, hot_buffer_size: int, scenario: str
) -> list[int]:
    if scenario == "full_hit":
        tokens = list(range(top_k))
        tokens.extend(range(top_k, top_k + hot_buffer_size - top_k))
        return tokens
    if scenario == "partial_miss":
        hits = top_k // 2
        tokens = list(range(hits))
        tokens.extend(range(top_k, top_k + hot_buffer_size - hits))
        return tokens
    if scenario == "full_miss":
        return list(range(top_k, top_k + hot_buffer_size))
    raise ValueError(f"Unknown scenario: {scenario}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(torch.version.hip is not None, reason="HIP not supported")
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("hot_buffer_multiplier", [1, 2, 4])
@pytest.mark.parametrize("scenario", ["full_hit", "partial_miss", "full_miss"])
@torch.inference_mode()
def test_sparse_cache_load_and_hits(
    top_k: int, hot_buffer_multiplier: int, scenario: str
) -> None:
    hot_buffer_size = (
        top_k + 1 if hot_buffer_multiplier == 1 else top_k * hot_buffer_multiplier
    )
    item_size_bytes = 128
    req_length = top_k + hot_buffer_size + 5
    buffer_tokens = _build_buffer_tokens(top_k, hot_buffer_size, scenario)
    max_token = max(req_length, max(buffer_tokens), top_k - 1)
    host_cache = _make_host_cache(max_token + 1, item_size_bytes)

    top_k_tokens = torch.arange(top_k, dtype=torch.int32, device="cuda")
    device_buffer_tokens = torch.tensor(
        buffer_tokens, dtype=torch.int32, device="cuda"
    )
    host_cache_locs = torch.arange(
        max_token + 1, dtype=torch.int64, device="cuda"
    )
    device_buffer_locs = torch.arange(
        hot_buffer_size, dtype=torch.int64, device="cuda"
    )
    device_buffer = torch.empty(
        (hot_buffer_size, item_size_bytes), dtype=torch.uint8, device="cuda"
    )
    top_k_device_locs = torch.empty(
        (top_k + 1,), dtype=torch.int64, device="cuda"
    )
    last_evicted_slot = torch.tensor([0], dtype=torch.int32, device="cuda")

    init_buffer_cpu = host_cache[torch.tensor(buffer_tokens, dtype=torch.int64)]
    device_buffer.copy_(init_buffer_cpu.to(device_buffer.device))

    expected_tokens, expected_slots, expected_last = _simulate_cache_update(
        top_k_tokens.cpu().tolist(),
        device_buffer_tokens.cpu().tolist(),
        hot_buffer_size,
        last_evicted_slot.item(),
        req_length,
    )

    torch.ops.sgl_kernel.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        item_size_bytes,
        req_length,
        last_evicted_slot,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        device_buffer_tokens.cpu(), torch.tensor(expected_tokens, dtype=torch.int32)
    )
    torch.testing.assert_close(
        top_k_device_locs.cpu(), torch.tensor(expected_slots, dtype=torch.int64)
    )
    assert last_evicted_slot.item() == expected_last

    slots = top_k_device_locs[:top_k]
    actual_data = device_buffer.index_select(0, slots).cpu()
    expected_data = host_cache.index_select(
        0, top_k_tokens.cpu().to(torch.int64)
    )
    torch.testing.assert_close(actual_data, expected_data)


if __name__ == "__main__":
    pytest.main([__file__])
