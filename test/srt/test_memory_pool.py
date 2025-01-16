import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool


@pytest.mark.parametrize("mem_pool_cls", [MHATokenToKVPool, MLATokenToKVPool])
@pytest.mark.parametrize("max_total_num_tokens", [2048])
def test_memory_pool_free_slot_index(mem_pool_cls, max_total_num_tokens):
    try:
        token_to_kv_pool = mem_pool_cls(
            max_total_num_tokens,
            torch.bfloat16,
            32,
            128,
            32,
            "cuda",
            False,
        )
    except:
        token_to_kv_pool = mem_pool_cls(
            max_total_num_tokens,
            torch.bfloat16,
            128,
            32,
            "cuda",
            False,
        )

    assert token_to_kv_pool.available_size() == max_total_num_tokens

    locs = []
    total_num_tokens = 0
    slots = token_to_kv_pool.slots.clone()
    for num_tokens in [7, 14, 9]:
        out_cache_loc = token_to_kv_pool.alloc(num_tokens)
        total_num_tokens += num_tokens
        assert token_to_kv_pool.free_slot_idx == total_num_tokens
        locs.append(out_cache_loc)
    for loc in reversed(locs):
        cur_free_slot_idx = token_to_kv_pool.free_slot_idx
        token_to_kv_pool.free(loc.clone())
        assert token_to_kv_pool.free_slot_idx == cur_free_slot_idx - loc.numel()
    assert torch.equal(slots, token_to_kv_pool.slots)


@pytest.mark.parametrize("mem_pool_cls", [MHATokenToKVPool, MLATokenToKVPool])
@pytest.mark.parametrize("max_total_num_tokens", [2048])
def test_memory_pool_free_slot_index_with_group(mem_pool_cls, max_total_num_tokens):
    try:
        token_to_kv_pool = mem_pool_cls(
            max_total_num_tokens,
            torch.bfloat16,
            32,
            128,
            32,
            "cuda",
            False,
        )
    except:
        token_to_kv_pool = mem_pool_cls(
            max_total_num_tokens,
            torch.bfloat16,
            128,
            32,
            "cuda",
            False,
        )
    assert token_to_kv_pool.available_size() == max_total_num_tokens
    locs = []
    total_num_tokens = 0
    slots = token_to_kv_pool.slots.clone()
    for num_tokens in [70, 142, 96]:
        out_cache_loc = token_to_kv_pool.alloc(num_tokens)
        total_num_tokens += num_tokens
        assert token_to_kv_pool.free_slot_idx == total_num_tokens
        locs.append(out_cache_loc)

    token_to_kv_pool.free_group_begin()
    for loc in locs:
        token_to_kv_pool.free(loc)
    token_to_kv_pool.free_group_end()
    assert token_to_kv_pool.free_slot_idx == 0
    assert torch.equal(slots, token_to_kv_pool.slots)


if __name__ == "__main__":
    test_memory_pool_free_slot_index(MHATokenToKVPool, 2048)
    test_memory_pool_free_slot_index_with_group(MHATokenToKVPool, 2048)
    test_memory_pool_free_slot_index(MLATokenToKVPool, 2048)
    test_memory_pool_free_slot_index_with_group(MLATokenToKVPool, 2048)
