from __future__ import annotations

import torch

from sglang.srt.kv_cache_canary.plan_input import walk_radix_cache_for_canary
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


def test_empty_radix_returns_zero_extras(device, make_radix_cache):
    cache = make_radix_cache([[]])
    slots, positions, prev_slots = walk_radix_cache_for_canary(
        radix_cache=cache, alive_running_req_pool_indices=set()
    )
    assert slots.numel() == 0
    assert positions.numel() == 0
    assert prev_slots.numel() == 0


def test_single_node_chain_positions_increase(device, make_radix_cache):
    chain = [10, 20, 30, 40]
    cache = make_radix_cache([[], chain])
    slots, positions, prev_slots = walk_radix_cache_for_canary(
        radix_cache=cache, alive_running_req_pool_indices=set()
    )
    assert slots.tolist() == chain
    assert positions.tolist() == [0, 1, 2, 3]
    assert prev_slots.tolist() == [-1, 10, 20, 30]


def test_child_node_first_slot_prev_is_parent_last(device, make_radix_cache):
    parent = [7, 8]
    child = [9, 10]
    cache = make_radix_cache([[], parent, child])
    slots, positions, prev_slots = walk_radix_cache_for_canary(
        radix_cache=cache, alive_running_req_pool_indices=set()
    )
    assert slots.tolist() == parent + child
    assert prev_slots.tolist()[len(parent)] == parent[-1]


def test_root_child_first_slot_prev_minus_one(device, make_radix_cache):
    cache = make_radix_cache([[], [42, 43]])
    _, _, prev_slots = walk_radix_cache_for_canary(
        radix_cache=cache, alive_running_req_pool_indices=set()
    )
    assert int(prev_slots[0]) == -1


def test_position_equals_depth_from_root(device, make_radix_cache):
    cache = make_radix_cache([[], [1, 2], [3], [4, 5]])
    slots, positions, _ = walk_radix_cache_for_canary(
        radix_cache=cache, alive_running_req_pool_indices=set()
    )
    assert positions.tolist() == [0, 1, 2, 3, 4]
    assert slots.tolist() == [1, 2, 3, 4, 5]


def test_skips_slots_owned_by_running_reqs(device, make_radix_cache):
    cache = make_radix_cache([[], [100, 101]])
    slots, _, _ = walk_radix_cache_for_canary(
        radix_cache=cache, alive_running_req_pool_indices=set()
    )
    assert set(slots.tolist()) == {100, 101}
    assert slots.dtype == torch.int32
