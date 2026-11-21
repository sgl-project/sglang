"""Correctness of the one-launch expert-parallel moe_align_ep_local kernel.

A plain-torch oracle of the documented contract runs everywhere, the Triton
interpreter included; on CUDA the op is also compared with the pair it replaces,
_map_topk_ids_to_local_experts plus moe_align_block_size(ignore_invalid_expert).
"""

import os
import sys

import pytest
import torch

from sglang.kernels.ops.moe.moe_align_ep_local import (
    ep_local_align_supported,
    moe_align_ep_local,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
NUM_GLOBAL, NUM_LOCAL, TOP_K = 256, 32, 8


def _expert_map(rank, num_local=NUM_LOCAL):
    expert_map = torch.full((NUM_GLOBAL,), -1, dtype=torch.int32)
    expert_map[rank * num_local : (rank + 1) * num_local] = torch.arange(
        num_local, dtype=torch.int32
    )
    return expert_map


def _skewed_ids(tokens, seed):
    """Distinct experts per token, with a few hot experts taking most routes."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.ones(NUM_GLOBAL)
    weights[:6] = 40.0
    return torch.stack(
        [
            torch.multinomial(weights, TOP_K, replacement=False, generator=generator)
            for _ in range(tokens)
        ]
    ).to(torch.int32)


def _align(topk_ids, expert_map, block, num_local=NUM_LOCAL):
    return moe_align_ep_local(
        topk_ids=topk_ids.to(DEVICE),
        expert_map=expert_map.to(DEVICE),
        num_local_experts=num_local,
        block_size=block,
    )


def _reference(topk_ids, expert_map, block, num_local):
    ids = topk_ids.long().flatten()
    in_range = (ids >= 0) & (ids < expert_map.numel())
    local = torch.full(ids.shape, -1, dtype=torch.int32)
    local[in_range] = expert_map[ids[in_range]]
    local[(local < 0) | (local >= num_local)] = -1
    sorted_ids, block_experts = [], []
    for expert in range(num_local):
        rows = (local == expert).nonzero().flatten().tolist()
        padded = -(-len(rows) // block) * block
        sorted_ids += rows + [ids.numel()] * (padded - len(rows))
        block_experts += [expert] * (padded // block)
    return local.view(topk_ids.shape), sorted_ids, block_experts


def _check(topk_ids, expert_map, block, num_local=NUM_LOCAL):
    local, sorted_ids, block_experts, total = (
        t.cpu() for t in _align(topk_ids, expert_map, block, num_local)
    )
    total = int(total.item())
    want_local, want_sorted, want_blocks = _reference(
        topk_ids, expert_map, block, num_local
    )
    assert local.dtype == torch.int32 and torch.equal(local, want_local)
    assert total == len(want_sorted)
    assert sorted_ids[:total].tolist() == want_sorted
    assert block_experts[: total // block].tolist() == want_blocks
    return total


@pytest.mark.parametrize("tokens", [1, 4, 8, 33, 128])
@pytest.mark.parametrize("block", [8, 16, 24, 48, 64])
@pytest.mark.parametrize("rank", [0, 3])
def test_matches_reference(tokens, block, rank):
    topk_ids = _skewed_ids(tokens, seed=tokens * 7 + block)
    if rank == 0:
        topk_ids[0] = torch.arange(TOP_K, dtype=torch.int32)
    _check(topk_ids, _expert_map(rank), block)


@pytest.mark.skipif(DEVICE == "cpu", reason="moe_align_block_size needs CUDA")
@pytest.mark.parametrize("tokens", [1, 8, 33, 128])
@pytest.mark.parametrize("block", [8, 24, 64])
def test_matches_the_pair_it_replaces(tokens, block):
    """Same buffers, blocks and routes per expert as the stock map and align;
    the stock kernel orders routes inside an expert by atomics."""
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        _map_topk_ids_to_local_experts,
    )

    expert_map = _expert_map(3).cuda()
    topk_ids = _skewed_ids(tokens, seed=tokens * 11 + block).cuda()
    mapped = _map_topk_ids_to_local_experts(topk_ids, expert_map).contiguous()
    want_sorted, want_blocks, want_total = moe_align_block_size(
        topk_ids=mapped,
        block_size=block,
        num_experts=NUM_LOCAL,
        ignore_invalid_expert=True,
    )
    local, sorted_ids, block_experts, total = _align(topk_ids, expert_map, block)
    assert torch.equal(local, mapped.to(torch.int32))
    assert sorted_ids.numel() == want_sorted.numel()
    assert block_experts.numel() == want_blocks.numel()
    num_blocks = int(total.item()) // block
    assert int(want_total.item()) // block == num_blocks
    assert torch.equal(block_experts[:num_blocks], want_blocks[:num_blocks])

    def routes_by_expert(sorted_rows, blocks):
        rows = sorted_rows[: num_blocks * block].view(num_blocks, block).cpu()
        out = {}
        for expert, row in zip(blocks[:num_blocks].tolist(), rows):
            out.setdefault(expert, set()).update(
                r for r in row.tolist() if r < mapped.numel()
            )
        return out

    assert routes_by_expert(sorted_ids, block_experts) == routes_by_expert(
        want_sorted, want_blocks
    )


def test_no_local_routes():
    topk_ids = torch.arange(TOP_K, dtype=torch.int32).repeat(4, 1)
    assert _check(topk_ids, _expert_map(7), block=8) == 0


@pytest.mark.parametrize("num_local", [16, 32])
def test_one_launch_takes_1024_routes_up_to_32_local_experts(num_local):
    """EP8 of 256 experts aligns up to 128 tokens at top 8 in one launch; past
    1024 routes the op refuses rather than spilling its register tile."""
    assert ep_local_align_supported(num_routes=1024, num_local_experts=num_local)
    assert not ep_local_align_supported(num_routes=1025, num_local_experts=num_local)
    expert_map = _expert_map(0, num_local)
    _check(_skewed_ids(1024 // TOP_K, seed=2), expert_map, 48, num_local)
    with pytest.raises(ValueError, match="does not cover"):
        _align(
            torch.zeros((1025, 1), dtype=torch.int32),
            expert_map,
            block=48,
            num_local=num_local,
        )


@pytest.mark.parametrize("num_local", [33, 64, 128])
def test_more_than_32_local_experts_never_take_one_launch(num_local):
    """The tile spills on sm90 at 64 and 128 local experts (EP4 and EP2 of 256)
    even with fewer routes, so those ranks keep the map and moe_align_block_size."""
    assert not ep_local_align_supported(num_routes=TOP_K, num_local_experts=num_local)
    with pytest.raises(ValueError, match="does not cover"):
        _align(_skewed_ids(1, seed=2), _expert_map(0, num_local), 8, num_local)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_invalid_ids_are_dropped(dtype):
    topk_ids = _skewed_ids(16, seed=3).to(dtype)
    topk_ids[:, 0] = 100
    topk_ids[1, 1] = -1
    topk_ids[2, 1] = NUM_GLOBAL
    if dtype == torch.int64:
        # Must not wrap to global id 100, a local expert on rank 3.
        topk_ids[3, 1] = 2**32 + 100
    _check(topk_ids, _expert_map(3), block=16)


def test_scattered_expert_map():
    """A non-contiguous map, and map values outside the local range, are honored."""
    generator = torch.Generator().manual_seed(4)
    expert_map = torch.full((NUM_GLOBAL,), -1, dtype=torch.int32)
    expert_map[torch.randperm(NUM_GLOBAL, generator=generator)[:NUM_LOCAL]] = (
        torch.randperm(NUM_LOCAL, generator=generator).to(torch.int32)
    )
    _check(_skewed_ids(64, seed=5), expert_map, block=16)
    expert_map = _expert_map(0)
    expert_map[5] = NUM_LOCAL
    expert_map[4] = -5
    _check(_skewed_ids(16, seed=6), expert_map, block=8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
