import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe.moe_align import moe_align_block_size
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_align import ceil_div, moe_align_block_size_triton

register_cuda_ci(est_time=28, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=50, stage="nightly", runner_config="1-gpu-large")


MOE_ALIGN_CASES = get_ci_test_range(
    list(
        itertools.product(
            [32, 64, 128, 256],  # block_size
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],  # num_tokens
            [1, 2, 4, 8, 16, 32, 64],  # topk
            [64, 128, 160, 256, 257, 260, 264],  # num_experts
            [True, False],  # pad_sorted_token_ids
        )
    ),
    [
        (32, 1, 1, 64, True),
        (128, 48, 1, 128, True),
        (256, 103, 4, 256, False),
        (128, 128, 1, 256, True),
        (256, 4096, 8, 256, False),
    ],
)


@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts,pad_sorted_token_ids",
    MOE_ALIGN_CASES,
)
def test_moe_align_block_size_compare_implementations(
    block_size, num_tokens, topk, num_experts, pad_sorted_token_ids
):
    topk_ids = torch.argsort(torch.rand(num_tokens, num_experts, device="cuda"), dim=1)[
        :, :topk
    ]

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    # The kernel pads sorted_ids with int4 stores.
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4

    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    if not pad_sorted_token_ids:
        sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.empty(
        num_experts + 2, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        cumsum_buffer,
        pad_sorted_token_ids,
    )

    moe_align_block_size_triton(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    assert torch.allclose(expert_ids_cuda, expert_ids_triton, atol=0, rtol=0), (
        f"Expert IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA expert_ids: {expert_ids_cuda}\n"
        f"Triton expert_ids: {expert_ids_triton}"
    )

    assert torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton, atol=0, rtol=0
    ), (
        f"Num tokens post pad mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA num_tokens_post_pad: {num_tokens_post_pad_cuda}\n"
        f"Triton num_tokens_post_pad: {num_tokens_post_pad_triton}"
    )

    # Select an expert to check
    expert_idx = expert_ids_cuda.max().item()

    # Get the first and last block id where expert_ids_cuda == expert_idx
    matching_indices = torch.where(expert_ids_cuda == expert_idx)[0]
    block_sorted_start = matching_indices[0].item() * block_size
    block_sorted_end = min(
        (matching_indices[-1].item() + 1) * block_size,
        num_tokens_post_pad_cuda.item(),
    )

    selected_sorted_ids_cuda = sorted_ids_cuda[
        block_sorted_start:block_sorted_end
    ].sort()[0]
    selected_sorted_ids_triton = sorted_ids_triton[
        block_sorted_start:block_sorted_end
    ].sort()[0]

    assert torch.allclose(
        selected_sorted_ids_cuda,
        selected_sorted_ids_triton,
        atol=0,
        rtol=0,
    ), (
        f"Sorted IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA sorted_ids: {selected_sorted_ids_cuda}\n"
        f"Triton sorted_ids: {selected_sorted_ids_triton}"
    )


@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts",
    list(
        itertools.product(
            [64, 128],  # block_size
            [1, 8, 32, 256],  # num_tokens
            [8],  # topk
            [
                1025,
                2048,
                4095,
            ],  # num_experts (>1024 to exercise v2 kernel, max 4095 real experts)
        )
    ),
)
def test_moe_align_block_size_v2_large_num_experts(
    block_size, num_tokens, topk, num_experts
):
    """Test moe_align_block_size v2 kernel for >1024 experts against Triton reference."""
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    # The kernel pads sorted_ids with int4 stores.
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4

    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.empty(
        num_experts + 2, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        cumsum_buffer,
        True,
    )

    moe_align_block_size_triton(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    assert torch.equal(num_tokens_post_pad_cuda, num_tokens_post_pad_triton), (
        f"Num tokens post pad mismatch: CUDA={num_tokens_post_pad_cuda.item()}, "
        f"Triton={num_tokens_post_pad_triton.item()}"
    )

    ntp = num_tokens_post_pad_cuda.item()
    num_blocks = ntp // block_size

    assert torch.equal(expert_ids_cuda[:num_blocks], expert_ids_triton[:num_blocks]), (
        f"Expert IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}, num_experts={num_experts}"
    )

    # Compare sorted_token_ids per expert block (order within block may differ)
    for b in range(num_blocks):
        s, e = b * block_size, (b + 1) * block_size
        block_cuda = sorted_ids_cuda[s:e].sort().values
        block_triton = sorted_ids_triton[s:e].sort().values
        assert torch.equal(block_cuda, block_triton), (
            f"Block {b} sorted_ids mismatch for num_experts={num_experts}, "
            f"num_tokens={num_tokens}"
        )


@pytest.mark.parametrize("remainder", [1, 2, 3])
@pytest.mark.parametrize(
    "num_experts,num_tokens",
    [
        (32, 8),  # small-batch kernel: <= 64 buckets and numel < 1024
        (128, 640),  # generic kernel: <= 1024 buckets
        (1025, 640),  # v2 kernel: > 1024 buckets
    ],
)
def test_moe_align_block_size_pad_fill_stays_in_bounds(
    num_experts, num_tokens, remainder
):
    """The pad fill writes sorted_token_ids with int4 stores; a buffer whose
    length is not a multiple of 4 must be filled to its end and no further."""
    block_size, topk = 64, 8
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    needed = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    length = needed + (remainder - needed) % 4
    assert length % 4 == remainder

    # sorted_ids is a prefix view of a longer buffer; a fill that runs past the
    # view lands in the guard words behind it.
    backing = torch.full((length + 4,), -1, dtype=torch.int32, device="cuda")
    sorted_ids = backing[:length]
    expert_ids = torch.zeros(
        (ceil_div(length, block_size),), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty(num_experts + 2, dtype=torch.int32, device="cuda")

    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    torch.cuda.synchronize()

    guard = backing[length:]
    assert torch.equal(guard, torch.full_like(guard, -1)), (
        f"pad fill wrote past sorted_token_ids for length={length}: {guard.tolist()}"
    )
    tail = sorted_ids[num_tokens_post_pad.item() :]
    assert torch.equal(tail, torch.full_like(tail, topk_ids.numel())), (
        f"pad fill left part of the tail unfilled for length={length}"
    )


@pytest.mark.parametrize("write_inactive_prefix_count", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tokens,num_experts", [(65, 4), (512, 128), (65, 1025)])
def test_moe_align_block_size_completes_expert_capacity(
    dtype, num_tokens, num_experts, write_inactive_prefix_count
):
    """Unused expert-map slots must be inactive after every graph replay."""
    block_size, topk = 4, 2
    original = (
        torch.arange(num_tokens * topk, device="cuda", dtype=dtype).reshape(
            num_tokens, topk
        )
        % num_experts
    )
    routes = original.clone()
    capacity = min(
        routes.numel() * block_size,
        routes.numel() + (num_experts + 1) * (block_size - 1),
    )
    capacity = ceil_div(capacity, 4) * 4
    sorted_ids = torch.empty(capacity, device="cuda", dtype=torch.int32)
    blocks = ceil_div(capacity, block_size)
    # Guard words make an out-of-bounds capacity fill observable.
    expert_storage = torch.full(
        (blocks + 4,), num_experts, device="cuda", dtype=torch.int32
    )
    expert_ids = expert_storage[:blocks]
    count = torch.empty(
        (2 if write_inactive_prefix_count else 1,), device="cuda", dtype=torch.int32
    )
    cumsum = torch.empty(num_experts + 2, device="cuda", dtype=torch.int32)

    def align():
        moe_align_block_size(
            routes,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            count,
            cumsum,
            True,
            pad_expert_ids=True,
            write_inactive_prefix_count=write_inactive_prefix_count,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        align()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        align()

    for live in (num_tokens, num_tokens // 2, 0, num_tokens):
        routes.copy_(original)
        routes[live:].fill_(-1)
        sorted_ids.fill_(-99)
        expert_storage.fill_(num_experts)
        graph.replay()
        used = count[0].item()
        if write_inactive_prefix_count:
            assert count[1].item() == ceil_div((routes < 0).sum().item(), block_size)
        assert used % block_size == 0
        assert torch.all(expert_ids[used // block_size :] == -1)
        assert torch.all(expert_storage[blocks:] == num_experts)
        assert torch.all(sorted_ids[used:] == routes.numel())
        # Every real route appears exactly once in its expert's contiguous region.
        flat_routes = routes.flatten()
        expected_start = 0
        for expert in torch.unique(flat_routes).tolist():
            expected = torch.where(flat_routes == expert)[0]
            padded = ceil_div(expected.numel(), block_size) * block_size
            end = expected_start + padded
            assert torch.all(
                expert_ids[expected_start // block_size : end // block_size] == expert
            )
            actual = sorted_ids[expected_start:end]
            actual = actual[actual < routes.numel()].long().sort().values
            assert torch.equal(actual, expected)
            expected_start = end
        assert expected_start == used


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows,num_experts",
    [(17, 8), (512, 65), (32, 1025), (17, 129), (0, 8), (0, 129)],
)
def test_moe_align_inactive_prefix_count_graph(rows, num_experts, dtype):
    """Owned count pairs describe only the used prefix, including replay changes."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size as align,
    )

    block_size, topk = 8, 3
    valid = (torch.arange(rows * topk, dtype=dtype) % num_experts).view(rows, topk)
    ids = valid.cuda()

    def run():
        return align(
            ids,
            block_size,
            num_experts,
            return_inactive_prefix_count=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sorted_ids, expert_ids, counts = run()
    assert counts.shape == (2,) and counts.dtype == torch.int32
    assert counts.is_contiguous() and counts.data_ptr() % 8 == 0

    mixed = valid.clone()
    mixed.reshape(-1)[::3] = -1
    inactive = torch.full_like(valid, -1)
    for routes in (mixed, inactive, valid, mixed):
        ids.copy_(routes)
        # Both words must be published on every replay, including empty inputs.
        counts.fill_(-123)
        graph.replay()
        expected_groups = []
        flat = routes.reshape(-1)
        for expert in sorted(set(flat.tolist())):
            positions = (flat == expert).nonzero().reshape(-1).tolist()
            padded = ceil_div(len(positions), block_size) * block_size
            expected_groups.append((expert, positions, padded))
        expected_total = sum(padded for _, _, padded in expected_groups)
        expected_prefix = sum(
            padded // block_size
            for expert, _, padded in expected_groups
            if expert == -1
        )
        assert counts.cpu().tolist() == [expected_total, expected_prefix]
        used_experts = expert_ids[: expected_total // block_size].cpu().tolist()
        expected_experts = [
            expert
            for expert, _, padded in expected_groups
            for _ in range(padded // block_size)
        ]
        assert used_experts == expected_experts
        used_ids = sorted_ids[:expected_total].cpu()
        offset = 0
        for _, positions, padded in expected_groups:
            group = used_ids[offset : offset + padded]
            assert sorted(group[group < flat.numel()].tolist()) == positions
            assert (group[group >= flat.numel()] == flat.numel()).all()
            offset += padded


@pytest.mark.parametrize("route_ids", [(-1, -1, -1), (2, -1, 0), (2, 0, 1)])
def test_moe_align_single_token_inactive_prefix_count(route_ids):
    from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

    block_size = 8
    ids = torch.tensor([route_ids], device="cuda", dtype=torch.int32)
    sorted_ids, expert_ids, counts = moe_align_single_token(
        ids, block_size, return_inactive_prefix_count=True
    )
    assert counts.cpu().tolist() == [len(route_ids) * block_size, route_ids.count(-1)]
    assert expert_ids.cpu().tolist() == sorted(route_ids)
    positions = sorted_ids.view(len(route_ids), block_size).cpu()
    assert (positions[:, 1:] == len(route_ids)).all()
    order = positions[:, 0].tolist()
    assert sorted(order) == list(range(len(route_ids)))
    assert [route_ids[i] for i in order] == sorted(route_ids)


@pytest.mark.parametrize("num_experts", [8, 65, 1025])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("write_inactive_prefix_count", [False, True])
def test_moe_align_count_publication_bounds(
    num_experts, empty, write_inactive_prefix_count
):
    """Empty producers publish zero; legacy capacity never certifies word 1."""
    ids = torch.tensor(
        [] if empty else [[-1, 0, num_experts - 1]],
        device="cuda",
        dtype=torch.int32,
    )
    block_size = 8
    total = 0 if empty else 24
    sorted_ids = torch.empty(total, device="cuda", dtype=torch.int32)
    expert_ids = torch.empty(total // block_size, device="cuda", dtype=torch.int32)
    count_backing = torch.full((2,), -123, device="cuda", dtype=torch.int32)
    cumsum = torch.empty(num_experts + 2, device="cuda", dtype=torch.int32)
    moe_align_block_size(
        ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        count_backing if write_inactive_prefix_count else count_backing[:1],
        cumsum,
        True,
        write_inactive_prefix_count=write_inactive_prefix_count,
    )
    prefix = (0 if empty else 1) if write_inactive_prefix_count else -123
    assert count_backing.cpu().tolist() == [total, prefix]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_experts", [8, 129, 1500])
@pytest.mark.parametrize("write_inactive_prefix_count", [False, True])
def test_moe_align_checked_ids_preserve_bounds_on_replay(
    dtype, num_experts, write_inactive_prefix_count
):
    """Invalid signed IDs occupy sentinel blocks without touching adjacent storage."""
    block_size, numel, poison = 8, 16, -100003
    valid = torch.tensor([0, num_experts - 1, 1, 0] * 4, dtype=dtype)
    invalid = [-1, -3, num_experts, 2 * num_experts]
    invalid += [torch.iinfo(dtype).min, torch.iinfo(dtype).max]
    if dtype == torch.int64:
        invalid += [-(2**40), 2**40]
    mixed = valid.clone()
    mixed[2 : 2 + len(invalid)] = torch.tensor(invalid, dtype=dtype)
    all_invalid = torch.tensor((invalid * numel)[:numel], dtype=dtype)
    ids = valid.cuda()

    def guarded(size):
        backing = torch.full((size + 8,), poison, device="cuda", dtype=torch.int32)
        return backing, backing[4 : 4 + size]

    max_tokens = numel * block_size
    buffers = [
        guarded(max_tokens),
        guarded(numel),
        guarded(2 if write_inactive_prefix_count else 1),
        guarded(num_experts + 2),
    ]
    sorted_ids, experts, counts, cumsum = [view for _, view in buffers]

    def run(routes=ids):
        moe_align_block_size(
            routes,
            num_experts + 1,
            block_size,
            sorted_ids,
            experts,
            counts,
            cumsum,
            True,
            write_inactive_prefix_count=write_inactive_prefix_count,
            map_invalid_experts_to_sentinel=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for routes in (valid, mixed, all_invalid, valid):
        ids.copy_(routes)
        for _, view in buffers:
            view.fill_(poison)
        graph.replay()
        torch.testing.assert_close(ids.cpu(), routes, rtol=0, atol=0)
        groups = {}
        for index, expert in enumerate(routes.tolist()):
            expert = expert if 0 <= expert < num_experts else -1
            groups.setdefault(expert, []).append(index)
        total = sum(
            ceil_div(len(group), block_size) * block_size for group in groups.values()
        )
        expected_counts = [total]
        if write_inactive_prefix_count:
            expected_counts.append(ceil_div(len(groups.get(-1, [])), block_size))
        assert counts.cpu().tolist() == expected_counts
        offset, expected_experts = 0, []
        for expert, indices in sorted(groups.items()):
            padded = ceil_div(len(indices), block_size) * block_size
            group = sorted_ids[offset : offset + padded].cpu()
            assert sorted(group[group < numel].tolist()) == indices
            assert bool((group[group >= numel] == numel).all())
            expected_experts += [expert] * (padded // block_size)
            offset += padded
        assert experts[: total // block_size].cpu().tolist() == expected_experts
        assert bool((experts[total // block_size :] == poison).all())
        for backing, _ in buffers:
            assert bool((backing[:4] == poison).all())
            assert bool((backing[-4:] == poison).all())

    for _, view in buffers:
        view.fill_(poison)
    run(ids[:0])
    assert counts.cpu().tolist() == ([0, 0] if write_inactive_prefix_count else [0])
    assert bool((experts == poison).all())
    for backing, _ in buffers:
        assert bool((backing[:4] == poison).all())
        assert bool((backing[-4:] == poison).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
