"""Correctness of the single-launch tiny-numel moe_align triton kernel.

The oracle is a plain-torch implementation of the documented contract, so it
does not depend on any other kernel's shape support; the AOT `sgl_kernel` path
is cross-checked on top of it to back the drop-in-replacement claim.
"""

import itertools
import sys

import pytest
import torch
import triton

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe import moe_align_block_size as cuda_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size as runner_moe_align_block_size,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Bucket counts above this are not uniformly supported by the AOT sgl_kernel
# path across wheel versions, so the cross-check against it stops here; the
# kernel under test has no expert limit and the oracle covers it past this bound.
CUDA_XCHECK_MAX_EXPERTS = 1023


def _reference(topk_ids, block_size, num_experts):
    """The contract, in plain torch, on CPU: bucket = expert + 1 (so EP-filtered
    -1 lands in bucket 0), each bucket padded to a block_size multiple, blocks in
    bucket order with expert_ids = bucket - 1, pad slots holding numel, pairs
    placed in ascending pair index within their bucket."""
    numel = topk_ids.numel()
    bucket = (topk_ids.flatten().to(torch.int64) + 1).cpu()
    counts = torch.bincount(bucket, minlength=num_experts + 1)
    padded = ((counts + block_size - 1) // block_size) * block_size
    offsets = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())

    non_empty = torch.nonzero(padded, as_tuple=True)[0]
    expert_ids = torch.repeat_interleave(
        non_empty - 1, padded[non_empty] // block_size
    ).to(torch.int32)

    sorted_ids = torch.full((total,), numel, dtype=torch.int32)
    cursor = offsets.clone()
    for pair in range(numel):
        b = int(bucket[pair])
        sorted_ids[cursor[b]] = pair
        cursor[b] += 1
    return sorted_ids, expert_ids, total


def _alloc(numel, block_size, num_experts):
    """Output buffers sized exactly as the moe_runner call site sizes them."""
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    return (
        torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda"),
        torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda"),
        torch.empty((1,), dtype=torch.int32, device="cuda"),
    )


def _run_triton(topk_ids, block_size, num_experts):
    sorted_ids, expert_ids, num_post_pad = _alloc(
        topk_ids.numel(), block_size, num_experts
    )
    moe_align_small_numel(
        topk_ids, num_experts + 1, block_size, sorted_ids, expert_ids, num_post_pad
    )
    return sorted_ids, expert_ids, num_post_pad


def _run_cuda(topk_ids, block_size, num_experts, ignore_invalid_expert=False):
    sorted_ids, expert_ids, num_post_pad = _alloc(
        topk_ids.numel(), block_size, num_experts
    )
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda")
    cuda_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_post_pad,
        cumsum_buffer,
        True,
        ignore_invalid_expert,
    )
    return sorted_ids, expert_ids, num_post_pad


def _assert_exact(got, ref, block_size):
    """Full equality, valid against the oracle because its intra-bucket order
    matches the kernel's (stable in pair index). The tail past the published
    total is left unwritten by design, so nothing is asserted there."""
    got_sorted, got_expert, got_total = got
    ref_sorted, ref_expert, ref_total = ref
    assert got_total.item() == ref_total, "num_tokens_post_pad"
    num_blocks = ref_total // block_size
    assert torch.equal(got_expert[:num_blocks].cpu(), ref_expert), "expert_ids"
    assert torch.equal(got_sorted[:ref_total].cpu(), ref_sorted), "sorted_token_ids"


def _assert_blockwise(got, ref, block_size):
    """Per-block multiset equality -- the comparison that also holds against the
    CUDA kernel, whose intra-bucket order is atomicAdd scheduling order."""
    got_sorted, got_expert, got_total = got
    ref_sorted, ref_expert, ref_total = ref
    assert got_total.item() == ref_total.item(), "num_tokens_post_pad"
    total = ref_total.item()
    num_blocks = total // block_size
    assert torch.equal(got_expert[:num_blocks], ref_expert[:num_blocks]), "expert_ids"
    got_blocks = got_sorted[:total].view(num_blocks, block_size).sort(dim=1).values
    ref_blocks = ref_sorted[:total].view(num_blocks, block_size).sort(dim=1).values
    assert torch.equal(got_blocks, ref_blocks), "sorted_token_ids block contents"


# num_experts straddles the CUDA small-batch kernel's 64-bucket limit (the corner
# this kernel exists to cover) and goes past what the AOT path handles at all.
ALIGN_CASES = get_ci_test_range(
    [
        (block_size, num_experts, topk, num_tokens)
        for block_size, num_experts, topk, num_tokens in itertools.product(
            [16, 32, 64, 128], [8, 64, 65, 129, 1024], [1, 2, 4, 8], [1, 4, 8]
        )
        if topk * num_tokens <= SMALL_NUMEL_LIMIT
    ],
    [
        (16, 65, 1, 1),
        (32, 65, 8, 8),
        (64, 129, 4, 4),
        (128, 1024, 8, 8),
        (128, 8, 2, 4),
    ],
)


@pytest.mark.parametrize("block_size,num_experts,topk,num_tokens", ALIGN_CASES)
def test_matches_reference(block_size, num_experts, topk, num_tokens):
    """Exact against the oracle, plus drop-in equivalence with the CUDA path
    this replaces wherever that path supports the bucket count."""
    torch.manual_seed(0)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    got = _run_triton(topk_ids, block_size, num_experts)
    _assert_exact(got, _reference(topk_ids, block_size, num_experts), block_size)
    if num_experts <= CUDA_XCHECK_MAX_EXPERTS:
        _assert_blockwise(got, _run_cuda(topk_ids, block_size, num_experts), block_size)


def test_ep_filtered_ids_map_to_expert_minus_one():
    """EP-filtered pairs (-1) collect in bucket 0, whose blocks carry -1 so
    fused_moe's filter_expert skips them."""
    torch.manual_seed(1)
    num_experts, block_size = 1024, 64
    topk_ids = torch.randint(0, num_experts, (8, 4), dtype=torch.int32, device="cuda")
    topk_ids[0] = -1
    topk_ids[3][2] = -1

    ref = _reference(topk_ids, block_size, num_experts)
    _assert_exact(_run_triton(topk_ids, block_size, num_experts), ref, block_size)
    assert -1 in ref[1].tolist(), "filtered pairs must produce an expert_id == -1 block"


@pytest.mark.parametrize(
    "numel", [SMALL_NUMEL_LIMIT - 1, SMALL_NUMEL_LIMIT, SMALL_NUMEL_LIMIT + 1]
)
def test_runner_dispatch_boundary(numel):
    """Both sides of the moe_runner gate must agree with the CUDA reference, so
    a future change to the limit cannot silently ship an unvalidated path."""
    torch.manual_seed(2)
    num_experts, block_size = 129, 32
    topk_ids = torch.randint(
        0, num_experts, (numel, 1), dtype=torch.int32, device="cuda"
    )
    _assert_blockwise(
        runner_moe_align_block_size(topk_ids, block_size, num_experts),
        _run_cuda(topk_ids, block_size, num_experts),
        block_size,
    )


def test_runner_defers_for_ignore_invalid_expert():
    """ignore_invalid_expert is a different contract than the '+1 offset'
    convention the triton kernel implements, so the runner must keep producing
    what the CUDA kernel produces under that flag."""
    torch.manual_seed(3)
    num_experts, block_size = 128, 32
    topk_ids = torch.randint(0, num_experts, (8, 4), dtype=torch.int32, device="cuda")
    topk_ids[1] = -1

    _assert_blockwise(
        runner_moe_align_block_size(
            topk_ids, block_size, num_experts, ignore_invalid_expert=True
        ),
        _run_cuda(topk_ids, block_size, num_experts, ignore_invalid_expert=True),
        block_size,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
