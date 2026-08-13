"""Correctness of the v2 moe_align kernels.

The oracle is a plain-torch implementation of the documented contract, so it does
not depend on any other kernel's shape support, and the AOT `sgl_kernel` path is
cross-checked on top of it to back the drop-in-replacement claim.

v2 puts three shapes of kernel behind one entry, and the parametrisations below
are chosen to land on each: one pair per warp up to `kWarpThreads` pairs, two
pairs per warp up to twice that, and the bucket-scanning fused/general pair above
it. The first two work purely on the pair axis, which is why they accept any
number of experts while the third is capped -- see
``test_tiny_batch_takes_any_expert_count``.
"""

import itertools
import sys

import pytest
import torch
import triton
from sgl_kernel import moe_align_block_size as aot_moe_align_block_size

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe.moe_align import (
    V2_MAX_BUCKETS,
    V2_SMALL_NUMEL_LIMIT,
    moe_align_block_size_out,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=90, suite="nightly-kernel-1-gpu", nightly=True)

# The entry takes a bucket count, which is one more than the expert count.
V2_MAX_EXPERTS = V2_MAX_BUCKETS - 1

# Bucket counts above this are not uniformly supported by the AOT sgl_kernel path
# across wheel versions, so the cross-check against it stops here.
AOT_XCHECK_MAX_EXPERTS = 1023


def _reference(topk_ids, block_size, num_experts, ignore_invalid_expert=False):
    """The contract, in plain torch, on CPU: bucket = expert + 1 (so EP-filtered
    -1 lands in bucket 0), each bucket padded to a block_size multiple, blocks in
    bucket order with expert_ids = bucket - 1, pad slots holding numel, pairs
    placed in ascending pair index within their bucket. Under
    ignore_invalid_expert the filtered pairs are dropped instead of bucketed."""
    numel = topk_ids.numel()
    flat = topk_ids.flatten().to(torch.int64).cpu()
    bucket = flat + 1
    live = (
        torch.nonzero(flat >= 0, as_tuple=True)[0]
        if ignore_invalid_expert
        else torch.arange(numel)
    )

    counts = torch.bincount(bucket[live], minlength=num_experts + 1)
    padded = ((counts + block_size - 1) // block_size) * block_size
    offsets = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())

    non_empty = torch.nonzero(padded, as_tuple=True)[0]
    expert_ids = torch.repeat_interleave(
        non_empty - 1, padded[non_empty] // block_size
    ).to(torch.int32)

    sorted_ids = torch.full((total,), numel, dtype=torch.int32)
    cursor = offsets.clone()
    for pair in live.tolist():
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
        torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda"),
    )


def _run_v2(topk_ids, block_size, num_experts, ignore_invalid_expert=False):
    buffers = _alloc(topk_ids.numel(), block_size, num_experts)
    moe_align_block_size_out(
        topk_ids,
        num_experts + 1,
        block_size,
        *buffers,
        True,
        ignore_invalid_expert,
        version=2,
    )
    return buffers[:3]


def _run_aot(topk_ids, block_size, num_experts):
    buffers = _alloc(topk_ids.numel(), block_size, num_experts)
    aot_moe_align_block_size(topk_ids, num_experts + 1, block_size, *buffers, True)
    return buffers[:3]


def _assert_exact(got, ref, block_size):
    """Full equality, valid against the oracle wherever the kernel's intra-bucket
    order is stable in pair index -- which is what the single-CTA kernels give,
    since a pair's slot is the count of earlier pairs sharing its bucket. The
    tail past the published total is not part of the contract."""
    got_sorted, got_expert, got_total = got
    ref_sorted, ref_expert, ref_total = ref
    assert got_total.item() == ref_total, "num_tokens_post_pad"
    num_blocks = ref_total // block_size
    assert torch.equal(got_expert[:num_blocks].cpu(), ref_expert), "expert_ids"
    assert torch.equal(got_sorted[:ref_total].cpu(), ref_sorted), "sorted_token_ids"


def _assert_bucketwise(got, ref, block_size):
    """Per-bucket multiset equality -- the comparison that still holds once the
    intra-bucket order becomes atomicAdd scheduling order, as it does on the
    bucket-scanning paths and in the AOT kernel. Blocks are not the unit: a
    bucket spanning several of them may hold its pairs in any of them. The
    buckets are located by expert_ids, which ascend and are asserted first."""
    got_sorted, got_expert, got_total = got
    ref_sorted, ref_expert, ref_total = ref
    ref_total = ref_total if isinstance(ref_total, int) else ref_total.item()
    assert got_total.item() == ref_total, "num_tokens_post_pad"

    num_blocks = ref_total // block_size
    got_expert = got_expert[:num_blocks].cpu()
    ref_expert = ref_expert[:num_blocks].cpu()
    assert torch.equal(got_expert, ref_expert), "expert_ids"
    if ref_total == 0:  # every pair filtered away; nothing else is written
        return

    got_sorted = got_sorted[:ref_total].cpu().to(torch.int64)
    ref_sorted = ref_sorted[:ref_total].cpu().to(torch.int64)
    # Sort each bucket's rows by value; the expert id keys the bucket, and the
    # pad sentinel is the largest value so it lands at the bucket's tail.
    key = torch.repeat_interleave(ref_expert.to(torch.int64), block_size)
    span = int(ref_sorted.max()) + 1
    canon = lambda v: v[torch.argsort(key * span + v)]
    assert torch.equal(canon(got_sorted), canon(ref_sorted)), "sorted_token_ids"


def _random_ids(num_tokens, topk, num_experts, seed, filtered=0.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    if filtered:
        mask = torch.rand(ids.shape, device="cuda", generator=generator) < filtered
        ids = torch.where(mask, torch.full_like(ids, -1), ids)
    return ids


# numel stays within the single-CTA capacity and straddles its two thresholds
# (one pair per warp, then two); num_experts runs from below the CUDA
# small-batch kernel's 64-bucket limit to well past what any bucket-scanning
# path accepts.
TINY_CASES = get_ci_test_range(
    [
        (block_size, num_experts, topk, num_tokens)
        for block_size, num_experts, topk, num_tokens in itertools.product(
            [4, 16, 64, 128],
            [8, 64, 65, 1023, 1024, 4096, 9000],
            [1, 2, 4, 8],
            [1, 4, 8],
        )
        if topk * num_tokens <= V2_SMALL_NUMEL_LIMIT
    ],
    [
        (4, 8, 1, 1),
        (16, 65, 4, 8),
        (64, 1023, 8, 4),
        (128, 4096, 8, 8),
        (128, 9000, 1, 1),
    ],
)

# Past the single-CTA capacity: the fused and general paths, both of which scan
# the bucket axis and are therefore capped at V2_MAX_EXPERTS.
BULK_CASES = get_ci_test_range(
    list(
        itertools.product(
            [4, 16, 64, 128],
            [8, 64, 257, V2_MAX_EXPERTS],
            [1, 2, 8],
            [16, 128, 1024, 4096],
        )
    ),
    [
        (4, 8, 1, 16),
        (16, 257, 2, 128),
        (64, V2_MAX_EXPERTS, 8, 1024),
        (128, 64, 1, 4096),
        (128, 257, 8, 4096),
    ],
)


@pytest.mark.parametrize("block_size,num_experts,topk,num_tokens", TINY_CASES)
def test_tiny_batch_matches_reference(block_size, num_experts, topk, num_tokens):
    """The single-CTA kernels, exactly: they place a pair at the count of earlier
    pairs sharing its bucket, so the oracle's order is reproducible."""
    topk_ids = _random_ids(num_tokens, topk, num_experts, seed=topk * num_tokens)
    _assert_exact(
        _run_v2(topk_ids, block_size, num_experts),
        _reference(topk_ids, block_size, num_experts),
        block_size,
    )


@pytest.mark.parametrize("block_size,num_experts,topk,num_tokens", BULK_CASES)
def test_bulk_matches_reference(block_size, num_experts, topk, num_tokens):
    """The bucket-scanning paths. Their intra-bucket order comes from atomicAdd,
    so only the per-block contents are pinned."""
    topk_ids = _random_ids(num_tokens, topk, num_experts, seed=num_tokens + topk)
    _assert_bucketwise(
        _run_v2(topk_ids, block_size, num_experts),
        _reference(topk_ids, block_size, num_experts),
        block_size,
    )


@pytest.mark.parametrize("num_experts", [64, 1023, 1024, 1025, 4096, 65535])
@pytest.mark.parametrize("numel", [1, V2_SMALL_NUMEL_LIMIT])
def test_tiny_batch_takes_any_expert_count(num_experts, numel):
    """A tiny batch is served whatever the expert count, including far past the
    bucket-scanning paths' cap: those kernels index shared memory by pair, never
    by bucket. Re-introducing a bucket bound on this path turns this red."""
    topk_ids = _random_ids(numel, 1, num_experts, seed=num_experts)
    _assert_exact(
        _run_v2(topk_ids, 16, num_experts),
        _reference(topk_ids, 16, num_experts),
        16,
    )


@pytest.mark.parametrize("numel", [1, V2_SMALL_NUMEL_LIMIT, V2_SMALL_NUMEL_LIMIT + 1])
def test_ep_filtered_ids_map_to_expert_minus_one(numel):
    """EP-filtered pairs (-1) collect in bucket 0, whose blocks carry -1 so
    fused_moe's filter_expert skips them."""
    num_experts, block_size = V2_MAX_EXPERTS, 64
    topk_ids = _random_ids(numel, 4, num_experts, seed=numel, filtered=0.3)
    topk_ids[0][0] = -1

    ref = _reference(topk_ids, block_size, num_experts)
    assert -1 in ref[1].tolist(), "filtered pairs must produce an expert_id == -1 block"
    _assert_bucketwise(_run_v2(topk_ids, block_size, num_experts), ref, block_size)


@pytest.mark.parametrize("numel", [1, V2_SMALL_NUMEL_LIMIT, V2_SMALL_NUMEL_LIMIT + 1])
@pytest.mark.parametrize("filtered", [0.3, 1.0])
def test_ignore_invalid_expert_drops_filtered_pairs(numel, filtered):
    """Under the flag the -1 pairs leave the output entirely rather than forming
    a sentinel bucket -- including when every pair is filtered, which leaves no
    live pair to publish the total."""
    num_experts, block_size = 257, 16
    topk_ids = _random_ids(numel, 4, num_experts, seed=numel, filtered=filtered)

    ref = _reference(topk_ids, block_size, num_experts, ignore_invalid_expert=True)
    assert -1 not in ref[1].tolist(), "no sentinel bucket under ignore_invalid_expert"
    _assert_bucketwise(
        _run_v2(topk_ids, block_size, num_experts, ignore_invalid_expert=True),
        ref,
        block_size,
    )


@pytest.mark.parametrize("numel", [1, V2_SMALL_NUMEL_LIMIT, 4096])
def test_matches_aot_kernel(numel):
    """v2 is a drop-in for the kernel it replaces, so it has to agree with the
    wheel and not merely with our reading of the contract."""
    num_experts, block_size = AOT_XCHECK_MAX_EXPERTS, 64
    topk_ids = _random_ids(numel, 4, num_experts, seed=numel, filtered=0.2)
    _assert_bucketwise(
        _run_v2(topk_ids, block_size, num_experts),
        _run_aot(topk_ids, block_size, num_experts),
        block_size,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
