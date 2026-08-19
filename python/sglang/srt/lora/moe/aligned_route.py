"""Sort token/expert pairs into whole route blocks, one group per virtual expert.

One kernel counts the pairs per group, one scans those counts into block-aligned
runs, one labels each block with its group and scatters the pairs into slots. The
group key is never materialized: both kernels that need it recompute it, which is
the [T,K] round trip the JIT path pays along with a ladder that stops at 32767
groups. routing.py switches here above FUSED_ALIGN_MIN_PAIRS and
FUSED_ALIGN_MIN_VIRTUAL_EXPERTS.

A shared-factor plan wants the same pairs grouped two ways at once, by virtual
expert for its inner factors and by adapter alone for its shared outer ones, so
every kernel carries both groupings behind NEED_PER_EXPERT and NEED_SHARED and
runs whichever the caller asked for. Doing the two as separate builds would read
the pair sources twice and run two three-kernel chains.
"""

from __future__ import annotations

import torch
import triton

from sglang.srt.lora.moe.route_kernels import (
    _hist_kernel,
    _place_kernel,
    _scan_kernel,
)
from sglang.srt.lora.moe.route_view import RouteView, RouteViewKind
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Launch tiles; see configs/README.md before changing them.
HIST_BLOCK = 512
HIST_WARPS = 8
EXPAND_BLOCK = 128
EXPAND_WARPS = 4
SCAN_CHUNK = 2048
SCAN_WARPS = 4

# Bin ceiling and smallest pair counts at which counting a block's pairs beats
# one atomic per pair; outside them the helpers below keep the per-pair path.
COUNT_MAX_BINS = 512
COUNT_MIN_PAIRS = 16384
CLAIM_MIN_PAIRS_PER_BUCKET = 12288


def count_bins(num_buckets: int, num_pairs: int) -> int:
    """Bins for counting inside a block, or 0 to add one pair at a time."""
    if num_buckets >= COUNT_MAX_BINS or num_pairs < COUNT_MIN_PAIRS:
        return 0
    return 1 << num_buckets.bit_length()  # one spare bin, for masked-off lanes


def _routing_capacity(
    num_pairs: int,
    block_size: int,
    num_virtual_experts: int,
) -> int:
    if num_pairs == 0:
        return 0
    max_nonempty_buckets = min(num_pairs, num_virtual_experts + 1)
    upper_bound = num_pairs + max_nonempty_buckets * (block_size - 1)
    return triton.cdiv(triton.cdiv(upper_bound, block_size) * block_size, 4) * 4


def _plan_scratch(
    workspace: MoeLoraWorkspace,
    *,
    prefix: str,
    num_buckets: int,
    capacity: int,
    block_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Route-owned scratch; counts are zeroed once because the scan restores that."""
    scratch: dict[str, object] = {
        "num_buckets": num_buckets,
        "capacity": capacity,
        "counts": workspace.tensor(
            f"{prefix}:counts",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        ),
        "block_cumulative": workspace.tensor(
            f"{prefix}:block_cumulative",
            (num_buckets + 1,),
            dtype=torch.int32,
            device=device,
        ),
        "cursor": workspace.tensor(
            f"{prefix}:cursor", (num_buckets,), dtype=torch.int32, device=device
        ),
        "bucket_end": workspace.tensor(
            f"{prefix}:bucket_end", (num_buckets,), dtype=torch.int32, device=device
        ),
        "padded_pairs": workspace.tensor(
            f"{prefix}:padded_pairs", (1,), dtype=torch.int32, device=device
        ),
    }
    scratch["sorted"] = workspace.tensor(
        f"{prefix}:sorted", (capacity,), dtype=torch.int32, device=device
    )
    scratch["block_ids"] = workspace.tensor(
        f"{prefix}:block_ids",
        (capacity // block_size,),
        dtype=torch.int32,
        device=device,
    )
    return scratch


def _run(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    per_expert: dict[str, object],
    shared: dict[str, object],
    need_per_expert: bool,
    need_shared: bool,
) -> None:
    """Launch count, scan and place over whichever groupings were asked for."""
    from sglang.kernels.jit.utils import is_arch_support_pdl

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    pe_buckets = per_expert["num_buckets"]
    sh_buckets = shared["num_buckets"]
    shape = dict(
        NEED_PER_EXPERT=need_per_expert,
        NEED_SHARED=need_shared,
        NUM_PER_EXPERT_BUCKETS=pe_buckets,
        NUM_SHARED_BUCKETS=sh_buckets,
        E_LOCAL=num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        USE_PDL=use_pdl,
    )

    _hist_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
        topk_ids,
        token_lora_mapping,
        per_expert["counts"],
        shared["counts"],
        num_pairs,
        num_local_experts,
        BLOCK=HIST_BLOCK,
        PER_EXPERT_BINS=count_bins(pe_buckets, num_pairs),
        SHARED_BINS=count_bins(sh_buckets, num_pairs),
        num_warps=HIST_WARPS,
        **shape,
    )
    _scan_kernel[(int(need_per_expert) + int(need_shared),)](
        per_expert["counts"],
        per_expert["block_cumulative"],
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["padded_pairs"],
        pe_buckets,
        shared["counts"],
        shared["block_cumulative"],
        shared["cursor"],
        shared["bucket_end"],
        shared["padded_pairs"],
        sh_buckets,
        NEED_PER_EXPERT=need_per_expert,
        NEED_SHARED=need_shared,
        BLOCK_SIZE_M=block_size,
        CHUNK=SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=SCAN_WARPS,
        **pdl_kwargs,
    )
    pe_blocks = per_expert["capacity"] // block_size
    sh_blocks = shared["capacity"] // block_size
    pe_labels = triton.cdiv(max(pe_blocks, 1), EXPAND_BLOCK) if need_per_expert else 0
    sh_labels = triton.cdiv(max(sh_blocks, 1), EXPAND_BLOCK) if need_shared else 0
    pair_programs = triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK)
    _place_kernel[(pe_labels + sh_labels + pair_programs,)](
        topk_ids,
        token_lora_mapping,
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["block_cumulative"],
        per_expert["sorted"],
        per_expert["block_ids"],
        pe_blocks,
        pe_labels,
        shared["cursor"],
        shared["bucket_end"],
        shared["block_cumulative"],
        shared["sorted"],
        shared["block_ids"],
        sh_blocks,
        sh_labels,
        num_pairs,
        num_local_experts,
        NUM_PER_EXPERT_VIRTUAL=pe_buckets - 1,
        NUM_SHARED_VIRTUAL=sh_buckets - 1,
        BLOCK=EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # The search picks one of NUM_BUCKETS + 1 answers, so it needs
        # num_buckets.bit_length() steps -- one fewer and a sentinel reads as 0.
        PER_EXPERT_SEARCH_STEPS=pe_buckets.bit_length(),
        SHARED_SEARCH_STEPS=sh_buckets.bit_length(),
        PER_EXPERT_CLAIM_PER_BLOCK=num_pairs >= CLAIM_MIN_PAIRS_PER_BUCKET * pe_buckets,
        SHARED_CLAIM_PER_BLOCK=num_pairs >= CLAIM_MIN_PAIRS_PER_BUCKET * sh_buckets,
        num_warps=EXPAND_WARPS,
        **pdl_kwargs,
        **shape,
    )


def build(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: MoeLoraWorkspace,
    tensor_prefix: str,
    need_per_expert: bool,
    need_shared: bool,
) -> tuple[RouteView | None, RouteView | None]:
    """Return the per-expert and shared-outer views; either is None if unasked."""
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if num_local_experts < 1 or max_loras < 1 or block_size < 1:
        raise ValueError("expert, adapter, and route block counts must be positive")
    num_pairs = topk_ids.numel()
    scratch: dict[str, dict[str, object]] = {}
    for name, virtual, wanted in (
        ("per_expert", num_local_experts * max_loras, need_per_expert),
        ("shared", max_loras, need_shared),
    ):
        if not wanted:
            continue
        capacity = _routing_capacity(num_pairs, block_size, virtual)
        if virtual + 1 >= 2**31 or capacity >= 2**31:
            raise ValueError(
                f"aligned routes use int32 plan math: {name} needs {virtual + 1} "
                f"buckets and {capacity} slots, both must be < 2**31"
            )
        scratch[name] = _plan_scratch(
            workspace,
            prefix=f"{tensor_prefix}:{name}",
            num_buckets=virtual + 1,
            capacity=capacity,
            block_size=block_size,
            device=topk_ids.device,
        )
    _run(
        topk_ids,
        token_lora_mapping,
        num_local_experts=num_local_experts,
        max_loras=max_loras,
        block_size=block_size,
        # An unbuilt route's slot mirrors the built one; its branches are
        # compiled out, so the pointers are never read.
        per_expert=scratch.get("per_expert") or scratch["shared"],
        shared=scratch.get("shared") or scratch["per_expert"],
        need_per_expert=need_per_expert,
        need_shared=need_shared,
    )

    def route(name: str, *, is_shared_outer: bool) -> RouteView:
        own = scratch[name]
        return RouteView(
            view=RouteViewKind.ALIGNED,
            block_size=block_size,
            topk_ids=topk_ids,
            token_lora_mapping=token_lora_mapping,
            num_local_experts=num_local_experts,
            is_shared_outer=is_shared_outer,
            max_loras=max_loras,
            maybe_sorted_pair_ids=own["sorted"],
            maybe_block_virtual_expert_ids=own["block_ids"],
            maybe_num_pairs_post_padded=own["padded_pairs"],
        )

    return (
        route("per_expert", is_shared_outer=False) if need_per_expert else None,
        route("shared", is_shared_outer=True) if need_shared else None,
    )
