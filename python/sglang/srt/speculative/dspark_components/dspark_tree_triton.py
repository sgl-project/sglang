"""NPU Triton helpers for DSpark tree budget pruning and layout finalization.

The Torch implementation in :mod:`dspark_tree` is kept as the CPU/reference
path. The kernels here follow the vLLM-Ascend tree layout kernels and are used
only when the layout tensors live on NPU.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _num_vectorcores() -> int:
    try:
        import torch_npu  # noqa: F401

        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        value = int(props.get("num_vectorcore", 0))
        if value > 0:
            return value
    except Exception:
        pass
    return 40


@triton.jit(do_not_specialize=["num_reqs", "num_candidates", "num_selected"])
def _prune_tree_layout_kernel(
    priority_ptr,
    tokens_ptr,
    depths_ptr,
    parents_ptr,
    out_tokens_ptr,
    out_depths_ptr,
    out_parents_ptr,
    num_reqs,
    num_candidates,
    num_selected,
    stride_priority_r,
    stride_tokens_r,
    stride_depths_r,
    stride_parents_r,
    stride_out_tokens_r,
    stride_out_depths_r,
    stride_out_parents_r,
    N_CAND_PAD: tl.constexpr,
    BUDGET: tl.constexpr,
    BUDGET_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, BUDGET_PAD)
    candidate_lanes = tl.arange(0, N_CAND_PAD)
    candidate_ids = candidate_lanes.to(tl.int64)
    big = tl.full((), N_CAND_PAD, tl.int64)
    neg_inf = -float("inf")

    req = pid
    while req < num_reqs:
        valid_candidates = candidate_lanes < num_candidates
        scores = tl.load(
            priority_ptr + req * stride_priority_r + candidate_lanes,
            mask=valid_candidates,
            other=neg_inf,
        ).to(tl.float32)

        taken = ~valid_candidates
        selected = tl.full((BUDGET_PAD,), -1, tl.int64)
        for rank in range(BUDGET_PAD):
            available = valid_candidates & ~taken
            is_nan = available & (scores != scores)
            has_nan = tl.sum(is_nan.to(tl.int32), axis=0) > 0
            values = tl.where(available & ~is_nan, scores, neg_inf)
            best_value = tl.max(values, axis=0)
            is_best = tl.where(has_nan, is_nan, (values == best_value) & available)
            best = tl.min(tl.where(is_best, candidate_ids, big), axis=0)
            selected = tl.where(
                lanes == rank,
                tl.where(rank < num_selected, best, -1),
                selected,
            )
            taken = taken | (candidate_ids == best)

        remaining = selected
        ordered = tl.full((BUDGET_PAD,), -1, tl.int64)
        for slot in range(BUDGET_PAD):
            minimum = tl.min(tl.where(remaining >= 0, remaining, big), axis=0)
            ordered = tl.where(
                lanes == slot,
                tl.where(slot < num_selected, minimum, -1),
                ordered,
            )
            remaining = tl.where(remaining == minimum, -1, remaining)

        valid_output = lanes < num_selected
        safe_slot = tl.where(valid_output, ordered, 0)
        old_node_id = safe_slot + 1
        token = tl.load(
            tokens_ptr + req * stride_tokens_r + safe_slot,
            valid_output,
            other=-1,
        )
        depth = tl.load(
            depths_ptr + req * stride_depths_r + safe_slot,
            valid_output,
            other=0,
        )
        old_parent = tl.load(
            parents_ptr + req * stride_parents_r + old_node_id,
            valid_output,
            other=0,
        )

        parent_match = ((ordered[None, :] + 1) == old_parent[:, None]) & valid_output[None, :]
        new_parent = tl.max(tl.where(parent_match, lanes[None, :] + 1, 0), axis=1)
        parent = tl.where(old_parent > 0, new_parent, 0)

        tl.store(
            out_tokens_ptr + req * stride_out_tokens_r + lanes,
            tl.where(valid_output, token, -1),
            lanes < BUDGET,
        )
        tl.store(
            out_depths_ptr + req * stride_out_depths_r + lanes,
            tl.where(valid_output, depth, 0),
            lanes < BUDGET,
        )
        tl.store(
            out_parents_ptr + req * stride_out_parents_r + lanes,
            tl.where(valid_output, parent, -1),
            lanes < BUDGET,
        )
        req += nprog


@triton.jit(do_not_specialize=["num_reqs"])
def _finalize_tree_layout_kernel(
    parent_ids_ptr,
    first_child_ptr,
    next_sibling_ptr,
    visibility_ptr,
    num_reqs,
    stride_par_r,
    stride_fc_r,
    stride_ns_r,
    stride_vis_r,
    stride_vis_i,
    NUM_NODES: tl.constexpr,
    BUDGET: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    req = pid
    while req < num_reqs:
        for n in range(BUDGET + 1):
            tl.store(first_child_ptr + req * stride_fc_r + n, -1)
            tl.store(next_sibling_ptr + req * stride_ns_r + n, -1)
        for slot in range(NUM_NODES):
            node_id = slot + 1
            parent_id = tl.load(parent_ids_ptr + req * stride_par_r + slot)
            cur_first = tl.load(first_child_ptr + req * stride_fc_r + parent_id)
            tl.store(next_sibling_ptr + req * stride_ns_r + node_id, cur_first)
            tl.store(first_child_ptr + req * stride_fc_r + parent_id, node_id)
        for slot in range(BUDGET):
            for j in range(BUDGET):
                tl.store(
                    visibility_ptr + req * stride_vis_r + slot * stride_vis_i + j,
                    0,
                )
        for slot in range(NUM_NODES):
            parent_id = tl.load(parent_ids_ptr + req * stride_par_r + slot)
            src = tl.where(parent_id > 0, parent_id - 1, 0)
            for j in range(slot):
                pv = tl.load(
                    visibility_ptr + req * stride_vis_r + src * stride_vis_i + j
                )
                val = tl.where((slot > 0) & (parent_id > 0), pv, 0)
                tl.store(
                    visibility_ptr + req * stride_vis_r + slot * stride_vis_i + j,
                    val,
                )
            tl.store(
                visibility_ptr + req * stride_vis_r + slot * stride_vis_i + slot,
                1,
            )
        req += nprog


def prune_tree_layout_triton(
    priority: torch.Tensor,
    tokens: torch.Tensor,
    depths: torch.Tensor,
    parents: torch.Tensor,
    out_tokens: torch.Tensor,
    out_depths: torch.Tensor,
    out_parents: torch.Tensor,
    num_nodes: int,
) -> None:
    num_reqs, num_candidates = priority.shape
    budget = out_tokens.shape[1]
    grid = min(max(num_reqs, 1), max(_num_vectorcores(), 1))
    _prune_tree_layout_kernel[(grid,)](
        priority,
        tokens,
        depths,
        parents,
        out_tokens,
        out_depths,
        out_parents,
        num_reqs,
        num_candidates,
        num_nodes,
        priority.stride(0),
        tokens.stride(0),
        depths.stride(0),
        parents.stride(0),
        out_tokens.stride(0),
        out_depths.stride(0),
        out_parents.stride(0),
        N_CAND_PAD=triton.next_power_of_2(max(num_candidates, 1)),
        BUDGET=budget,
        BUDGET_PAD=triton.next_power_of_2(budget),
    )


def finalize_tree_layout_triton(
    parent_ids: torch.Tensor,
    first_child: torch.Tensor,
    next_sibling: torch.Tensor,
    visibility: torch.Tensor,
    num_nodes: int,
) -> None:
    vis = visibility.view(torch.int8) if visibility.dtype == torch.bool else visibility
    num_reqs = parent_ids.shape[0]
    budget = visibility.shape[1]
    grid = min(max(num_reqs, 1), max(_num_vectorcores(), 1))
    _finalize_tree_layout_kernel[(grid,)](
        parent_ids,
        first_child,
        next_sibling,
        vis,
        num_reqs,
        parent_ids.stride(0),
        first_child.stride(0),
        next_sibling.stride(0),
        vis.stride(0),
        vis.stride(1),
        NUM_NODES=num_nodes,
        BUDGET=budget,
    )
