from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
    _apply_cfg_postprocess,
    _unwrap,
    _wrap,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_gather,
    cfg_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
        CFGBranch,
        CFGPolicy,
    )

# Tracks (n_branches, cfg_world_size, cfg_rank) tuples already logged so the
# dispatch table is printed once per unique configuration, not once per step.
_logged_dispatch_keys: set[tuple[int, int, int]] = set()


def _run(
    predict_fn: Callable[["CFGBranch"], "torch.Tensor | tuple[torch.Tensor, ...]"],
    bid: int,
    branches,
) -> tuple[torch.Tensor, ...]:
    branch = branches[bid]
    device = get_local_torch_device()
    local_branch = dataclasses.replace(
        branch,
        kwargs={
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in branch.kwargs.items()
        },
    )
    raw = predict_fn(local_branch)
    return _wrap(raw)


def run_cfg_parallel(
    policy: "CFGPolicy",
    predict_fn: Callable[["CFGBranch"], "torch.Tensor | tuple[torch.Tensor, ...]"],
) -> "list[torch.Tensor | tuple[torch.Tensor, ...]]":
    """Dispatch CFG branches across ranks, all-gather results, return in branch order.

    ``predict_fn`` is a closure capturing all step-varying state
    (latent_model_input, timestep, model, etc.).  It is called with each
    assigned ``CFGBranch`` and must return the raw ``_predict_noise`` output.

    Idle ranks (cfg_world_size > n_branches) run branch 0 as a dummy forward
    to obtain tensor shapes for the all-gather.

    Returns a list indexed to match ``policy.branches``, identical on every rank.
    """

    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()
    branches = policy.branches
    n_branches = len(branches)
    assignments = dispatch_branches(n_branches, cfg_world_size)
    branches_assigned_to_local_rank = assignments[cfg_rank]
    max_num_branches_per_rank = max(len(a) for a in assignments)

    if cfg_world_size > n_branches:
        logger.warning_once(
            "cfg_parallel_size=%d > n_branches=%d; %d GPU(s) will be idle for CFG",
            cfg_world_size,
            n_branches,
            cfg_world_size - n_branches,
        )

    dispatch_key = (n_branches, cfg_world_size, cfg_rank)
    if dispatch_key not in _logged_dispatch_keys:
        _logged_dispatch_keys.add(dispatch_key)
        branch_names = (
            [branches[i].name for i in branches_assigned_to_local_rank]
            if branches_assigned_to_local_rank
            else ["(idle)"]
        )
        logger.info(
            "CFG parallel dispatch: rank %d/%d -> [%s]",
            cfg_rank,
            cfg_world_size,
            ", ".join(branch_names),
        )

    # perform the forward for local branches
    predicts_from_local_branches: list[tuple[torch.Tensor, ...]] = [
        _run(predict_fn, bid, branches) for bid in branches_assigned_to_local_rank
    ]

    if not predicts_from_local_branches:  # idle rank: run branch 0 for tensor shapes
        predicts_from_local_branches.append(_run(predict_fn, 0, branches))

    # pad the predicts to the length of max_num_branches_per_rank, to prepare for the all-gather later
    ref = predicts_from_local_branches[0]
    while len(predicts_from_local_branches) < max_num_branches_per_rank:
        # TODO: cache this zero
        predicts_from_local_branches.append(tuple(torch.zeros_like(t) for t in ref))

    # All-gather each slot and output element with separate_tensors=True.
    # all_slots[slot][elem] = list[Tensor] indexed by CFG rank; no reshape.
    all_slots: list[list[list[torch.Tensor]]] = [
        [
            cfg_model_parallel_all_gather(p, dim=0, separate_tensors=True)
            for p in slot_pred
        ]
        for slot_pred in predicts_from_local_branches
    ]

    # reorder the results in branch order: branch bid -> owner rank, slot.
    n_elems = len(ref)
    final: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
    for bid in range(n_branches):
        owner = bid % cfg_world_size
        slot = bid // cfg_world_size
        elems = tuple(all_slots[slot][ei][owner] for ei in range(n_elems))
        final.append(_unwrap(elems))
    return final


def run_two_branch_cfg_parallel(
    policy: "CFGPolicy",
    predict_fn: Callable[["CFGBranch"], "torch.Tensor | tuple[torch.Tensor, ...]"],
    cfg_scale: float,
    batch,
    pipeline_config,
) -> "torch.Tensor | tuple[torch.Tensor, ...]":
    """Run standard two-pass CFG with the old all-reduce combine.

    This keeps the existing WAN baselines: it avoids gathering both branch
    predictions, and it preserves the bf16 arithmetic order used before the
    multi-branch CFG dispatcher was added.
    """

    cfg_rank = get_classifier_free_guidance_rank()
    pred_t = _run(predict_fn, cfg_rank, policy.branches)

    if cfg_rank == 0:
        partial = tuple(cfg_scale * p for p in pred_t)
        cond_t = pred_t
    else:
        partial = tuple((1 - cfg_scale) * p for p in pred_t)
        cond_t = tuple(torch.empty_like(p) for p in pred_t)

    results = [cfg_model_parallel_all_reduce(p) for p in partial]
    cond_t = tuple(get_cfg_group().broadcast(p, src=0) for p in cond_t)
    results[0] = _apply_cfg_postprocess(results[0], cond_t[0], batch, pipeline_config)
    return _unwrap(tuple(results))


def dispatch_branches(n_branches: int, n_ranks: int) -> list[list[int]]:
    """Assign branches to ranks in Round-robin fashion

    Returns a list of length ``n_ranks`` where element ``r`` contains the
    branch indices assigned to rank ``r``.  Branch ``i`` goes to rank
    ``i % n_ranks``.

    Example: 4 passes, 2 GPUs:
        rank 0 -> [0, 2],  rank 1 -> [1, 3]
    """
    assignments: list[list[int]] = [[] for _ in range(n_ranks)]
    for i in range(n_branches):
        assignments[i % n_ranks].append(i)
    return assignments
