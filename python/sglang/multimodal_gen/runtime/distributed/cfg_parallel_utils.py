from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGBranch, CFGPolicy

try:
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
    logger = init_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


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
    from sglang.multimodal_gen.runtime.distributed.cfg_policy import _wrap, _unwrap

    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()
    branches = policy.branches
    n_branches = len(branches)
    assignments = dispatch_branches(n_branches, cfg_world_size)
    my_indices = assignments[cfg_rank]
    max_per_rank = max(len(a) for a in assignments)

    if cfg_world_size > n_branches:
        logger.warning_once(
            "cfg_parallel_size=%d > n_branches=%d; %d GPU(s) will be idle for CFG",
            cfg_world_size,
            n_branches,
            cfg_world_size - n_branches,
        )

    def _run(bid: int) -> tuple[torch.Tensor, ...]:
        raw = predict_fn(branches[bid])
        return tuple(p.float() for p in _wrap(raw))

    my_preds: list[tuple[torch.Tensor, ...]] = [_run(bid) for bid in my_indices]

    if not my_preds:  # idle rank: run branch 0 for tensor shapes
        my_preds.append(_run(0))

    ref = my_preds[0]
    while len(my_preds) < max_per_rank:
        my_preds.append(tuple(torch.zeros_like(t) for t in ref))

    # All-gather each slot × output element with separate_tensors=True.
    # all_slots[slot][elem] = list[Tensor] indexed by CFG rank — no reshape.
    all_slots: list[list[list[torch.Tensor]]] = [
        [cfg_model_parallel_all_gather(p, dim=0, separate_tensors=True) for p in slot_pred]
        for slot_pred in my_preds
    ]

    # Reconstruct in branch order: branch bid → owner rank (bid % W), slot (bid // W).
    n_elems = len(ref)
    final: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
    for bid in range(n_branches):
        owner = bid % cfg_world_size
        slot = bid // cfg_world_size
        elems = tuple(all_slots[slot][ei][owner] for ei in range(n_elems))
        final.append(_unwrap(elems))
    return final


def dispatch_branches(n_branches: int, n_ranks: int) -> list[list[int]]:
    """Round-robin branch-to-rank assignment.

    Returns a list of length ``n_ranks`` where element ``r`` contains the
    branch indices assigned to rank ``r``.  Branch ``i`` goes to rank
    ``i % n_ranks``.

    Example — 4 passes, 2 GPUs:
        rank 0 → [0, 2],  rank 1 → [1, 3]
    """
    assignments: list[list[int]] = [[] for _ in range(n_ranks)]
    for i in range(n_branches):
        assignments[i % n_ranks].append(i)
    return assignments
