# SPDX-License-Identifier: Apache-2.0
"""The expert map of a load-diff snapshot (D2 of #50943).

``FusedMoE.weight_loader`` places each checkpoint expert in a physical slot by
the process's physical-to-logical expert map. An owner's expert rows are
therefore laid out by the owner's map, while a reader picks each token's slot
by its own. The map travels in the manifest, and a reader takes it without
moving a row:

- ``export_expert_map``: owner, the map it placed its experts by.
- ``check_expert_map``: reader, before anything is attached: nothing to take,
  the owner's map, or a refusal.
- ``adopt_expert_map``: reader, once the owner's tensors are installed: this
  process dispatches by the owner's map.
- ``agree_on_expert_map``: every rank, once its model is loaded: the ranks
  all dispatch by the same map.

Not handled: a draft load. The process's map is its target model's, and the
weight cache refuses speculative decoding at startup. A snapshot source that
serves draft models has to say whose map a draft's snapshot carries.

The EPLB and runtime modules are imported inside the functions: load_diff
imports this module and stays light to import.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from sglang.srt.weight_cache.load_diff_format import (
    ExpertMap,
    SnapshotError,
    expert_map_digest,
)

logger = logging.getLogger(__name__)

_REFUSED = "this reader cannot take the owner's expert map"


def export_expert_map() -> Optional[ExpertMap]:
    """Owner: the map this process placed its experts by, or None (a model
    without one)."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return None
    rows = metadata.physical_to_logical_map_cpu.tolist()
    return ExpertMap(
        num_logical_experts=metadata.num_logical_experts,
        ep_size=metadata.ep_size,
        rows=rows,
        digest=expert_map_digest(rows),
    )


def check_expert_map(owner: Optional[ExpertMap]) -> Optional[torch.Tensor]:
    """Reader, before anything is attached: what this process takes from the
    ``owner``'s map (as validate_manifest left it). None: nothing, there is no
    map on either side or the values are this process's own. A tensor: the
    owner's map, for adopt_expert_map. Raises SnapshotError for a map this
    process cannot dispatch by."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    metadata = get_global_expert_location_metadata()
    if owner is None and metadata is None:
        return None
    if metadata is None:
        raise SnapshotError(
            _REFUSED,
            ["the owner placed its experts by a map, and this process has none"],
        )
    if owner is None:
        raise SnapshotError(
            _REFUSED,
            ["this process dispatches by a map, and the owner's snapshot has none"],
        )
    mine = metadata.physical_to_logical_map_cpu.to(torch.int64)
    here = (*mine.shape, metadata.num_logical_experts, metadata.ep_size)
    there = (
        len(owner.rows),
        len(owner.rows[0]),
        owner.num_logical_experts,
        owner.ep_size,
    )
    if there != here:
        raise SnapshotError(
            _REFUSED,
            [
                "another shape: (layers, physical slots, logical experts, EP "
                f"size) is {there} on the owner and {here} here"
            ],
        )
    theirs = torch.tensor(owner.rows, dtype=torch.int64)
    if torch.equal(theirs, mine):
        return None
    why = _why_not_another_layout()
    if why is not None:
        changed = int(torch.count_nonzero(theirs != mine))
        raise SnapshotError(
            _REFUSED,
            [
                f"{changed} of {mine.numel()} physical slots hold another "
                f"expert on the owner than here, and {why}"
            ],
        )
    return theirs


def _why_not_another_layout() -> Optional[str]:
    """Why this process cannot dispatch by another layout than its own, or
    None. Its MoE layers pick a token's slot through the process's map only
    with a dispatch algorithm (ExpertLocationDispatchInfo.init_new gives none
    without one) and with remapping on (is_expert_remapping_enabled: EPLB, a
    non-trivial initial layout or redundant experts). Otherwise a token runs
    in the slot of its expert's number, whatever the map says."""
    from sglang.srt.layers.moe.topk import is_expert_remapping_enabled
    from sglang.srt.runtime_context import get_exec

    moe = get_exec().moe
    if moe.is_ep_scale_joiner:
        return (
            "an elastic EP scale joiner computes its layout for a shifted "
            "rank and takes it from the ranks it joins"
        )
    if moe.ep_dispatch_algorithm is None or not is_expert_remapping_enabled():
        return (
            "this engine's MoE layers do not pick a token's slot through the "
            "map (that takes --ep-dispatch-algorithm, here "
            f"{moe.ep_dispatch_algorithm!r}, and one of --init-expert-location, "
            "--enable-eplb or --ep-num-redundant-experts). Start this engine "
            "with the owner's --init-expert-location"
        )
    if moe.kt_weight_path is not None:
        return (
            "this engine places the experts it runs on the CPU by its own map "
            "(--kt-weight-path). Start this engine with the owner's "
            "--init-expert-location"
        )
    return None


def adopt_expert_map(theirs: torch.Tensor) -> Dict[str, Any]:
    """Reader, once the owner's tensors are installed: dispatch by the owner's
    map (``theirs``: what check_expert_map returned). No row moves: the
    installed expert rows are in that layout already.

    The process's ExpertLocationMetadata is updated in place, as a rebalance
    updates it: the expert distribution recorder keeps the object, and the
    dispatch reads its tensors at forward time. The LPLB solvers, built from
    this process's own map before the load, are built again. Returns what
    changed, for the install report."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.runtime_context import get_exec, get_parallel

    metadata = get_global_expert_location_metadata()
    mine = metadata.physical_to_logical_map_cpu.to(torch.int64, copy=True)
    metadata.adopt_layout(theirs, moe_ep_rank=get_parallel().moe_ep_rank)
    lplb = get_exec().moe.ep_dispatch_algorithm == "lp"
    if lplb:
        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            rebuild_lplb_solvers,
        )

        rebuild_lplb_solvers()
    if not torch.equal(metadata.physical_to_logical_map_cpu.to(torch.int64), theirs):
        raise SnapshotError(
            "cannot dispatch by the owner's expert map",
            ["this process's map is not the owner's after taking it"],
        )
    report = {
        "digest": expert_map_digest(theirs.tolist()).hex(),
        "changed_physical_slots": int(torch.count_nonzero(mine != theirs)),
        "physical_slots": theirs.numel(),
        "lplb_solvers_rebuilt": lplb,
    }
    logger.warning(
        "load-diff: adopted the owner's expert map. This process set another "
        "expert layout than the owner placed its experts by; it now dispatches "
        "tokens by the owner's, and its own is not used. %s",
        report,
    )
    return report


def agree_on_expert_map(group) -> None:
    """Every rank of ``group``, once its model is loaded: the ranks all
    dispatch by the same expert map, or this raises the same report on each
    of them. ``group``: a GroupCoordinator of every rank that exchanges
    tokens by physical slot; one small object per rank goes over its CPU
    group.

    A reader takes its owner's map on its own. A rank that found no snapshot
    loads from disk and keeps the map it set, and nothing else compares the
    maps that the owners of different ranks exported."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    metadata = get_global_expert_location_metadata()
    mine = (
        "no expert map"
        if metadata is None
        else expert_map_digest(metadata.physical_to_logical_map_cpu.tolist()).hex()
    )
    ranks: Dict[str, list] = {}
    for rank, digest in enumerate(group.all_gather_object(mine)):
        ranks.setdefault(digest, []).append(rank)
    if len(ranks) > 1:
        raise SnapshotError(
            "the ranks of this engine would dispatch tokens by different expert "
            "maps. A rank that installs a snapshot dispatches by its owner's "
            "map, a rank that loads from disk by its own. Start every rank and "
            "every owner with the same --init-expert-location, or have every "
            "rank load the snapshot",
            [f"ranks {held}: {digest}" for digest, held in ranks.items()],
        )
