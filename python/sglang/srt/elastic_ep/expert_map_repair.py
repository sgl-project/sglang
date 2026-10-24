"""Expert-map repair for an elastic shrink. Survivors repair independently without
exchanging maps, so the rule must be deterministic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List

import torch

from sglang.srt.elastic_ep.errors import ElasticLayoutFatal
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.runtime_context import get_context

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def local_relabelled_logicals(
    old_p2l: torch.Tensor,
    new_p2l: torch.Tensor,
    *,
    num_local: int,
    ep_rank: int,
) -> Dict[int, List[int]]:
    """Per layer, logicals this rank's own slots now hold but did not before; slots
    past the old width count as changed."""
    lo = num_local * ep_rank
    hi = min(lo + num_local, new_p2l.shape[1])
    if hi <= lo:
        return {}
    window = new_p2l[:, lo:hi]
    changed = torch.ones_like(window, dtype=torch.bool)
    overlap = min(old_p2l.shape[1], hi) - lo
    if overlap > 0:
        changed[:, :overlap] = window[:, :overlap] != old_p2l[:, lo : lo + overlap]
    changed &= window >= 0
    return {
        layer_id: sorted(set(logicals))
        for layer_id in range(window.shape[0])
        if (logicals := window[layer_id][changed[layer_id]].tolist())
    }


def repair_orphan_logicals(
    *, shrunk_p2l: torch.Tensor, num_logical: int
) -> Dict[int, List[int]]:
    """Give every logical a replica by relabelling duplicated slots of ``shrunk_p2l``
    in place. Returns the relabelled logicals per layer; the caller must reload them."""
    num_layers, new_num_physical = shrunk_p2l.shape
    if new_num_physical < num_logical:
        raise RuntimeError(
            f"Shrink leaves {new_num_physical} slots for {num_logical} logicals; "
            "increase --ep-num-redundant-experts."
        )

    # Keyed by the global layer id the weight-name filter parses.
    repaired: Dict[int, List[int]] = {}
    for layer_id in range(num_layers):
        row = shrunk_p2l[layer_id].tolist()
        counts = [0] * num_logical
        for value in row:
            if 0 <= value < num_logical:
                counts[value] += 1
        orphans = [l for l in range(num_logical) if counts[l] == 0]
        if not orphans:
            continue
        repaired[layer_id] = orphans
        for orphan in orphans:
            for slot_idx, value in enumerate(row):
                if 0 <= value < num_logical and counts[value] >= 2:
                    counts[value] -= 1
                    row[slot_idx] = orphan
                    counts[orphan] = 1
                    break
            else:
                raise RuntimeError(
                    f"Layer {layer_id}: no duplicate to cover logical {orphan}; "
                    "increase --ep-num-redundant-experts."
                )
        shrunk_p2l[layer_id] = torch.tensor(
            row, dtype=shrunk_p2l.dtype, device=shrunk_p2l.device
        )

    if repaired:
        logger.warning(
            "[Elastic EP] shrink orphaned %d logical expert(s) across %d layer(s); "
            "reloading their weights (the donor slots hold the wrong ones): %s",
            sum(len(v) for v in repaired.values()),
            len(repaired),
            dict(list(repaired.items())[:4]),
        )
    return repaired


def shrink_expert_metadata(
    *,
    model_config: ModelConfig,
    from_ep_size: int,
    effective_size: int,
    moe_ep_rank: int,
    reload_relabelled: Callable[[Dict[int, List[int]]], None],
) -> None:
    """Truncate physical_to_logical_map to survivor slots + repair orphaned logicals."""
    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return
    old_num_physical = metadata.num_physical_experts
    new_num_physical = old_num_physical // from_ep_size * effective_size
    if new_num_physical >= old_num_physical:
        return

    get_context().override("elastic_ep.scale", ep_size=effective_size)
    # clone(), not contiguous(): a single-MoE-layer slice is already contiguous, so
    # the repair below would write through into the still-installed metadata.
    shrunk_p2l = metadata.physical_to_logical_map[:, :new_num_physical].clone()
    relabelled = repair_orphan_logicals(
        shrunk_p2l=shrunk_p2l, num_logical=metadata.num_logical_experts
    )
    metadata.adopt_scaled_in_place(
        ExpertLocationMetadata.init_by_mapping(
            model_config, physical_to_logical_map=shrunk_p2l, moe_ep_rank=moe_ep_rank
        )
    )
    # After install so the loader follows the map. Fatal on failure: the map says a
    # slot holds the orphan while it still holds its donor.
    try:
        reload_relabelled(relabelled)
    except Exception as exc:
        raise ElasticLayoutFatal(
            "Elastic EP shrink left this rank's expert weights out of sync with "
            f"its expert map ({exc}). It must not continue serving."
        ) from exc
