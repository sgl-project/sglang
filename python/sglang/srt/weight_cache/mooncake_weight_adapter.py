# SPDX-License-Identifier: Apache-2.0
"""Adapt immutable SGLang weight inventories to Mooncake contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import msgspec

from .weight_runtime_manifest import WeightManifestError


class _ImmutableWeightAllocationToken:
    def __init__(self, fence: Any) -> None:
        self._fence = fence

    @property
    def fence(self) -> Any:
        return self._fence

    def release_after_terminal(self, terminal_state: Any) -> None:
        pass


class _ImmutableWeightAllocationGuard:
    """Attest one binding owned by a non-updating weight-cache daemon."""

    def __init__(self, binding: Any) -> None:
        self._binding = binding

    def acquire(
        self,
        *,
        transfer_id: str,
        expected_binding: Any,
        required_fragment_ids: Sequence[str],
    ) -> Any:
        from mooncake.reshard.weight.lifetime import (
            AcquiredWeightBinding,
            weight_allocation_fence,
        )

        if expected_binding != self._binding:
            raise WeightManifestError(
                "immutable daemon weight binding differs from the planned binding"
            )
        fragment_ids = {fragment.fragment_id for fragment in self._binding.fragments}
        missing = set(required_fragment_ids) - fragment_ids
        if missing:
            raise WeightManifestError(
                f"immutable daemon weight binding is missing fragments: {sorted(missing)}"
            )
        token = _ImmutableWeightAllocationToken(
            weight_allocation_fence(
                self._binding,
                required_fragment_ids,
                token_id=(
                    f"immutable:{self._binding.instance_id}:"
                    f"{self._binding.participant_id}:{transfer_id}"
                ),
            )
        )
        return AcquiredWeightBinding(binding=self._binding, token=token)


def immutable_weight_allocation_guards(
    bindings: Sequence[Any],
) -> dict[tuple[str, str], _ImmutableWeightAllocationGuard]:
    return {
        (binding.instance_id, binding.participant_id): (
            _ImmutableWeightAllocationGuard(binding)
        )
        for binding in bindings
    }


def _parallel_axes(
    tensor: dict[str, Any],
    *,
    tp_size: int,
    pp_size: int,
    ep_size: int,
) -> tuple[Any, ...]:
    from mooncake.reshard.weight import (
        OwnershipAxis,
        ReplicatedAxis,
        SplitAxis,
    )

    axes = []
    if pp_size > 1:
        axes.append(OwnershipAxis("pp"))

    shard_dims = tuple(tensor.get("shard_dims") or ())
    ep_dim = None
    if tensor.get("expert_id") is not None:
        axes.append(OwnershipAxis("ep"))
    elif ep_size > 1 and 0 in shard_dims:
        ep_dim = 0
        axes.append(SplitAxis("ep", dim=ep_dim))

    tp_dims = tuple(dim for dim in shard_dims if dim != ep_dim)
    if tp_dims:
        if len(tp_dims) != 1:
            raise WeightManifestError(
                "runtime tensor has unsupported TP shard semantics"
            )
        axes.append(SplitAxis("tp", dim=tp_dims[0]))
    elif tp_size > 1 and ep_dim is None and tensor.get("expert_id") is None:
        axes.append(ReplicatedAxis("tp"))
    return tuple(axes)


def build_mooncake_weight_manifests(
    runtime_inventories: Sequence[Any],
    *,
    placement_set_id: str,
    tp_size: int,
    pp_size: int,
    ep_size: int,
):
    """Build one Mooncake placement and its per-participant runtime bindings."""
    from mooncake.reshard.weight import (
        ParallelRank,
        ParallelTopology,
        PlacementFragment,
        RuntimeBindingFragment,
        TensorDescriptor,
        TopologyParticipant,
        WeightPlacementManifest,
        WeightPlacementPart,
        WeightRuntimeBindingManifest,
    )

    inventories = tuple(msgspec.to_builtins(item) for item in runtime_inventories)
    if not inventories:
        raise WeightManifestError("runtime inventory set must not be empty")

    local_parts = []
    identity = None
    for inventory in inventories:
        current_identity = (
            inventory["model_id"],
            inventory["revision"],
            inventory["generation"],
        )
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise WeightManifestError(
                "runtime inventories have different model identities"
            )

        tensors = inventory["tensors"]
        if not tensors:
            raise WeightManifestError("runtime inventory must contain tensors")
        ranks = {
            tuple(tensor["rank"][axis] for axis in ("dp", "tp", "pp", "ep"))
            for tensor in tensors
        }
        if len(ranks) != 1:
            raise WeightManifestError("runtime inventory spans multiple parallel ranks")
        dp_rank, tp_rank, pp_rank, ep_rank = next(iter(ranks))
        rank = ParallelRank(
            dp=dp_rank,
            tp=tp_rank,
            pp=pp_rank,
            ep=ep_rank,
        )
        participant_id = f"dp{dp_rank}:pp{pp_rank}:ep{ep_rank}:tp{tp_rank}"

        aliases_by_storage = {}
        for tensor in tensors:
            alias_key = (
                tensor["address"],
                tensor["nbytes"],
                tuple(tensor["global_offset"]),
                tuple(tensor["local_shape"]),
                tensor["dtype"],
            )
            aliases_by_storage.setdefault(alias_key, set()).add(tensor["tensor_id"])

        descriptors = {}
        placement_fragments = []
        placement_id_by_runtime_fragment = {}
        storage_ends = {}
        runtime_storage = {}
        for tensor in tensors:
            itemsize = int(tensor["itemsize"])
            storage_offset_bytes = int(tensor["storage_offset"]) * itemsize
            storage_address = int(tensor["address"]) - storage_offset_bytes
            storage_ends[storage_address] = max(
                storage_ends.get(storage_address, 0),
                storage_offset_bytes + int(tensor["nbytes"]),
            )
            runtime_storage[tensor["fragment_id"]] = (
                storage_address,
                storage_offset_bytes,
            )

            descriptor = TensorDescriptor(
                tensor_id=tensor["tensor_id"],
                global_shape=tuple(tensor["global_shape"]),
                dtype=tensor["dtype"],
                itemsize=itemsize,
                shard_dims=tuple(tensor.get("shard_dims") or ()),
                layout_fingerprint=tensor["layout_fingerprint"],
                parallel_axes=_parallel_axes(
                    tensor,
                    tp_size=tp_size,
                    pp_size=pp_size,
                    ep_size=ep_size,
                ),
                layer_id=tensor.get("layer_id"),
                expert_id=tensor.get("expert_id"),
            )
            previous = descriptors.setdefault(descriptor.tensor_id, descriptor)
            if previous != descriptor:
                raise WeightManifestError(
                    f"runtime tensor descriptors disagree: {descriptor.tensor_id}"
                )

            alias_key = (
                tensor["address"],
                tensor["nbytes"],
                tuple(tensor["global_offset"]),
                tuple(tensor["local_shape"]),
                tensor["dtype"],
            )
            aliases = tuple(sorted(aliases_by_storage[alias_key]))
            fragment = PlacementFragment(
                tensor_id=tensor["tensor_id"],
                global_offset=tuple(tensor["global_offset"]),
                local_shape=tuple(tensor["local_shape"]),
                nbytes=int(tensor["nbytes"]),
                rank=rank,
                aliases=aliases if len(aliases) > 1 else (),
            )
            placement_fragments.append(fragment)
            placement_id_by_runtime_fragment[tensor["fragment_id"]] = (
                fragment.placement_fragment_id
            )

        runtime_fragments = []
        for tensor in tensors:
            storage_address, storage_offset_bytes = runtime_storage[
                tensor["fragment_id"]
            ]
            runtime_fragments.append(
                RuntimeBindingFragment(
                    placement_fragment_id=placement_id_by_runtime_fragment[
                        tensor["fragment_id"]
                    ],
                    fragment_id=tensor["fragment_id"],
                    address=int(tensor["address"]),
                    nbytes=int(tensor["nbytes"]),
                    worker_id=tensor["worker_id"],
                    endpoint=tensor["endpoint"],
                    device=tensor["device"],
                    itemsize=int(tensor["itemsize"]),
                    local_shape=tuple(tensor["local_shape"]),
                    strides_bytes=tuple(
                        int(stride) * int(tensor["itemsize"])
                        for stride in tensor["stride"]
                    ),
                    storage_address=storage_address,
                    storage_nbytes=storage_ends[storage_address],
                    storage_offset_bytes=storage_offset_bytes,
                )
            )

        local_parts.append(
            (
                inventory,
                participant_id,
                rank,
                tuple(descriptors.values()),
                tuple(placement_fragments),
                tuple(runtime_fragments),
            )
        )

    topology = ParallelTopology(
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        dp_size=1,
        participants=tuple(
            TopologyParticipant(participant_id, rank)
            for _, participant_id, rank, _, _, _ in local_parts
        ),
    )
    placement_parts = tuple(
        WeightPlacementPart(
            resource_id=inventory["model_id"],
            revision=inventory["revision"],
            weight_generation=int(inventory["generation"]),
            placement_set_id=placement_set_id,
            topology_id=topology.topology_id,
            participant_id=participant_id,
            rank=rank,
            tensors=descriptors,
            fragments=placement_fragments,
        )
        for (
            inventory,
            participant_id,
            rank,
            descriptors,
            placement_fragments,
            _,
        ) in local_parts
    )
    first = placement_parts[0]
    placement = WeightPlacementManifest(
        resource_id=first.resource_id,
        revision=first.revision,
        weight_generation=first.weight_generation,
        placement_set_id=placement_set_id,
        topology=topology,
        parts=placement_parts,
    )
    bindings = tuple(
        WeightRuntimeBindingManifest(
            resource_id=placement.resource_id,
            revision=placement.revision,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id=inventory["instance_id"],
            participant_id=participant_id,
            generation=int(inventory["generation"]),
            lease_id=f"immutable:{inventory['instance_id']}",
            fragments=runtime_fragments,
        )
        for inventory, participant_id, _, _, _, runtime_fragments in local_parts
    )
    return placement, bindings
