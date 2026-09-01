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


def _geometry_signature(tensors: Sequence[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            {
                (
                    tuple(tensor["global_offset"]),
                    tuple(tensor["local_shape"]),
                )
                for tensor in tensors
            }
        )
    )


def _index_tensor_geometry(
    inventories: Sequence[dict[str, Any]],
) -> dict[str, dict[tuple[int, int, int], list[dict[str, Any]]]]:
    """Index fragments by logical tensor and (PP, EP, MoE-TP) rank."""
    result: dict[
        str, dict[tuple[int, int, int], list[dict[str, Any]]]
    ] = {}
    for inventory in inventories:
        for tensor in inventory["tensors"]:
            rank = tensor["rank"]
            coordinate = (
                int(rank["pp"]),
                int(rank["ep"]),
                int(rank["tp"]),
            )
            result.setdefault(tensor["tensor_id"], {}).setdefault(
                coordinate, []
            ).append(tensor)
    return result


def _infer_replicated_tensor_ids(
    geometry_index: dict[
        str, dict[tuple[int, int, int], list[dict[str, Any]]]
    ],
    *,
    axis: str,
    axis_size: int,
) -> frozenset[str]:
    """Find tensors whose complete fragment geometry repeats over one axis."""
    if axis_size <= 1:
        return frozenset()
    if axis not in ("ep", "tp"):
        raise ValueError(f"unsupported replication axis: {axis}")

    axis_index = 1 if axis == "ep" else 2
    fixed_indices = (0, 2) if axis == "ep" else (0, 1)
    expected = set(range(axis_size))
    replicated = set()
    for tensor_id, geometry_by_rank in geometry_index.items():
        if axis == "ep" and any(
            tensor.get("expert_id") is not None
            for tensors in geometry_by_rank.values()
            for tensor in tensors
        ):
            continue
        groups: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = {}
        for coordinate, tensors in geometry_by_rank.items():
            fixed = tuple(coordinate[index] for index in fixed_indices)
            groups.setdefault(fixed, {})[coordinate[axis_index]] = tensors
        if all(
            set(geometry_by_axis) == expected
            and len(
                {
                    _geometry_signature(tensors)
                    for tensors in geometry_by_axis.values()
                }
            )
            == 1
            for geometry_by_axis in groups.values()
        ):
            replicated.add(tensor_id)
    return frozenset(replicated)


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for begin, end in sorted(set(intervals)):
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((begin, end) for begin, end in merged)


def _infer_split_dims(
    geometry_index: dict[
        str, dict[tuple[int, int, int], list[dict[str, Any]]]
    ],
    *,
    axis: str,
    axis_size: int,
) -> dict[str, tuple[int, ...]]:
    """Infer process split dims from complete cross-rank logical geometry."""
    if axis_size <= 1:
        return {}
    if axis not in ("ep", "tp"):
        raise ValueError(f"unsupported split axis: {axis}")

    axis_index = 1 if axis == "ep" else 2
    expected = set(range(axis_size))
    result = {}
    for tensor_id, geometry_by_rank in geometry_index.items():
        if axis == "ep" and any(
            tensor.get("expert_id") is not None
            for tensors in geometry_by_rank.values()
            for tensor in tensors
        ):
            continue

        geometry_by_pp: dict[
            int, dict[tuple[int, int, int], list[dict[str, Any]]]
        ] = {}
        for coordinate, tensors in geometry_by_rank.items():
            geometry_by_pp.setdefault(coordinate[0], {})[coordinate] = tensors

        inferred_by_pp = []
        for pp_geometry in geometry_by_pp.values():
            all_tensors = [
                tensor
                for tensors in pp_geometry.values()
                for tensor in tensors
            ]
            candidates = sorted(
                {
                    dim
                    for tensor in all_tensors
                    for dim in tuple(tensor.get("shard_dims") or ())
                    if int(tensor["local_shape"][dim])
                    != int(tensor["global_shape"][dim])
                }
            )
            split_dims = []
            for dim in candidates:
                extents = {
                    int(tensor["global_shape"][dim])
                    for tensor in all_tensors
                }
                if len(extents) != 1:
                    raise WeightManifestError(
                        f"global shape differs across fragments: {tensor_id}"
                    )
                intervals_by_rank: dict[int, list[tuple[int, int]]] = {}
                for coordinate, tensors in pp_geometry.items():
                    rank = coordinate[axis_index]
                    for tensor in tensors:
                        begin = int(tensor["global_offset"][dim])
                        end = begin + int(tensor["local_shape"][dim])
                        intervals_by_rank.setdefault(rank, []).append(
                            (begin, end)
                        )
                if set(intervals_by_rank) != expected:
                    continue
                owned = sorted(
                    (begin, end, rank)
                    for rank, intervals in intervals_by_rank.items()
                    for begin, end in _merge_intervals(intervals)
                )
                cursor = 0
                for begin, end, _ in owned:
                    if begin != cursor:
                        break
                    cursor = end
                else:
                    if cursor == next(iter(extents)):
                        split_dims.append(dim)
            inferred_by_pp.append(tuple(split_dims))

        if len(set(inferred_by_pp)) == 1:
            dims = inferred_by_pp[0]
            if dims:
                result[tensor_id] = dims
    return result


def _parallel_axes(
    tensor: dict[str, Any],
    *,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    ep_split_dims: tuple[int, ...] = (),
    tp_split_dims: tuple[int, ...] = (),
    ep_replicated: bool = False,
    tp_replicated: bool = False,
) -> tuple[Any, ...]:
    from mooncake.reshard.weight import (
        OwnershipAxis,
        ReplicatedAxis,
        SplitAxis,
    )

    if len(ep_split_dims) > 1 or len(tp_split_dims) > 1:
        raise WeightManifestError(
            "runtime tensor has unsupported multi-dimensional parallel shards: "
            f"{tensor['tensor_id']}"
        )

    axes = []
    if pp_size > 1:
        axes.append(OwnershipAxis("pp"))

    if tensor.get("expert_id") is not None:
        if ep_size > 1:
            axes.append(OwnershipAxis("ep"))
    elif ep_split_dims:
        axes.append(SplitAxis("ep", dim=ep_split_dims[0]))
    elif ep_size > 1:
        axes.append(
            ReplicatedAxis("ep") if ep_replicated else OwnershipAxis("ep")
        )

    if tp_split_dims:
        axes.append(SplitAxis("tp", dim=tp_split_dims[0]))
    elif tp_size > 1:
        axes.append(
            ReplicatedAxis("tp") if tp_replicated else OwnershipAxis("tp")
        )
    return tuple(axes)


def build_mooncake_weight_manifests(
    runtime_inventories: Sequence[Any],
    *,
    placement_set_id: str,
    tp_size: int,
    pp_size: int,
    ep_size: int,
):
    """Build a placement using SGLang global TP factored as EP × MoE-TP."""
    from mooncake.reshard.weight import (
        ParallelRank,
        ParallelTopology,
        PlacementFragment,
        RuntimeBindingFragment,
        SplitAxis,
        TensorDescriptor,
        TopologyParticipant,
        WeightPlacementManifest,
        WeightPlacementPart,
        WeightRuntimeBindingManifest,
    )

    inventories = tuple(msgspec.to_builtins(item) for item in runtime_inventories)
    if not inventories:
        raise WeightManifestError("runtime inventory set must not be empty")
    if tp_size % ep_size != 0:
        raise WeightManifestError(
            f"tp_size={tp_size} must be divisible by ep_size={ep_size}"
        )
    moe_tp_size = tp_size // ep_size
    geometry_index = _index_tensor_geometry(inventories)
    ep_replicated_tensor_ids = _infer_replicated_tensor_ids(
        geometry_index,
        axis="ep",
        axis_size=ep_size,
    )
    tp_replicated_tensor_ids = _infer_replicated_tensor_ids(
        geometry_index,
        axis="tp",
        axis_size=moe_tp_size,
    )
    ep_split_dims = _infer_split_dims(
        geometry_index,
        axis="ep",
        axis_size=ep_size,
    )
    tp_split_dims = _infer_split_dims(
        geometry_index,
        axis="tp",
        axis_size=moe_tp_size,
    )

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

            parallel_axes = _parallel_axes(
                tensor,
                tp_size=moe_tp_size,
                pp_size=pp_size,
                ep_size=ep_size,
                ep_split_dims=ep_split_dims.get(tensor["tensor_id"], ()),
                tp_split_dims=tp_split_dims.get(tensor["tensor_id"], ()),
                ep_replicated=(
                    tensor["tensor_id"] in ep_replicated_tensor_ids
                ),
                tp_replicated=tensor["tensor_id"] in tp_replicated_tensor_ids,
            )
            descriptor = TensorDescriptor(
                tensor_id=tensor["tensor_id"],
                global_shape=tuple(tensor["global_shape"]),
                dtype=tensor["dtype"],
                itemsize=itemsize,
                shard_dims=tuple(
                    sorted(
                        {
                            axis.dim
                            for axis in parallel_axes
                            if isinstance(axis, SplitAxis)
                        }
                    )
                ),
                layout_fingerprint=tensor["layout_fingerprint"],
                parallel_axes=parallel_axes,
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

    expected_ranks = {
        (0, tp_rank, pp_rank, ep_rank)
        for pp_rank in range(pp_size)
        for ep_rank in range(ep_size)
        for tp_rank in range(moe_tp_size)
    }
    actual_ranks = {
        (rank.dp, rank.tp, rank.pp, rank.ep)
        for _, _, rank, _, _, _ in local_parts
    }
    if actual_ranks != expected_ranks:
        raise WeightManifestError(
            "runtime inventories do not form a complete PP×EP×MoE-TP topology: "
            f"missing={sorted(expected_ranks - actual_ranks)[:5]}, "
            f"unexpected={sorted(actual_ranks - expected_ranks)[:5]}"
        )

    topology = ParallelTopology(
        tp_size=moe_tp_size,
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
