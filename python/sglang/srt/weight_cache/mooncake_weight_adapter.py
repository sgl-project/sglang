# SPDX-License-Identifier: Apache-2.0
"""Adapt immutable SGLang weight manifests to Mooncake contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
        # weight_allocation_fence rejects fragment ids this binding does not own,
        # so the ownership contract is enforced once, immediately below.
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


def _alias_key(tensor: dict[str, Any]) -> tuple[Any, ...]:
    """Identify the exact bytes a fragment occupies, so aliases collapse."""
    return (
        tensor["address"],
        tensor["nbytes"],
        tuple(tensor["global_offset"]),
        tuple(tensor["local_shape"]),
        tensor["dtype"],
    )


def _storage_base(tensor: dict[str, Any]) -> tuple[int, int]:
    """Recover the owning allocation base and this view's offset, in bytes."""
    storage_offset_bytes = int(tensor["storage_offset"]) * int(tensor["itemsize"])
    return int(tensor["address"]) - storage_offset_bytes, storage_offset_bytes


def _axis_index(axis: str) -> int:
    """Map a parallel axis onto its slot in the (PP, EP, MoE-TP) coordinate."""
    if axis == "ep":
        return 1
    if axis == "tp":
        return 2
    raise ValueError(f"unsupported parallel axis: {axis}")


def _has_expert_fragments(
    geometry_by_rank: dict[tuple[int, int, int], list[dict[str, Any]]],
) -> bool:
    """Report whether any fragment carries a per-expert identity."""
    return any(
        tensor.get("expert_id") is not None
        for tensors in geometry_by_rank.values()
        for tensor in tensors
    )


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
    manifests: Sequence[dict[str, Any]],
) -> dict[str, dict[tuple[int, int, int], list[dict[str, Any]]]]:
    """Index fragments by logical tensor and (PP, EP, MoE-TP) rank."""
    result: dict[
        str, dict[tuple[int, int, int], list[dict[str, Any]]]
    ] = {}
    for manifest in manifests:
        for tensor in manifest["tensors"]:
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
    axis_index = _axis_index(axis)
    fixed_indices = (0, 2) if axis == "ep" else (0, 1)
    expected = set(range(axis_size))
    replicated = set()
    for tensor_id, geometry_by_rank in geometry_index.items():
        if axis == "ep" and _has_expert_fragments(geometry_by_rank):
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


def _is_exact_partition(
    owned: Sequence[tuple[int, int, int]],
    extent: int,
) -> bool:
    """Report whether sorted owned intervals tile [0, extent) without gaps."""
    cursor = 0
    for begin, end, _ in owned:
        if begin != cursor:
            return False
        cursor = end
    return cursor == extent


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
    axis_index = _axis_index(axis)
    expected = set(range(axis_size))
    result = {}
    for tensor_id, geometry_by_rank in geometry_index.items():
        if axis == "ep" and _has_expert_fragments(geometry_by_rank):
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
                if _is_exact_partition(owned, next(iter(extents))):
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


@dataclass(frozen=True)
class _ParallelGeometry:
    """Per-tensor split/replication structure recovered from reported geometry."""

    ep_replicated_tensor_ids: frozenset[str]
    tp_replicated_tensor_ids: frozenset[str]
    ep_split_dims: dict[str, tuple[int, ...]]
    tp_split_dims: dict[str, tuple[int, ...]]


def _infer_parallel_geometry(
    manifests: Sequence[dict[str, Any]],
    *,
    ep_size: int,
    moe_tp_size: int,
) -> _ParallelGeometry:
    """Recover how each tensor is split or replicated across the two axes.

    The structure is derived by comparing the geometry every rank reported, not
    read from model configuration, so no per-architecture knowledge is needed.
    """
    geometry_index = _index_tensor_geometry(manifests)
    return _ParallelGeometry(
        ep_replicated_tensor_ids=_infer_replicated_tensor_ids(
            geometry_index,
            axis="ep",
            axis_size=ep_size,
        ),
        tp_replicated_tensor_ids=_infer_replicated_tensor_ids(
            geometry_index,
            axis="tp",
            axis_size=moe_tp_size,
        ),
        ep_split_dims=_infer_split_dims(
            geometry_index,
            axis="ep",
            axis_size=ep_size,
        ),
        tp_split_dims=_infer_split_dims(
            geometry_index,
            axis="tp",
            axis_size=moe_tp_size,
        ),
    )


@dataclass(frozen=True)
class _ParticipantPart:
    """One rank's contribution, split into logical and physical halves."""

    manifest: dict[str, Any]
    participant_id: str
    rank: Any
    descriptors: tuple[Any, ...]
    placement_fragments: tuple[Any, ...]
    runtime_fragments: tuple[Any, ...]


def _resolve_rank(tensors: Sequence[dict[str, Any]]):
    """Return the single parallel rank every fragment in one manifest shares."""
    from mooncake.reshard.weight import ParallelRank

    ranks = {
        tuple(tensor["rank"][axis] for axis in ("dp", "tp", "pp", "ep"))
        for tensor in tensors
    }
    if len(ranks) != 1:
        raise WeightManifestError("runtime manifest spans multiple parallel ranks")
    dp_rank, tp_rank, pp_rank, ep_rank = next(iter(ranks))
    return ParallelRank(dp=dp_rank, tp=tp_rank, pp=pp_rank, ep=ep_rank)


def _build_participant_part(
    manifest: dict[str, Any],
    *,
    geometry: _ParallelGeometry,
    moe_tp_size: int,
    pp_size: int,
    ep_size: int,
) -> _ParticipantPart:
    """Translate one rank's manifest into placement and runtime fragments.

    Three passes are required and cannot be merged: aliases must be known before
    fragments are emitted, and a storage's full extent is only known once every
    fragment sharing it has been seen.
    """
    from mooncake.reshard.weight import (
        PlacementFragment,
        RuntimeBindingFragment,
        SplitAxis,
        TensorDescriptor,
    )

    tensors = manifest["tensors"]
    if not tensors:
        raise WeightManifestError("runtime manifest must contain tensors")
    rank = _resolve_rank(tensors)
    participant_id = f"dp{rank.dp}:pp{rank.pp}:ep{rank.ep}:tp{rank.tp}"

    aliases_by_storage: dict[tuple[Any, ...], set[str]] = {}
    for tensor in tensors:
        aliases_by_storage.setdefault(_alias_key(tensor), set()).add(
            tensor["tensor_id"]
        )

    descriptors: dict[str, Any] = {}
    placement_fragments = []
    placement_id_by_runtime_fragment = {}
    storage_ends: dict[int, int] = {}
    for tensor in tensors:
        storage_address, storage_offset_bytes = _storage_base(tensor)
        storage_ends[storage_address] = max(
            storage_ends.get(storage_address, 0),
            storage_offset_bytes + int(tensor["nbytes"]),
        )

        tensor_id = tensor["tensor_id"]
        parallel_axes = _parallel_axes(
            tensor,
            tp_size=moe_tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            ep_split_dims=geometry.ep_split_dims.get(tensor_id, ()),
            tp_split_dims=geometry.tp_split_dims.get(tensor_id, ()),
            ep_replicated=tensor_id in geometry.ep_replicated_tensor_ids,
            tp_replicated=tensor_id in geometry.tp_replicated_tensor_ids,
        )
        descriptor = TensorDescriptor(
            tensor_id=tensor_id,
            global_shape=tuple(tensor["global_shape"]),
            dtype=tensor["dtype"],
            itemsize=int(tensor["itemsize"]),
            # TensorDescriptor validates rather than derives this, so the split
            # dimensions have to be projected out of the axes here.
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

        aliases = tuple(sorted(aliases_by_storage[_alias_key(tensor)]))
        fragment = PlacementFragment(
            tensor_id=tensor_id,
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
        storage_address, storage_offset_bytes = _storage_base(tensor)
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

    return _ParticipantPart(
        manifest=manifest,
        participant_id=participant_id,
        rank=rank,
        descriptors=tuple(descriptors.values()),
        placement_fragments=tuple(placement_fragments),
        runtime_fragments=tuple(runtime_fragments),
    )


def _require_single_model_identity(manifests: Sequence[dict[str, Any]]) -> None:
    """Reject a manifest set that mixes models, revisions, or generations."""
    identities = {
        (manifest["model_id"], manifest["revision"], manifest["generation"])
        for manifest in manifests
    }
    if len(identities) != 1:
        raise WeightManifestError("runtime manifests have different model identities")


def _require_complete_topology(
    parts: Sequence[_ParticipantPart],
    *,
    moe_tp_size: int,
    pp_size: int,
    ep_size: int,
) -> None:
    """Reject a partial grid.

    ParallelTopology only range-checks each rank component, so a missing or
    duplicated participant would otherwise reach the planner unnoticed.
    """
    expected_ranks = {
        (0, tp_rank, pp_rank, ep_rank)
        for pp_rank in range(pp_size)
        for ep_rank in range(ep_size)
        for tp_rank in range(moe_tp_size)
    }
    actual_ranks = {
        (part.rank.dp, part.rank.tp, part.rank.pp, part.rank.ep) for part in parts
    }
    if actual_ranks != expected_ranks:
        raise WeightManifestError(
            "runtime manifests do not form a complete PP×EP×MoE-TP topology: "
            f"missing={sorted(expected_ranks - actual_ranks)[:5]}, "
            f"unexpected={sorted(actual_ranks - expected_ranks)[:5]}"
        )


def build_mooncake_placement_and_bindings(
    runtime_manifests: Sequence[Any],
    *,
    placement_set_id: str,
    tp_size: int,
    pp_size: int,
    ep_size: int,
):
    """Translate per-rank weight manifests into Mooncake placement and bindings.

    SGLang's flat global TP is factored as EP × MoE-TP because Mooncake models
    those as separate axes. The address-free placement is what the planner
    diffs; the bindings carry the addresses used only at late binding time.
    """
    from mooncake.reshard.weight import (
        ParallelTopology,
        TopologyParticipant,
        WeightPlacementManifest,
        WeightPlacementPart,
        WeightRuntimeBindingManifest,
    )

    manifests = tuple(msgspec.to_builtins(item) for item in runtime_manifests)
    if not manifests:
        raise WeightManifestError("runtime manifest set must not be empty")
    _require_single_model_identity(manifests)

    moe_tp_size = tp_size // ep_size
    geometry = _infer_parallel_geometry(
        manifests,
        ep_size=ep_size,
        moe_tp_size=moe_tp_size,
    )
    parts = tuple(
        _build_participant_part(
            manifest,
            geometry=geometry,
            moe_tp_size=moe_tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
        )
        for manifest in manifests
    )
    _require_complete_topology(
        parts,
        moe_tp_size=moe_tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
    )

    topology = ParallelTopology(
        tp_size=moe_tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        dp_size=1,
        participants=tuple(
            TopologyParticipant(part.participant_id, part.rank) for part in parts
        ),
    )
    placement_parts = tuple(
        WeightPlacementPart(
            resource_id=part.manifest["model_id"],
            revision=part.manifest["revision"],
            weight_generation=int(part.manifest["generation"]),
            placement_set_id=placement_set_id,
            topology_id=topology.topology_id,
            participant_id=part.participant_id,
            rank=part.rank,
            tensors=part.descriptors,
            fragments=part.placement_fragments,
        )
        for part in parts
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
            instance_id=part.manifest["instance_id"],
            participant_id=part.participant_id,
            generation=int(part.manifest["generation"]),
            lease_id=f"immutable:{part.manifest['instance_id']}",
            fragments=part.runtime_fragments,
        )
        for part in parts
    )
    return placement, bindings
