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


@dataclass(frozen=True)
class _Distribution:
    tp_size: int
    ep_size: int
    dp_size: int
    # Indexed by physical global-TP rank; values are (DP replica, TP shard, EP).
    ranks: tuple[tuple[int, int, int], ...]

    def axes(self, tensor, pp_size):
        from mooncake.reshard.weight import OwnershipAxis, ReplicatedAxis, SplitAxis

        axes = [ReplicatedAxis("dp")] if self.dp_size > 1 else []
        if pp_size > 1:
            axes.append(OwnershipAxis("pp"))
        if self.ep_size > 1:
            axes.append(
                OwnershipAxis("ep")
                if tensor.expert_id is not None
                else SplitAxis("ep", dim=0)
            )
        if self.tp_size > 1:
            axes.append(
                ReplicatedAxis("tp")
                if tensor.partition.dim is None or tensor.partition.size == 1
                else SplitAxis("tp", dim=tensor.partition.dim)
            )
        return tuple(axes)


class _ManifestIndex:
    """Index explicit declarations once; never infer process axes from boxes."""

    def __init__(self, manifests, layout):
        from .weight_runtime_manifest import WeightRuntimeManifest

        self.layout = layout
        self.moe_tp_size = layout.tp_size // layout.ep_size
        by_rank = {}
        for value in manifests:
            manifest = (
                value
                if isinstance(value, WeightRuntimeManifest)
                else msgspec.convert(value, type=WeightRuntimeManifest)
            )
            rank = manifest.rank
            if (
                rank.dp != 0
                or not 0 <= rank.pp < layout.pp_size
                or not 0 <= rank.ep < layout.ep_size
                or not 0 <= rank.tp < self.moe_tp_size
            ):
                raise WeightManifestError(
                    "runtime manifest rank is outside its topology"
                )
            ordinal = rank.pp * layout.tp_size + self.global_tp(rank)
            if ordinal in by_rank:
                raise WeightManifestError("duplicate runtime manifest rank")
            by_rank[ordinal] = manifest
        if set(by_rank) != set(range(layout.tp_size * layout.pp_size)):
            raise WeightManifestError(
                "runtime manifests do not form a complete PP/TP topology"
            )
        self.manifests = tuple(by_rank[i] for i in sorted(by_rank))
        if len({(m.model_id, m.revision, m.generation) for m in self.manifests}) != 1:
            raise WeightManifestError(
                "runtime manifests have different model identities"
            )
        self.tensors, self.declarations, self.fragments = {}, {}, {}
        for index, manifest in enumerate(self.manifests):
            local = {t.tensor_id: t for t in manifest.tensors}
            if len(local) != len(manifest.tensors):
                raise WeightManifestError("duplicate logical tensor declaration")
            for name, tensor in local.items():
                previous = self.tensors.setdefault(name, tensor)
                if previous.definition() != tensor.definition():
                    raise WeightManifestError(f"tensor declarations disagree: {name}")
                self.declarations.setdefault(name, []).append((index, tensor.partition))
                self.fragments.setdefault(name, [])
            for fragment in manifest.fragments:
                if fragment.tensor_id not in local:
                    raise WeightManifestError(
                        f"fragment has no local declaration: {fragment.tensor_id}"
                    )
                self.fragments[fragment.tensor_id].append((index, fragment))
        if not self.tensors:
            raise WeightManifestError("runtime manifests contain no logical tensors")
        self.distributions = {name: self._distribution(name) for name in self.tensors}

    def global_tp(self, rank):
        return rank.ep * self.moe_tp_size + rank.tp

    def _distribution(self, name):
        tensor = self.tensors[name]
        if tensor.partition.kind == "expert":
            owners = {}
            for index, partition in self.declarations[name]:
                rank = self.manifests[index].rank
                if partition.size != self.moe_tp_size or partition.rank != rank.tp:
                    raise WeightManifestError(
                        f"expert partition differs from MoE-TP: {name}"
                    )
                owners.setdefault((rank.pp, rank.ep), set()).add(rank.tp)
            if any(ranks != set(range(self.moe_tp_size)) for ranks in owners.values()):
                raise WeightManifestError(f"incomplete expert partition: {name}")
            return _Distribution(
                self.moe_tp_size,
                self.layout.ep_size,
                1,
                tuple(
                    (0, rank % self.moe_tp_size, rank // self.moe_tp_size)
                    for rank in range(self.layout.tp_size)
                ),
            )
        if tensor.expert_id is not None:
            raise WeightManifestError(
                f"expert tensor declares a dense partition: {name}"
            )
        by_pp = {}
        for index, partition in self.declarations[name]:
            rank = self.manifests[index].rank
            by_pp.setdefault(rank.pp, {})[self.global_tp(rank)] = partition.rank
        mapping = None
        size = tensor.partition.size
        for declared in by_pp.values():
            if set(declared) != set(range(self.layout.tp_size)):
                raise WeightManifestError(
                    f"incomplete dense partition declaration: {name}"
                )
            replicas = {shard: [] for shard in range(size)}
            for rank, shard in sorted(declared.items()):
                replicas[shard].append(rank)
            if (
                any(not ranks for ranks in replicas.values())
                or len({len(ranks) for ranks in replicas.values()}) != 1
            ):
                raise WeightManifestError(f"uneven partition replicas: {name}")
            projected = {}
            for shard, ranks in replicas.items():
                for replica, rank in enumerate(ranks):
                    projected[rank] = (replica, shard, 0)
            current = tuple(projected[rank] for rank in range(self.layout.tp_size))
            if mapping is not None and mapping != current:
                raise WeightManifestError(
                    f"partition mapping differs between PP stages: {name}"
                )
            mapping = current
        if size == 1:
            # Fully replicated tensors can share the ordinary global-TP
            # placement. Only partial GQA replication needs a TP x DP projection.
            return _Distribution(
                self.layout.tp_size,
                1,
                1,
                tuple((0, rank, 0) for rank in range(self.layout.tp_size)),
            )
        return _Distribution(size, 1, self.layout.tp_size // size, mapping)


def _participant_id(rank):
    return f"dp{rank.dp}:pp{rank.pp}:ep{rank.ep}:tp{rank.tp}"


def _compile_group(index, tensor_ids, distribution, placement_set_id):
    """Construct final Mooncake descriptors, placement fragments and bindings once."""
    from mooncake.reshard.weight import (
        ParallelRank,
        ParallelTopology,
        PlacementFragment,
        RuntimeBindingFragment,
        SplitAxis,
        TensorDescriptor,
        TopologyParticipant,
        WeightPlacementManifest,
        WeightRuntimeBindingManifest,
    )

    from .weight_runtime_manifest import _contiguous_stride

    ranks = []
    for manifest in index.manifests:
        dp, tp, ep = distribution.ranks[index.global_tp(manifest.rank)]
        ranks.append(ParallelRank(dp=dp, tp=tp, ep=ep, pp=manifest.rank.pp))
    topology = ParallelTopology(
        dp_size=distribution.dp_size,
        tp_size=distribution.tp_size,
        ep_size=distribution.ep_size,
        pp_size=index.layout.pp_size,
        participants=tuple(
            TopologyParticipant(_participant_id(rank), rank) for rank in ranks
        ),
    )
    descriptors, fragments = [], []
    runtime = [[] for _ in index.manifests]
    for name in tensor_ids:
        tensor = index.tensors[name]
        axes = distribution.axes(tensor, index.layout.pp_size)
        descriptors.append(
            TensorDescriptor(
                tensor_id=name,
                global_shape=tensor.global_shape,
                dtype=tensor.dtype,
                itemsize=tensor.itemsize,
                shard_dims=tuple(
                    sorted(axis.dim for axis in axes if isinstance(axis, SplitAxis))
                ),
                parallel_axes=axes,
                layout_fingerprint=tensor.layout_fingerprint,
                expert_id=tensor.expert_id,
            )
        )
        for owner, view in index.fragments[name]:
            manifest = index.manifests[owner]
            fragment = PlacementFragment(
                tensor_id=name,
                global_offset=view.global_offset,
                local_shape=view.local_shape,
                nbytes=view.nbytes,
                rank=ranks[owner],
            )
            fragments.append(fragment)
            runtime[owner].append(
                RuntimeBindingFragment(
                    placement_fragment_id=fragment.placement_fragment_id,
                    fragment_id=fragment.fragment_id,
                    address=view.address,
                    nbytes=view.nbytes,
                    worker_id=manifest.worker_id,
                    endpoint=manifest.endpoint,
                    device=manifest.device,
                    itemsize=tensor.itemsize,
                    local_shape=view.local_shape,
                    strides_bytes=tuple(
                        s * tensor.itemsize
                        for s in _contiguous_stride(view.local_shape)
                    ),
                    storage_address=view.storage_address,
                    storage_nbytes=view.storage_nbytes,
                    storage_offset_bytes=view.storage_offset_bytes,
                )
            )
    first = index.manifests[0]
    placement = WeightPlacementManifest.from_fragments(
        resource_id=first.model_id,
        revision=first.revision,
        weight_generation=first.generation,
        placement_set_id=placement_set_id,
        topology=topology,
        tensors=tuple(descriptors),
        fragments=tuple(fragments),
    )
    bindings = tuple(
        WeightRuntimeBindingManifest(
            resource_id=placement.resource_id,
            revision=placement.revision,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id=manifest.instance_id,
            participant_id=_participant_id(rank),
            generation=manifest.generation,
            lease_id=f"immutable:{manifest.instance_id}",
            fragments=tuple(owned),
        )
        for manifest, rank, owned in zip(index.manifests, ranks, runtime)
    )
    return placement, bindings


def build_mooncake_transfer_groups(
    source_runtime_manifests,
    target_runtime_manifests,
    *,
    source_parallel_layout,
    target_parallel_layout,
):
    """Co-group tensors only when both sides have identical process mappings.

    In particular all compatible GQA K/V tensors share a placement, while tensors
    with the same source mapping but different target mappings remain separate.
    """
    source = _ManifestIndex(source_runtime_manifests, source_parallel_layout)
    target = _ManifestIndex(target_runtime_manifests, target_parallel_layout)
    if source.tensors.keys() != target.tensors.keys():
        raise WeightManifestError("source and target logical tensor inventories differ")
    groups = {}
    for name in sorted(source.tensors):
        key = (source.distributions[name], target.distributions[name])
        groups.setdefault(key, []).append(name)
    result = {}
    for ordinal, ((source_distribution, target_distribution), names) in enumerate(
        groups.items()
    ):
        group_id = f"weights:{ordinal}"
        result[group_id] = (
            *_compile_group(
                source, names, source_distribution, f"weight-cache-source:{group_id}"
            ),
            *_compile_group(
                target, names, target_distribution, f"weight-cache-target:{group_id}"
            ),
        )
    return result
