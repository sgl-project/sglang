# SPDX-License-Identifier: Apache-2.0
"""SGLang storage snapshots; Mooncake contracts are built once at the boundary."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Sequence

import msgspec

# Mooncake's placement and runtime-binding contracts require a weight generation.
# Weight-cache daemon tensors are immutable after publication, so they always
# belong to this single generation.
IMMUTABLE_WEIGHT_GENERATION = 0


class WeightManifestError(RuntimeError):
    pass


class WeightPartition(msgspec.Struct, frozen=True, kw_only=True):
    """Explicit module partition, independent of fragment geometry.

    Dense ranks with the same partition rank are replicas. Expert partitions
    use the daemon's EP/MoE-TP coordinates; a tensor without an expert_id has a
    leading logical expert dimension, otherwise EP expresses ownership.
    """

    kind: Literal["dense", "expert"] = "dense"
    rank: int = 0
    size: int = 1
    dim: int | None = None

    def __post_init__(self):
        if (
            self.kind not in ("dense", "expert")
            or self.size < 1
            or not 0 <= self.rank < self.size
        ):
            raise WeightManifestError("invalid declared weight partition")
        if self.dim is not None and self.dim < 0:
            raise WeightManifestError("partition dimension must be nonnegative")


class LogicalWeightTensor(msgspec.Struct, frozen=True, kw_only=True):
    tensor_id: str
    global_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition: WeightPartition
    layout_fingerprint: str
    expert_id: int | None = None

    def definition(self):
        """The descriptor fields that must agree across declaring ranks."""
        return (
            self.global_shape,
            self.dtype,
            self.itemsize,
            self.layout_fingerprint,
            self.expert_id,
            self.partition.kind,
            self.partition.size,
            self.partition.dim,
        )


class WeightParallelRank(msgspec.Struct, frozen=True, kw_only=True):
    dp: int = 0
    tp: int = 0
    pp: int = 0
    ep: int = 0


class WeightParallelTopology(msgspec.Struct, frozen=True, kw_only=True):
    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1

    def __post_init__(self) -> None:
        ranks = (
            self.dp_rank,
            self.tp_rank,
            self.pp_rank,
            self.ep_rank,
        )
        sizes = (
            self.dp_size,
            self.tp_size,
            self.pp_size,
            self.ep_size,
        )
        if any(rank < 0 for rank in ranks) or any(size <= 0 for size in sizes):
            raise ValueError("parallel ranks and sizes must be positive")
        if any(rank >= size for rank, size in zip(ranks, sizes)):
            raise ValueError("parallel rank is outside its topology")

    def rank(self) -> WeightParallelRank:
        return WeightParallelRank(
            dp=self.dp_rank,
            tp=self.tp_rank,
            pp=self.pp_rank,
            ep=self.ep_rank,
        )


class RuntimeWeightFragment(msgspec.Struct, frozen=True, kw_only=True):
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    address: int
    nbytes: int
    storage_address: int
    storage_nbytes: int
    storage_offset_bytes: int


class WeightRuntimeManifest(msgspec.Struct, frozen=True, kw_only=True):
    model_id: str
    revision: str
    instance_id: str
    generation: int
    worker_id: str
    endpoint: str
    device: str
    rank: WeightParallelRank
    tensors: tuple[LogicalWeightTensor, ...]
    fragments: tuple[RuntimeWeightFragment, ...]


def model_identity_from_config(config: Any) -> str:
    """Return a path-independent identity for one model configuration."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): normalize(item)
                for key, item in value.items()
                if not (
                    isinstance(key, str)
                    and (key.startswith("_") or key == "transformers_version")
                )
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    if not hasattr(config, "to_dict"):
        raise WeightManifestError("model config cannot provide a stable identity")
    config_data = normalize(config.to_dict())
    payload = json.dumps(
        config_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    model_type = str(config_data.get("model_type") or type(config).__name__)
    return f"{model_type}:{hashlib.sha256(payload.encode()).hexdigest()}"


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(shape)
    value = 1
    for index in range(len(shape) - 1, -1, -1):
        result[index] = value
        value *= shape[index]
    return tuple(result)


def _storage_key(parameter: Any) -> tuple:
    return (
        int(parameter.untyped_storage().data_ptr()),
        int(parameter.storage_offset()),
        tuple(int(value) for value in parameter.shape),
        tuple(int(value) for value in parameter.stride()),
        str(parameter.dtype).removeprefix("torch."),
    )


class ImmutableWeightRuntimeManifestBuilder:
    """Bind the already validated layout to current SGLang-owned allocations.

    Logical declarations occur once per tensor, physical locations once per
    fragment, and worker metadata once per manifest. No temporary Mooncake
    descriptors or placement fragments are created here.
    """

    def __init__(
        self,
        *,
        model: Any,
        layout_plan: Any,
        topology: WeightParallelTopology,
        allowed_devices: Sequence[str] = ("cuda",),
    ) -> None:
        self.model, self.plan, self.topology = model, layout_plan, topology
        self.allowed_devices = frozenset(allowed_devices)

    def build(
        self,
        *,
        model_id: str,
        revision: str,
        instance_id: str,
        worker_id: str,
        endpoint: str,
    ) -> WeightRuntimeManifest:
        if not all((model_id, revision, instance_id, worker_id, endpoint)):
            raise WeightManifestError("manifest identifiers must not be empty")
        self.plan.validate_bindings(self.model)
        devices = {tensor.device.type for tensor in self.model.parameters()}
        devices.update(view.parameter.device.type for view in self.plan.views)
        if len(devices) != 1 or not devices <= self.allowed_devices:
            raise WeightManifestError(f"unsupported runtime devices: {devices}")
        fragments = []
        for view in self.plan.views:
            tensor = view.parameter
            storage = tensor.untyped_storage()
            address = tensor.data_ptr() + view.byte_offset
            if address <= 0 or view.nbytes <= 0:
                raise WeightManifestError(
                    f"tensor has no transferable storage: {view.tensor_id}"
                )
            fragments.append(
                RuntimeWeightFragment(
                    tensor_id=view.tensor_id,
                    global_offset=view.global_offset or (0,),
                    local_shape=view.local_shape or (1,),
                    address=address,
                    nbytes=view.nbytes,
                    storage_address=storage.data_ptr(),
                    storage_nbytes=storage.nbytes(),
                    storage_offset_bytes=address - storage.data_ptr(),
                )
            )
        return WeightRuntimeManifest(
            model_id=model_id,
            revision=revision,
            instance_id=instance_id,
            generation=IMMUTABLE_WEIGHT_GENERATION,
            worker_id=worker_id,
            endpoint=endpoint,
            device=next(iter(devices)),
            rank=self.topology.rank(),
            tensors=self.plan.tensors,
            fragments=tuple(
                sorted(
                    fragments, key=lambda f: (f.tensor_id, f.global_offset, f.address)
                )
            ),
        )
