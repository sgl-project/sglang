from __future__ import annotations

import hashlib
import json
from math import prod
from typing import Any, Sequence

import msgspec

# Mooncake's placement and runtime-binding contracts require a weight generation.
# Weight-cache daemon tensors are immutable after publication, so they always
# belong to this single generation.
IMMUTABLE_WEIGHT_GENERATION = 0


class WeightManifestError(RuntimeError):
    pass


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


class RuntimeWeightTensor(msgspec.Struct, frozen=True, kw_only=True):
    fragment_id: str
    tensor_id: str
    runtime_name: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    shard_dims: tuple[int, ...]
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    address: int
    nbytes: int
    byte_offset: int
    stride: tuple[int, ...]
    storage_offset: int
    device: str
    worker_id: str
    endpoint: str
    rank: WeightParallelRank


class WeightLoadTensorMetadata(msgspec.Struct, frozen=True, kw_only=True):
    tensor_id: str
    shape: tuple[int, ...]
    dtype: str
    itemsize: int
    scalar_value: bool | int | float | None = None


class WeightRuntimeManifest(msgspec.Struct, frozen=True, kw_only=True):
    model_id: str
    revision: str
    instance_id: str
    generation: int
    load_tensors: tuple[WeightLoadTensorMetadata, ...]
    tensors: tuple[RuntimeWeightTensor, ...]


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
    """Bind the recorded logical regions to the model's final live allocations.

    Each view is indexed once. The complete cross-rank geometry and placement
    validation belong to Mooncake; this boundary checks SGLang storage ownership
    and that every final parameter is accounted for.
    """

    def __init__(
        self,
        *,
        model: Any,
        load_plan: Any,
        topology: WeightParallelTopology,
        allowed_devices: Sequence[str] = ("cuda",),
    ) -> None:
        self.model, self.plan, self.topology = model, load_plan, topology
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
        from mooncake.reshard.weight import (
            ParallelRank,
            PlacementFragment,
            TensorDescriptor,
        )
        from mooncake.reshard.weight.types import validate_fragment_geometry

        if not all((model_id, revision, instance_id, worker_id, endpoint)):
            raise WeightManifestError("manifest identifiers must not be empty")
        # Shape/address checks cannot detect this in-place post-load permutation.
        # Reject both source and target until its logical mapping is supported.
        for name, module in self.model.named_modules():
            if getattr(
                getattr(module, "quant_method", None), "w13_swiglu_interleaved", False
            ):
                raise WeightManifestError(
                    f"interleaved SwiGLU weight layout is unsupported for reshard: {name}; "
                    "disable SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED on source and target"
                )
        regions = (*self.plan.views, *self.plan.initializations)
        initialized = {id(region) for region in self.plan.initializations}
        views_by_id, views_by_name = {}, {}
        for view in regions:
            views_by_id.setdefault(id(view.parameter), []).append(view)
            for name in view.parameter_names:
                views_by_name.setdefault(name, []).append(view)

        groups = {}
        for is_param, entries in (
            (True, self.model.named_parameters(remove_duplicate=False)),
            (False, self.model.named_buffers(remove_duplicate=False)),
        ):
            for name, tensor in entries:
                views = {id(v): v for v in views_by_id.get(id(tensor), ())}
                if not is_param:
                    views.update((id(v), v) for v in views_by_name.get(name, ()))
                    if not views:
                        continue  # Model-generated buffers are rebuilt locally.
                key = _storage_key(tensor)
                if any(_storage_key(v.parameter) != key for v in views.values()):
                    raise WeightManifestError(
                        f"recorded tensor storage changed: {name}"
                    )
                group = groups.setdefault(key, (tensor, [], {}))
                group[1].append(name)
                group[2].update(views)

        bound = {key for _, _, views in groups.values() for key in views}
        if any(id(view) not in bound for view in regions):
            raise WeightManifestError(
                "recorded tensor is no longer bound to model storage"
            )

        tensors, logical_keys = [], set()
        rank = self.topology.rank()
        mooncake_rank = ParallelRank(dp=rank.dp, tp=rank.tp, pp=rank.pp, ep=rank.ep)
        for tensor, names, indexed_views in groups.values():
            name = min(names)
            if tensor.is_sparse or not tensor.is_contiguous():
                raise WeightManifestError(f"non-contiguous runtime tensor: {name}")
            if tensor.device.type not in self.allowed_devices:
                raise WeightManifestError(
                    f"tensor device is unsupported: {name}: {tensor.device}"
                )
            itemsize, address = tensor.element_size(), tensor.data_ptr()
            total = tensor.numel() * itemsize
            if address <= 0 or total <= 0:
                raise WeightManifestError(f"tensor has no transferable storage: {name}")
            views = sorted(indexed_views.values(), key=lambda v: v.byte_offset)
            cursor = 0
            for view in views:
                nbytes = view.nbytes
                if view.byte_offset != cursor:
                    raise WeightManifestError(
                        f"recorded writes overlap or leave a gap in {name}"
                    )
                cursor += nbytes
            if cursor != total:
                raise WeightManifestError(
                    f"native weight-load recorder did not cover complete parameter: {name}"
                )

            dtype = str(tensor.dtype).removeprefix("torch.")
            for view in views:
                if id(view) in initialized:
                    continue
                shape = view.global_shape or (1,)
                offset = view.global_offset or (0,)
                local = view.local_shape or (1,)
                nbytes = prod(local) * itemsize
                key = (view.tensor_id, offset, local)
                if key in logical_keys:
                    raise WeightManifestError(
                        f"duplicate logical view: {view.tensor_id}"
                    )
                logical_keys.add(key)
                # Construct canonical local geometry now; process-axis semantics
                # are inferred only after all ranks have reported their pieces.
                descriptor = TensorDescriptor(
                    tensor_id=view.tensor_id,
                    global_shape=shape,
                    dtype=dtype,
                    itemsize=itemsize,
                    shard_dims=(),
                    parallel_axes=(),
                    layout_fingerprint=view.layout_fingerprint,
                    expert_id=view.expert_id,
                )
                fragment = PlacementFragment(
                    tensor_id=view.tensor_id,
                    global_offset=offset,
                    local_shape=local,
                    nbytes=nbytes,
                    rank=mooncake_rank,
                )
                validate_fragment_geometry(
                    descriptor,
                    fragment_id=fragment.fragment_id,
                    global_offset=offset,
                    local_shape=local,
                    nbytes=nbytes,
                )
                tensors.append(
                    RuntimeWeightTensor(
                        fragment_id=fragment.fragment_id,
                        tensor_id=view.tensor_id,
                        runtime_name=name,
                        global_shape=shape,
                        global_offset=offset,
                        local_shape=local,
                        dtype=dtype,
                        itemsize=itemsize,
                        shard_dims=view.shard_dims,
                        layer_id=None,
                        expert_id=view.expert_id,
                        layout_fingerprint=view.layout_fingerprint,
                        address=address + view.byte_offset,
                        nbytes=nbytes,
                        byte_offset=view.byte_offset,
                        stride=_contiguous_stride(local),
                        storage_offset=tensor.storage_offset()
                        + view.byte_offset // itemsize,
                        device=tensor.device.type,
                        worker_id=worker_id,
                        endpoint=endpoint,
                        rank=rank,
                    )
                )
        return WeightRuntimeManifest(
            model_id=model_id,
            revision=revision,
            instance_id=instance_id,
            generation=IMMUTABLE_WEIGHT_GENERATION,
            load_tensors=tuple(
                WeightLoadTensorMetadata(
                    tensor_id=m.tensor_id,
                    shape=m.shape,
                    dtype=m.dtype,
                    itemsize=m.itemsize,
                    scalar_value=m.scalar_value,
                )
                for m in self.plan.logical_weights
            ),
            tensors=tuple(
                sorted(
                    tensors, key=lambda t: (t.tensor_id, t.global_offset, t.fragment_id)
                )
            ),
        )
