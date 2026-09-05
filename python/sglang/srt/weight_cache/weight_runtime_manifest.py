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


class LogicalTensorView(msgspec.Struct, frozen=True, kw_only=True):
    tensor_id: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    partition_dim: int | None
    byte_offset: int
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    shard_dims: tuple[int, ...] | None = None


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


class WeightRuntimeManifest(msgspec.Struct, frozen=True, kw_only=True):
    model_id: str
    revision: str
    instance_id: str
    generation: int
    load_tensors: tuple[WeightLoadTensorMetadata, ...]
    tensors: tuple[RuntimeWeightTensor, ...]


class _PhysicalParameter(msgspec.Struct, frozen=True, kw_only=True):
    names: tuple[str, ...]
    parameters: tuple[Any, ...]
    is_parameter: bool
    address: int
    nbytes: int
    storage_offset: int
    dtype: str
    itemsize: int
    device: str


def _dtype_name(dtype: Any) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")


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
        _dtype_name(parameter.dtype),
    )


def _inspect_parameter(
    *,
    names: tuple[str, ...],
    parameter: Any,
    parameters: tuple[Any, ...],
    is_parameter: bool,
    allowed_devices: frozenset[str],
) -> _PhysicalParameter:
    runtime_name = names[0]
    kind = "parameter" if is_parameter else "buffer"
    if getattr(parameter, "is_sparse", False):
        raise WeightManifestError(f"sparse {kind} is unsupported: {runtime_name}")
    layout = getattr(parameter, "layout", None)
    if layout is not None and str(layout) not in ("strided", "torch.strided"):
        raise WeightManifestError(
            f"non-strided {kind} is unsupported: {runtime_name}"
        )
    if not parameter.is_contiguous():
        raise WeightManifestError(
            f"non-contiguous {kind} is unsupported: {runtime_name}"
        )

    device = str(parameter.device.type)
    if device not in allowed_devices:
        raise WeightManifestError(
            f"{kind} device is unsupported: {runtime_name}: {device}"
        )
    itemsize = int(parameter.element_size())
    nbytes = int(parameter.numel()) * itemsize
    address = int(parameter.data_ptr())
    if address <= 0 or itemsize <= 0 or nbytes <= 0:
        raise WeightManifestError(
            f"{kind} has no transferable storage: {runtime_name}"
        )
    return _PhysicalParameter(
        names=names,
        parameters=parameters,
        is_parameter=is_parameter,
        address=address,
        nbytes=nbytes,
        storage_offset=int(parameter.storage_offset()),
        dtype=_dtype_name(parameter.dtype),
        itemsize=itemsize,
        device=device,
    )


def _view_shard_dims(view: LogicalTensorView) -> tuple[int, ...]:
    if view.shard_dims is None:
        return () if view.partition_dim is None else (view.partition_dim,)
    shard_dims = view.shard_dims
    if (
        not isinstance(shard_dims, tuple)
        or any(type(dim) is not int for dim in shard_dims)
        or tuple(sorted(shard_dims)) != shard_dims
        or len(set(shard_dims)) != len(shard_dims)
    ):
        raise WeightManifestError(f"invalid shard axes for {view.tensor_id}")
    if view.partition_dim is not None and shard_dims != (view.partition_dim,):
        raise WeightManifestError(
            f"partition axis conflicts with shard axes for {view.tensor_id}"
        )
    return shard_dims


def _validate_view(view: LogicalTensorView, physical: _PhysicalParameter) -> int:
    ndim = len(view.global_shape)
    if (
        not view.tensor_id
        or not view.layout_fingerprint
        or len(view.global_offset) != ndim
        or len(view.local_shape) != ndim
    ):
        raise WeightManifestError(
            f"invalid logical view for {physical.names[0]}: {view.tensor_id}"
        )
    if view.partition_dim is not None and (
        type(view.partition_dim) is not int or not 0 <= view.partition_dim < ndim
    ):
        raise WeightManifestError(f"invalid partition axis for {view.tensor_id}")
    shard_dims = _view_shard_dims(view)
    if any(dim < 0 or dim >= ndim for dim in shard_dims):
        raise WeightManifestError(f"invalid shard axes for {view.tensor_id}")
    for offset, extent, total in zip(
        view.global_offset, view.local_shape, view.global_shape
    ):
        if offset < 0 or extent <= 0 or offset + extent > total:
            raise WeightManifestError(f"view is out of bounds: {view.tensor_id}")
    for dim, (offset, extent, total) in enumerate(
        zip(view.global_offset, view.local_shape, view.global_shape)
    ):
        if dim not in shard_dims and (offset != 0 or extent != total):
            raise WeightManifestError(
                f"view uses a non-shard axis: {view.tensor_id}: {dim}"
            )
    nbytes = prod(view.local_shape) * physical.itemsize
    if (
        view.byte_offset < 0
        or view.byte_offset % physical.itemsize != 0
        or view.byte_offset + nbytes > physical.nbytes
    ):
        raise WeightManifestError(
            f"view exceeds registered tensor storage: {view.tensor_id}"
        )
    return nbytes


def _runtime_fragment_id(
    *,
    instance_id: str,
    worker_id: str,
    tensor_id: str,
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    byte_offset: int,
) -> str:
    identity = (
        instance_id,
        worker_id,
        tensor_id,
        global_offset,
        local_shape,
        byte_offset,
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:24]


class ImmutableWeightRuntimeManifestBuilder:
    """Build one address-bearing manifest for immutable daemon weights."""

    def __init__(
        self,
        *,
        model: Any,
        load_plan: Any,
        topology: WeightParallelTopology,
        allowed_devices: Sequence[str] = ("cuda",),
    ) -> None:
        self._model = model
        self._load_plan = load_plan
        self._topology = topology
        self._allowed_devices = frozenset(allowed_devices)

    def build(
        self,
        *,
        model_id: str,
        revision: str,
        instance_id: str,
        worker_id: str,
        endpoint: str,
    ) -> WeightRuntimeManifest:
        if not model_id or not revision:
            raise WeightManifestError("manifest identifiers must not be empty")
        if not all((instance_id, worker_id, endpoint)):
            raise WeightManifestError("runtime identifiers must not be empty")

        tensors = self._build_runtime_tensors(
            physical=self._collect_physical_parameters(),
            instance_id=instance_id,
            worker_id=worker_id,
            endpoint=endpoint,
        )
        return WeightRuntimeManifest(
            model_id=model_id,
            revision=revision,
            instance_id=instance_id,
            generation=IMMUTABLE_WEIGHT_GENERATION,
            load_tensors=tuple(
                WeightLoadTensorMetadata(
                    tensor_id=item.tensor_id,
                    shape=item.shape,
                    dtype=item.dtype,
                    itemsize=item.itemsize,
                )
                for item in self._load_plan.logical_weights
            ),
            tensors=tensors,
        )

    def _collect_physical_parameters(self) -> tuple[_PhysicalParameter, ...]:
        grouped: dict[tuple, tuple[Any, list[Any], list[str], list[str]]] = {}
        recorded_views_by_id: dict[int, list[Any]] = {}
        recorded_views_by_name: dict[str, list[Any]] = {}
        for view in self._load_plan.views:
            recorded_views_by_id.setdefault(id(view.parameter), []).append(view)
            for name in view.parameter_names:
                recorded_views_by_name.setdefault(name, []).append(view)
        for name, parameter in self._model.named_parameters(remove_duplicate=False):
            key = _storage_key(parameter)
            if key not in grouped:
                grouped[key] = (parameter, [], [], [])
            grouped[key][1].append(parameter)
            grouped[key][2].append(name)
        for name, buffer in self._model.named_buffers(remove_duplicate=False):
            candidate_views = {
                id(view): view
                for view in (
                    *recorded_views_by_id.get(id(buffer), ()),
                    *recorded_views_by_name.get(name, ()),
                )
            }
            if not candidate_views:
                continue
            key = _storage_key(buffer)
            for view in candidate_views.values():
                if _storage_key(view.parameter) != key:
                    raise WeightManifestError(
                        "recorded tensor storage changed before manifest build: "
                        f"{name}"
                    )
            if key not in grouped:
                grouped[key] = (buffer, [], [], [])
            grouped[key][1].append(buffer)
            grouped[key][1].extend(
                view.parameter for view in candidate_views.values()
            )
            grouped[key][3].append(name)

        physical = []
        for (
            parameter,
            parameters,
            parameter_names,
            buffer_names,
        ) in grouped.values():
            unique_parameters = tuple(
                {id(item): item for item in parameters}.values()
            )
            is_parameter = bool(parameter_names)
            names = tuple(
                dict.fromkeys(
                    (*sorted(set(parameter_names)), *sorted(set(buffer_names)))
                )
            )
            physical.append(
                _inspect_parameter(
                    names=names,
                    parameter=parameter,
                    parameters=unique_parameters,
                    is_parameter=is_parameter,
                    allowed_devices=self._allowed_devices,
                )
            )
        physical = tuple(sorted(physical, key=lambda item: item.names))
        bound_view_ids = {
            id(view)
            for item in physical
            for view in self._load_plan.views_for_parameters(item.parameters)
        }
        missing_views = [
            view
            for view in self._load_plan.views
            if id(view) not in bound_view_ids
        ]
        if missing_views:
            raise WeightManifestError(
                "recorded tensor is no longer bound to model storage: "
                f"{missing_views[0].tensor_id}"
            )
        return physical

    def _build_runtime_tensors(
        self,
        *,
        physical: tuple[_PhysicalParameter, ...],
        instance_id: str,
        worker_id: str,
        endpoint: str,
    ) -> tuple[RuntimeWeightTensor, ...]:
        rank = self._topology.rank()
        tensors = []
        logical_keys = set()
        for item in physical:
            recorded_views = self._load_plan.views_for_parameters(item.parameters)
            views = tuple(
                LogicalTensorView(
                    tensor_id=view.tensor_id,
                    global_shape=view.global_shape,
                    global_offset=view.global_offset,
                    local_shape=view.local_shape,
                    partition_dim=(
                        view.shard_dims[0] if len(view.shard_dims) == 1 else None
                    ),
                    byte_offset=view.byte_offset,
                    layer_id=None,
                    expert_id=view.expert_id,
                    layout_fingerprint=view.layout_fingerprint,
                    shard_dims=view.shard_dims,
                )
                for view in recorded_views
            )
            if not views:
                raise WeightManifestError(
                    "native weight-load recorder did not cover parameter: "
                    f"{item.names[0]}"
                )
            kind = "parameter" if item.is_parameter else "buffer"
            intervals = sorted(
                (
                    view.byte_offset,
                    view.byte_offset + prod(view.local_shape) * item.itemsize,
                    view.tensor_id,
                )
                for view in views
            )
            cursor = 0
            for begin, end, tensor_id in intervals:
                if begin != cursor:
                    relation = "overlaps" if begin < cursor else "leaves a gap in"
                    raise WeightManifestError(
                        f"recorded tensor {tensor_id} {relation} {kind} "
                        f"{item.names[0]} at byte {begin}; expected {cursor}"
                    )
                cursor = end
            if cursor != item.nbytes:
                raise WeightManifestError(
                    f"native weight-load recorder did not cover complete {kind}: "
                    f"{item.names[0]}: covered={cursor}, nbytes={item.nbytes}"
                )
            for view in views:
                nbytes = _validate_view(view, item)
                logical_key = (
                    view.tensor_id,
                    view.global_offset,
                    view.local_shape,
                )
                if logical_key in logical_keys:
                    raise WeightManifestError(
                        f"duplicate logical view: {view.tensor_id}"
                    )
                logical_keys.add(logical_key)
                tensors.append(
                    RuntimeWeightTensor(
                        fragment_id=_runtime_fragment_id(
                            instance_id=instance_id,
                            worker_id=worker_id,
                            tensor_id=view.tensor_id,
                            global_offset=view.global_offset,
                            local_shape=view.local_shape,
                            byte_offset=view.byte_offset,
                        ),
                        tensor_id=view.tensor_id,
                        runtime_name=item.names[0],
                        global_shape=view.global_shape,
                        global_offset=view.global_offset,
                        local_shape=view.local_shape,
                        dtype=item.dtype,
                        itemsize=item.itemsize,
                        shard_dims=_view_shard_dims(view),
                        layer_id=view.layer_id,
                        expert_id=view.expert_id,
                        layout_fingerprint=view.layout_fingerprint,
                        address=item.address + view.byte_offset,
                        nbytes=nbytes,
                        byte_offset=view.byte_offset,
                        stride=_contiguous_stride(view.local_shape),
                        storage_offset=(
                            item.storage_offset + view.byte_offset // item.itemsize
                        ),
                        device=item.device,
                        worker_id=worker_id,
                        endpoint=endpoint,
                        rank=rank,
                    )
                )
        return tuple(
            sorted(
                tensors,
                key=lambda item: (
                    item.tensor_id,
                    item.global_offset,
                    item.fragment_id,
                ),
            )
        )


def create_weight_runtime_manifest_builder(
    *,
    model: Any,
    load_plan: Any,
    topology: WeightParallelTopology,
    allowed_devices: Sequence[str] = ("cuda",),
):
    return ImmutableWeightRuntimeManifestBuilder(
        model=model,
        load_plan=load_plan,
        topology=topology,
        allowed_devices=allowed_devices,
    )
