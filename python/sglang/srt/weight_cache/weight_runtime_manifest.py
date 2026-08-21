from __future__ import annotations

import hashlib
import json
from math import prod
from typing import Any, Protocol, Sequence

import msgspec

# Mooncake's runtime-manifest schema requires a generation on the manifest and
# each fragment. Weight-cache daemon tensors are immutable after publication,
# so they always belong to this single generation; there is no application-level
# lease lifecycle to acquire, renew, expire, or release.
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
    moe_tp_rank: int = 0
    moe_tp_size: int = 1
    attention_tp_rank: int = 0
    attention_tp_size: int = 1

    def __post_init__(self) -> None:
        ranks = (
            self.dp_rank,
            self.tp_rank,
            self.pp_rank,
            self.ep_rank,
            self.moe_tp_rank,
            self.attention_tp_rank,
        )
        sizes = (
            self.dp_size,
            self.tp_size,
            self.pp_size,
            self.ep_size,
            self.moe_tp_size,
            self.attention_tp_size,
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
    aliases: tuple[str, ...]
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition_dim: int | None
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
    is_contiguous: bool
    worker_id: str
    endpoint: str
    rank: WeightParallelRank
    lease_generation: int


class WeightRuntimeManifest(msgspec.Struct, frozen=True, kw_only=True):
    model_id: str
    revision: str
    instance_id: str
    generation: int
    tensors: tuple[RuntimeWeightTensor, ...]
    format_version: int = 2


class WeightSemanticsAdapter(Protocol):
    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]: ...


class _PhysicalParameter(msgspec.Struct, frozen=True, kw_only=True):
    names: tuple[str, ...]
    parameter: Any
    address: int
    nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
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
    allowed_devices: frozenset[str],
) -> _PhysicalParameter:
    runtime_name = names[0]
    if getattr(parameter, "is_sparse", False):
        raise WeightManifestError(f"sparse parameter is unsupported: {runtime_name}")
    layout = getattr(parameter, "layout", None)
    if layout is not None and str(layout) not in ("strided", "torch.strided"):
        raise WeightManifestError(
            f"non-strided parameter is unsupported: {runtime_name}"
        )
    if not parameter.is_contiguous():
        raise WeightManifestError(
            f"non-contiguous parameter is unsupported: {runtime_name}"
        )

    device = str(parameter.device.type)
    if device not in allowed_devices:
        raise WeightManifestError(
            f"parameter device is unsupported: {runtime_name}: {device}"
        )
    shape = tuple(int(value) for value in parameter.shape)
    itemsize = int(parameter.element_size())
    nbytes = int(parameter.numel()) * itemsize
    address = int(parameter.data_ptr())
    if address <= 0 or itemsize <= 0 or nbytes <= 0:
        raise WeightManifestError(
            f"parameter has no transferable storage: {runtime_name}"
        )
    return _PhysicalParameter(
        names=names,
        parameter=parameter,
        address=address,
        nbytes=nbytes,
        shape=shape,
        stride=tuple(int(value) for value in parameter.stride()),
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
        raise WeightManifestError(f"view exceeds parameter storage: {view.tensor_id}")
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
        adapter: WeightSemanticsAdapter,
        topology: WeightParallelTopology,
        allowed_devices: Sequence[str] = ("cuda",),
    ) -> None:
        self._model = model
        self._adapter = adapter
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
            tensors=tensors,
        )

    def _collect_physical_parameters(self) -> tuple[_PhysicalParameter, ...]:
        grouped: dict[tuple, tuple[Any, list[str]]] = {}
        for name, parameter in self._model.named_parameters(remove_duplicate=False):
            key = _storage_key(parameter)
            if key not in grouped:
                grouped[key] = (parameter, [])
            grouped[key][1].append(name)

        physical = [
            _inspect_parameter(
                names=tuple(sorted(names)),
                parameter=parameter,
                allowed_devices=self._allowed_devices,
            )
            for parameter, names in grouped.values()
        ]
        return tuple(sorted(physical, key=lambda item: item.names))

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
            views = self._adapter.describe_parameter(
                names=item.names,
                parameter=item.parameter,
                topology=self._topology,
            )
            if not views:
                raise WeightManifestError(
                    f"adapter returned no views for {item.names[0]}"
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
                        aliases=item.names,
                        global_shape=view.global_shape,
                        global_offset=view.global_offset,
                        local_shape=view.local_shape,
                        dtype=item.dtype,
                        itemsize=item.itemsize,
                        partition_dim=view.partition_dim,
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
                        is_contiguous=True,
                        worker_id=worker_id,
                        endpoint=endpoint,
                        rank=rank,
                        lease_generation=IMMUTABLE_WEIGHT_GENERATION,
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
    config: Any,
    topology: WeightParallelTopology,
    is_multimodal: bool = False,
    moe_runner_backend: str | None = None,
):
    model_type = getattr(config, "model_type", None)
    text_model_types = (
        "qwen3",
        "qwen3_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
        "qwen3_next",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_v4",
    )
    deepseek_model_types = (
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_v4",
    )
    multimodal_model_types = ("qwen3_5", "qwen3_5_moe")
    if is_multimodal and model_type not in multimodal_model_types:
        raise WeightManifestError(
            f"unsupported multimodal model type for weight manifests: {model_type}"
        )
    if not is_multimodal and model_type not in text_model_types:
        raise WeightManifestError(
            f"unsupported model type for weight manifests: {model_type}"
        )
    if model_type == "qwen3_next" and moe_runner_backend != "triton":
        raise WeightManifestError(
            "Qwen3-Next weight manifests require the canonical triton MoE "
            f"runner backend; got {moe_runner_backend!r}"
        )
    if model_type in deepseek_model_types:
        if int(getattr(config, "n_routed_experts", 0) or 0) > 0 and (
            moe_runner_backend != "triton"
        ):
            raise WeightManifestError(
                "DeepSeek MoE weight manifests require the canonical triton MoE "
                f"runner backend; got {moe_runner_backend!r}"
            )

    from .weight_semantics.qwen3_5 import (
        Qwen35MultimodalWeightSemanticsAdapter,
        Qwen35WeightSemanticsAdapter,
    )
    from .weight_semantics.qwen3 import (
        Qwen3WeightSemanticsAdapter,
    )
    from .weight_semantics.qwen3_next import (
        Qwen3NextWeightSemanticsAdapter,
    )

    up_first_w13_parameters = set()
    modules = getattr(model, "modules", None)
    if modules is not None:
        for module in modules():
            parameter = getattr(module, "w13_weight", None)
            if parameter is None:
                continue
            quant_method = getattr(module, "quant_method", None)
            if bool(getattr(module, "use_flashinfer_trtllm_moe", False)) or bool(
                getattr(quant_method, "load_up_proj_weight_first", False)
            ):
                up_first_w13_parameters.add(id(parameter))

    if is_multimodal:
        text_config = getattr(config, "text_config", None)
        vision_config = getattr(config, "vision_config", None)
        if text_config is None or vision_config is None:
            raise WeightManifestError(
                "Qwen3.5 multimodal config is missing text_config or vision_config"
            )
        adapter = Qwen35MultimodalWeightSemanticsAdapter(
            text_config=text_config,
            vision_config=vision_config,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )
    elif model_type == "qwen3_next":
        adapter = Qwen3NextWeightSemanticsAdapter(
            config=config,
            up_first_w13_parameter_ids=up_first_w13_parameters,
            num_fused_shared_experts=int(getattr(model, "num_fused_shared_experts", 0)),
        )
    elif model_type in deepseek_model_types:
        from .weight_semantics.deepseek_v2 import (
            DeepseekV2WeightSemanticsAdapter,
        )
        from .weight_semantics.deepseek_v4 import (
            DeepseekV4WeightSemanticsAdapter,
        )

        adapter_class = (
            DeepseekV4WeightSemanticsAdapter
            if model_type == "deepseek_v4"
            else DeepseekV2WeightSemanticsAdapter
        )
        adapter = adapter_class(
            config=config,
            up_first_w13_parameter_ids=up_first_w13_parameters,
            num_fused_shared_experts=int(getattr(model, "num_fused_shared_experts", 0)),
        )
    elif model_type in ("qwen3", "qwen3_moe"):
        adapter = Qwen3WeightSemanticsAdapter(
            config=config,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )
    else:
        adapter = Qwen35WeightSemanticsAdapter(
            config=config,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )

    return ImmutableWeightRuntimeManifestBuilder(
        model=model,
        adapter=adapter,
        topology=topology,
    )
