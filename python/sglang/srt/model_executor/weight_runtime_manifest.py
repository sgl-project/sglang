from __future__ import annotations

import hashlib
import threading
import time
from math import prod
from typing import Any, Callable, Protocol, Sequence
from uuid import uuid4

import msgspec

DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 300
MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 30
MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 3600


def validate_remote_instance_weight_transfer_lease_timeout(
    lease_timeout_sec: int,
) -> int:
    if isinstance(lease_timeout_sec, bool) or not isinstance(lease_timeout_sec, int):
        raise ValueError("lease_timeout_sec must be an integer")
    if not (
        MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        <= lease_timeout_sec
        <= MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
    ):
        raise ValueError(
            "lease_timeout_sec must be between "
            f"{MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC} and "
            f"{MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC}"
        )
    return lease_timeout_sec


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
    lease_id: str
    tensors: tuple[RuntimeWeightTensor, ...]
    format_version: int = 1


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
    if view.partition_dim is not None and not 0 <= view.partition_dim < ndim:
        raise WeightManifestError(f"invalid partition axis for {view.tensor_id}")
    for offset, extent, total in zip(
        view.global_offset, view.local_shape, view.global_shape
    ):
        if offset < 0 or extent <= 0 or offset + extent > total:
            raise WeightManifestError(f"view is out of bounds: {view.tensor_id}")
    nbytes = prod(view.local_shape) * physical.itemsize
    if (
        view.byte_offset < 0
        or view.byte_offset % physical.itemsize != 0
        or view.byte_offset + nbytes > physical.nbytes
    ):
        raise WeightManifestError(f"view exceeds parameter storage: {view.tensor_id}")
    return nbytes


def _fragment_id(
    *,
    instance_id: str,
    worker_id: str,
    generation: int,
    view: LogicalTensorView,
) -> str:
    value = (
        f"{instance_id}|{worker_id}|{generation}|{view.tensor_id}|"
        f"{view.global_offset}|{view.local_shape}|{view.byte_offset}"
    ).encode()
    return hashlib.sha256(value).hexdigest()[:24]


class WeightSnapshotCoordinator:
    """Serializes in-place updates with address-bearing runtime snapshots."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._generation = 1
        self._healthy = True
        self._needs_revision_commit = False
        self._last_update_success = True
        self._update_token: str | None = None
        self._leases: dict[str, tuple[int, float | None]] = {}

    def _prune_expired_leases_locked(self) -> None:
        now = self._clock()
        expired = [
            lease_id
            for lease_id, (_, deadline) in self._leases.items()
            if deadline is not None and deadline <= now
        ]
        for lease_id in expired:
            del self._leases[lease_id]

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def begin_update(self) -> str:
        with self._lock:
            self._prune_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is already in progress")
            if self._leases:
                raise WeightManifestError("a weight snapshot lease is active")
            token = uuid4().hex
            self._update_token = token
            return token

    def finish_update(self, token: str, *, success: bool) -> None:
        with self._lock:
            if not token or token != self._update_token:
                raise WeightManifestError("weight update token does not match")
            self._generation += 1
            self._healthy = False
            self._needs_revision_commit = True
            self._last_update_success = bool(success)
            self._update_token = None

    def commit_revision(self) -> int:
        with self._lock:
            self._prune_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is in progress")
            if self._leases:
                raise WeightManifestError("a weight snapshot lease is active")
            if not self._needs_revision_commit:
                return self._generation
            if not self._last_update_success:
                raise WeightManifestError(
                    "the last weight update failed; a full successful update is required"
                )
            self._healthy = True
            self._needs_revision_commit = False
            return self._generation

    def acquire_snapshot(
        self, *, lease_timeout_sec: int | None = None
    ) -> tuple[str, int]:
        if lease_timeout_sec is not None:
            validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        with self._lock:
            self._prune_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is in progress")
            if not self._healthy:
                if self._needs_revision_commit and self._last_update_success:
                    raise WeightManifestError(
                        "updated weights require an explicit revision commit"
                    )
                raise WeightManifestError(
                    "the last weight update failed; a full successful update is required"
                )
            lease_id = uuid4().hex
            deadline = (
                None if lease_timeout_sec is None else self._clock() + lease_timeout_sec
            )
            self._leases[lease_id] = (self._generation, deadline)
            return lease_id, self._generation

    def renew_snapshot(self, lease_id: str, *, lease_timeout_sec: int) -> None:
        validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        with self._lock:
            self._prune_expired_leases_locked()
            lease = self._leases.get(lease_id)
            if lease is None:
                raise WeightManifestError("weight snapshot lease does not exist")
            generation, _ = lease
            self._leases[lease_id] = (
                generation,
                self._clock() + lease_timeout_sec,
            )

    def has_snapshot(self, lease_id: str) -> bool:
        with self._lock:
            self._prune_expired_leases_locked()
            return lease_id in self._leases

    def release_snapshot(self, lease_id: str) -> None:
        with self._lock:
            self._prune_expired_leases_locked()
            if lease_id not in self._leases:
                raise WeightManifestError("weight snapshot lease does not exist")
            del self._leases[lease_id]

    def invalidate(self) -> None:
        token = self.begin_update()
        self.finish_update(token, success=True)
        self.commit_revision()

    def poison_uncoordinated_mutation(self, lease_id: str) -> None:
        with self._lock:
            self._prune_expired_leases_locked()
            if lease_id not in self._leases:
                raise WeightManifestError("weight snapshot lease does not exist")
            del self._leases[lease_id]
            self._generation += 1
            self._healthy = False
            self._needs_revision_commit = True
            self._last_update_success = False


class WeightRuntimeManifestManager:
    def __init__(
        self,
        *,
        model: Any,
        adapter: WeightSemanticsAdapter,
        topology: WeightParallelTopology,
        allowed_devices: Sequence[str] = ("cuda",),
        coordinator: WeightSnapshotCoordinator | None = None,
    ) -> None:
        self._model = model
        self._adapter = adapter
        self._topology = topology
        self._allowed_devices = frozenset(allowed_devices)
        self.coordinator = coordinator or WeightSnapshotCoordinator()
        self._last_signature: tuple | None = None
        self._last_generation: int | None = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        self.coordinator.invalidate()

    def release(self, lease_id: str) -> None:
        self.coordinator.release_snapshot(lease_id)

    def renew(self, lease_id: str, *, lease_timeout_sec: int) -> None:
        self.coordinator.renew_snapshot(lease_id, lease_timeout_sec=lease_timeout_sec)

    def has_lease(self, lease_id: str) -> bool:
        return self.coordinator.has_snapshot(lease_id)

    def commit_revision(self) -> int:
        return self.coordinator.commit_revision()

    def snapshot(
        self,
        *,
        model_id: str,
        revision: str,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
    ) -> WeightRuntimeManifest:
        if not all((model_id, revision, instance_id, worker_id, endpoint)):
            raise WeightManifestError("runtime manifest identifiers must not be empty")
        lease_id, generation = self.coordinator.acquire_snapshot(
            lease_timeout_sec=lease_timeout_sec
        )
        release_on_error = True
        try:
            with self._lock:
                physical = self._collect_physical_parameters()
                signature = tuple(
                    (
                        item.names,
                        item.address,
                        item.nbytes,
                        item.shape,
                        item.stride,
                        item.storage_offset,
                        item.dtype,
                        item.device,
                    )
                    for item in physical
                )
                if (
                    self._last_signature is not None
                    and signature != self._last_signature
                    and generation == self._last_generation
                ):
                    self.coordinator.poison_uncoordinated_mutation(lease_id)
                    release_on_error = False
                    raise WeightManifestError(
                        "parameter storage changed outside the update coordinator"
                    )
                tensors = self._build_tensors(
                    physical=physical,
                    instance_id=instance_id,
                    worker_id=worker_id,
                    endpoint=endpoint,
                    generation=generation,
                )
                self._last_signature = signature
                self._last_generation = generation
                manifest = WeightRuntimeManifest(
                    model_id=model_id,
                    revision=revision,
                    instance_id=instance_id,
                    generation=generation,
                    lease_id=lease_id,
                    tensors=tensors,
                )
            if not self.coordinator.has_snapshot(lease_id):
                raise WeightManifestError("weight snapshot lease expired")
            release_on_error = False
            return manifest
        finally:
            if release_on_error and self.coordinator.has_snapshot(lease_id):
                self.coordinator.release_snapshot(lease_id)

    def _collect_physical_parameters(self) -> tuple[_PhysicalParameter, ...]:
        grouped: dict[tuple, tuple[Any, list[str]]] = {}
        for name, parameter in self._model.named_parameters(remove_duplicate=False):
            key = _storage_key(parameter)
            if key not in grouped:
                grouped[key] = (parameter, [])
            grouped[key][1].append(name)
        result = []
        for parameter, names in grouped.values():
            result.append(
                _inspect_parameter(
                    names=tuple(sorted(names)),
                    parameter=parameter,
                    allowed_devices=self._allowed_devices,
                )
            )
        result.sort(key=lambda item: item.names)
        return tuple(result)

    def _build_tensors(
        self,
        *,
        physical: tuple[_PhysicalParameter, ...],
        instance_id: str,
        worker_id: str,
        endpoint: str,
        generation: int,
    ) -> tuple[RuntimeWeightTensor, ...]:
        rank = self._topology.rank()
        result = []
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
                result.append(
                    RuntimeWeightTensor(
                        fragment_id=_fragment_id(
                            instance_id=instance_id,
                            worker_id=worker_id,
                            generation=generation,
                            view=view,
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
                        lease_generation=generation,
                    )
                )
        result.sort(
            key=lambda item: (
                item.tensor_id,
                item.global_offset,
                item.fragment_id,
            )
        )
        return tuple(result)


class UnavailableWeightRuntimeManifestManager:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def invalidate(self) -> None:
        return None

    def snapshot(self, **kwargs) -> WeightRuntimeManifest:
        del kwargs
        raise WeightManifestError(self._reason)

    def release(self, lease_id: str) -> None:
        del lease_id
        raise WeightManifestError(self._reason)

    def commit_revision(self) -> int:
        raise WeightManifestError(self._reason)


def _topology_from_sglang(
    *, parallel_state: Any, parallel: Any
) -> WeightParallelTopology:
    if parallel_state.dp_rank is not None:
        dp_rank = parallel_state.dp_rank
        dp_size = parallel_state.dp_size
    else:
        dp_rank = parallel_state.moe_dp_rank or 0
        dp_size = parallel_state.moe_dp_size
    return WeightParallelTopology(
        dp_rank=dp_rank,
        dp_size=dp_size,
        tp_rank=parallel_state.tp_rank,
        tp_size=parallel_state.tp_size,
        pp_rank=parallel_state.pp_rank,
        pp_size=parallel_state.pp_size,
        ep_rank=parallel_state.moe_ep_rank,
        ep_size=parallel_state.moe_ep_size,
        moe_tp_rank=parallel.moe_tp_rank,
        moe_tp_size=parallel.moe_tp_size,
        attention_tp_rank=parallel_state.attn_tp_rank,
        attention_tp_size=parallel_state.attn_tp_size,
    )


def create_sglang_weight_runtime_manifest_manager(
    *,
    model: Any,
    config: Any,
    parallel_state: Any,
    parallel: Any,
    allowed_devices: Sequence[str] = ("cuda",),
    quantization: str | None = None,
    lora_enabled: bool = False,
    is_multimodal: bool = False,
    dynamic_expert_placement: bool = False,
    dp_attention_enabled: bool = False,
    coordinator: WeightSnapshotCoordinator | None = None,
):
    return create_weight_runtime_manifest_manager(
        model=model,
        config=config,
        topology=_topology_from_sglang(
            parallel_state=parallel_state,
            parallel=parallel,
        ),
        allowed_devices=allowed_devices,
        quantization=quantization,
        lora_enabled=lora_enabled,
        is_multimodal=is_multimodal,
        dynamic_expert_placement=dynamic_expert_placement,
        dp_attention_enabled=dp_attention_enabled,
        coordinator=coordinator,
    )


def create_weight_runtime_manifest_manager(
    *,
    model: Any,
    config: Any,
    topology: WeightParallelTopology,
    allowed_devices: Sequence[str] = ("cuda",),
    quantization: str | None = None,
    lora_enabled: bool = False,
    is_multimodal: bool = False,
    dynamic_expert_placement: bool = False,
    dp_attention_enabled: bool = False,
    coordinator: WeightSnapshotCoordinator | None = None,
):
    if quantization is not None:
        return UnavailableWeightRuntimeManifestManager(
            f"quantized weight manifests are unsupported: {quantization}"
        )
    if lora_enabled:
        return UnavailableWeightRuntimeManifestManager(
            "LoRA weight manifests are unsupported"
        )
    if dp_attention_enabled:
        return UnavailableWeightRuntimeManifestManager(
            "DP attention weight manifests are unsupported"
        )
    model_type = getattr(config, "model_type", None)
    text_model_types = ("qwen3_5_text", "qwen3_5_moe_text")
    multimodal_model_types = ("qwen3_5", "qwen3_5_moe")
    if is_multimodal and model_type not in multimodal_model_types:
        return UnavailableWeightRuntimeManifestManager(
            f"unsupported multimodal model type for weight manifests: {model_type}"
        )
    if not is_multimodal and model_type not in text_model_types:
        return UnavailableWeightRuntimeManifestManager(
            f"unsupported model type for weight manifests: {model_type}"
        )

    from sglang.srt.model_executor.weight_semantics.qwen3_5 import (
        Qwen35MultimodalWeightSemanticsAdapter,
        Qwen35WeightSemanticsAdapter,
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
            return UnavailableWeightRuntimeManifestManager(
                "Qwen3.5 multimodal config is missing text_config or vision_config"
            )
        adapter = Qwen35MultimodalWeightSemanticsAdapter(
            text_config=text_config,
            vision_config=vision_config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )
    else:
        adapter = Qwen35WeightSemanticsAdapter(
            config=config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )

    return WeightRuntimeManifestManager(
        model=model,
        adapter=adapter,
        topology=topology,
        allowed_devices=allowed_devices,
        coordinator=coordinator,
    )
