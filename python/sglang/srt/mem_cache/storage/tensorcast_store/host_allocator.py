# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Configuration and host-allocation boundary for TensorCast HiCache."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator

if TYPE_CHECKING:
    from tensorcast.api.store import (
        RegionBackedArtifactSession,
        RegionBackedArtifactSessionOptions,
    )


_SUPPORTED_LAYOUT_IO_PAIRS = frozenset(
    {
        ("page_first", "kernel"),
        ("page_first_direct", "direct"),
    }
)


class TensorcastTransferMode(str, Enum):
    """SGLang's supported TensorCast host-transfer modes.
    allocator (default): TensorCast daemon allocates and owns a shared memory slab
    with direct RDMA
    scratch: uses SGlang-owned common HostTensorAllocator. TensorCast uses a scratch memory slab as staging buffer to copy KV pages into it. More compatible with performance overhead.
    """

    ALLOCATOR = "allocator"
    SCRATCH = "scratch"


class _FrozenConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class TensorcastScratchConfig(_FrozenConfig):
    """Fixed per-direction scratch-arena configuration."""

    capacity_bytes: Annotated[int, Field(strict=True, gt=0)] = 16 * 1024 * 1024


class TensorcastConfig(_FrozenConfig):
    """Validated TensorCast-specific HiCache configuration."""

    daemon_address: Annotated[str, Field(strict=True)]
    namespace: Annotated[str, Field(strict=True)] = "default"
    transfer_mode: TensorcastTransferMode = TensorcastTransferMode.ALLOCATOR
    model_id: Annotated[str, Field(strict=True)] | None = None
    model_version: Annotated[str, Field(strict=True)] = "unversioned"
    session_name_prefix: Annotated[str, Field(strict=True)] = "sglang"
    region_name_prefix: Annotated[str, Field(strict=True)] = "sglang_tensorcast"
    exists_timeout_s: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False),
    ] = 30.0
    transfer_timeout_s: (
        Annotated[
            float,
            Field(gt=0.0, allow_inf_nan=False),
        ]
        | None
    ) = None
    scratch: TensorcastScratchConfig = TensorcastScratchConfig()

    @field_validator(
        "namespace",
        "model_version",
        "session_name_prefix",
        "region_name_prefix",
    )
    @classmethod
    def validate_required_string(cls, value: str) -> str:
        return _strip_non_empty(value)

    @field_validator("model_id")
    @classmethod
    def validate_optional_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_non_empty(value)


TensorcastConfigSource: TypeAlias = TensorcastConfig | str | Mapping[str, object]


class TensorcastSessionRegistryError(RuntimeError):
    """Raised when process-local TensorCast Session admission is invalid."""


@dataclass(slots=True)
class _TensorcastSessionRegistryState:
    session: RegionBackedArtifactSession | None = None
    session_options: RegionBackedArtifactSessionOptions | None = None
    early_attach_completed: bool = False
    active_store_owner: object | None = None
    terminal: bool = False


_PROCESS_SESSION_REGISTRY_LOCK = threading.Lock()
_PROCESS_SESSION_REGISTRY = _TensorcastSessionRegistryState()


def normalize_tensorcast_config(source: TensorcastConfigSource) -> TensorcastConfig:
    """Normalize raw or controller-parsed HiCache extra configuration."""

    if isinstance(source, TensorcastConfig):
        return source

    raw = _load_extra_config(source) if isinstance(source, str) else source
    if "tensorcast" not in raw:
        raise ValueError(
            "TensorCast HiCache configuration requires a 'tensorcast' object"
        )

    return TensorcastConfig.model_validate(raw["tensorcast"])


def format_tensorcast_rank_label(world_rank: int, world_size: int) -> str:
    """Return the stable diagnostic label for one SGLang rank."""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_rank < 0 or world_rank >= world_size:
        raise ValueError(
            "world_rank must be in [0, world_size), got "
            f"world_rank={world_rank}, world_size={world_size}"
        )
    return f"rank{world_rank}of{world_size}"


def build_tensorcast_session_options(
    config: TensorcastConfig,
    *,
    world_rank: int | None = None,
    world_size: int | None = None,
) -> RegionBackedArtifactSessionOptions:
    """Build public TensorCast Session options without exposing SDK internals."""

    try:
        from tensorcast.api.store import (
            AllocatorTransferOptions,
            RegionBackedArtifactSessionOptions,
            ScratchTransferOptions,
        )
    except ImportError as exc:
        raise ImportError(
            "The TensorCast HiCache backend requires the optional 'tensorcast' "
            "Python package. Install TensorCast before selecting "
            "--hicache-storage-backend tensorcast."
        ) from exc

    if (world_rank is None) != (world_size is None):
        raise ValueError("world_rank and world_size must be provided together")
    if world_rank is None:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        world_rank = parallel.world_rank
        world_size = parallel.world_size

    rank_label = format_tensorcast_rank_label(world_rank, cast(int, world_size))
    if config.transfer_mode == TensorcastTransferMode.ALLOCATOR:
        transfer = AllocatorTransferOptions()
    else:
        transfer = ScratchTransferOptions(
            capacity_bytes=config.scratch.capacity_bytes,
        )

    return RegionBackedArtifactSessionOptions(
        daemon_address=config.daemon_address,
        session_name=f"{config.session_name_prefix}-{rank_label}",
        transfer=transfer,
        transfer_timeout_s=config.transfer_timeout_s,
        exists_timeout_s=config.exists_timeout_s,
        region_name_prefix=f"{config.region_name_prefix}-{rank_label}",
    )


def _attach_process_session(
    options: RegionBackedArtifactSessionOptions,
) -> RegionBackedArtifactSession:
    from tensorcast.api.store import RegionBackedArtifactSession

    return RegionBackedArtifactSession.attach(options)


def attach_early_process_session(
    options: RegionBackedArtifactSessionOptions,
) -> RegionBackedArtifactSession:
    """Attach once before HostPool allocation and retain the process Session."""

    with _PROCESS_SESSION_REGISTRY_LOCK:
        state = _PROCESS_SESSION_REGISTRY
        if state.terminal:
            raise TensorcastSessionRegistryError(
                "the TensorCast process Session is terminal; restart the rank process"
            )
        if state.early_attach_completed:
            if options != state.session_options:
                raise TensorcastSessionRegistryError(
                    "TensorCast early attach conflicts with the process Session options"
                )
            if state.session is None:
                raise RuntimeError(
                    "TensorCast Session registry is inconsistent after early attach"
                )
            return state.session

        try:
            session = _attach_process_session(options)
        except BaseException:
            state.terminal = True
            raise

        state.session = session
        state.session_options = options
        state.early_attach_completed = True
        return session


def claim_tensorcast_store_session(
    options: RegionBackedArtifactSessionOptions,
    *,
    owner: object,
) -> RegionBackedArtifactSession:
    """Give one Store ownership of the already attached process Session."""

    if owner is None:
        raise ValueError("TensorCast Store owner must not be None")
    with _PROCESS_SESSION_REGISTRY_LOCK:
        state = _PROCESS_SESSION_REGISTRY
        if state.terminal:
            raise TensorcastSessionRegistryError(
                "the TensorCast process Session is terminal; restart the rank process"
            )
        if not state.early_attach_completed or state.session is None:
            raise TensorcastSessionRegistryError(
                "TensorCast Store cannot attach for the first time; the HostPool "
                "allocator must complete early Session attach"
            )
        if options != state.session_options:
            raise TensorcastSessionRegistryError(
                "TensorCast Store options conflict with the early process Session"
            )
        if state.active_store_owner is None:
            state.active_store_owner = owner
        elif state.active_store_owner is not owner:
            raise TensorcastSessionRegistryError(
                "another TensorCast Store already owns the process Session"
            )
        return state.session


def terminate_tensorcast_store_session(*, owner: object) -> bool:
    """Make Store admission terminal and terminate the Session exactly once."""

    with _PROCESS_SESSION_REGISTRY_LOCK:
        state = _PROCESS_SESSION_REGISTRY
        if state.active_store_owner is not owner:
            raise TensorcastSessionRegistryError(
                "only the active TensorCast Store owner may terminate the process Session"
            )
        if state.terminal:
            return False
        if state.session is None:
            raise RuntimeError(
                "TensorCast Session registry has an owner without an attached Session"
            )
        state.terminal = True
        session = state.session

    session.terminate_process_session()
    return True


class TensorcastHostTensorAllocator(HostTensorAllocator):
    """Delegate exact HostPool tensor allocations to one TensorCast Session."""

    def __init__(self, session: RegionBackedArtifactSession) -> None:
        super().__init__()
        self._session = session
        self._allocation_sequence = 0

    @property
    def session(self) -> RegionBackedArtifactSession:
        return self._session

    def allocate(
        self,
        dims: tuple[int, ...],
        dtype: torch.dtype,
        device: str,
    ) -> torch.Tensor:
        if device != "cpu":
            raise ValueError(
                f"TensorCast host allocation requires CPU memory, got device={device!r}"
            )
        self.dims = dims
        self.dtype = dtype
        self._allocation_sequence += 1
        return self._session.allocate_host_tensor(
            dims,
            dtype,
            name=f"host-pool-{self._allocation_sequence}",
        )


def create_tensorcast_host_allocator(
    source: TensorcastConfigSource,
    *,
    host_memory_mode: str,
    host_layout: str,
    io_backend: str,
    platform_name: str,
    is_cuda_backend: bool,
    world_rank: int | None = None,
    world_size: int | None = None,
) -> HostTensorAllocator:
    """Validate, attach, and select the configured TensorCast host allocator."""

    config = normalize_tensorcast_config(source)
    validate_tensorcast_startup_configuration(
        host_memory_mode=host_memory_mode,
        host_layout=host_layout,
        io_backend=io_backend,
        platform_name=platform_name,
        is_cuda_backend=is_cuda_backend,
    )
    options = build_tensorcast_session_options(
        config,
        world_rank=world_rank,
        world_size=world_size,
    )
    session = attach_early_process_session(options)
    if config.transfer_mode == TensorcastTransferMode.SCRATCH:
        return HostTensorAllocator()
    return TensorcastHostTensorAllocator(session)


def get_tensorcast_host_allocator_from_runtime() -> HostTensorAllocator:
    """Select the TensorCast allocator from the published SGLang config."""

    from sglang.srt.platforms import current_platform
    from sglang.srt.runtime_context import get_memory

    memory = get_memory()
    source = memory.hicache_storage_backend_extra_config
    if source is None:
        raise ValueError(
            "--hicache-storage-backend tensorcast requires "
            "--hicache-storage-backend-extra-config with tensorcast.daemon_address"
        )
    return create_tensorcast_host_allocator(
        source,
        host_memory_mode=memory.hicache_host_memory_mode,
        host_layout=memory.hicache_mem_layout,
        io_backend=memory.hicache_io_backend,
        platform_name=sys.platform,
        is_cuda_backend=current_platform.is_cuda(),
    )


def resolve_tensorcast_model_id(
    config: TensorcastConfig,
    storage_model_name: str | None,
) -> str:
    """Resolve the artifact model identity at later Store registration."""

    if config.model_id is not None:
        return config.model_id
    if storage_model_name is None:
        raise ValueError(
            "TensorCast artifact identity requires tensorcast.model_id or an "
            "SGLang storage model name"
        )
    try:
        return _strip_non_empty(storage_model_name)
    except ValueError as exc:
        raise ValueError(
            "TensorCast artifact identity requires tensorcast.model_id or a "
            "non-empty SGLang storage model name"
        ) from exc


def validate_tensorcast_startup_configuration(
    *,
    host_memory_mode: str,
    host_layout: str,
    io_backend: str,
    platform_name: str,
    is_cuda_backend: bool,
) -> None:
    """Reject initial-scope runtime combinations before Session attachment."""

    if platform_name != "linux":
        raise ValueError(
            "TensorCast HiCache initially supports Linux only, got "
            f"platform={platform_name!r}"
        )
    if not is_cuda_backend:
        raise ValueError("TensorCast HiCache initially supports the CUDA backend only")
    if host_memory_mode != "cache":
        raise ValueError(
            "TensorCast HiCache requires --hicache-host-memory-mode=cache, got "
            f"{host_memory_mode!r}"
        )
    if (host_layout, io_backend) not in _SUPPORTED_LAYOUT_IO_PAIRS:
        raise ValueError(
            "TensorCast HiCache requires (page_first, kernel) or "
            "(page_first_direct, direct), got "
            f"layout={host_layout!r}, io_backend={io_backend!r}"
        )


def _strip_non_empty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be empty")
    return stripped


def _load_extra_config(source: str) -> Mapping[str, object]:
    if not source.startswith("@"):
        return _require_mapping(json.loads(source), source_name="inline JSON")

    path_text = source[1:]
    if not path_text:
        raise ValueError("TensorCast HiCache config path must not be empty")
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as config_file:
            parsed = json.load(config_file)
    elif suffix == ".toml":
        import tomllib

        with path.open("rb") as config_file:
            parsed = tomllib.load(config_file)
    elif suffix in {".yaml", ".yml"}:
        import yaml

        with path.open(encoding="utf-8") as config_file:
            parsed = yaml.safe_load(config_file)
    else:
        raise ValueError(
            f"Unsupported TensorCast HiCache config file extension {suffix!r}"
        )
    return _require_mapping(parsed, source_name=str(path))


def _require_mapping(value: object, *, source_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"TensorCast HiCache config from {source_name} must be an object"
        )
    return cast(Mapping[str, object], value)
