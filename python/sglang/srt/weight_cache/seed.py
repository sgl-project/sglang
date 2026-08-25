# SPDX-License-Identifier: Apache-2.0
"""Daemon-to-daemon weight seeding for weight_cache.

This is deliberately a *separate* abstraction from
``WeightCacheTransportBackend`` (transport.py), even though both move weights
around:

- ``WeightCacheTransportBackend`` is daemon -> engine and has **mapping**
  semantics: the engine's ``param.data`` ends up pointing at the daemon's own
  GPU memory, so the engine must watch daemon liveness and can never outlive it.
- ``WeightCacheSeedSource`` (here) is daemon -> daemon and has **copy**
  semantics: the mirror daemon ends up owning private memory with the same
  bytes, and the two daemons are fully decoupled afterwards.

Collapsing them would force one of the two lifetime contracts onto the other.

A mirror daemon never builds an ``nn.Module``: it asks the source daemon for a
manifest of every exported tensor (shape/dtype/is_param), allocates them
locally, fills them through a seed source, and re-exports them via the normal
transport backend. That is what lets it skip disk read, TP sharding and
``process_weights_after_loading`` entirely.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from sglang.srt.platforms import current_platform
from sglang.srt.utils import MultiprocessingSerializer

from .protocol import visible_daemon_keys

logger = logging.getLogger(__name__)

PEER_IPC_SEED_SOURCE = "peer_ipc"
RDMA_SEED_SOURCE = "rdma"


def build_manifest(
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
) -> Dict[str, Dict[str, Any]]:
    """Describe every exported tensor well enough for a mirror to allocate it.

    Covers the source daemon's *entire* export set -- post-quantization
    parameters and non-persistent buffers included -- not just
    ``named_parameters()``. A mirror has no model to fall back on, so anything
    missing here would be missing for good.
    """
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "is_param": is_param,
        }
        for name, (tensor, is_param) in state_tensors.items()
    }


def manifest_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a manifest dtype string (``"bfloat16"``) to a ``torch.dtype``."""
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"[weight_cache:seed] manifest names dtype {dtype_name!r}, which this "
            f"torch build does not provide. The source daemon runs a different "
            f"torch version than this mirror."
        )
    return dtype


def manifest_nbytes(manifest: Mapping[str, Dict[str, Any]]) -> int:
    """Total payload size described by a manifest, for throughput logging."""
    total = 0
    for meta in manifest.values():
        numel = 1
        for dim in meta["shape"]:
            numel *= dim
        total += numel * manifest_dtype(meta["dtype"]).itemsize
    return total


class WeightCacheSeedSource(ABC):
    """Moves weight bytes from a source daemon into a mirror daemon.

    ``prepare_seed`` runs on the source; whatever it returns is embedded in the
    ``fetch_manifest`` response and handed back to ``fill`` on the mirror.
    """

    name: str

    @abstractmethod
    def prepare_seed(
        self,
        state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
        *,
        transport_entries: Mapping[str, Dict[str, Any]],
        gpu_id: int,
    ) -> Dict[str, Any]:
        """Source side: publish whatever the mirror needs to pull the bytes.

        ``transport_entries`` is the daemon's already-built export metadata, so a
        source whose mover happens to match the transport's representation can
        reuse it instead of exporting the same tensors twice.
        """

    @abstractmethod
    def fill(
        self,
        manifest: Mapping[str, Dict[str, Any]],
        seed_meta: Mapping[str, Any],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Mirror side: allocate on ``device`` and return the filled tensors."""

    def close(self) -> None:
        """Release any source-side resources (registrations, engines)."""


class PeerIpcSeedSource(WeightCacheSeedSource):
    """Same-node mover: ``dst.copy_(src)`` across a CUDA IPC mapping.

    torch's ``_share_cuda_`` handle carries the exporting process's device
    index, so deserializing it in the mirror yields a tensor that *lives on the
    source card*. Copying out of it is therefore a ``cudaMemcpyPeerAsync`` over
    NVLink -- strictly better than routing the same bytes through a NIC, which
    is why the intra-node case does not use RDMA.
    """

    name = PEER_IPC_SEED_SOURCE

    def prepare_seed(
        self,
        state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
        *,
        transport_entries: Mapping[str, Dict[str, Any]],
        gpu_id: int,
    ) -> Dict[str, Any]:
        # The torch_ipc transport already exported exactly the handles a peer
        # copy needs, so reuse them rather than calling _share_cuda_ a second
        # time on the same storages.
        handles = {
            name: entry["handle"]
            for name, entry in transport_entries.items()
            if entry.get("handle") is not None
        }
        missing = set(state_tensors) - set(handles)
        if missing:
            raise RuntimeError(
                f"[weight_cache:seed] the active transport backend did not "
                f"produce CUDA IPC handles for {len(missing)} tensor(s) "
                f"(e.g. {sorted(missing)[:3]}), so it cannot seed a peer daemon "
                f"via {PEER_IPC_SEED_SOURCE}."
            )
        return {
            "backend": self.name,
            "handles": handles,
            # Physical identity of the source card plus the logical index baked
            # into the handles; the mirror needs both to validate that it can
            # actually open them (see _assert_source_reachable).
            "source_local_index": gpu_id,
            "source_daemon_key": _daemon_key_of_local_index(gpu_id),
        }

    def fill(
        self,
        manifest: Mapping[str, Dict[str, Any]],
        seed_meta: Mapping[str, Any],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        handles = seed_meta["handles"]
        self._assert_source_reachable(seed_meta, device)

        tic = time.perf_counter()
        result: Dict[str, torch.Tensor] = {}
        # Keep the mappings into the source process alive until every copy has
        # retired, then drop them all: after the synchronize below the mirror
        # holds only its own memory and the source may die freely.
        source_refs: List[torch.Tensor] = []
        # Identical handle bytes mean the identical tensor view of the identical
        # storage, so tied weights map (and copy) once and stay tied here too.
        by_handle: Dict[str, torch.Tensor] = {}

        for name, meta in manifest.items():
            handle = handles.get(name)
            if handle is None:
                raise RuntimeError(
                    f"[weight_cache:seed] source daemon listed {name!r} in its "
                    f"manifest but published no handle for it."
                )
            cached = by_handle.get(handle)
            if cached is not None:
                result[name] = cached
                continue

            src = MultiprocessingSerializer.deserialize(handle)
            source_refs.append(src)
            dtype = manifest_dtype(meta["dtype"])
            shape = torch.Size(meta["shape"])
            if src.shape != shape or src.dtype != dtype:
                raise RuntimeError(
                    f"[weight_cache:seed] manifest/handle disagree for {name!r}: "
                    f"manifest says {shape}/{dtype}, mapped tensor is "
                    f"{src.shape}/{src.dtype}."
                )
            dst = torch.empty(shape, dtype=dtype, device=device)
            # Issued on the current (default) stream so the allocator's
            # stream bookkeeping stays trivially correct; one synchronize at the
            # end replaces what would otherwise be thousands of blocking copies.
            dst.copy_(src, non_blocking=True)
            result[name] = dst
            by_handle[handle] = dst

        current_platform.synchronize()
        source_refs.clear()

        elapsed = time.perf_counter() - tic
        total_bytes = manifest_nbytes(manifest)
        logger.info(
            "[weight_cache:seed] peer-copied %d tensors (%.2f GiB) in %.2fs "
            "(%.1f GiB/s)",
            len(result),
            total_bytes / 1024**3,
            elapsed,
            total_bytes / 1024**3 / max(elapsed, 1e-9),
        )
        return result

    @staticmethod
    def _assert_source_reachable(
        seed_meta: Mapping[str, Any], device: torch.device
    ) -> None:
        """Fail with actionable text before cudaIpcOpenMemHandle fails opaquely.

        The handles name the source card by the source process's *logical* index.
        A mirror can only open them if that same index resolves to the same
        physical card here, which is exactly what a mismatched
        CUDA_VISIBLE_DEVICES breaks -- the most common way this path is
        misconfigured.
        """
        source_index = seed_meta["source_local_index"]
        source_key = seed_meta["source_daemon_key"]
        local_index = _local_index_of_daemon_key(source_key)

        if local_index is None:
            raise RuntimeError(
                f"[weight_cache:seed] the source daemon runs on physical device "
                f"{source_key!r}, which is not visible to this mirror "
                f"(visible: {visible_daemon_keys()}). CUDA IPC requires the "
                f"source card to be visible here. Widen CUDA_VISIBLE_DEVICES to "
                f"include both the source and the mirror card."
            )
        if local_index != source_index:
            raise RuntimeError(
                f"[weight_cache:seed] physical device {source_key!r} is logical "
                f"index {source_index} in the source daemon but {local_index} "
                f"here, and CUDA IPC handles carry the source's index. Launch "
                f"both replicas with the same CUDA_VISIBLE_DEVICES (or leave it "
                f"unset on both) so the device numbering agrees."
            )

        if device.index is not None and device.index != source_index:
            try:
                if not torch.cuda.can_device_access_peer(device.index, source_index):
                    logger.warning(
                        "[weight_cache:seed] no peer access between device %d and "
                        "source device %d; the copies will be staged through host "
                        "memory. Results stay correct but the transfer is much "
                        "slower than NVLink -- check the throughput logged below.",
                        device.index,
                        source_index,
                    )
            except Exception as e:  # pragma: no cover - probe is advisory only
                logger.debug("[weight_cache:seed] peer-access probe failed: %s", e)


class RdmaSeedSource(WeightCacheSeedSource):
    """Cross-machine mover: mooncake TransferEngine, mirror pulls (read).

    The mirror is the active side (``batch_transfer_sync_read``), matching the
    existing remote-instance weight loader, so the source only has to register
    memory and publish addresses.
    """

    name = RDMA_SEED_SOURCE

    def __init__(self, ib_device: Optional[str] = None):
        self.ib_device = ib_device
        self._engine = None
        self._registered_ptrs: List[int] = []

    def _get_engine(self, gpu_id: int):
        if self._engine is None:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                MooncakeTransferEngine,
            )
            from sglang.srt.utils.network import get_local_ip_auto

            self._engine = MooncakeTransferEngine(
                hostname=get_local_ip_auto(),
                gpu_id=gpu_id,
                ib_device=self.ib_device,
            )
        return self._engine

    @staticmethod
    def _region_of(name: str, tensor: torch.Tensor) -> Tuple[int, int]:
        # RDMA addresses a flat byte range, so a strided view would transfer the
        # wrong bytes. The daemon exports post-processing outputs, which are
        # contiguous in practice; assert rather than silently corrupt.
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"[weight_cache:seed] {name!r} is not contiguous, so it cannot "
                f"be seeded over RDMA (a flat byte range would not match the "
                f"tensor's layout)."
            )
        return tensor.data_ptr(), tensor.numel() * tensor.element_size()

    def prepare_seed(
        self,
        state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
        *,
        transport_entries: Mapping[str, Dict[str, Any]],
        gpu_id: int,
    ) -> Dict[str, Any]:
        engine = self._get_engine(gpu_id)

        regions: Dict[str, Tuple[int, int]] = {}
        ptrs: List[int] = []
        lengths: List[int] = []
        seen: Dict[int, int] = {}
        for name, (tensor, _is_param) in state_tensors.items():
            ptr, nbytes = self._region_of(name, tensor)
            regions[name] = (ptr, nbytes)
            # Tied weights share a pointer; registering it twice is an error in
            # mooncake, so register the largest span once.
            if seen.get(ptr, -1) < nbytes:
                seen[ptr] = nbytes

        for ptr, nbytes in seen.items():
            ptrs.append(ptr)
            lengths.append(nbytes)

        # One batched registration: per-tensor registration of a full model is
        # thousands of round trips into the NIC driver.
        tic = time.perf_counter()
        ret = engine.batch_register(ptrs, lengths)
        if ret != 0:
            raise RuntimeError(
                f"[weight_cache:seed] mooncake batch_register_memory failed "
                f"(ret={ret}) for {len(ptrs)} weight region(s); this daemon "
                f"cannot serve as an RDMA seed."
            )
        self._registered_ptrs = ptrs
        logger.info(
            "[weight_cache:seed] registered %d weight region(s) with mooncake "
            "in %.2fs",
            len(ptrs),
            time.perf_counter() - tic,
        )

        return {
            "backend": self.name,
            "session_id": engine.get_session_id(),
            "regions": regions,
        }

    def fill(
        self,
        manifest: Mapping[str, Dict[str, Any]],
        seed_meta: Mapping[str, Any],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            MooncakeTransferEngine,
        )
        from sglang.srt.utils.network import get_local_ip_auto

        regions = seed_meta["regions"]
        session_id = seed_meta["session_id"]

        engine = MooncakeTransferEngine(
            hostname=get_local_ip_auto(),
            gpu_id=device.index if device.index is not None else 0,
            ib_device=self.ib_device,
        )

        result: Dict[str, torch.Tensor] = {}
        by_region: Dict[Tuple[int, int, str], torch.Tensor] = {}
        local_ptrs: List[int] = []
        remote_ptrs: List[int] = []
        lengths: List[int] = []

        for name, meta in manifest.items():
            region = regions.get(name)
            if region is None:
                raise RuntimeError(
                    f"[weight_cache:seed] source daemon listed {name!r} in its "
                    f"manifest but published no RDMA region for it."
                )
            remote_ptr, remote_nbytes = region
            dtype = manifest_dtype(meta["dtype"])
            shape = torch.Size(meta["shape"])
            key = (remote_ptr, remote_nbytes, f"{shape}/{dtype}")
            cached = by_region.get(key)
            if cached is not None:
                result[name] = cached
                continue

            dst = torch.empty(shape, dtype=dtype, device=device)
            nbytes = dst.numel() * dst.element_size()
            if nbytes != remote_nbytes:
                raise RuntimeError(
                    f"[weight_cache:seed] size mismatch for {name!r}: source "
                    f"region is {remote_nbytes} bytes, local allocation is "
                    f"{nbytes} bytes."
                )
            result[name] = dst
            by_region[key] = dst
            local_ptrs.append(dst.data_ptr())
            remote_ptrs.append(remote_ptr)
            lengths.append(nbytes)

        ret = engine.batch_register(local_ptrs, lengths)
        if ret != 0:
            raise RuntimeError(
                f"[weight_cache:seed] mooncake batch_register_memory failed "
                f"(ret={ret}) for {len(local_ptrs)} local weight region(s)."
            )

        tic = time.perf_counter()
        ret = engine.engine.batch_transfer_sync_read(
            session_id, local_ptrs, remote_ptrs, lengths
        )
        if ret < 0:
            raise RuntimeError(
                f"[weight_cache:seed] mooncake batch_transfer_sync_read failed "
                f"(ret={ret}) while pulling weights from {session_id}."
            )
        elapsed = time.perf_counter() - tic

        total_bytes = sum(lengths)
        logger.info(
            "[weight_cache:seed] pulled %d tensors (%.2f GiB) from %s over RDMA "
            "in %.2fs (%.1f GiB/s)",
            len(result),
            total_bytes / 1024**3,
            session_id,
            elapsed,
            total_bytes / 1024**3 / max(elapsed, 1e-9),
        )
        # The mirror owns these bytes now; keep the engine alive only as long as
        # this call so the mirror does not depend on the source afterwards.
        engine.batch_deregister(local_ptrs)
        return result

    def close(self) -> None:
        if self._engine is not None and self._registered_ptrs:
            self._engine.batch_deregister(self._registered_ptrs)
            self._registered_ptrs = []


_SEED_SOURCES = {
    PEER_IPC_SEED_SOURCE: PeerIpcSeedSource,
    RDMA_SEED_SOURCE: RdmaSeedSource,
}


def get_seed_source(name: str, **kwargs) -> WeightCacheSeedSource:
    """Instantiate a seed source by name."""
    cls = _SEED_SOURCES.get(name)
    if cls is None:
        raise RuntimeError(
            f"[weight_cache:seed] unknown seed source {name!r}; known sources: "
            f"{sorted(_SEED_SOURCES)}"
        )
    return cls(**kwargs)


def _daemon_key_of_local_index(gpu_id: int) -> str:
    from .protocol import compute_daemon_key

    return compute_daemon_key(gpu_id)


def _local_index_of_daemon_key(daemon_key: str) -> Optional[int]:
    """Logical index of a physical device key in *this* process's namespace.

    Returns None when the key names a device this process cannot see.
    """
    keys = visible_daemon_keys()
    if keys is None:
        # Unrestricted visibility: logical index == physical index, which is
        # only expressible when the key is a plain index (a UUID key means the
        # source narrowed its visibility and we cannot resolve it).
        return int(daemon_key) if daemon_key.isdigit() else None
    return keys.index(daemon_key) if daemon_key in keys else None
