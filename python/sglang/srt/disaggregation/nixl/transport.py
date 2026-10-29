from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec
import numpy as np
import torch

from sglang.srt.disaggregation.transport.base import (
    BufferKey,
    Copy,
    Handle,
    RemoteBuffers,
    Result,
    Status,
    Transport,
)
from sglang.srt.runtime_context import get_device
from sglang.srt.utils.common import run_with_deadline

logger = logging.getLogger(__name__)


class NixlResult(Result, frozen=True):
    # FAILED only: the write already stopped touching memory, as when NIXL
    # reports a posted transfer as terminated.
    quiescent: bool = False


class Registration(msgspec.Struct, frozen=True):
    key: BufferKey
    ptr: int
    length: int
    # GPU index, or -1 for page-locked host memory.
    device: int


class StridedRun(msgspec.Struct, frozen=True):
    """`count` blocks of `length` bytes in buffer `key`; block i starts at
    offset + i * stride."""

    key: BufferKey
    offset: int
    length: int
    stride: int
    count: int


_DONE = NixlResult(Status.DONE)
_PENDING = NixlResult(Status.PENDING)

# (ptr, length, device): one engine registration.
_Region = Tuple[int, int, int]


def _mem_kind(device: int) -> str:
    return "DRAM" if device < 0 else "VRAM"


def _nixl_device_id(device: int) -> int:
    # NIXL names page-locked host memory as device 0.
    return max(device, 0)


def _expand_runs(runs: np.ndarray) -> np.ndarray:
    """Expand Nx5 runs into one (addr, len, dev_id) descriptor per block."""
    parts = []
    for addr, length, dev_id, stride, count in runs.tolist():
        out = np.empty((count, 3), dtype=np.uint64)
        out[:, 0] = addr + np.arange(count, dtype=np.uint64) * stride
        out[:, 1] = length
        out[:, 2] = dev_id
        parts.append(out)
    return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.uint64)


class NixlRemoteBuffers(RemoteBuffers):
    def __init__(
        self,
        *,
        agent_name: str,
        metadata: bytes,
        bases: Dict[BufferKey, int],
        devices: Dict[BufferKey, int],
    ):
        self.agent_name = agent_name
        # Kept so that reload() can restore a peer NIXL dropped.
        self.metadata = metadata
        self.bases = bases
        self.devices = devices


class NixlHandle(Handle):
    def __init__(
        self,
        *,
        xfer: Any = None,
        result: Optional[NixlResult] = None,
        error: Optional[BaseException] = None,
    ):
        self.xfer = xfer
        # Set once final; a handle that never posted starts final.
        self.result = result
        # What NIXL raised creating, posting or polling this transfer, if anything.
        self.error = error


class NixlPreparedList:
    """A NIXL dlist prepared once; write_prepared() selects its blocks by flat
    index, runs in order."""

    def __init__(self, *, dlist: Any):
        self.dlist = dlist


def create_nixl_agent(
    *,
    backend: str,
    backend_params: Dict[str, str],
    num_threads: int,
    gpu_id: int,
    init_timeout_s: float,
) -> Any:
    try:
        from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
    except ImportError as e:
        raise ImportError(
            "Please install NIXL by following the instructions at "
            "https://github.com/ai-dynamo/nixl/blob/main/README.md "
            "to run SGLang with NixlTransferEngine."
        ) from e

    # NixlKVManager.transfer_worker and _start_bootstrap_thread run concurrently
    # so we cannot use sync_mode=None which is thread-unsafe.
    agent_config = nixl_agent_config(
        backends=[],
        num_threads=num_threads,
        sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
    )
    agent = nixl_agent(str(uuid.uuid4()), agent_config)
    backend_params = dict(backend_params)
    if num_threads > 0:
        # TODO: Remove this once NIXL passes thread parameters from
        # nixl_agent_config to explicitly-created backends.
        if backend == "UCX" or backend == "OBJ":
            backend_params.setdefault("num_threads", str(num_threads))
        elif backend == "GDS_MT":
            backend_params.setdefault("thread_count", str(num_threads))
        elif backend == "UCCL":
            backend_params.setdefault("num_cpus", str(num_threads))

    def create_backend():
        torch.get_device_module(get_device().device).set_device(gpu_id)
        return agent.create_backend(backend, backend_params)

    run_with_deadline(
        create_backend,
        timeout_s=init_timeout_s,
        what=f"NIXL create_backend({backend!r}, {backend_params})",
    )

    available_plugins = agent.get_plugin_list()
    if backend not in available_plugins:
        raise ValueError(
            f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
            f"Please install the required NIXL plugin or choose from: {available_plugins}"
        )
    return agent


class NixlTransport(Transport):
    """Transport over a NIXL agent made by create_nixl_agent().

    write() and write_prepared() post without waiting. An ERR that NIXL reports
    for a posted transfer is final and quiescent; a rejected post or an
    unreadable state is not, since NIXL may still have writes on the wire.
    """

    def __init__(self, *, agent: Any):
        # One agent per KVManager, owned here: close() also drops its peers.
        self._agent = agent
        # (ptr, length, device) each key is registered with. Keys sharing a
        # region share one engine registration.
        self._registrations: Dict[BufferKey, _Region] = {}
        self._remotes: Dict[str, NixlRemoteBuffers] = {}
        # Newer NIXL prepares compressed (addr, len, dev_id, stride, count) runs
        # directly; cleared on the first TypeError to expand per block instead.
        self._use_strided_descs = True

    @property
    def agent_name(self) -> str:
        return self._agent.name

    @property
    def agent_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    @property
    def supports_strided_runs(self) -> bool:
        return self._use_strided_descs

    def register(self, key: BufferKey, ptr: int, length: int, device: int) -> None:
        self.register_all(
            [Registration(key=key, ptr=ptr, length=length, device=device)]
        )

    def register_all(self, registrations: Sequence[Registration]) -> None:
        """Register in one register_memory call per memory kind: a NIXL
        registration list holds a single memory kind."""
        self.deregister(
            [
                r.key
                for r in registrations
                if self._registrations.get(r.key, (r.ptr, r.length, r.device))
                != (r.ptr, r.length, r.device)
            ]
        )
        registered_regions = set(self._registrations.values())
        new_regions = []
        for r in registrations:
            if r.length == 0:
                # An elided entry keeps its position; nothing writes it.
                continue
            region = (r.ptr, r.length, r.device)
            if region not in registered_regions:
                registered_regions.add(region)
                new_regions.append(region)
            self._registrations[r.key] = region
        for kind, regions in self._by_mem_kind(new_regions).items():
            descs = self._agent.register_memory(self._reg_descs(regions), kind)
            if not descs:
                raise RuntimeError(
                    f"NIXL memory registration failed for {len(regions)} {kind} regions"
                )

    def deregister(self, keys: Sequence[BufferKey]) -> None:
        """Unregister `keys`, preserving regions still held by other keys."""
        released_regions = {
            self._registrations.pop(key) for key in keys if key in self._registrations
        }
        unused_regions = released_regions - set(self._registrations.values())
        for kind, regions in self._by_mem_kind(unused_regions).items():
            self._agent.deregister_memory(
                self._agent.get_reg_descs(self._reg_descs(regions), kind)
            )

    @staticmethod
    def _reg_descs(regions: List[_Region]) -> List[Tuple[int, int, int, str]]:
        return [(ptr, length, _nixl_device_id(dev), "") for ptr, length, dev in regions]

    @staticmethod
    def _by_mem_kind(regions: Iterable[_Region]) -> Dict[str, List[_Region]]:
        grouped: Dict[str, List[_Region]] = {}
        for region in regions:
            grouped.setdefault(_mem_kind(region[2]), []).append(region)
        return grouped

    def import_registration(
        self,
        *,
        agent_name: str,
        metadata: bytes,
        bases: Dict[BufferKey, int],
        devices: Dict[BufferKey, int],
    ) -> NixlRemoteBuffers:
        """Import the current registration fields and load the peer's agent."""
        self._agent.add_remote_agent(metadata)
        remote = NixlRemoteBuffers(
            agent_name=agent_name, metadata=metadata, bases=bases, devices=devices
        )
        self._remotes[agent_name] = remote
        return remote

    def is_loaded(self, remote: NixlRemoteBuffers) -> bool:
        """False once NIXL dropped the peer, as it does after a remote disconnect."""
        return self._agent.check_remote_metadata(remote.agent_name)

    def reload(self, remote: NixlRemoteBuffers) -> None:
        """Load a dropped peer again; lists prepared over it before stay invalid."""
        self._agent.add_remote_agent(remote.metadata)

    def unload(self, remote: NixlRemoteBuffers) -> None:
        self._agent.remove_remote_agent(remote.agent_name)

    def write(
        self,
        remote: NixlRemoteBuffers,
        src: BufferKey,
        dst: BufferKey,
        ranges: np.ndarray,
    ) -> NixlHandle:
        return self.write_all(remote, [Copy(src=src, dst=dst, ranges=ranges)])[0]

    def write_all(
        self, remote: NixlRemoteBuffers, copies: Sequence[Copy]
    ) -> List[NixlHandle]:
        """One NIXL transfer per (source, destination) memory kind: a NIXL
        descriptor list holds a single memory kind."""
        groups: Dict[Tuple[str, str], List[Copy]] = {}
        for c in copies:
            kinds = (
                _mem_kind(self._registrations[c.src][2]),
                _mem_kind(remote.devices[c.dst]),
            )
            groups.setdefault(kinds, []).append(c)
        return [
            self._post_copies(remote=remote, kinds=kinds, copies=group)
            for kinds, group in groups.items()
        ]

    def _post_copies(
        self, *, remote: NixlRemoteBuffers, kinds: Tuple[str, str], copies: List[Copy]
    ) -> NixlHandle:
        src_reqs = np.vstack(
            [self._descs(*self._locate(None, c.src), c.ranges[:, 0], c) for c in copies]
        )
        dst_reqs = np.vstack(
            [
                self._descs(*self._locate(remote, c.dst), c.ranges[:, 1], c)
                for c in copies
            ]
        )
        try:
            xfer = self._agent.initialize_xfer(
                "WRITE",
                self._agent.get_xfer_descs(src_reqs, kinds[0]),
                self._agent.get_xfer_descs(dst_reqs, kinds[1]),
                remote.agent_name,
            )
        except Exception as e:
            return self._never_posted(f"NIXL could not create the transfer: {e}", e)
        if not xfer:
            return self._never_posted("NIXL could not create the transfer")
        return self._post(xfer)

    def _locate(
        self, remote: Optional[NixlRemoteBuffers], key: BufferKey
    ) -> Tuple[int, int]:
        """(base address, device) of `key` on this side (remote None) or a peer."""
        if remote is None:
            ptr, _, device = self._registrations[key]
            return ptr, device
        return remote.bases[key], remote.devices[key]

    @staticmethod
    def _descs(base: int, device: int, offsets: np.ndarray, copy: Copy) -> np.ndarray:
        # uint64: Intel XPU addresses have bit 63 set and overflow int64.
        out = np.empty((len(offsets), 3), dtype=np.uint64)
        out[:, 0] = np.uint64(base) + offsets.astype(np.uint64, copy=False)
        out[:, 1] = copy.ranges[:, 2]
        out[:, 2] = _nixl_device_id(device)
        return out

    def prepare(
        self,
        remote: Optional[NixlRemoteBuffers],
        runs: Sequence[StridedRun],
        *,
        allow_expansion: bool = True,
    ) -> NixlPreparedList:
        """Prepare a NIXL dlist handle from runs over local buffers (remote None)
        or a peer's.

        Tries the strided API first and falls back to expanding one descriptor
        per block when the installed NIXL rejects the Nx5 layout; with
        allow_expansion False that raises TypeError instead.
        """
        peer_name = "" if remote is None else remote.agent_name
        located = [self._locate(remote, r.key) for r in runs]
        # The protocol prepares one memory kind per list (see _kv_xfer_mem_segments).
        kind = _mem_kind(located[0][1])
        # uint64: Intel XPU addresses have bit 63 set and overflow int64.
        array = np.array(
            [
                (base + r.offset, r.length, _nixl_device_id(device), r.stride, r.count)
                for r, (base, device) in zip(runs, located)
            ],
            dtype=np.uint64,
        ).reshape(-1, 5)
        if self._use_strided_descs:
            try:
                dlist = self._agent.prep_xfer_dlist(peer_name, array, kind)
                return NixlPreparedList(dlist=dlist)
            except TypeError:
                logger.warning_once(
                    "Installed NIXL does not support strided descriptors, "
                    "falling back to per-block descriptors."
                )
                self._use_strided_descs = False
        if not allow_expansion:
            raise TypeError("Installed NIXL does not support strided descriptors")
        descs = self._agent.get_xfer_descs(_expand_runs(array), kind)
        return NixlPreparedList(
            dlist=self._agent.prep_xfer_dlist(peer_name, descs, kind)
        )

    def write_prepared(
        self,
        src: NixlPreparedList,
        src_indices: np.ndarray,
        dst: NixlPreparedList,
        dst_indices: np.ndarray,
    ) -> List[NixlHandle]:
        try:
            xfer = self._agent.make_prepped_xfer(
                "WRITE", src.dlist, src_indices, dst.dlist, dst_indices
            )
        except Exception as e:
            return [self._never_posted(f"NIXL could not create the transfer: {e}", e)]
        if not xfer:
            return [self._never_posted("NIXL could not create the transfer")]
        return [self._post(xfer)]

    def _post(self, xfer: Any) -> NixlHandle:
        # The NIXL Python API raises on a failed post rather than returning "ERR".
        try:
            state = self._agent.transfer(xfer)
        except Exception as e:
            reason, error = f"NIXL rejected the post: {e}", e
        else:
            if state != "ERR":
                return NixlHandle(xfer=xfer)
            reason, error = "NIXL rejected the post", None
        return NixlHandle(
            xfer=xfer, result=NixlResult(Status.FAILED, reason=reason), error=error
        )

    @staticmethod
    def _never_posted(reason: str, error: Optional[BaseException] = None) -> NixlHandle:
        return NixlHandle(
            result=NixlResult(Status.FAILED, reason=reason, quiescent=True),
            error=error,
        )

    def status(self, handle: NixlHandle) -> NixlResult:
        if handle.result is not None:
            return handle.result
        try:
            state = self._agent.check_xfer_state(handle.xfer)
        except Exception as e:
            # Not proof that the write into the peer's memory is over.
            result = NixlResult(Status.FAILED, reason=f"NIXL state unreadable: {e}")
            handle.error = e
        else:
            if state == "DONE":
                result = _DONE
            elif state == "ERR":
                result = NixlResult(
                    Status.FAILED, reason="NIXL reported ERR", quiescent=True
                )
            else:
                return _PENDING
        handle.result = result
        return result

    def close(self) -> None:
        for remote in list(self._remotes.values()):
            if self.is_loaded(remote):
                self.unload(remote)
        self._remotes.clear()
        self.deregister(list(self._registrations))
