# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""GPU-resident Mooncake sink for SpecForge hidden-state capture.

Captured CUDA tensors are copied on-device into publisher-owned buffers and
remain resident until the trainer acknowledges consumption.  Mooncake carries
the tensor payload over MNNVL/NVLink or RDMA; a small TCP endpoint carries only
descriptors and release messages.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import json
import logging
import os
import secrets
import socketserver
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)

_DTYPE_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_DTYPE = {value: key for key, value in _DTYPE_STR.items()}
_MAX_CONTROL_BATCH = 4096


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _load_cudart() -> ctypes.CDLL:
    for name in (
        ctypes.util.find_library("cudart"),
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so",
    ):
        if not name:
            continue
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        library.cudaMemcpyAsync.restype = ctypes.c_int
        return library
    raise RuntimeError("unable to load libcudart for MNNVL D2D staging")


_CUDART: Optional[ctypes.CDLL] = None
_CUDART_LOCK = threading.Lock()


def _copy_to_address(
    destination: int,
    source: int,
    length: int,
    *,
    stream: torch.cuda.Stream,
) -> None:
    global _CUDART
    if _CUDART is None:
        with _CUDART_LOCK:
            if _CUDART is None:
                _CUDART = _load_cudart()
    status = _CUDART.cudaMemcpyAsync(
        ctypes.c_void_p(destination),
        ctypes.c_void_p(source),
        ctypes.c_size_t(length),
        3,  # cudaMemcpyDeviceToDevice
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"cudaMemcpyAsync D2D failed with status {status}")


@dataclass
class _PublishedBuffer:
    address: int
    nbytes: int
    allocation_offset: int
    allocation_nbytes: int
    shape: Tuple[int, ...]
    dtype: str
    tensor: Optional[torch.Tensor]
    registered: bool

    def descriptor(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass
class _ResidentSample:
    generation: int
    buffers: Dict[str, _PublishedBuffer]
    result: Dict[str, Any]


class _ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: Tuple[str, int], owner: "SpecCaptureSink"):
        self.owner = owner
        super().__init__(address, _ControlHandler)


class _ControlHandler(socketserver.StreamRequestHandler):
    disable_nagle_algorithm = True

    def handle(self) -> None:
        while True:
            try:
                payload = self.rfile.readline(1 << 20)
                if not payload:
                    return
                response = self.server.owner.handle_control(  # type: ignore[attr-defined]
                    json.loads(payload.decode("utf-8"))
                )
            except BaseException as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write(
                json.dumps(response, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            self.wfile.flush()


class SpecCaptureSink:
    _ARENA_ALIGNMENT = 256

    def __init__(self, server_args) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-direct spec capture requires CUDA")
        self.backend = str(server_args.spec_capture_transfer_backend)
        if self.backend not in {"nvlink", "nvlink_intra", "rdma"}:
            raise ValueError(f"unsupported spec-capture backend {self.backend!r}")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.aux_layer_ids = (
            list(server_args.spec_capture_aux_layer_ids)
            if server_args.spec_capture_aux_layer_ids
            else None
        )
        self.max_resident_bytes = server_args.spec_capture_max_resident_bytes
        if self.max_resident_bytes is not None and self.max_resident_bytes < 1:
            raise ValueError("spec_capture_max_resident_bytes must be positive")
        if server_args.spec_capture_max_pending_batches < 1:
            raise ValueError("spec_capture_max_pending_batches must be positive")

        advertise_host = server_args.spec_capture_control_host or get_local_ip_auto()
        self._lock = threading.RLock()
        self._engine_lock = threading.RLock()
        self._engine = self._initialize_engine(server_args, advertise_host)
        self.session_id = f"{advertise_host}:{int(self._engine.get_rpc_port())}"
        self._arena_address = 0
        self._arena_bytes = 0
        self._arena_free: List[Tuple[int, int]] = []
        if self.backend in {"nvlink", "nvlink_intra"}:
            if self.max_resident_bytes is None:
                raise ValueError(
                    "NVLink GPU-direct capture requires "
                    "spec_capture_max_resident_bytes for its fixed arena"
                )
            self._arena_bytes = int(self.max_resident_bytes)
            with self._engine_lock:
                self._arena_address = int(
                    self._engine.allocate_managed_buffer(self._arena_bytes)
                )
            if self._arena_address == 0:
                raise MemoryError(
                    f"Mooncake MNNVL arena allocation failed "
                    f"({self._arena_bytes} bytes)"
                )
            self._arena_free = [(0, self._arena_bytes)]
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="spec-capture-gpudirect",
        )
        self._token = secrets.token_hex(24)
        self._residents: Dict[str, _ResidentSample] = {}
        self._reserved_bytes = 0
        self._closed = False
        self._stats = {
            "published_samples": 0,
            "released_samples": 0,
            "device_staging_bytes": 0,
            "host_payload_bytes": 0,
            "control_requests": 0,
            "control_batch_requests": 0,
            "control_batch_items": 0,
        }

        self._control_server = _ControlServer(
            (advertise_host, int(server_args.spec_capture_control_port)), self
        )
        control_port = int(self._control_server.server_address[1])
        self.control_endpoint = f"{advertise_host}:{control_port}"
        self._control_thread = threading.Thread(
            target=self._control_server.serve_forever,
            name="spec-capture-control",
            daemon=True,
        )
        self._control_thread.start()
        atexit.register(self.close)
        logger.info(
            "GPU-direct spec-capture sink initialized: backend=%s session=%s "
            "control=%s arena_address=%s arena_bytes=%d",
            self.backend,
            self.session_id,
            self.control_endpoint,
            hex(self._arena_address) if self._arena_address else None,
            self._arena_bytes,
        )

    def _initialize_engine(self, server_args, advertise_host: str):
        try:
            from mooncake import engine as mooncake_engine
        except ImportError as exc:
            raise ImportError(
                "GPU-direct spec capture requires mooncake-transfer-engine"
            ) from exc
        if self.backend == "nvlink":
            os.environ["MC_FORCE_MNNVL"] = "true"
            os.environ.pop("MC_INTRA_NVLINK", None)
            os.environ.pop("MC_FORCE_HCA", None)
            if not bool(getattr(mooncake_engine, "SUPPORT_MNNVL", False)):
                raise RuntimeError("installed Mooncake wheel has SUPPORT_MNNVL=False")
        elif self.backend == "nvlink_intra":
            os.environ.pop("MC_FORCE_MNNVL", None)
            os.environ["MC_INTRA_NVLINK"] = "true"
            os.environ.pop("MC_FORCE_HCA", None)
            if not bool(
                getattr(mooncake_engine, "SUPPORT_INTRA_NVLINK", False)
            ):
                raise RuntimeError(
                    "installed Mooncake wheel has SUPPORT_INTRA_NVLINK=False"
                )
        else:
            os.environ.pop("MC_FORCE_MNNVL", None)
            os.environ.pop("MC_INTRA_NVLINK", None)
            os.environ["MC_FORCE_HCA"] = "true"
        engine = mooncake_engine.TransferEngine()
        ib_device = ""
        if self.backend == "rdma":
            ib_device = (
                server_args.spec_capture_ib_device
                or server_args.mooncake_ib_device
                or server_args.disaggregation_ib_device
                or ""
            )
        status = engine.initialize(
            advertise_host,
            "P2PHANDSHAKE",
            self.backend,
            ib_device,
        )
        if int(status) != 0:
            raise RuntimeError(
                f"Mooncake {self.backend} TransferEngine initialization failed: {status}"
            )
        return engine

    def _resident_bytes_locked(self) -> int:
        return sum(
            buffer.nbytes
            for sample in self._residents.values()
            for buffer in sample.buffers.values()
        )

    def _resident_allocated_bytes_locked(self) -> int:
        return sum(
            buffer.allocation_nbytes
            for sample in self._residents.values()
            for buffer in sample.buffers.values()
        )

    @classmethod
    def _aligned_nbytes(cls, nbytes: int) -> int:
        alignment = cls._ARENA_ALIGNMENT
        return (nbytes + alignment - 1) & ~(alignment - 1)

    def _arena_allocate_locked(self, nbytes: int) -> Tuple[int, int]:
        allocation_nbytes = self._aligned_nbytes(nbytes)
        candidates = [
            (length, index, offset)
            for index, (offset, length) in enumerate(self._arena_free)
            if length >= allocation_nbytes
        ]
        if not candidates:
            largest = max((length for _, length in self._arena_free), default=0)
            raise MemoryError(
                f"GPU-direct arena cannot fit {allocation_nbytes} bytes; "
                f"largest free span is {largest} bytes"
            )
        length, index, offset = min(candidates)
        remaining = length - allocation_nbytes
        if remaining:
            self._arena_free[index] = (
                offset + allocation_nbytes,
                remaining,
            )
        else:
            self._arena_free.pop(index)
        return offset, allocation_nbytes

    def _arena_free_locked(self, offset: int, nbytes: int) -> None:
        if nbytes == 0:
            return
        self._arena_free.append((offset, nbytes))
        self._arena_free.sort()
        merged: List[Tuple[int, int]] = []
        for current_offset, current_length in self._arena_free:
            if merged and merged[-1][0] + merged[-1][1] == current_offset:
                prior_offset, prior_length = merged[-1]
                merged[-1] = (
                    prior_offset,
                    prior_length + current_length,
                )
            else:
                merged.append((current_offset, current_length))
        self._arena_free = merged

    def _allocate_buffer(self, source: torch.Tensor) -> _PublishedBuffer:
        source = source.detach().contiguous()
        nbytes = _nbytes(source)
        dtype = _DTYPE_STR.get(
            source.dtype, str(source.dtype).replace("torch.", "")
        )
        if self.backend in {"nvlink", "nvlink_intra"}:
            with self._lock:
                allocation_offset, allocation_nbytes = (
                    self._arena_allocate_locked(nbytes)
                )
            address = self._arena_address + allocation_offset
            _copy_to_address(
                address,
                source.data_ptr(),
                nbytes,
                stream=self._copy_stream,
            )
            source.record_stream(self._copy_stream)
            return _PublishedBuffer(
                address=address,
                nbytes=nbytes,
                allocation_offset=allocation_offset,
                allocation_nbytes=allocation_nbytes,
                shape=tuple(source.shape),
                dtype=dtype,
                tensor=None,
                registered=False,
            )

        staging = torch.empty_like(source, memory_format=torch.contiguous_format)
        staging.copy_(source, non_blocking=True)
        source.record_stream(self._copy_stream)
        status = self._engine.register_memory(staging.data_ptr(), nbytes)
        if status is not None and int(status) != 0:
            raise RuntimeError(
                f"Mooncake RDMA CUDA registration failed with status {status}"
            )
        return _PublishedBuffer(
            address=staging.data_ptr(),
            nbytes=nbytes,
            allocation_offset=0,
            allocation_nbytes=nbytes,
            shape=tuple(staging.shape),
            dtype=dtype,
            tensor=staging,
            registered=True,
        )

    def _free_buffer(self, buffer: _PublishedBuffer) -> None:
        if self.backend in {"nvlink", "nvlink_intra"}:
            with self._lock:
                self._arena_free_locked(
                    buffer.allocation_offset,
                    buffer.allocation_nbytes,
                )
                buffer.allocation_nbytes = 0
            status = 0
        elif buffer.registered:
            status = self._engine.unregister_memory(buffer.address)
        else:
            status = 0
        if status is not None and int(status) != 0:
            raise RuntimeError(
                f"Mooncake buffer release failed with status {status}"
            )
        buffer.tensor = None

    def _free_sample_locked(self, sample_id: str, generation: int) -> int:
        resident = self._residents.get(sample_id)
        if resident is None:
            return 0
        if resident.generation != generation:
            raise RuntimeError(
                f"stale release for {sample_id!r}: generation {generation} != "
                f"{resident.generation}"
            )
        freed = 0
        for buffer in resident.buffers.values():
            self._free_buffer(buffer)
            freed += buffer.nbytes
        self._residents.pop(sample_id, None)
        return freed

    def _prepare_tensors(
        self,
        spec: Dict[str, Any],
        aux: Optional[torch.Tensor],
        last_hidden: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        requested_transport = spec.get("transport")
        if requested_transport is not None and str(requested_transport) != self.backend:
            raise ValueError(
                f"request expects transport {requested_transport!r}; "
                f"server uses {self.backend!r}"
            )
        features = dict(spec.get("features") or {})
        tensors: Dict[str, torch.Tensor] = {}
        aux_name = features.get("aux")
        if aux_name is not None:
            if aux is None:
                raise RuntimeError("spec_capture requested aux but none was captured")
            tensors[str(aux_name)] = aux.unsqueeze(0)
        last_name = features.get("last_hidden")
        if last_name is not None:
            if last_hidden is None:
                raise RuntimeError(
                    "spec_capture requested last_hidden but none was captured"
                )
            tensors[str(last_name)] = last_hidden.unsqueeze(0)

        for item in spec.get("passthrough") or []:
            dtype = _STR_DTYPE.get(str(item.get("dtype", "int64")))
            if dtype is None:
                raise TypeError(
                    f"unsupported passthrough dtype {item.get('dtype')!r}"
                )
            tensors[str(item["name"])] = torch.tensor(
                item["data"],
                dtype=dtype,
                device=self.device,
            ).reshape([int(dim) for dim in item["shape"]])
        if not tensors:
            raise ValueError("spec_capture request contains no features")
        return tensors

    def _stage_samples(self, samples):
        prepared = []
        new_bytes = 0
        new_allocated_bytes = 0
        for spec, aux, last_hidden in samples:
            tensors = self._prepare_tensors(spec, aux, last_hidden)
            prepared.append((spec, tensors))
            new_bytes += sum(_nbytes(tensor) for tensor in tensors.values())
            new_allocated_bytes += sum(
                self._aligned_nbytes(_nbytes(tensor))
                for tensor in tensors.values()
            )

        # Reserve capacity before calling into Mooncake.  The previous ordering
        # allocated every buffer first and checked the cap afterwards, allowing
        # the native allocator to exhaust device memory before backpressure.
        with self._lock:
            for spec, _ in prepared:
                prior = self._residents.get(str(spec["sample_id"]))
                if prior is not None:
                    if not bool(spec.get("replace", False)):
                        raise RuntimeError(
                            f"sample {spec['sample_id']!r} is already resident"
                        )
            projected = (
                self._resident_allocated_bytes_locked()
                + self._reserved_bytes
                + new_allocated_bytes
            )
            if (
                self.max_resident_bytes is not None
                and projected > self.max_resident_bytes
            ):
                raise MemoryError(
                    f"GPU-direct capture residency {projected} exceeds "
                    f"{self.max_resident_bytes} bytes"
                )
            self._reserved_bytes += new_allocated_bytes

        staged = []
        source_stream = torch.cuda.current_stream(self.device)
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_stream(source_stream)
                try:
                    for spec, tensors in prepared:
                        buffers: Dict[str, _PublishedBuffer] = {}
                        try:
                            for name, tensor in tensors.items():
                                buffers[name] = self._allocate_buffer(tensor)
                        except BaseException:
                            self._copy_stream.synchronize()
                            for buffer in buffers.values():
                                self._free_buffer(buffer)
                            raise
                        staged.append((spec, buffers))
                    done = torch.cuda.Event()
                    done.record(self._copy_stream)
                except BaseException:
                    self._copy_stream.synchronize()
                    for _, buffers in staged:
                        for buffer in buffers.values():
                            self._free_buffer(buffer)
                    raise

            with self._lock:
                for spec, _ in staged:
                    sample_id = str(spec["sample_id"])
                    prior = self._residents.get(sample_id)
                    if prior is not None:
                        self._free_sample_locked(sample_id, prior.generation)
        except BaseException:
            with self._lock:
                self._reserved_bytes -= new_allocated_bytes
            raise
        return done, staged, new_allocated_bytes

    def _commit_samples(self, event: torch.cuda.Event, staged, reserved_bytes: int):
        committed_sample_ids = []
        reservation_released = False
        try:
            event.synchronize()
            results = []
            with self._lock:
                for spec, buffers in staged:
                    sample_id = str(spec["sample_id"])
                    store_id = str(spec["store_id"])
                    generation = int(spec.get("gen", 1))
                    result = {
                        "sample_id": sample_id,
                        "store_id": store_id,
                        "gen": generation,
                        "transport": self.backend,
                        "session_id": self.session_id,
                        "control_endpoint": self.control_endpoint,
                        "control_token": self._token,
                        "aux_layer_ids": self.aux_layer_ids,
                        "features": {
                            name: buffer.descriptor()
                            for name, buffer in buffers.items()
                        },
                    }
                    self._residents[sample_id] = _ResidentSample(
                        generation=generation,
                        buffers=buffers,
                        result=result,
                    )
                    committed_sample_ids.append(sample_id)
                    results.append(result)
                    sample_bytes = sum(
                        buffer.nbytes for buffer in buffers.values()
                    )
                    self._stats["published_samples"] += 1
                    self._stats["device_staging_bytes"] += sample_bytes
                self._reserved_bytes -= reserved_bytes
                reservation_released = True
            return results
        except BaseException:
            with self._lock:
                if not reservation_released:
                    self._reserved_bytes -= reserved_bytes
                for sample_id in committed_sample_ids:
                    self._residents.pop(sample_id, None)
                for _, buffers in staged:
                    for buffer in buffers.values():
                        try:
                            self._free_buffer(buffer)
                        except Exception:
                            logger.exception("failed to reclaim staged capture buffer")
            raise

    def submit_samples(self, samples) -> Future[List[Dict[str, Any]]]:
        try:
            event, staged, reserved_bytes = self._stage_samples(samples)
        except BaseException as exc:
            future: Future[List[Dict[str, Any]]] = Future()
            future.set_exception(exc)
            return future
        return self._executor.submit(
            self._commit_samples, event, staged, reserved_bytes
        )

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": self.backend,
                "session_id": self.session_id,
                "control_endpoint": self.control_endpoint,
                "resident_samples": len(self._residents),
                "resident_bytes": self._resident_bytes_locked(),
                "resident_allocated_bytes": self._resident_allocated_bytes_locked(),
                "reserved_bytes": self._reserved_bytes,
                "arena_address": self._arena_address,
                "arena_bytes": self._arena_bytes,
                "arena_free_bytes": sum(length for _, length in self._arena_free),
                "arena_largest_free_span": max(
                    (length for _, length in self._arena_free), default=0
                ),
                **self._stats,
            }

    def handle_control(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._stats["control_requests"] += 1
        if request.get("op") == "health":
            return {"ok": True, "health": self.health()}
        if request.get("token") != self._token:
            return {"ok": False, "error": "invalid control token"}
        operation = request.get("op")
        if operation not in {"release", "abort", "release_batch", "abort_batch"}:
            return {"ok": False, "error": "unsupported control operation"}
        if operation in {"release_batch", "abort_batch"}:
            items = request.get("items")
            if not isinstance(items, list) or not items:
                return {"ok": False, "error": "control batch requires items"}
            if len(items) > _MAX_CONTROL_BATCH:
                return {
                    "ok": False,
                    "error": (
                        f"control batch has {len(items)} items; "
                        f"limit is {_MAX_CONTROL_BATCH}"
                    ),
                }
            normalized = [
                (str(item["sample_id"]), int(item["generation"]))
                for item in items
            ]
        else:
            normalized = [
                (str(request["sample_id"]), int(request["generation"]))
            ]
        with self._lock:
            # Validate every live generation before mutating any resident entry.
            # This makes a retry after a failed batch safe and deterministic.
            for sample_id, generation in normalized:
                resident = self._residents.get(sample_id)
                if resident is not None and resident.generation != generation:
                    raise RuntimeError(
                        f"stale release for {sample_id!r}: generation "
                        f"{generation} != {resident.generation}"
                    )
            freed = sum(
                self._free_sample_locked(sample_id, generation)
                for sample_id, generation in normalized
            )
            released = len(normalized)
            self._stats["released_samples"] += released
            if operation in {"release_batch", "abort_batch"}:
                self._stats["control_batch_requests"] += 1
                self._stats["control_batch_items"] += released
        return {"ok": True, "freed_bytes": freed, "released_samples": released}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            for sample_id, resident in list(self._residents.items()):
                self._free_sample_locked(sample_id, resident.generation)
        if self._arena_address:
            with self._engine_lock:
                status = self._engine.free_managed_buffer(
                    self._arena_address, self._arena_bytes
                )
            if status is not None and int(status) != 0:
                logger.error(
                    "Mooncake arena release failed with status %s", status
                )
            self._arena_address = 0
        self._control_server.shutdown()
        self._control_server.server_close()
        self._control_thread.join(timeout=5.0)


_SINK: Optional[SpecCaptureSink] = None


def maybe_init_sink(server_args) -> None:
    global _SINK
    if server_args.enable_spec_capture and _SINK is None:
        _SINK = SpecCaptureSink(server_args)


def get_sink() -> Optional[SpecCaptureSink]:
    return _SINK


__all__ = ["SpecCaptureSink", "get_sink", "maybe_init_sink"]
