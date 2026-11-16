"""HIXL one-sided expert reads. All registrations own their backing storage."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _load_bridge():
    from torch.utils.cpp_extension import load

    home = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    if not (home / "include/hixl/hixl.h").is_file():
        raise RuntimeError("Async EPLB requires CANN HIXL headers and libcann_hixl.so")
    library = load(
        name="sglang_hixl_bridge",
        sources=[str(Path(__file__).with_name("hixl_bridge.cpp"))],
        extra_include_paths=[str(home / "include")],
        extra_ldflags=[
            f"-L{home / 'lib64'}",
            "-lcann_hixl",
            "-lmetadef",
            f"-Wl,-rpath,{home / 'lib64'}",
        ],
        is_python_module=False,
    )
    lib = ctypes.CDLL(library)
    pointer, string, integer = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
    signatures = {
        "create": [string, ctypes.POINTER(pointer)],
        "register": [
            pointer,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(pointer),
        ],
        "connect": [pointer, string, integer],
        "read": [
            pointer,
            string,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            integer,
        ],
        "disconnect": [pointer, string, integer],
        "deregister": [pointer, pointer],
        "destroy": [pointer],
    }
    for name, args in signatures.items():
        fn = getattr(lib, "sgl_hixl_" + name)
        fn.argtypes, fn.restype = args, ctypes.c_uint
    return lib


class HixlExpertTransport:
    def __init__(self, weights, staging, group, endpoint: str, timeout: float = 60):
        import acl

        self.weights, self.staging = weights, staging
        self.group, self.timeout_ms = group, int(timeout * 1000)
        self.rank, self.size = dist.get_rank(group), dist.get_world_size(group)
        self._engine = ctypes.c_void_p()
        self._handles, self._storage_refs, self._peers = [], [], []
        self._poisoned = False
        self._context, status = acl.rt.get_context()
        self._check(status, "get ACL context")
        # All phases agree on failure before entering the next collective.
        self._phase(lambda: setattr(self, "_lib", _load_bridge()))
        self._phase(
            lambda: self._check(
                self._lib.sgl_hixl_create(
                    endpoint.encode(), ctypes.byref(self._engine)
                ),
                "initialize",
            )
        )
        self._phase(self._register)
        local = {
            layer: [(t.data_ptr(), t.shape[0], t.shape[1]) for t in tensors]
            for layer, tensors in weights.items()
        }
        self.remote = [None] * self.size
        dist.all_gather_object(self.remote, (endpoint, local), group=group)

        def connect():
            for rank, (peer, metadata) in enumerate(self.remote):
                if metadata.keys() != local.keys() or any(
                    [v[1:] for v in metadata[layer]] != [v[1:] for v in local[layer]]
                    for layer in local
                ):
                    raise ValueError("HIXL expert geometry differs across ranks")
                if rank != self.rank:
                    self._check(
                        self._lib.sgl_hixl_connect(
                            self._engine, peer.encode(), self.timeout_ms
                        ),
                        "connect",
                    )
                    self._peers.append(peer)

        self._phase(connect)

    def _phase(self, operation):
        error = None
        try:
            operation()
        except Exception as exc:
            error = exc
        flag = torch.tensor([int(error is not None)], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
        if flag.item():
            # Startup failure is fatal; retain the partial registration owner.
            _FAILED_TRANSPORTS.append(self)
            raise RuntimeError(
                "HIXL initialization failed on at least one rank"
            ) from error

    def _check(self, status, operation):
        if status:
            self._poisoned = True
            raise RuntimeError(f"HIXL {operation} failed: status={status}")

    def _register(self):
        # CANN registrations use 2 MiB pages. Aliases already include NZ bytes.
        alignment = 2 << 20
        spans = []
        for tensors in [*self.weights.values(), self.staging]:
            for tensor in tensors:
                storage = tensor.untyped_storage()
                start, end = tensor.data_ptr(), tensor.data_ptr() + tensor.numel()
                if (
                    not storage.data_ptr()
                    <= start
                    < end
                    <= storage.data_ptr() + storage.nbytes()
                ):
                    raise ValueError("Expert byte span exceeds its backing storage")
                self._storage_refs.append(storage)
                spans.append(
                    (
                        start // alignment * alignment,
                        (end + alignment - 1) // alignment * alignment,
                    )
                )
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            handle = ctypes.c_void_p()
            self._check(
                self._lib.sgl_hixl_register(
                    self._engine, start, end - start, ctypes.byref(handle)
                ),
                "register",
            )
            self._handles.append(handle)

    def read(self, layer: int, reads) -> None:
        import acl

        self._check(acl.rt.set_context(self._context), "restore ACL context")
        by_peer = {}
        for rank, source, destination in reads:
            descriptors = by_peer.setdefault(rank, [])
            for buffer, (address, _, row_bytes) in zip(
                self.staging, self.remote[rank][1][layer], strict=True
            ):
                descriptors.extend(
                    (
                        buffer[destination].data_ptr(),
                        address + source * row_bytes,
                        row_bytes,
                    )
                )
        for rank, triples in by_peer.items():
            payload = (ctypes.c_uint64 * len(triples))(*triples)
            # ctypes releases the GIL while the native worker waits for DMA.
            self._check(
                self._lib.sgl_hixl_read(
                    self._engine,
                    self.remote[rank][0].encode(),
                    payload,
                    len(triples) // 3,
                    self.timeout_ms,
                ),
                "READ",
            )

    def drain(self):
        # The coordinator joins the sole read worker before calling this.
        # An error/timeout does not establish that external DMA has stopped.
        if self._poisoned:
            raise RuntimeError(
                "HIXL failed; registrations must survive until worker exit"
            )

    def close(self):
        self.drain()
        for peer in self._peers:
            self._check(
                self._lib.sgl_hixl_disconnect(
                    self._engine, peer.encode(), self.timeout_ms
                ),
                "disconnect",
            )
        dist.barrier(group=self.group)
        for handle in reversed(self._handles):
            self._check(
                self._lib.sgl_hixl_deregister(self._engine, handle), "deregister"
            )
        self._check(self._lib.sgl_hixl_destroy(self._engine), "finalize")
        self._handles.clear()
        self._storage_refs.clear()


_FAILED_TRANSPORTS = []
