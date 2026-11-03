"""One launch over borrowed independent QPs; callers close group before contexts."""

import ctypes as C

import torch


class ReadGroup:
    def __init__(self, contexts):
        if not 1 <= len(contexts) <= 16 or len({id(c) for c in contexts}) != len(
            contexts
        ):
            raise ValueError("need1..16distinct contexts")
        if any(not c.handle or c.gpu != contexts[0].gpu for c in contexts):
            raise ValueError("READ contexts must be live on one GPU")
        if any(len(c.buffers) != 1 or not c.buffers[0].is_cuda for c in contexts):
            raise ValueError("READ group requires registered GPU destinations")
        first = contexts[0].buffers[0]
        if any(
            c.buffers[0].data_ptr() != first.data_ptr()
            or c.buffers[0].numel() != first.numel()
            for c in contexts
        ):
            raise ValueError("READ group must share one destination allocation")
        self.contexts = list(contexts)
        self.gpu = contexts[0].gpu
        self.lib = contexts[0].library
        self.lib.engram_read_group_create.argtypes = [C.POINTER(C.c_void_p), C.c_uint32]
        self.lib.engram_read_group_create.restype = C.c_void_p
        self.lib.engram_read_group_fetch.argtypes = [
            C.c_void_p,
            C.c_uint64,
            C.c_uint32,
            C.c_size_t,
            C.c_size_t,
            C.c_uint32,
            C.c_uint32,
            C.c_uint64,
            C.c_uint64,
            C.c_uint64,
            C.c_uint64,
            C.c_int,
        ]
        self.lib.engram_read_group_destroy.argtypes = [C.c_void_p]
        if hasattr(self.lib, "engram_read_group_fetch_mapped"):
            self.lib.engram_read_group_fetch_mapped.argtypes = (
                self.lib.engram_read_group_fetch.argtypes + [C.c_uint64]
            )
        if hasattr(self.lib, "engram_read_group_fetch_compacted"):
            self.lib.engram_read_group_fetch_compacted.argtypes = (
                self.lib.engram_read_group_fetch.argtypes + [C.c_uint64] * 3
            )
        handles = (C.c_void_p * len(contexts))(*(c.handle for c in contexts))
        self.handle = self.lib.engram_read_group_create(handles, len(contexts))
        if not self.handle:
            raise RuntimeError(self.lib.engram_last_error().decode())
        self.active = 0

    def fetch(
        self,
        ids,
        count,
        payload,
        dump,
        table,
        window,
        stats,
        lo,
        hi,
        pending=False,
        representatives=None,
        query_list=None,
        list_count=None,
    ):
        if not self.handle or any(not c.handle for c in self.contexts):
            raise RuntimeError("closed READ group/context")
        if (
            ids.dtype != torch.int64
            or not ids.is_cuda
            or not ids.is_contiguous()
            or not 0 < count <= ids.numel()
        ):
            raise ValueError("invalid grouped READ IDs")
        if (
            stats.dtype != torch.uint64
            or not stats.is_cuda
            or not stats.is_contiguous()
            or tuple(stats.shape) != (len(self.contexts), 8)
        ):
            raise ValueError("invalid grouped READ stats")
        if (
            ids.device.index != self.gpu
            or stats.device.index != self.gpu
            or not 0 <= lo <= hi
            or min(payload, dump) < 0
        ):
            raise ValueError("invalid grouped READ device/range")
        if not all(0 <= table < c.remote_count for c in self.contexts):
            raise ValueError("invalid source table")
        stream = torch.cuda.current_stream(self.gpu)
        args = (
            self.handle,
            ids.data_ptr(),
            count,
            payload,
            dump,
            table,
            window,
            stats.data_ptr(),
            stream.cuda_stream,
            lo,
            hi,
            int(pending),
        )
        if representatives is not None:
            if (
                representatives.dtype != torch.int64
                or representatives.shape != ids.shape
                or representatives.device != ids.device
                or not representatives.is_contiguous()
            ):
                raise ValueError("invalid representatives")
            if query_list is not None:
                if (
                    query_list.dtype != torch.int64
                    or query_list.shape != ids.shape
                    or query_list.device != ids.device
                    or list_count.dtype != torch.uint64
                    or list_count.numel() != 1
                    or list_count.device != ids.device
                ):
                    raise ValueError("invalid compacted work list")
                self.contexts[0].check(
                    self.lib.engram_read_group_fetch_compacted(
                        *args,
                        representatives.data_ptr(),
                        query_list.data_ptr(),
                        list_count.data_ptr(),
                    )
                )
                query_list.record_stream(stream)
                list_count.record_stream(stream)
            else:
                self.contexts[0].check(
                    self.lib.engram_read_group_fetch_mapped(
                        *args, representatives.data_ptr()
                    )
                )
            representatives.record_stream(stream)
        else:
            self.contexts[0].check(self.lib.engram_read_group_fetch(*args))
        ids.record_stream(stream)
        stats.record_stream(stream)
        capacity = min(count, len(self.contexts))
        chunk = (count + capacity - 1) // capacity
        self.active = (count + chunk - 1) // chunk
        for i in range(self.active):
            self.contexts[i].last_buffers = (
                ids,
                stats,
                representatives,
                query_list,
                list_count,
            )

    def set_pipeline(self, depth):
        if depth not in (1, 2):
            raise ValueError("completion depth must be1or2")
        if not hasattr(self.lib, "engram_read_group_pipeline"):
            if depth == 1:
                return
            raise RuntimeError("runtime lacks FIFO pipeline capability")
        self.lib.engram_read_group_pipeline.argtypes = [C.c_void_p, C.c_uint32]
        self.contexts[0].check(self.lib.engram_read_group_pipeline(self.handle, depth))

    def close(self):
        if self.handle:
            self.contexts[0].check(self.lib.engram_read_group_destroy(self.handle))
            self.handle = None
