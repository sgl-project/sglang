"""HiCache row storage backed by Mooncake's local EngramStore."""

import contextlib
import logging

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheEngramStorage

logger = logging.getLogger(__name__)


class MooncakeEngramStorage(HiCacheEngramStorage):
    def __init__(self, *, local_dir: str):
        self._local_dir = local_dir
        self._store = None

    def initialize(self, rows: torch.Tensor) -> None:
        from mooncake.engram import EngramStore

        if self._store is not None:
            raise RuntimeError("Engram tables are immutable; already initialized")
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up Engram tables before CUDA graph capture")
        if rows.device.type != "cpu" or rows.dtype != torch.uint8 or rows.ndim != 2:
            raise ValueError("Engram initialization requires a CPU byte matrix")
        row_bytes = rows.shape[1]
        store = EngramStore(
            tables={"rows": {"num_rows": rows.shape[0], "row_bytes": row_bytes}},
            namespace="sglang-v1",
            local_dir=self._local_dir,
            persistent=False,
        )
        try:
            store.populate("rows", rows.detach().contiguous().numpy())
        except BaseException:
            store.close()
            raise
        self._row_bytes = row_bytes
        self._store = store
        logger.info("Mooncake Engram: published %s rows of %s bytes", *rows.shape)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        if self._store is None:
            raise RuntimeError("Initialize Engram storage before lookup")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Engram row IDs must be int32 or int64")
        indices = ids.reshape(-1, 1).contiguous()
        output = torch.zeros(
            (indices.shape[0], 1, self._row_bytes),
            dtype=torch.uint8,
            device=ids.device,
        )
        # Empty idle batches must not launch a zero-sized CUDA grid.
        if indices.numel():
            if ids.is_cuda:
                self._store.lookup(
                    ["rows"],
                    indices,
                    output,
                    stream=torch.cuda.current_stream(ids.device).cuda_stream,
                )
            else:
                self._store.lookup(["rows"], indices.numpy(), output.numpy())
        return output.view(*ids.shape, self._row_bytes)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
