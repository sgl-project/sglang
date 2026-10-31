"""HiCache row storage backed by Mooncake's local EngramStore."""

import contextlib
import logging

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheEngramStorage

logger = logging.getLogger(__name__)


class MooncakeEngramStorage(HiCacheEngramStorage):
    def __init__(
        self, *, local_dir: str, remote=None, head_bounds=None, namespace=None
    ):
        self._local_dir = local_dir
        self._store = None
        self.remote = remote
        self._namespace = namespace
        if remote is not None:
            if (
                not namespace
                or not head_bounds
                or head_bounds[0] != 0
                or any(a >= b for a, b in zip(head_bounds, head_bounds[1:]))
            ):
                raise ValueError(
                    "Remote Engram requires a named table and increasing head boundaries"
                )
            heads = len(head_bounds) - 1
            shards, shard = remote["num_shards"], remote["shard_id"]
            if shards > heads:
                raise ValueError("More node shards than Engram hash heads")
            self._bounds = tuple(head_bounds)
            self._head_begin, self._head_end = (
                heads * shard // shards,
                heads * (shard + 1) // shards,
            )
            self.row_start, self.row_end = (
                head_bounds[self._head_begin],
                head_bounds[self._head_end],
            )
            self._names = tuple(f"head{i}" for i in range(heads))
            self._owners = {
                name: ((i + 1) * shards - 1) // heads
                for i, name in enumerate(self._names)
            }

    def initialize(self, rows: torch.Tensor) -> None:
        from mooncake.engram import EngramStore

        if self._store is not None:
            raise RuntimeError("Engram tables are immutable; already initialized")
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up Engram tables before CUDA graph capture")
        if rows.device.type != "cpu" or rows.dtype != torch.uint8 or rows.ndim != 2:
            raise ValueError("Engram initialization requires a CPU byte matrix")
        row_bytes = rows.shape[1]
        if self.remote is not None:
            return self._initialize_remote(rows, row_bytes)
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

    def _initialize_remote(self, rows, row_bytes):
        from mooncake.engram import EngramStore

        if rows.shape[0] != self.row_end - self.row_start:
            raise ValueError("Loaded rows must match this node's storage heads")
        tables = {
            name: {
                "num_rows": self._bounds[i + 1] - self._bounds[i],
                "row_bytes": row_bytes,
            }
            for i, name in enumerate(self._names)
        }
        local = {
            self._names[i]: rows[
                self._bounds[i] - self.row_start : self._bounds[i + 1] - self.row_start
            ]
            for i in range(self._head_begin, self._head_end)
        }
        store = EngramStore(
            tables=tables,
            namespace=self._namespace,
            local_dir=self._local_dir,
            persistent=False,
        )
        options = dict(self.remote)
        options.setdefault("caller_id", torch.cuda.current_device())
        if "rdma_device" in options:
            options["rdma_device"] = options["rdma_device"].format(
                local_rank=torch.cuda.current_device()
            )
        try:
            store.initialize(local, table_owners=self._owners, **options)
        except BaseException:
            store.close()
            raise
        self._store, self._row_bytes = store, row_bytes
        logger.info(
            "Mooncake Engram %s: node shard %s/%s, stored heads [%s,%s)",
            self._namespace,
            options["shard_id"],
            options["num_shards"],
            self._head_begin,
            self._head_end,
        )

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        if self._store is None:
            raise RuntimeError("Initialize Engram storage before lookup")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Engram row IDs must be int32 or int64")
        if self.remote is not None:
            if not ids.is_cuda or ids.numel() % len(self._names):
                raise ValueError(
                    "Remote lookup requires CUDA token-major hash-head IDs"
                )
            indices = ids.reshape(-1, len(self._names))
            output = torch.zeros(
                (*indices.shape, self._row_bytes), dtype=torch.uint8, device=ids.device
            )
            self._store.lookup(
                self._names,
                indices,
                output,
                offsets=self._bounds[:-1],
                stream=torch.cuda.current_stream(ids.device).cuda_stream,
            )
            return output.view(*ids.shape, self._row_bytes)
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
