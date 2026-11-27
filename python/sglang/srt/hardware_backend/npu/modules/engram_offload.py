"""NPU Engram offload through MemFabric SHARED DRAM pools.

Owners pack layer chunks into one slot. A3 reads packed codes/scales directly;
A5 uses entry_gather with device-resident scales. Independent process groups
synchronize loading through a separate TCPStore.
"""

from __future__ import annotations

import atexit
import ctypes
import datetime
import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.arg_groups.engram_config import EngramPoolConfig
from sglang.srt.hardware_backend.npu.utils import is_ascend_910c
from sglang.srt.layers.engram import EngramLayout, _e8m0_to_float
from sglang.srt.mem_cache.pool_host.npu_host_memory import (
    memfabric_host_memory_enabled,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils.device import is_device_stream_capturing

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# One row must fit one 120KB UB ping-pong slot inside the entry_gather AIV
# kernel (OFFLOAD_ENTRY_GATHER_MAX_ENTRY_BYTES in acc_offload.h).
_MAX_ENTRY_BYTES = 120 * 1024
# The share pool holds at most 16 node-local ranks.
_MAX_LOCAL_WORLD = 16


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _host_tensor(addr: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A torch view over pool memory at `addr` (same trick as offload.empty)."""
    numel = 1
    for size in shape:
        numel *= size
    itemsize = torch.tensor([], dtype=dtype).element_size()
    buf = (ctypes.c_int8 * (numel * itemsize)).from_address(addr)
    return torch.frombuffer(buf, dtype=dtype).reshape(shape)


def _current_device_id() -> int:
    return torch.npu.current_device()


@dataclass(frozen=True)
class EngramOffloadTable:
    """Layer chunks use base + owner_ranks[i // rows] * slot + block_offset
    + (i % rows) * entry_bytes for logical row i.
    """

    num_embeddings: int
    # host view of this rank's chunk, [rows, entry_bytes] uint8
    pool_weight: torch.Tensor
    # chunk rows per slot (ceil sharding; the last chunk's tail is unused)
    rows: int
    # chunk start inside every slot
    block_offset: int
    # row pitch in bytes (head_dim for arch35 fp8)
    entry_bytes: int


class EngramOffloadManager:
    """Owns the process-wide acc_offload SHARED entity and the engram layout."""

    def __init__(
        self,
        entity,
        local_world: int,
        rank: int,
        slot: int,
        tables: list[EngramOffloadTable],
        owner_ranks: list[int],
        load_store=None,
    ):
        self._entity = entity
        self.local_world = local_world
        self.rank = rank
        self.slot = slot
        self._tables = tables
        self.owner_ranks = owner_ranks
        self.owner_index = owner_ranks.index(rank) if rank in owner_ranks else None
        self.rows_per_slot = sum(table.rows for table in tables)
        self._owners: Optional[torch.Tensor] = None
        self._load_store = load_store

    def prepare_owners(self, device: torch.device) -> None:
        if self._owners is None and self.owner_ranks != list(
            range(len(self.owner_ranks))
        ):
            self._owners = torch.tensor(
                self.owner_ranks, dtype=torch.int64, device=device
            )

    def encode_ids(self, ids: torch.Tensor, table: EngramOffloadTable) -> torch.Tensor:
        self.prepare_owners(ids.device)
        owner = ids // table.rows
        if self._owners is not None:
            owner = self._owners[owner]
        return (
            owner * self.rows_per_slot
            + table.block_offset // table.entry_bytes
            + ids % table.rows
        )

    def finish_load(self, layer_index: int) -> None:
        if self._load_store is not None:
            self._load_store.set(f"loaded/{layer_index}/{self.rank}", b"1")
            self._load_store.wait(
                [f"loaded/{layer_index}/{rank}" for rank in self.owner_ranks]
            )

    def table(self, layer_hash_index: int) -> EngramOffloadTable:
        return self._tables[layer_hash_index]

    def entry_gather(self, dst, ids: torch.Tensor, count, device: torch.device) -> int:
        """Async gather on the current NPU stream; ordering with the following
        dequant ops comes from the stream itself."""
        return self._entity.entry_gather(dst, ids, count, device)

    def uninitialize(self) -> None:
        logger.info(
            "[engram_offload] rank %d: pool uninitialize (collective teardown)",
            self.rank,
        )
        self._entity.uninitialize()


def _node_local_pool_coords(tp_size: int, tp_rank: int) -> tuple[int, int]:
    """Dense coordinates across all scheduler processes sharing this node."""
    world = os.environ.get("SGLANG_NODE_LOCAL_WORLD")
    rank = os.environ.get("SGLANG_NODE_LOCAL_RANK")
    if world is not None or rank is not None:
        if world is None or rank is None:
            raise ValueError(
                "Set both SGLANG_NODE_LOCAL_WORLD and SGLANG_NODE_LOCAL_RANK"
            )
        return int(world), int(rank)
    parallel = get_parallel()
    if parallel.pp_size != 1 or tp_size % parallel.nnodes:
        raise RuntimeError(
            "Engram pool needs explicit node-local coordinates for this TP/PP layout"
        )
    local_tp = tp_size // parallel.nnodes
    replica = (parallel.dp_rank or 0) if parallel.dp_size > 1 else 0
    # LOCAL_RANK belongs to the parent launcher and is inherited by every worker.
    return local_tp * parallel.dp_size, replica * local_tp + tp_rank % local_tp


def _make_offload_entity():
    """Keep Engram separate from HiCache when the wheel supports instances."""
    try:
        from memfabric_hybrid import offload as offload_module
    except ImportError as e:
        raise RuntimeError(
            "the NPU engram host offload needs the memfabric_hybrid "
            "acc_offload module: install the memfabric-hybrid wheel and point "
            f"MEMFABRIC_HYBRID_EXTEND_LIB_PATH at the built kernel library "
            f"(import failed: {e}); use --engram-config placement 'gpu' for "
            "device-resident Engram tables"
        ) from e
    if hasattr(offload_module, "Offload"):
        return offload_module.Offload()
    if memfabric_host_memory_enabled():
        raise RuntimeError(
            "engram host offload cannot share the memfabric acc_offload "
            "entity with SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric on this "
            "memfabric_hybrid wheel (no offload.Offload multi-instance API); "
            "upgrade the wheel, or unset one of the two switches"
        )
    logger.warning(
        "[engram_offload] this memfabric_hybrid wheel has no offload.Offload "
        "multi-instance API; using the module-level default entity (upgrade "
        "the wheel to coexist with the HiCache memfabric host pool)"
    )
    return offload_module


def _resolve_pool_topology(pool: EngramPoolConfig):
    parallel = get_parallel()
    # Same-node PP stages load different layers into the same slot offsets.
    assert max(parallel.pp_size // parallel.nnodes, 1) == 1, (
        "engram offload needs every node-local scheduler in one TP group; "
        f"same-node pipeline parallelism is unsupported (pp_size="
        f"{parallel.pp_size}, nnodes={parallel.nnodes})"
    )
    local_world, rank = _node_local_pool_coords(parallel.tp_size, parallel.tp_rank)
    assert 1 <= local_world <= _MAX_LOCAL_WORLD, (
        f"engram offload pool needs 1..{_MAX_LOCAL_WORLD} node-local ranks, "
        f"got {local_world}"
    )

    world = pool.world_size or local_world
    rank_offset = pool.rank_offset
    if not 0 <= rank < local_world:
        raise ValueError("Engram node-local rank must be within local_world_size")
    rank += rank_offset
    if (
        world < local_world
        or world % local_world
        or rank_offset % local_world
        or not 0 <= rank < world
    ):
        raise ValueError(
            "Engram pool ranks must be dense, machine-contiguous and within world_size"
        )
    owner_ranks = (
        pool.owner_ranks if pool.owner_ranks is not None else list(range(world))
    )
    if (
        not owner_ranks
        or len(set(owner_ranks)) != len(owner_ranks)
        or any(r < 0 or r >= world for r in owner_ranks)
    ):
        raise ValueError(
            "Engram owner ranks must be unique pool ranks within world_size"
        )
    address = urlsplit(pool.store_url)
    independent_groups = world != local_world or parallel.dp_size > 1
    if independent_groups and (
        address.scheme != "tcp"
        or not address.hostname
        or not address.port
        or address.port >= 65535
    ):
        raise ValueError(
            "Cross-instance or ordinary-DP Engram pools require an explicit tcp://host:port store URL"
        )
    return (
        local_world,
        world,
        rank,
        owner_ranks,
        address if independent_groups else None,
    )


def _create_pool_tables(chunks, slot_base: int, entry_bytes: int, rank: int):
    tables = []
    for layer_hash_index, (block_offset, rows, num_embeddings) in enumerate(chunks):
        logger.info(
            "[engram_offload] rank %d: registered table %d (E=%d "
            "rows_per_owner=%d block_off=%d entry=%dB)",
            rank,
            layer_hash_index,
            num_embeddings,
            rows,
            block_offset,
            entry_bytes,
        )
        tables.append(
            EngramOffloadTable(
                num_embeddings=num_embeddings,
                pool_weight=(
                    _host_tensor(
                        slot_base + block_offset, (rows, entry_bytes), torch.uint8
                    )
                    if slot_base
                    else torch.empty((0, entry_bytes), dtype=torch.uint8, device="cpu")
                ),
                rows=rows,
                block_offset=block_offset,
                entry_bytes=entry_bytes,
            )
        )
    return tables


def _create_load_store(
    address, world: int, rank: int, layout: EngramLayout, owner_ranks
):
    # Independent TP groups must wait for every owner before reading the pool.
    load_store = torch.distributed.TCPStore(
        address.hostname,
        address.port + 1,
        world,
        rank == 0,
        timeout=datetime.timedelta(minutes=30),
        wait_for_workers=False,
    )
    fingerprint = repr((layout, tuple(owner_ranks))).encode()
    load_store.set(f"layout/{rank}", fingerprint)
    keys = [f"layout/{peer}" for peer in range(world)]
    load_store.wait(keys)
    if any(load_store.get(key) != fingerprint for key in keys):
        raise ValueError("Engram pool members disagree on table layout or owner ranks")
    return load_store


def _plan_pool_layout(layout: EngramLayout, owner_count: int):
    entry_bytes = layout.head_dim
    if is_ascend_910c():
        entry_bytes += layout.head_dim // FP8_BLOCK_SIZE
    assert 1 <= entry_bytes <= _MAX_ENTRY_BYTES, (
        f"engram row {entry_bytes}B exceeds the {_MAX_ENTRY_BYTES}B entry_gather limit"
    )
    # Each owner packs the same layer offsets; ceil sharding leaves unused tail rows.
    chunks = []
    offset = 0
    for num_embeddings in layout.num_embeddings:
        rows = -(-num_embeddings // owner_count)
        chunks.append((offset, rows, num_embeddings))
        offset += rows * entry_bytes
    used = offset
    slot = _align_up(used, _GIB)
    return entry_bytes, chunks, used, slot


def _allocate_pool(entity, config, entry_bytes: int, rows_per_slot: int):
    ret = entity.initialize(config)
    if ret != 0:
        detail = ""
        err_fn = getattr(entity, "get_last_err_msg", None)
        if err_fn is not None:
            detail = f": {err_fn()}"
        raise RuntimeError(
            f"[engram_offload] offload.initialize failed ({ret}){detail}; on "
            "the SHARED scene every rank of the node must pass the same "
            "config (check --engram-config pool)"
        )

    slot_base = entity.malloc(config.alloc_size, 0) if config.alloc_size else 0
    if config.alloc_size and slot_base == 0:
        raise RuntimeError(
            f"[engram_offload] offload.malloc({config.alloc_size}) failed: host DRAM "
            "exhausted for the engram pool"
        )

    ret = entity.register_entry_table(entry_bytes, rows_per_slot)
    if ret != 0:
        raise RuntimeError(f"[engram_offload] register_entry_table failed ({ret})")
    return slot_base


def _open_engram_offload(layout: EngramLayout) -> EngramOffloadManager:
    pool = get_exec().offload.engram_config.pool
    local_world, world, rank, owner_ranks, load_address = _resolve_pool_topology(pool)
    is_owner = rank in owner_ranks
    store_url = pool.store_url

    entity = _make_offload_entity()
    entry_bytes, chunks, used, slot = _plan_pool_layout(layout, len(owner_ranks))
    logger.info(
        "[engram_offload] pool init: slot=%.2f GiB, tables=%.2f MiB (%.1f%% "
        "used), row=%dB, world=%d rank=%d owners=%s store=%s",
        slot / _GIB,
        used / (1 << 20),
        100.0 * used / slot,
        entry_bytes,
        world,
        rank,
        owner_ranks,
        store_url or "<node-local>",
    )

    device_id = _current_device_id()
    config = entity.OffloadConfig()
    config.device_id = device_id
    config.reserve_size = slot
    # Consumers reserve the same virtual stride without allocating host DRAM.
    config.alloc_size = slot if is_owner else 0
    config.world_size = world
    if world != local_world:
        config.local_world_size = local_world
    config.rank_id = rank
    config.scene = entity.Scene.SHARED
    config.flags = 0  # vmm mode: dva == hva == gva, peer slots readable
    if store_url:
        try:
            config.store_url = store_url
        except AttributeError as e:
            raise RuntimeError(
                "Engram pool.store_url needs a memfabric_hybrid "
                "wheel whose OffloadConfig carries store_url"
            ) from e
    slot_base = _allocate_pool(entity, config, entry_bytes, used // entry_bytes)
    tables = _create_pool_tables(chunks, slot_base, entry_bytes, rank)
    load_store = None
    if load_address is not None:
        load_store = _create_load_store(load_address, world, rank, layout, owner_ranks)
    manager = EngramOffloadManager(
        entity, local_world, rank, slot, tables, owner_ranks, load_store
    )
    # Collective: every rank of the node must leave together.
    atexit.register(manager.uninitialize)
    return manager


_MANAGER: Optional[EngramOffloadManager] = None


def get_engram_offload_manager(layout: EngramLayout) -> EngramOffloadManager:
    """Reuse one pool; every rank must construct layers in the same order."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = _open_engram_offload(layout)
    return _MANAGER


class _OffloadForwardBuffers:
    """Entry-aligned device buffers retained for the lifetime of captured graphs."""

    def __init__(self, capacity: int, entry_bytes: int, device: torch.device):
        self.capacity = capacity
        self.ids = torch.empty(capacity, dtype=torch.int64, device=device)
        self.count = torch.empty(1, dtype=torch.int32, device=device)
        # entry_gather packs the gathered rows at dst + i * entry_bytes.
        self._backing = torch.empty(
            capacity * entry_bytes + entry_bytes - 1, dtype=torch.uint8, device=device
        )
        self.staging = self._backing[-self._backing.data_ptr() % entry_bytes :]


class NpuEngramOffloadBackend:
    """Full-table A3/A5 lookup through the process-wide MemFabric pool."""

    def __init__(
        self, layout: EngramLayout, dim: int, layer_id: int, storage: dict | None = None
    ):
        layer_hash_index = layout.layer_ids.index(layer_id)
        self._layer_hash_index = layer_hash_index
        self._dim = dim
        self._n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self._code_dtype = torch.int8 if storage is not None else torch.float8_e4m3fn
        self._packed = is_ascend_910c()
        self._manager = get_engram_offload_manager(layout)
        self._table = self._manager.table(layer_hash_index)
        # Chunk ownership follows owner_ranks, not the physical pool rank.
        self.rows = self._table.rows if self._manager.owner_index is not None else 0
        self.row_start = (self._manager.owner_index or 0) * self._table.rows
        self._finalized = False
        self._buffers: Optional[_OffloadForwardBuffers] = None
        self._scale: Optional[torch.Tensor] = None
        self._mapped_bases: Optional[torch.Tensor] = None

    def install_params(self, embed: nn.Module) -> None:
        """Install checkpoint loaders; codes are loaded directly into pool DRAM."""
        embed._derived_weight_cache_error = (
            "Online weight updates are not supported with MemFabric Engram offload; "
            "restart the service to load new weights."
        )
        device = torch.empty(0).device
        embed.rows = self.rows
        embed.row_start = self.row_start
        embed.weight = nn.Parameter(
            torch.empty(self.rows, self._dim, dtype=self._code_dtype, device="meta"),
            requires_grad=False,
        )
        embed.weight.weight_loader = self._load_weight_rows
        embed.scale = nn.Parameter(
            torch.empty(
                self.rows if self._packed else self._table.num_embeddings,
                self._dim // FP8_BLOCK_SIZE,
                dtype=torch.float8_e8m0fnu,
                device="meta" if self._packed else device,
            ),
            requires_grad=False,
        )
        embed.scale.weight_loader = self._load_scale_rows
        self._scale = embed.scale
        if self._packed:
            self._init_mapped_bases(device)
        logger.info(
            "[engram_offload] rank %d: layer %d OFFLOAD mode engaged "
            "(E=%d chunk_rows=%d entry=%dB block_off=%d; "
            "mapped=%s code_dtype=%s)",
            self._manager.rank,
            self._layer_hash_index,
            self._table.num_embeddings,
            self.rows,
            self._table.entry_bytes,
            self._table.block_offset,
            self._packed,
            self._code_dtype,
        )

    def _init_mapped_bases(self, device: torch.device) -> None:
        query = getattr(self._manager._entity, "get_mapped_pool", None)
        base, stride = query() if query is not None else (0, 0)
        if not base or stride != self._manager.slot:
            raise RuntimeError(
                "A3 Engram offload requires MemFabric mapped-pool support"
            )
        self._mapped_bases = torch.tensor(
            [
                base + rank * stride + self._table.block_offset
                for rank in self._manager.owner_ranks
            ],
            dtype=torch.int64,
            device=device,
        )

    def _load_weight_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self._finalized:
            raise RuntimeError("Engram offload does not support online weight updates")
        expected = (self._table.num_embeddings, self._dim)
        if (
            loaded_weight.dtype != self._code_dtype
            or tuple(loaded_weight.shape) != expected
        ):
            raise ValueError(
                f"Engram checkpoint {loaded_weight.dtype} {tuple(loaded_weight.shape)} "
                f"does not match {self._code_dtype} {expected}"
            )
        rows = loaded_weight[self.row_start : self.row_start + self.rows]
        codes = self._table.pool_weight[:, : self._dim]
        codes[: rows.shape[0]].copy_(rows.view(torch.uint8))
        codes[rows.shape[0] :].zero_()

    def _load_scale_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self._finalized:
            raise RuntimeError("Engram offload does not support online weight updates")
        expected = (self._table.num_embeddings, self._dim // FP8_BLOCK_SIZE)
        if (
            loaded_weight.dtype not in (torch.float8_e8m0fnu, torch.uint8)
            or tuple(loaded_weight.shape) != expected
        ):
            raise ValueError(
                "Engram checkpoint scales must match the E8M0 table layout"
            )
        scales = loaded_weight.view(torch.uint8)
        if self._packed:
            rows = scales[self.row_start : self.row_start + self.rows]
            target = self._table.pool_weight[:, self._dim :]
            target[: rows.shape[0]].copy_(rows)
            target[rows.shape[0] :].zero_()
        else:
            param.data.view(torch.uint8).copy_(scales)

    def finish_load(self, label: str) -> None:
        """Publish completed CPU writes before preparing graph metadata."""
        if self._finalized:
            return
        assert self._scale is not None, "install_params must run before finish_load"
        self._manager.finish_load(self._layer_hash_index)
        self._finalized = True
        logger.info(
            "[engram_offload] rank %d: %s pool chunk ready "
            "(%d rows x %dB @ slot off %d)",
            self._manager.rank,
            label,
            self.rows,
            self._table.entry_bytes,
            self._table.block_offset,
        )
        if self._mapped_bases is not None:
            return
        self._manager.prepare_owners(self._scale.device)
        # Captured graphs retain these addresses; include expanded MTP batches.
        chunked_prefill = get_schedule().chunked_prefill_size or 0
        decode_bs = get_exec().graph.cuda_graph_config.decode.max_bs or 0
        decode_width = max_speculative_num_draft_tokens() or 1
        max_tokens = max(chunked_prefill, decode_bs * decode_width, 0)
        if max_tokens:
            self._buffers = _OffloadForwardBuffers(
                max_tokens * self._n_hash_cols,
                self._table.entry_bytes,
                self._scale.device,
            )
            logger.info(
                "[engram_offload] rank %d: %s forward buffers pre-sized and "
                "frozen (capacity=%d ids for max %d tokens x %d cols)",
                self._manager.rank,
                label,
                self._buffers.capacity,
                max_tokens,
                self._n_hash_cols,
            )

    def _ensure_buffers(
        self, count: int, device: torch.device
    ) -> _OffloadForwardBuffers:
        buffers = self._buffers
        if buffers is None:
            self._buffers = _OffloadForwardBuffers(
                count, self._table.entry_bytes, device
            )
            return self._buffers
        if buffers.capacity >= count:
            return buffers
        if is_device_stream_capturing(device):
            raise RuntimeError(
                f"[engram_offload] frozen buffers (capacity {buffers.capacity}) "
                f"cannot serve a capture of {count} ids; preallocation "
                "undersized the graph batch (check chunked_prefill_size and "
                "the decode max batch size)"
            )
        # Eager allocations must not replace buffers referenced by existing graphs.
        logger.warning(
            "[engram_offload] rank %d: layer %d eager lookup of %d ids "
            "exceeds frozen capacity %d; using a throwaway buffer",
            self._manager.rank,
            self._layer_hash_index,
            count,
            buffers.capacity,
        )
        return _OffloadForwardBuffers(count, self._table.entry_bytes, device)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [T, n_hash_cols] -> bf16 [T, n_hash_cols, head_dim]."""
        assert self._finalized, "engram offload rows are not flushed yet"
        if self._mapped_bases is not None:
            from sglang.kernels.ops.embeddings.engram_mapped_gather import (
                mapped_engram_gather,
            )

            return mapped_engram_gather(
                self._mapped_bases,
                indices,
                self._table.rows,
                self._table.num_embeddings,
                self._dim,
                FP8_BLOCK_SIZE,
                self._code_dtype == torch.float8_e4m3fn,
            )
        num_tokens, num_cols = indices.shape
        count = indices.numel()
        if count == 0:
            # An idle DP batch must not freeze a zero-capacity buffer.
            return torch.empty(
                num_tokens,
                num_cols,
                self._dim,
                dtype=torch.bfloat16,
                device=indices.device,
            )
        buffers = self._ensure_buffers(count, indices.device)
        # Layer IDs are a strided [T, L, H] slice; preserve [T, H] for copy_.
        encoded = self._manager.encode_ids(indices, self._table)
        buffers.ids[:count].view(indices.shape).copy_(encoded)
        buffers.count.fill_(count)
        ret = self._manager.entry_gather(
            buffers.staging,
            buffers.ids,
            buffers.count,
            indices.device,
        )
        if ret != 0:
            raise RuntimeError(f"[engram_offload] entry_gather failed with code {ret}")
        staged = buffers.staging[: count * self._table.entry_bytes]
        # A5 has no E8M0 cast; use the device-table path's byte decoder.
        rows = (
            staged.view(self._code_dtype)
            .reshape(num_tokens, num_cols, self._dim)
            .float()
            .unflatten(-1, (-1, FP8_BLOCK_SIZE))
        )
        scales = _e8m0_to_float(self._scale.view(torch.uint8)[indices]).unsqueeze(-1)
        return (rows * scales).flatten(-2).to(torch.bfloat16)
