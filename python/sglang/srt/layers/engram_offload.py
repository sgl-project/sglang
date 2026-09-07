"""Host offload of the DeepSeek-V4.1 engram hash tables via memfabric_hybrid acc_offload.

SHARED mode maps the same GVA window into every rank of one node: world_size
contiguous per-rank host slots, each physically backed by its own rank. The
engram fp8 weight tables are row-sharded across those node-local slots —
numbered over every scheduler process on the node, dp replicas included,
which load identical bytes so the node keeps a single replica; never across
machines, so a multi-node run keeps one replica per node — and forward
gathers the selected rows with the acc_offload entry_gather AIV kernel over a
slot-segmented layout: each rank's slot holds the row chunks back to back
from the slot start (row pitch = engram head dim, no padding) and whatever
the slot cannot hold stays empty at its tail, which the kernel addressing
strides over. Neither the pool nor the kernel interprets the row width —
the per-layer layout is registered once after init
(offload_register_entry_table) and the gather call carries only the table id.
The e8m0 scale tables are 128x smaller and stay resident on the device.

The pool is reserved and the layouts registered once per process, identically
on every rank (model build order is rank-uniform), which is what lets the NPU
graph capture the lookup. Requests only reach a node's schedulers after the
launch controller has seen every local scheduler ready (weight load done,
chunks flushed), so no gather can race a peer's flush.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# One row must fit one 120KB UB ping-pong slot inside the entry_gather AIV
# kernel (OFFLOAD_ENTRY_GATHER_MAX_ENTRY_BYTES in acc_offload.h).
_MAX_ENTRY_BYTES = 120 * 1024
# A3 share-pool ceiling: at most 16 node-local ranks per pool.
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


@dataclass(frozen=True)
class EngramOffloadTable:
    """One layer's weight table chunk inside this rank's pool slot.

    Row `i` of the layer table lives at
    `pool_base + block_offset + (i // rows) * slot + (i % rows) * entry_bytes`
    — chunk rank is the node-local rank, the slot stride hops over each
    slot's tail gap, and the kernel resolves the mapping from the registered
    layout, so forward passes the raw hash row ids through.
    """

    layer_hash_index: int
    num_embeddings: int
    head_dim: int
    table_id: int  # handle from offload_register_entry_table
    pool_weight: torch.Tensor  # host view of this rank's chunk, [rows, entry_bytes] uint8
    rows: int  # chunk rows per slot (ceil sharding; the last chunk's tail is unused)
    block_offset: int  # chunk start inside every slot
    entry_bytes: int  # row pitch in bytes (head_dim for fp8, head_dim*2 for bf16)


class EngramOffloadManager:
    """Owns the process-wide acc_offload SHARED pool and the engram layout."""

    def __init__(self, offload, device_id: int, local_world: int, rank: int,
                 slot: int, tables: list[EngramOffloadTable]):
        self._offload = offload
        self._device_id = device_id
        self.local_world = local_world
        self.rank = rank
        self.slot = slot
        self._tables = tables

    def table(self, layer_hash_index: int) -> EngramOffloadTable:
        return self._tables[layer_hash_index]

    def entry_gather(
        self, dst, ids: torch.Tensor, count: torch.Tensor, table_id: int, device: torch.device
    ) -> int:
        """Async gather on the current NPU stream; ordering with the following
        dequant ops comes from the stream itself."""
        return self._offload.entry_gather(dst, ids, count, table_id, device)

    def uninitialize(self) -> None:
        logger.info(
            "[engram_offload] rank %d: pool uninitialize (collective teardown)",
            self.rank,
        )
        self._offload.uninitialize()


def _node_local_pool_coords(tp_size: int, tp_rank: int) -> tuple[int, int]:
    """(world, rank) of the scheduler set that shares this node's pool.

    The DP controller stamps every scheduler it spawns with a dense
    node-local index spanning all dp groups and the node's pp/tp slices; one
    controller runs per node, so the numbering is node-unique by
    construction. Torchrun-style launches carry the same fact in
    LOCAL_WORLD_SIZE/LOCAL_RANK. Directly invoked schedulers (router mode,
    embedded engines) have neither; they fall back to the TP group, which
    then must fit one node.
    """
    for world_var, rank_var in (
        ("SGLANG_NODE_LOCAL_WORLD", "SGLANG_NODE_LOCAL_RANK"),
        ("LOCAL_WORLD_SIZE", "LOCAL_RANK"),
    ):
        world, rank = os.environ.get(world_var), os.environ.get(rank_var)
        if world is not None and rank is not None:
            return int(world), int(rank)
    return tp_size, tp_rank


def _open_engram_offload(config) -> EngramOffloadManager:
    try:
        from memfabric_hybrid import offload
    except ImportError as e:
        raise RuntimeError(
            "SGLANG_OPT_ENGRAM_HOST_OFFLOAD=1 needs the memfabric_hybrid "
            "acc_offload module: install the memfabric-hybrid wheel with "
            "acc_offload support and point MEMFABRIC_HYBRID_EXTEND_LIB_PATH "
            f"at the built kernel library (import failed: {e})"
        ) from e

    parallel = get_parallel()
    # Same-node pipeline stages hold different layers' tables, so they cannot
    # share one slot space (a stage would flush its chunk over another
    # stage's); fail fast while the corruption would otherwise be silent.
    assert max(parallel.pp_size // parallel.nnodes, 1) == 1, (
        "engram offload needs every node-local scheduler in one TP group; "
        f"same-node pipeline parallelism is unsupported (pp_size="
        f"{parallel.pp_size}, nnodes={parallel.nnodes})"
    )
    # One pool per node, shared by every scheduler process on it: dp
    # replicas hold identical weights, so sharding the chunks across all of
    # them keeps a single replica per node. Members are numbered by the
    # node-local index above — tp_rank alone would collide across same-node
    # dp groups in the acc_offload store.
    local_world, rank = _node_local_pool_coords(parallel.tp_size, parallel.tp_rank)
    assert 1 <= local_world <= _MAX_LOCAL_WORLD, (
        f"engram offload pool needs 1..{_MAX_LOCAL_WORLD} node-local ranks, "
        f"got {local_world}"
    )

    head_dim = config.engram_head_dim
    # A3 stores bf16 rows (2 bytes/element); A5 (arch35) stores fp8 (1 byte).
    # The pool and entry_gather treat entry_bytes as the row pitch; head_dim
    # stays the element count for downstream reshape.
    is_arch35 = is_npu_arch35()
    entry_bytes = head_dim if is_arch35 else head_dim * 2
    assert 1 <= entry_bytes <= _MAX_ENTRY_BYTES, (
        f"engram row {entry_bytes}B exceeds the {_MAX_ENTRY_BYTES}B entry_gather limit"
    )
    # Per-layer chunk layout in layer_hash_index order, packed back to back
    # from the slot start (row pitch = head_dim, no padding); every rank runs
    # the same arithmetic, so block offsets inside each slot are identical.
    chunks = []
    offset = 0
    for num_embeddings in config.engram_num_embeddings:
        # ceil: the last chunk's tail rows stay unused (hash ids stay below
        # num_embeddings, so the slack is never addressed).
        rows = -(-num_embeddings // local_world)
        block_offset = offset
        offset = block_offset + rows * entry_bytes
        chunks.append((block_offset, rows, num_embeddings))
    used = offset
    slot = _align_up(used, _GIB)
    logger.info(
        "[engram_offload] pool init: slot=%.2f GiB, tables=%.2f MiB (%.1f%% used), "
        "row=%dB, world=%d rank=%d",
        slot / _GIB,
        used / (1 << 20),
        100.0 * used / slot,
        entry_bytes,
        local_world,
        rank,
    )

    device_id = torch.npu.current_device()
    offload_config = offload.OffloadConfig()
    offload_config.device_id = device_id
    offload_config.reserve_size = slot
    # Every rank physically backs its own slot: together the node holds a
    # full replica of all engram tables.
    offload_config.alloc_size = slot
    offload_config.world_size = local_world
    offload_config.rank_id = rank
    offload_config.scene = offload.Scene.SHARED
    offload_config.flags = 0  # vmm mode: dva == hva == gva, peer slots readable
    assert offload.initialize(offload_config) == 0, (
        f"rank {rank}: offload.initialize failed"
    )

    # One whole-slot malloc; the chunks are carved at the fixed offsets above
    # so a peer chunk is one registered slot stride away.
    slot_base = offload.malloc(slot, 0)
    assert slot_base != 0, f"rank {rank}: offload.malloc({slot}) failed"

    tables = []
    for layer_hash_index, (block_offset, rows, num_embeddings) in enumerate(chunks):
        table_id = offload.register_entry_table(entry_bytes, rows, block_offset)
        assert table_id == layer_hash_index, (
            f"table id {table_id} diverged from registration order "
            f"{layer_hash_index}; every rank must register the same layouts"
        )
        logger.info(
            "[engram_offload] rank %d: registered table %d "
            "(E=%d rows_per_slot=%d block_off=%d entry=%dB table_id=%d)",
            rank,
            layer_hash_index,
            num_embeddings,
            rows,
            block_offset,
            entry_bytes,
            table_id,
        )
        tables.append(
            EngramOffloadTable(
                layer_hash_index=layer_hash_index,
                num_embeddings=num_embeddings,
                head_dim=head_dim,
                table_id=table_id,
                pool_weight=_host_tensor(
                    slot_base + block_offset, (rows, entry_bytes), torch.uint8
                ),
                rows=rows,
                block_offset=block_offset,
                entry_bytes=entry_bytes,
            )
        )
    manager = EngramOffloadManager(offload, device_id, local_world, rank, slot, tables)
    # Collective: every rank of the node must leave together.
    atexit.register(manager.uninitialize)
    return manager


_MANAGER: Optional[EngramOffloadManager] = None


def get_engram_offload_manager(config=None) -> Optional[EngramOffloadManager]:
    """Process-wide singleton; `config` only matters for the first call, which
    happens in the first EngramEmbedding constructor (model build order is
    rank-uniform, so the collective initialize is safe)."""
    global _MANAGER
    if _MANAGER is None:
        if config is None:
            return None
        _MANAGER = _open_engram_offload(config)
    return _MANAGER
