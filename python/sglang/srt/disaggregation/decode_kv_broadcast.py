"""Decode-side KV broadcast for MLA (SGLANG_ENABLE_DISAGG_MLA_DECODE_KV_BROADCAST).

With MLA the decode KV pool is replicated across attention TP ranks, so every
rank otherwise pulls a byte-identical copy over one network link. With this
enabled only attn TP rank 0 pulls, then relays to its peers over NVLink. The
network side of the skip lives in the KV receiver; this module owns the relay.

Everything the transfer would have written has to be relayed: the latent KV,
the state cache riding alongside it, and the draft pool's copies of both under
MTP. Buffers sharing a row size and an index space form one `RelayGroup`, so
KV and state are separate groups: `kv_buffer` has one row per pool slot, the
state buffer one row per device-pool page.

Watch out: the KV indices here are pool slots, not the page ids the receiver
puts on the wire; they are captured before `kv_to_page_indices` in
`DecodePreallocQueue`. State indices are the same page ids the wire carries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = logging.getLogger(__name__)

ENV_NAME = "SGLANG_ENABLE_DISAGG_MLA_DECODE_KV_BROADCAST"
CHUNK_ENV_NAME = "SGLANG_DISAGG_MLA_DECODE_KV_BROADCAST_CHUNK_MIB"

# attn TP rank that owns the network pull and sources the broadcast.
KV_BROADCAST_SRC_RANK = 0

# Triton block size for the gather/scatter, in int32 elements.
BLOCK_SIZE = 1024

# Rows are copied as int32 regardless of buffer dtype, so one kernel covers the
# fp8, bf16 and uint8 pools.
RELAY_DTYPE = torch.int32


def default_chunk_bytes() -> int:
    # Uniform across attn TP ranks because they share the launcher's env;
    # a rank reading a different value would desynchronize the relay.
    chunk_mib = envs.SGLANG_DISAGG_MLA_DECODE_KV_BROADCAST_CHUNK_MIB.get()
    if chunk_mib <= 0:
        raise RuntimeError(
            f"{CHUNK_ENV_NAME} must be a positive number of MiB, got {chunk_mib}."
        )
    return chunk_mib * 1024 * 1024


class RelayableBuffers(msgspec.Struct, frozen=True):
    """Every buffer one pool type contributes to the relay."""

    # A pool -> its per-layer row buffers.
    Accessor = Callable[[Any], Sequence[torch.Tensor]]

    kv: Accessor
    states: Dict[StateType, Accessor] = {}


# Exact pool type -> what to relay for it. Adding a pool means declaring its
# buffers: matching by `isinstance` would let a subclass's extra buffer (a
# quantized pool's scales, say) go unrelayed, which is silent corruption rather
# than a startup error.
RELAYABLE_POOLS: Dict[type, RelayableBuffers] = {
    MLATokenToKVPool: RelayableBuffers(kv=lambda pool: pool.kv_buffer),
    DSATokenToKVPool: RelayableBuffers(
        kv=lambda pool: pool.kv_buffer,
        states={StateType.DSA: lambda pool: pool.index_k_with_scale_buffer},
    ),
}


@triton.jit
def _relay_copy_kernel(
    layer_ptrs,
    indices,
    relay_buf,
    per_layer_elems,
    ELEMS_PER_ROW: tl.constexpr,
    TO_RELAY_BUF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy rows between every layer's buffer and a packed relay chunk.

    The row-indexed sibling of the `layer_ptrs` kernels in
    `common/staging_buffer.py`, without their page expansion and head sharding.
    One launch covers all layers: the per-layer buffers are separate
    allocations, so they are reached through a device array of base pointers
    rather than a stride. Relay layout is [num_layers, num_rows, row].
    """
    layer_id = tl.program_id(0)
    block_id = tl.program_id(1)

    layer_ptr = tl.load(layer_ptrs + layer_id).to(relay_buf.dtype)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < per_layer_elems

    row_id = offsets // ELEMS_PER_ROW
    elem_id = offsets % ELEMS_PER_ROW
    loc = tl.load(indices + row_id, mask=mask, other=0)

    buffer_offsets = loc.to(tl.int64) * ELEMS_PER_ROW + elem_id
    relay_offsets = layer_id.to(tl.int64) * per_layer_elems.to(tl.int64) + offsets

    if TO_RELAY_BUF:
        tl.store(
            relay_buf + relay_offsets,
            tl.load(layer_ptr + buffer_offsets, mask=mask),
            mask=mask,
        )
    else:
        tl.store(
            layer_ptr + buffer_offsets,
            tl.load(relay_buf + relay_offsets, mask=mask),
            mask=mask,
        )


class RelayGroup(msgspec.Struct, frozen=True):
    """One set of per-layer row buffers relayed under a shared index space."""

    name: str
    # int64 device array of per-layer row-buffer base addresses.
    layer_ptrs: torch.Tensor
    num_layers: int
    elems_per_row: int
    rows_per_chunk: int

    def relay_elems(self, num_rows: int) -> int:
        return self.num_layers * num_rows * self.elems_per_row


def _build_relay_group(
    name: str,
    buffers: Sequence[torch.Tensor],
    chunk_bytes: int,
) -> RelayGroup:
    layers = [buffer.flatten(1).view(RELAY_DTYPE) for buffer in buffers]
    for layer, buffer in zip(layers, buffers):
        # `layer_ptrs` snapshots the view; a copy would leave it dangling.
        if layer.data_ptr() != buffer.data_ptr():
            raise RuntimeError(
                f"{ENV_NAME} requires each {name} layer buffer to be viewable "
                "without a copy; got a non-aliasing layout."
            )
    elems_per_row = layers[0].shape[1]
    if any(layer.shape[1] != elems_per_row for layer in layers):
        raise RuntimeError(
            f"{ENV_NAME} requires a uniform row size within a relay group, got "
            f"{sorted({layer.shape[1] for layer in layers})} for {name}."
        )
    row_bytes = elems_per_row * layers[0].element_size()
    return RelayGroup(
        name=name,
        layer_ptrs=torch.tensor(
            [layer.data_ptr() for layer in layers],
            dtype=torch.int64,
            device=layers[0].device,
        ),
        num_layers=len(layers),
        elems_per_row=elems_per_row,
        rows_per_chunk=max(1, chunk_bytes // (len(layers) * row_bytes)),
    )


def _build_kv_groups(
    pools: Sequence[Tuple[str, Any]],
    chunk_bytes: int,
) -> List[RelayGroup]:
    return [
        _build_relay_group(
            name=f"{tag}_kv",
            buffers=RELAYABLE_POOLS[type(pool)].kv(pool),
            chunk_bytes=chunk_bytes,
        )
        for tag, pool in pools
    ]


def _build_state_groups(
    pools: Sequence[Tuple[str, Any]],
    chunk_bytes: int,
) -> List[RelayGroup]:
    return [
        _build_relay_group(
            name=f"{tag}_{state_type.value}",
            buffers=buffers(pool),
            chunk_bytes=chunk_bytes,
        )
        for tag, pool in pools
        for state_type, buffers in RELAYABLE_POOLS[type(pool)].states.items()
    ]


def _build_relay_groups(
    token_to_kv_pool,
    draft_token_to_kv_pool,
    chunk_bytes: int,
) -> Tuple[List[RelayGroup], List[RelayGroup]]:
    pools = [("target", token_to_kv_pool)]
    if draft_token_to_kv_pool is not None:
        pools.append(("draft", draft_token_to_kv_pool))
    for tag, pool in pools:
        if type(pool) not in RELAYABLE_POOLS:
            raise RuntimeError(
                f"{ENV_NAME} requires the {tag} pool to be one of "
                f"{[cls.__name__ for cls in RELAYABLE_POOLS]}, got "
                f"{type(pool).__name__}."
            )

    return (
        _build_kv_groups(pools=pools, chunk_bytes=chunk_bytes),
        _build_state_groups(pools=pools, chunk_bytes=chunk_bytes),
    )


def resolve_decode_kv_broadcast(
    *,
    state_types: Sequence[StateType],
    is_mla_backend: bool,
    attn_tp_size: int,
    supports_decode_kv_broadcast: bool,
    enable_hisparse: bool,
    enable_staging: bool,
    enable_decode_hicache: bool,
) -> bool:
    """Decide whether this decode instance relays KV instead of each rank pulling.

    Off unless the env var is set; then rejects, rather than silently falls
    back on, configurations where a single pulled copy is not equivalent. That
    covers the parallel layouts with nothing to relay, the state living outside
    the KV pools, and the paths that do not implement the network-side skip;
    the pool requirement itself is enforced in `_build_relay_groups`.
    """
    if not envs.SGLANG_ENABLE_DISAGG_MLA_DECODE_KV_BROADCAST.get():
        return False
    if not is_mla_backend:
        raise RuntimeError(
            f"{ENV_NAME} needs an MLA KV pool; a non-MLA pool is sharded over "
            "the attn TP ranks, so no rank holds another's rows."
        )
    if attn_tp_size <= 1:
        raise RuntimeError(
            f"{ENV_NAME} needs attn_tp_size > 1, got {attn_tp_size}; with one "
            "attn TP rank per KV replica there is no duplicate pull to remove. "
            "Pure DP attention lands here; DP attention over several attn TP "
            "ranks does not and is supported."
        )
    if not supports_decode_kv_broadcast:
        raise RuntimeError(
            f"{ENV_NAME} needs a transfer backend that can skip a receiver's "
            "KV; only mooncake implements it."
        )
    relayed_states = {
        state_type for relay in RELAYABLE_POOLS.values() for state_type in relay.states
    }
    unrelayed_states = [st for st in state_types if st not in relayed_states]
    if unrelayed_states:
        raise RuntimeError(
            f"{ENV_NAME} does not support these state caches: {unrelayed_states}; "
            f"only {sorted(relayed_states)} is relayed."
        )
    if enable_hisparse:
        raise RuntimeError(f"{ENV_NAME} is incompatible with hisparse.")
    if enable_staging:
        raise RuntimeError(
            f"{ENV_NAME} is incompatible with SGLANG_DISAGG_STAGING_BUFFER; the "
            "staging path writes the pool from its own buffers."
        )
    if enable_decode_hicache:
        raise RuntimeError(
            f"{ENV_NAME} is untested with the decode hicache path, which fills the "
            "pool from a local restore as well as from the transfer."
        )
    return True


class DecodeKVBroadcaster:
    """Relays freshly transferred rows from attn TP rank 0 to its peers.

    Each rank applies its own indices, so pool layouts never have to agree --
    only the row count per group, which holds because attention TP ranks
    schedule in lockstep.

    Called from the scheduler thread and runs on the caller's stream,
    serialized between the previous forward and the next one (see `broadcast`).
    """

    def __init__(
        self,
        token_to_kv_pool,
        draft_token_to_kv_pool,
        relay_comm: PyNcclCommunicator,
        attn_tp_rank: int,
        attn_tp_size: int,
        forward_stream: torch.cuda.Stream,
        chunk_bytes: Optional[int] = None,
    ):
        self.relay_comm = relay_comm
        self.forward_stream = forward_stream
        self.is_src = attn_tp_rank == KV_BROADCAST_SRC_RANK

        # The relay must span exactly the ranks the KV manager gated on and
        # place the source on the rank that pulls: a communicator over any
        # other group would relay from a rank holding no transferred KV.
        if (relay_comm.rank, relay_comm.world_size) != (attn_tp_rank, attn_tp_size):
            raise RuntimeError(
                f"{ENV_NAME} needs the relay communicator to span the attn TP "
                f"group, but it is rank {relay_comm.rank}/{relay_comm.world_size} "
                f"where the KV manager uses {attn_tp_rank}/{attn_tp_size}."
            )

        # PyNcclCommunicator starts disabled (normally enabled only for
        # CUDA-graph capture) and a disabled broadcast returns silently,
        # which would let non-source ranks scatter from an uninitialized
        # relay buffer. Clear it here so the broadcaster is self-contained.
        if not relay_comm.available:
            raise RuntimeError(
                f"{ENV_NAME} needs a working NCCL communicator for the relay, "
                "but PyNcclCommunicator failed to initialize."
            )
        relay_comm.disabled = False
        if chunk_bytes is None:
            chunk_bytes = default_chunk_bytes()
        self.kv_groups, self.state_groups = _build_relay_groups(
            token_to_kv_pool=token_to_kv_pool,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            chunk_bytes=chunk_bytes,
        )
        all_groups = self.kv_groups + self.state_groups
        self.relay_buf = torch.empty(
            max(
                relay_group.relay_elems(relay_group.rows_per_chunk)
                for relay_group in all_groups
            ),
            dtype=RELAY_DTYPE,
            device=token_to_kv_pool.device,
        )
        logger.info(
            f"Decode KV broadcast enabled, src attn_tp_rank={KV_BROADCAST_SRC_RANK}, "
            f"chunk={chunk_bytes / 2**20:g} MiB, "
            "relay groups="
            + ", ".join(
                f"{relay_group.name}: {relay_group.num_layers} layers x "
                f"{relay_group.elems_per_row} int32/row, "
                f"{relay_group.rows_per_chunk} rows/chunk"
                for relay_group in all_groups
            )
        )

    def broadcast(
        self,
        kv_indices_list: List[torch.Tensor],
        state_indices_list: List[torch.Tensor],
    ) -> None:
        """Broadcast the rows the completed transfers wrote.

        KV indices are pool slots; state indices are device-pool pages, and
        cover only the `RELAYABLE_POOLS` state buffers (today the DSA
        indexer's K cache).

        Collective: every attention TP rank must call this with the same number
        of indices per kind, in the same iteration.
        """
        relays = (
            (self.kv_groups, _cat_indices(kv_indices_list)),
            (self.state_groups, _cat_indices(state_indices_list)),
        )
        if all(indices is None for _, indices in relays):
            return

        # Two NCCL kernels from different communicators running concurrently on
        # one device can deadlock, so the relay never overlaps the forward's
        # collectives. This orders the relay behind the in-flight forward; the
        # other direction is the scheduler's forward_stream.wait_stream(
        # schedule_stream) at each launch site, which is also what fences the
        # next forward's KV reads behind the relay's writes.
        torch.cuda.current_stream().wait_stream(self.forward_stream)

        for groups, indices in relays:
            if indices is None:
                continue
            for relay_group in groups:
                self._relay(relay_group=relay_group, indices=indices)

    def _relay(self, relay_group: RelayGroup, indices: torch.Tensor) -> None:
        for start in range(0, indices.numel(), relay_group.rows_per_chunk):
            chunk_indices = indices[start : start + relay_group.rows_per_chunk]
            per_layer_elems = chunk_indices.numel() * relay_group.elems_per_row
            relay_buf = self.relay_buf[: relay_group.relay_elems(chunk_indices.numel())]
            grid = (
                relay_group.num_layers,
                triton.cdiv(per_layer_elems, BLOCK_SIZE),
            )
            if self.is_src:
                _relay_copy_kernel[grid](
                    relay_group.layer_ptrs,
                    chunk_indices,
                    relay_buf,
                    per_layer_elems,
                    ELEMS_PER_ROW=relay_group.elems_per_row,
                    TO_RELAY_BUF=True,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            self.relay_comm.broadcast(relay_buf, src=KV_BROADCAST_SRC_RANK)
            if not self.is_src:
                _relay_copy_kernel[grid](
                    relay_group.layer_ptrs,
                    chunk_indices,
                    relay_buf,
                    per_layer_elems,
                    ELEMS_PER_ROW=relay_group.elems_per_row,
                    TO_RELAY_BUF=False,
                    BLOCK_SIZE=BLOCK_SIZE,
                )


def _cat_indices(indices_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not indices_list:
        return None
    indices = torch.cat(indices_list).to(torch.int64)
    return indices if indices.numel() else None
