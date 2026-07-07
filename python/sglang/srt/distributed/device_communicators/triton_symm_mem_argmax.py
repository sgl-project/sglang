# SPDX-License-Identifier: Apache-2.0
"""Distributed argmax over a tensor-parallel-sharded vocab.

Each rank argmaxes its local vocab shard; only the per-rank
``(max_value, global_token_id)`` pair crosses the wire, so the greedy draft loop
avoids the all-gather of the full ``[T, V]`` logits (traffic ``O(tp*T*V)`` ->
``O(tp*T)``). Two transports return identical ids: ``nccl`` (default,
capture-safe) and an opt-in ``multimem.st`` NVLink path
(``SGLANG_DIST_ARGMAX_USE_MULTIMEM=1``).

Tie-break matches ``torch.argmax`` over the full logits (smallest global id on
ties): each rank owns an ascending contiguous vocab range and ``torch.max``
returns the lowest local index, so the cross-rank reduction reproduces it.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Packed int64 key = order-preserving fp32 value (32 bits) << 31 | complemented
# global id (31 bits). Staying within 63 bits keeps the key non-negative so a
# plain signed-int64 max reduces it correctly.
_ID_BITS = 31
_ID_MASK = (1 << _ID_BITS) - 1


def _use_multimem() -> bool:
    return os.environ.get("SGLANG_DIST_ARGMAX_USE_MULTIMEM", "0") == "1"


def _orderable_u32(values: torch.Tensor) -> torch.Tensor:
    # Radix-sort float key: flip all bits of negatives, set sign bit otherwise,
    # so unsigned int order matches float order. Held in int64 to avoid sign
    # truncation.
    u = values.to(torch.float32).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    is_neg = (u & 0x80000000) != 0
    return torch.where(is_neg, (~u) & 0xFFFFFFFF, u | 0x80000000)


def pack_keys(local_val: torch.Tensor, global_id: torch.Tensor) -> torch.Tensor:
    key32 = _orderable_u32(local_val)
    # Complement the id so a larger key means a smaller id on ties.
    id_key = (_ID_MASK - (global_id.to(torch.int64) & _ID_MASK)) & _ID_MASK
    return (key32 << _ID_BITS) | id_key


def unpack_ids(best_key: torch.Tensor) -> torch.Tensor:
    return _ID_MASK - (best_key & _ID_MASK)


def fits_packing(vocab_size: int) -> bool:
    return vocab_size <= _ID_MASK


def reduce_pairs(
    gathered_val: torch.Tensor, gathered_ids: torch.Tensor
) -> torch.Tensor:
    """Reduce ``[world_size, T]`` ``(value, id)`` pairs to the global argmax id."""
    best_rank = torch.argmax(gathered_val, dim=0, keepdim=True)
    return torch.gather(gathered_ids, 0, best_rank).view(gathered_ids.shape[1])


def nccl_pair_argmax(
    local_val: torch.Tensor,
    global_id: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """Global argmax token id per token via an all-gather of ``(value, id)``."""
    world_size = int(tp_group.world_size)
    num_tokens = int(local_val.shape[0])

    local_val = local_val.contiguous()
    global_id = global_id.to(torch.int64).contiguous()

    gathered_val = torch.empty(
        world_size * num_tokens, dtype=local_val.dtype, device=local_val.device
    )
    gathered_ids = torch.empty(
        world_size * num_tokens, dtype=torch.int64, device=global_id.device
    )
    tp_group.all_gather_into_tensor(gathered_val, local_val)
    tp_group.all_gather_into_tensor(gathered_ids, global_id)

    return reduce_pairs(
        gathered_val.view(world_size, num_tokens),
        gathered_ids.view(world_size, num_tokens),
    )


@dataclass
class _DistArgmaxState:
    group: Any
    rank_in_group: int
    world_size: int
    device: torch.device
    max_tokens: int
    comm_buff: torch.Tensor  # symmetric [max_tokens, world_size] int64 buffer
    symm_mem_hdl: Any


def _create_multimem_state(
    group,
    rank_in_group: int,
    world_size: int,
    max_tokens: int,
    device: torch.device,
) -> Optional[_DistArgmaxState]:
    try:
        import torch.distributed._symmetric_memory as symm_mem

        from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
            _MAX_BLOCKS,
        )

        pad_bytes = _MAX_BLOCKS * world_size * 4
        symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
        with torch.inference_mode(False), torch.no_grad():
            comm_buff = symm_mem.empty(
                (max_tokens, world_size), dtype=torch.int64, device=device
            )
        hdl = symm_mem.rendezvous(comm_buff, group=group)
        if hdl.multicast_ptr == 0:
            # No multicast for this world size / arch; multimem.st writes nowhere.
            logger.warning(
                "distributed_argmax multimem disabled (no multicast for "
                "world_size=%d); falling back to NCCL",
                world_size,
            )
            return None
        return _DistArgmaxState(
            group=group,
            rank_in_group=rank_in_group,
            world_size=world_size,
            device=device,
            max_tokens=max_tokens,
            comm_buff=comm_buff,
            symm_mem_hdl=hdl,
        )
    except Exception as e:  # pragma: no cover - hardware/setup dependent
        logger.warning("distributed_argmax multimem disabled (%s)", e)
        return None


def _build_multimem_kernel():
    # Lazy so importing this module never requires triton.
    import triton
    import triton.language as tl

    from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
        _blockwise_barrier,
        _get_flat_tid,
        _sync_threads,
    )

    @triton.jit
    def _multimem_st_b64(mc_ptr, x, mask):
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.eq.s32 %p0, $3, 1;
                @!%p0 bra end;
                multimem.st.relaxed.sys.global.u64 [$1], $2;
                end:
            }
            """,
            "=r,l,l,r",
            args=[mc_ptr, x, mask.to(tl.int32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _argmax_scatter_kernel(
        key_ptr,
        multicast_ptr,
        signal_pad_ptr,
        total_tokens,
        WORLD_SIZE: tl.constexpr,
        RANK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SKIP_ENTRY_SYNC: tl.constexpr,
    ):
        if SKIP_ENTRY_SYNC == 0:
            _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="relaxed")
            _sync_threads()

        pid = tl.program_id(axis=0)
        tid = _get_flat_tid()
        block_start = pid * BLOCK_SIZE
        base = multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.int64))
        while block_start < total_tokens:
            tok = block_start + tid
            mask = tok < total_tokens
            key = tl.load(key_ptr + tok, mask=mask, other=0)
            # Broadcast this rank's key into column RANK of every peer's buffer.
            out_ptr = base + tok * WORLD_SIZE + RANK
            _multimem_st_b64(out_ptr, key, mask)
            block_start += tl.num_programs(axis=0) * BLOCK_SIZE

        _sync_threads()
        _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="acq_rel")

    return _argmax_scatter_kernel


_MULTIMEM_KERNEL = None
_MULTIMEM_BLOCK_THREADS = 1024
_MULTIMEM_MIN_BLOCKS = 4


def multimem_argmax(
    state: _DistArgmaxState,
    local_val: torch.Tensor,
    global_id: torch.Tensor,
    skip_entry_sync: bool = False,
) -> torch.Tensor:
    global _MULTIMEM_KERNEL
    if _MULTIMEM_KERNEL is None:
        _MULTIMEM_KERNEL = _build_multimem_kernel()

    num_tokens = int(local_val.shape[0])
    keys = pack_keys(local_val, global_id).contiguous()
    hdl = state.symm_mem_hdl
    grid = (_MULTIMEM_MIN_BLOCKS, 1, 1)
    _MULTIMEM_KERNEL[grid](
        key_ptr=keys,
        multicast_ptr=hdl.multicast_ptr,
        signal_pad_ptr=hdl.signal_pad_ptrs_dev,
        total_tokens=num_tokens,
        WORLD_SIZE=state.world_size,
        RANK=hdl.rank,
        BLOCK_SIZE=_MULTIMEM_BLOCK_THREADS,
        SKIP_ENTRY_SYNC=1 if skip_entry_sync else 0,
        num_warps=_MULTIMEM_BLOCK_THREADS // 32,
    )
    gathered = state.comm_buff[:num_tokens]  # [T, world_size] int64
    best_key, _ = torch.max(gathered, dim=1)
    return unpack_ids(best_key)


class DistributedArgmax:
    """Guarded distributed argmax: NCCL baseline, optional multimem fast path.

    Guards use TP-replicated quantities so every rank picks the same transport.
    """

    _UNINIT = object()

    def __init__(self, max_tokens: int, *, enabled: bool = True):
        self._max_tokens = int(max_tokens)
        self._enabled = enabled
        # None => NCCL only; _UNINIT => try to build multimem on first eager call.
        self._state = self._UNINIT if (enabled and _use_multimem()) else None

    def __call__(
        self,
        local_val: torch.Tensor,
        global_id: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``[T]`` int64 argmax ids, or ``None`` when TP is disabled."""
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return None

        num_tokens = int(local_val.shape[0])
        state = self._state
        if state is self._UNINIT:
            state = self._maybe_build(tp_group)
            if state is not self._UNINIT:
                self._state = state

        if (
            state is not None
            and state is not self._UNINIT
            and 0 < num_tokens <= state.max_tokens
        ):
            try:
                return multimem_argmax(
                    state, local_val, global_id, skip_entry_sync=False
                )
            except Exception as e:  # pragma: no cover - hardware dependent
                logger.warning("distributed_argmax multimem failed (%s); using NCCL", e)
                self._state = None

        return nccl_pair_argmax(local_val, global_id, tp_group)

    def _maybe_build(self, tp_group):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # Can't allocate symmetric memory under capture; retry later.
            return self._UNINIT
        return _create_multimem_state(
            group=tp_group.device_group,
            rank_in_group=tp_group.rank_in_group,
            world_size=tp_group.world_size,
            max_tokens=self._max_tokens,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
