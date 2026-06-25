# SPDX-License-Identifier: Apache-2.0
"""NVIDIA multimem all-gather along the hidden (last) dim, ported from the
tokenspeed_kernel TritonRSAG ``all_gather_inner`` path.

This is a symmetric-memory + ``multimem.st`` Triton kernel that gathers
per-rank hidden-dim shards ``[T, H/TP]`` into the replicated ``[T, H]`` tensor.
Each rank stores its shard once into the symmetric multicast buffer with a
single 128-bit ``multimem.st`` (which lands the bytes on every peer at once),
so the gather costs one NVLink store pass rather than an NCCL ring.

Only the NVIDIA inner (concat-along-hidden) path is ported; AMD / token-dim /
reduce-scatter / dp-sampling paths from tokenspeed_kernel are intentionally
omitted. Buffers are allocated and rendezvous'd once in ``create_state`` (a
collective), so the steady-state ``all_gather_inner`` launch is CUDA-graph
capturable.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# A CTA runs _BLOCK_THREADS threads; each thread moves _NUMEL_PER_THREAD bf16
# elements with one 128-bit multimem op (16 bytes / 2 bytes per bf16). The
# kernel grid-strides, so the block count is a free tuning knob bounded by
# [_MIN_BLOCKS, _MAX_BLOCKS]; the signal pad is sized for _MAX_BLOCKS.
_BLOCK_THREADS = 1024
_NUMEL_PER_THREAD = 8
_MIN_BLOCKS = 4
_MAX_BLOCKS = 32


# ------------------------------------------------------------------------------
# Low-level PTX helpers
# ------------------------------------------------------------------------------


@triton.jit
def _multimem_st_128(multicast_ptrs, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _local_ld_128(in_ptr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[in_ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():
    tid_x, tid_y, tid_z = _get_tid()
    ntid_x, ntid_y, _ = _get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _send_signal(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.relaxed.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _send_signal_release(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.relaxed.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal_acquire(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _blockwise_barrier(
    signal_pad_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    block_id = (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )
    flat_tid = _get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if flat_tid < world_size:
        if sem == "relaxed":
            _send_signal(send_addrs)
            _wait_signal(wait_addrs)
        else:
            _send_signal_release(send_addrs)
            _wait_signal_acquire(wait_addrs)


@triton.jit
def _all_gather_kernel_inner(
    input_ptr,
    multicast_ptr,
    signal_pad_ptr,
    total_tokens,
    hidden_offset,
    LOCAL_HIDDEN: tl.constexpr,
    TOTAL_HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SKIP_ENTRY_SYNC: tl.constexpr,
) -> None:
    if SKIP_ENTRY_SYNC == 0:
        _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="relaxed")
        _sync_threads()

    chunks_per_row: tl.constexpr = LOCAL_HIDDEN // NUMEL_PER_THREAD
    total_hidden_chunks: tl.constexpr = TOTAL_HIDDEN // NUMEL_PER_THREAD
    hidden_offset_chunks = hidden_offset // NUMEL_PER_THREAD
    total_chunks = total_tokens * chunks_per_row

    pid = tl.program_id(axis=0)
    tid = _get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < total_chunks:
        chunk = block_start + tid
        mask = chunk < total_chunks
        row = chunk // chunks_per_row
        col_chunk = chunk % chunks_per_row

        in_ptr = input_ptr.to(tl.pointer_type(tl.uint64)) + chunk * 2
        out_chunk = row * total_hidden_chunks + hidden_offset_chunks + col_chunk
        out_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64)) + out_chunk * 2
        )
        x, y, z, w = _local_ld_128(in_ptr, mask)
        _multimem_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    _sync_threads()
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="acq_rel")


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


@dataclass
class MultimemAllGatherState:
    group: dist.ProcessGroup
    rank_in_group: int
    world_size: int
    device: torch.device
    max_token_num: int
    hidden_dim: int
    comm_buff: torch.Tensor
    # Cached rendezvous handle: its multicast_ptr / signal_pad are stable for
    # the buffer's lifetime, so we resolve it once and reuse on every launch
    # (a fresh rendezvous() per call is wasteful and noisy under graph capture).
    symm_mem_hdl: Any


def create_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device | None = None,
) -> MultimemAllGatherState:
    """Allocate and rendezvous the symmetric-memory output buffer.

    This is a collective (every rank in ``group`` must call it with identical
    ``max_tokens``/``hidden_size``). Call it once, outside CUDA-graph capture.
    """
    assert type(group) is dist.ProcessGroup, f"Expected ProcessGroup, got {type(group)}"
    assert hidden_size % _NUMEL_PER_THREAD == 0, (
        f"hidden_size={hidden_size} must be a multiple of {_NUMEL_PER_THREAD} "
        f"bf16 for 16-byte multimem.st row alignment"
    )
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

    # blockwise_barrier indexes the pad at block_id * world_size + rank, so a
    # _MAX_BLOCKS-CTA grid needs _MAX_BLOCKS * world_size uint32 slots. max()
    # only grows the pad so we never shrink one another module enlarged.
    pad_bytes = _MAX_BLOCKS * group.size() * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
    with torch.inference_mode(False), torch.no_grad():
        comm_buff = symm_mem.empty(
            (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
    hdl = symm_mem.rendezvous(comm_buff, group=group)
    assert hdl.rank == rank_in_group, (
        f"symm_mem handle rank {hdl.rank} != rank_in_group {rank_in_group}; the "
        f"hidden-shard offset would be wrong"
    )
    return MultimemAllGatherState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=group.size(),
        device=device,
        max_token_num=max_tokens,
        hidden_dim=hidden_size,
        comm_buff=comm_buff,
        symm_mem_hdl=hdl,
    )


def _launch_config(local_numel: int):
    assert local_numel % _NUMEL_PER_THREAD == 0
    return _MIN_BLOCKS, _BLOCK_THREADS, _BLOCK_THREADS // 32, _NUMEL_PER_THREAD


def all_gather_inner(
    state: MultimemAllGatherState,
    hidden_states: torch.Tensor,
    tp_hidden_dim: int,
    skip_entry_sync: bool = False,
    safe: bool = True,
) -> torch.Tensor:
    """All-gather ``[T, H/TP]`` shards into ``[T, H]`` along the hidden dim.

    ``tp_hidden_dim`` is the full (gathered) hidden width ``H``; it must be an
    even multiple of ``world_size`` and each per-rank shard a multiple of 8 bf16
    (16-byte ``multimem.st`` alignment). Returns a clone when ``safe`` (default),
    or a view into the symmetric buffer when ``safe=False`` (valid only until the
    next collective overwrites the buffer).
    """
    world_size = state.world_size
    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported"
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.data_ptr() % 16 == 0, (
        f"hidden_states.data_ptr()={hex(hidden_states.data_ptr())} must be "
        f"16-byte aligned for 128-bit multimem.st"
    )
    assert (
        tp_hidden_dim % world_size == 0
    ), f"tp_hidden_dim={tp_hidden_dim} must be divisible by world_size={world_size}"
    local_hidden = tp_hidden_dim // world_size
    assert local_hidden % _NUMEL_PER_THREAD == 0, (
        f"per-rank hidden shard ({local_hidden}) must be a multiple of "
        f"{_NUMEL_PER_THREAD} bf16"
    )
    assert tp_hidden_dim <= state.hidden_dim, (
        f"comm buffer too narrow: tp_hidden_dim={tp_hidden_dim} > "
        f"state.hidden_dim={state.hidden_dim}"
    )
    total_tokens, in_hidden = hidden_states.shape
    assert (
        in_hidden == local_hidden
    ), f"input hidden ({in_hidden}) != this rank's shard ({local_hidden})"
    assert (
        total_tokens <= state.max_token_num
    ), f"total_tokens={total_tokens} exceeds max_token_num={state.max_token_num}"

    hidden_offset = local_hidden * state.rank_in_group
    symm_mem_hdl = state.symm_mem_hdl
    num_blocks, block_size, num_warps, numel_per_thread = _launch_config(
        total_tokens * local_hidden
    )
    grid = (num_blocks, 1, 1)
    _all_gather_kernel_inner[grid](
        input_ptr=hidden_states,
        multicast_ptr=symm_mem_hdl.multicast_ptr,
        signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
        total_tokens=total_tokens,
        hidden_offset=hidden_offset,
        LOCAL_HIDDEN=local_hidden,
        TOTAL_HIDDEN=state.hidden_dim,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        SKIP_ENTRY_SYNC=1 if skip_entry_sync else 0,
        num_warps=num_warps,
    )
    output = state.comm_buff[:total_tokens, :tp_hidden_dim]
    return output.clone() if safe else output


# ------------------------------------------------------------------------------
# Guarded wrapper: single all-gather path for every caller (fc, LM-head logits, …)
# ------------------------------------------------------------------------------


def recommended_max_tokens(include_prefill: bool, floor: int = 0) -> int:
    """Size a comm buffer from server args: the steady spec-decode token count
    (``max_running_requests * max(num_draft_tokens, eagle_topk)``), optionally
    grown to also cover a prefill chunk. The per-call guard still falls back to
    NCCL when a batch overflows the buffer, so this only needs to cover the case
    we want on the fast path. Returns ``floor`` if server args are unavailable.
    """

    def _pos(v):
        return v if isinstance(v, int) and v > 0 else 0

    try:
        from sglang.srt.server_args import get_global_server_args

        sa = get_global_server_args()
        decode = _pos(getattr(sa, "max_running_requests", 0)) * max(
            _pos(getattr(sa, "speculative_num_draft_tokens", 0)),
            _pos(getattr(sa, "speculative_eagle_topk", 0)),
            1,
        )
        prefill = 0
        if include_prefill:
            prefill = max(
                _pos(getattr(sa, "chunked_prefill_size", 0)),
                _pos(getattr(sa, "max_prefill_tokens", 0)),
            )
        return max(decode, prefill, floor)
    except Exception:
        return floor


class MultimemAllGatherer:
    """Guarded multimem all-gather along the last dim, with an NCCL fallback.

    This is the single entry point every caller uses (the draft ``fc`` gather,
    the vocab-parallel LM-head logits gather, …) so the kernel, guards, and
    fallback live in exactly one place. It owns one symmetric-memory buffer,
    built lazily on the first eager call (``create_state`` is a TP-group
    collective + allocation, illegal under CUDA-graph capture) once the gathered
    width ``x.shape[-1] * world_size`` is known. On each call it uses the
    multimem kernel when the input meets the kernel's dtype/shape/alignment
    contract and fits the buffer, else the standard NCCL all-gather. Every guard
    depends only on TP-replicated quantities, so all ranks pick the same path.

    ``skip_entry_sync`` drops the kernel's entry barrier; pass ``True`` only when
    a cross-rank sync (e.g. a TP all-reduce in the surrounding layer) is
    guaranteed between consecutive calls — see ``all_gather_inner``.
    """

    _UNINIT = object()

    def __init__(
        self,
        max_tokens: int,
        *,
        enabled: bool = True,
        skip_entry_sync: bool = False,
    ):
        self._max_tokens = int(max_tokens)
        self._skip_entry_sync = skip_entry_sync
        # None => permanently disabled (always NCCL); _UNINIT => build on the
        # first eager call; a MultimemAllGatherState => fast path available.
        self._state = self._UNINIT if enabled else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        state = self._state
        if state is self._UNINIT:
            state = self._build(x)
            if state is not self._UNINIT:  # cache real state / permanent disable
                self._state = state
        if (
            state is not None
            and state is not self._UNINIT
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.is_contiguous()
            and 0 < x.shape[0] <= state.max_token_num
            and x.data_ptr() % 16 == 0
            and x.shape[-1] * state.world_size <= state.hidden_dim
        ):
            return all_gather_inner(
                state,
                x,
                tp_hidden_dim=x.shape[-1] * state.world_size,
                skip_entry_sync=self._skip_entry_sync,
                safe=False,
            )
        # Lazy import keeps this device communicator from depending on the
        # high-level distributed facade at module load.
        from sglang.srt.distributed import tensor_model_parallel_all_gather

        return tensor_model_parallel_all_gather(x, dim=-1)

    def _build(self, x: torch.Tensor):
        if x.dim() != 2 or x.dtype != torch.bfloat16:
            return None
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # Can't allocate / rendezvous under capture; retry on a later call.
            return self._UNINIT
        if x.shape[-1] % _NUMEL_PER_THREAD != 0:
            return None
        try:
            from sglang.srt.distributed import get_tp_group

            tp_group = get_tp_group()
            if tp_group.world_size <= 1:
                return None
            return create_state(
                group=tp_group.device_group,
                rank_in_group=tp_group.rank_in_group,
                max_tokens=self._max_tokens,
                hidden_size=x.shape[-1] * tp_group.world_size,
            )
        except Exception as e:
            logger.warning("multimem all-gather disabled (%s)", e)
            return None
