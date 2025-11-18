from __future__ import annotations

import logging
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.distributed import (
    GroupCoordinator,
    get_attn_cp_group,
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
)
from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils import get_bool_env_var, is_cpu, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_DP_RANK: Optional[int] = None
_ATTN_DP_SIZE: Optional[int] = None

_is_hip = is_hip()
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")
_is_cpu = is_cpu()


class DpPaddingMode(IntEnum):

    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        dp_size = get_attention_dp_size()

        # When is_extend_in_batch and dp_size > 1, use SUM_LEN to avoid padding
        # overhead from uneven token distribution.
        # For dp_size=1, max_len equals sum_len, so prefer MAX_LEN mode
        # to enable symmetric memory optimization (needed for DSA CP, etc.).
        if is_extend_in_batch and dp_size > 1:
            # Hybrid-SSM models materialize idle ranks via the MAX_LEN
            # fabricated-row conversion; other models keep mainline SUM_LEN.
            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN

        # we choose the mode that minimizes the communication cost
        # prefer MAX_LEN when communication cost is equal to enable symmetric memory
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 >= max_len * dp_size:
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
        # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
        # it can be safely removed later, once RCCL fixed
        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:
    """Facade for the DP gathered-buffer state: allocation metadata lives on
    ``flags.dp`` (set once at initialize_dp_attention). The per-forward
    sizing quartet stays as class attributes: the values are read inside
    torch.compile-traced model code, and attribute-source ints get dynamo's
    automatic-dynamic treatment, while contextvars are untraceable and dict
    slots value-guard into the recompile limit (one recompile per distinct
    size)."""

    _global_dp_buffer_len: int
    _local_dp_buffer_len: int
    _dp_max_padding: bool
    _global_num_tokens: Optional[List[int]]

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        dp.buffer_hidden_size = hidden_size
        dp.buffer_dtype = dtype
        dp.buffer_device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens

    @classmethod
    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._global_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._local_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_hidden_size(cls) -> int:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_device

    @classmethod
    def is_dp_max_padding(cls) -> bool:
        return cls._dp_max_padding


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    dp_max_padding: bool,
    global_num_tokens: Optional[List[int]] = None,
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len, local_dp_buffer_len, dp_max_padding, global_num_tokens
    )


def get_global_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer(group=group)


def get_local_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_local_dp_buffer(group=group)


def get_global_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def get_dp_global_num_tokens() -> List[int]:
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):
    # Sticky within the thread: every ForwardBatch construction writes it,
    # graph runners force False around capture; readers are the EP
    # dispatchers on the same (single) forward thread.
    from sglang.srt.runtime_context import get_forward

    get_forward().set("is_extend_in_batch", is_extend_in_batch)


def get_is_extend_in_batch() -> bool:
    from sglang.srt.runtime_context import get_forward

    return get_forward().is_extend_in_batch


def is_dp_max_padding() -> bool:
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    attn_dp_size = dp_size if enable_dp_attention else 1
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:
        # Rank layout is (dp, cp, tp) where tp is the fastest-changing dim:
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    dp = get_flags().dp
    dp.max_len_with_idle = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
    )
    enable_dp_attention = server_args.enable_dp_attention
    dp_size = server_args.dp_size
    moe_dense_tp_size = server_args.moe_dense_tp_size
    attn_cp_size = server_args.attn_cp_size

    dp.enabled = enable_dp_attention

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    _, _, _ATTN_DP_RANK, _ = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    _ATTN_DP_SIZE = dp_size if enable_dp_attention else 1

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )


def is_dp_attention_enabled() -> bool:
    return get_flags().dp.enabled


def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_dp_rank() -> int:
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


@contextmanager
def disable_dp_size():
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _ATTN_DP_SIZE
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _ATTN_DP_SIZE
    _ATTN_DP_SIZE = 1
    try:
        yield
    finally:
        _ATTN_DP_SIZE = old_dp_size


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.
    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:
    # CPU (start, length) slice for DP-local data in a rank-padded buffer.
    # Returns Python ints (no D2H sync) and handles the cuda-graph-padded layout.
    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = get_attention_dp_rank()
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens


from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton


# TODO: write c++ kernel for cpu
def memcpy_cpu(dst, src, dim, offset, sz, offset_src):
    assert dim == 0, "Only dim=0 supported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same trailing shape"

    total_rows_dst, total_rows_src = dst.shape[0], src.shape[0]
    dst_start, src_start = 0, 0

    if offset_src:
        # src[offset:] → dst[0:]
        src_start = offset
        dst_start = 0
    else:
        # src[0:] → dst[offset:]
        src_start = 0
        dst_start = offset

    dst_end = min(dst_start + sz, total_rows_dst)
    src_end = min(src_start + sz, total_rows_src)
    actual_sz = min(dst_end - dst_start, src_end - src_start)

    if actual_sz <= 0:
        return

    dst[dst_start : dst_start + actual_sz].copy_(src[src_start : src_start + actual_sz])


memcpy_func = memcpy_cpu if _is_cpu else memcpy_triton


def memcpy(dst, src, dim, offset, sz, offset_src):
    memcpy_func(dst, src, dim, offset, sz, offset_src)


def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    if local_tokens.shape[0] > 0 and (
        is_partial or get_attn_tensor_model_parallel_rank() == 0
    ):
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"

        memcpy(global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False)

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    NUM_GPUS_PER_NODE = 8
    if (
        not local_tokens.dtype.is_floating_point
        and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
    ):
        from sglang.srt.distributed.parallel_state import inplace_all_reduce

        inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)

    else:
        global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if get_attn_tensor_model_parallel_world_size() == 1:
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_attn_tensor_model_parallel_rank() != 0:
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(
        get_attn_tensor_model_parallel_world_size()
    )[get_attn_tensor_model_parallel_rank()]
    get_attn_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


# Variable-length DP-MoE gather (reference https://github.com/ROCm/ATOM/pull/930): instead of padding every
# rank to max_len (all_gather) or all-reducing a sum_len zero-buffer (all_reduce),
# gather exactly sum(per-rank tokens) via all_gatherv. Env-gated; only the simple
# tp_size==dp_size (attn_tp_size==1) case is supported for now (e.g. tp8dp8).
_USE_DP_GATHERV = get_bool_env_var("SGLANG_DP_USE_GATHERV")


def is_dp_gatherv_active() -> bool:
    """Variable-length DP-MoE gather/scatter (all_gatherv + reduce_scatterv) is
    enabled and applicable to the CURRENT forward. Requires:
      - env SGLANG_DP_USE_GATHERV (default off),
      - supported layout (attn_tp_size==1, tp_size==dp_size),
      - SUM_LEN padding mode. The gatherv pair (all_gatherv + reduce_scatterv) is
        only valid under SUM_LEN; under MAX_LEN the buffer is equal-padded and the
        gather/combine use all_gather / (aiter) reduce_scatter instead. Reading the
        per-forward padding via _DpGatheredBufferWrapper.is_dp_max_padding() (set by
        set_dp_buffer_len) keeps callers that lack a ForwardBatch (e.g.
        dp_reduce_scatter_tensor) consistent."""
    return (
        _USE_DP_GATHERV
        and get_attn_tensor_model_parallel_world_size() == 1
        and get_tensor_model_parallel_world_size() == get_attention_dp_size()
        and not _DpGatheredBufferWrapper.is_dp_max_padding()
    )


def _dp_gatherv_sizes(forward_batch) -> Optional[List[int]]:
    """Per-rank CPU token counts for the buffer being gathered. The MoE gather
    passes a ForwardBatch (global_num_tokens_cpu); the logits gather passes a
    LogitsMetadata (global_num_tokens_for_logprob_cpu). Return the sizes that
    match the LOCAL tensor for this context, or None to fall back."""
    sizes = getattr(forward_batch, "global_num_tokens_for_logprob_cpu", None)
    if sizes is None:
        sizes = getattr(forward_batch, "global_num_tokens_cpu", None)
    if sizes is None:
        return None
    try:
        return [int(x) for x in sizes]
    except (TypeError, ValueError):
        return None


def _dp_gather_via_all_gatherv(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    sizes: List[int],
):
    # attn_tp_size == 1: each DP rank contributes exactly `sizes[rank]` rows.
    # CRITICAL: the MoE downstream runs on the WHOLE `global_tokens` buffer
    # (M = global_tokens.shape[0]), so the gather MUST fill every row. We pad
    # each rank's local tensor up to sizes[rank] with zeros (matching the
    # buffer's reserved per-rank slot) so sum(sizes) == buffer rows and there
    # is no uninitialized tail for the MoE to read.
    rank = get_attention_dp_rank()
    local_rows = sizes[rank]
    if local_tokens.shape[0] == local_rows:
        local_real = local_tokens
    elif local_tokens.shape[0] > local_rows:
        local_real = local_tokens[:local_rows]
    else:
        local_real = local_tokens.new_zeros((local_rows, *local_tokens.shape[1:]))
        local_real[: local_tokens.shape[0]].copy_(local_tokens)
    # sum(sizes) == global_tokens.shape[0] is guaranteed by the caller (else it
    # falls back to all_reduce). Pass global_tokens as the NCCL output buffer so
    # the gather writes directly into it -- avoids the previous extra full-buffer
    # torch.cat + copy_ (two ~sum(sizes)*hidden DtoD copies, ~700us/layer at c512).
    get_tp_group().all_gatherv(local_real, sizes=sizes, output=global_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if (
        is_dp_gatherv_active()
        and forward_batch.dp_padding_mode is not None
        and not forward_batch.dp_padding_mode.is_max_len()
    ):
        # The gatherv per-rank sizes MUST sum to the pre-allocated global buffer
        # (the MoE runs on the whole buffer, so any unfilled tail = garbage).
        # The buffer was sized from the ceil_align'd global_num_tokens stored via
        # set_dp_buffer_len (forward_batch_info), so the authoritative sizes are
        # get_dp_global_num_tokens() — the SAME source the reduce_scatterv combine
        # uses (symmetric). _dp_gatherv_sizes() reads the raw (un-aligned, and for
        # the MoE-gather context the logprob-token) counts, which do NOT match the
        # buffer for prefill steps -> would force an all_reduce fallback.
        # Prefer the buffer-aligned sizes; fall back to the per-batch sizes only
        # if they happen to match (e.g. the logits gather path).
        _gatherv_sizes = get_dp_global_num_tokens()
        if _gatherv_sizes is None or sum(_gatherv_sizes) != global_tokens.shape[0]:
            _gatherv_sizes = _dp_gatherv_sizes(forward_batch)
        if _gatherv_sizes is not None and sum(_gatherv_sizes) == global_tokens.shape[0]:
            _dp_gather_via_all_gatherv(
                global_tokens, local_tokens, forward_batch, is_partial, _gatherv_sizes
            )
            return
    if forward_batch.dp_padding_mode.is_max_len():
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_scatter(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between local_tokens and global_tokens not allowed"

        memcpy(local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True)


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    if is_dp_gatherv_active():
        # Variable-length combine matching all_gatherv dispatch: scatter the
        # global (sum_len) tensor back to per-rank token counts. Fall through to
        # the default reduce-scatter path if per-rank sizes are unavailable.
        sizes = get_dp_global_num_tokens()
        if sizes is not None:
            get_tp_group().reduce_scatterv(input, output=output, sizes=sizes)
            return
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        get_attn_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


# ---------------------------------------------------------------------------
# Two-batch-overlap (non-EP / DP TP-MoE) async gather + combine.
#
# The DP TP-MoE path (deepseek_v4) gathers local hidden -> a global buffer
# before the experts and reduce-scatters back after. For TBO we run those two
# collectives on a single shared comm stream (mirroring the mori dispatcher's
# _comm_stream) and return a CUDA event, so the op engine can yield and let the
# OTHER ubatch's attn+MoE compute run on the compute stream while this ubatch's
# gather/combine proceeds on the comm stream. Both ubatches share ONE comm
# stream -> their collectives serialize in-order (no concurrent-collective
# deadlock on the RCCL communicator), each overlapping the other's compute.
# ---------------------------------------------------------------------------
def get_dp_tbo_comm_stream() -> torch.cuda.Stream:
    from sglang.srt.runtime_context import get_stream

    return get_stream("dp_tbo_comm")


# Persistent reusable CUDA events for non-EP DP TBO, keyed by (kind, subbatch).
# CRITICAL: do NOT create a fresh event per gather/combine -- that is ~244 new
# torch.cuda.Event per forward (61 layers x 2 ubatches x 2), and the HSA signal
# pool is exhausted after a few hundred forwards -> HSA_STATUS_ERROR_OUT_OF_RESOURCES
# ("...create internal OS-specific events"). Reuse one event per (kind, subbatch)
# and just re-record it (mirrors the mori CommStreamPool event reuse).
def _tbo_event(key) -> torch.cuda.Event:
    from sglang.srt.runtime_context import get_resources

    pool = get_resources().tbo_event_pool
    ev = pool.get(key)
    if ev is None:
        ev = torch.cuda.Event()
        pool[key] = ev
    return ev


def dp_gather_partial_async(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    event_key=("gather", 0),
) -> torch.cuda.Event:
    """Launch `dp_gather_partial` (all_gatherv) on the shared DP TBO comm stream;
    re-record + return a PERSISTENT event (keyed by `event_key`) that fires when
    the gather completes. Caller yields, then `compute_stream.wait_event(ev)`
    before reading `global_tokens`."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    # Keep buffers alive across streams (caching allocator).
    local_tokens.record_stream(comm)
    global_tokens.record_stream(comm)
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)  # inputs were produced on the compute stream
        dp_gather_partial(global_tokens, local_tokens, forward_batch)
        ev.record(comm)
    return ev


# Persistent grow-only buffers for non-EP DP TBO, keyed by (kind, tbo_subbatch).
# Reused across ALL layers (and forwards) so the caching allocator does not churn
# a fresh per-layer `torch.empty` for the 8x DP-gather / combine buffers. That
# churn (different sizes per forward x 2 ubatches x 61 layers, kept alive by the
# comm-stream record_stream) ballooned `reserved` to ~270GB and tripped
# HSA_STATUS_ERROR_OUT_OF_RESOURCES at large prefill chunks, even though the live
# (allocated) working set was only ~10GB.
_TBO_PERSIST_BUF: dict = {}


def get_tbo_persistent_buffer(
    key, rows: int, hidden: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return a [rows, hidden] view of a grow-only persistent buffer for `key`.
    Reallocates only when the request exceeds the cached capacity / changes
    dtype|hidden. Caller must treat the returned view as scratch (overwritten)."""
    buf = _TBO_PERSIST_BUF.get(key)
    cap = 0 if buf is None else buf.shape[0]
    if buf is None or rows > cap or buf.shape[1] != hidden or buf.dtype != dtype:
        new_rows = max(rows, cap)
        buf = torch.empty((new_rows, hidden), dtype=dtype, device=device)
        _TBO_PERSIST_BUF[key] = buf
    return buf[:rows]


def dp_reduce_scatterv_async(
    output_local: torch.Tensor,
    global_tokens: torch.Tensor,
    sizes: List[int],
    event_key=("combine", 0),
) -> torch.cuda.Event:
    """Launch the variable-length reduce_scatterv (combine) on the shared DP TBO
    comm stream; re-record + return a PERSISTENT event (keyed by `event_key`).
    Matches the gatherv (SUM_LEN) path."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        get_tp_group().reduce_scatterv(global_tokens, output=output_local, sizes=sizes)
        ev.record(comm)
    return ev


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    return get_attn_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().all_gather_into_tensor(output, input)


def get_moe_cp_group() -> GroupCoordinator:
    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""
    return _get_moe_dp_group()


def get_moe_cp_rank() -> int:
    return _get_moe_dp_group().rank_in_group


def get_moe_cp_size() -> int:
    return _get_moe_dp_group().world_size


def is_enable_moe_cp_allgather() -> bool:
    """True when moe_dp_size < attn_cp_size, requiring allgather across CP ranks before MoE."""
    from sglang.srt.runtime_context import get_server_args

    sa = get_server_args()
    return sa.attn_cp_size > sa.moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return _get_moe_dp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_attn_tp_group().all_gather(input, output_tensor_list=output_list)
