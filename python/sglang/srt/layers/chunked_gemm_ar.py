"""Row chunks keep the GEMM bit-identical per row, but chunking can reassociate
NCCL's cross-rank sum (a last-bit difference)."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class _Knobs(msgspec.Struct, frozen=True):
    min_tokens: int
    num_chunks: int
    ratio: float
    reserve_sms: int


_ENABLED: Optional[bool] = None
_KNOBS: Optional[_Knobs] = None
_TOTAL_SMS: Optional[int] = None
_LOGGED = False


def chunked_gemm_ar_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = envs.SGLANG_OPT_CHUNKED_GEMM_AR.get()
    return _ENABLED


def _knobs() -> _Knobs:
    global _KNOBS
    if _KNOBS is None:
        _KNOBS = _Knobs(
            min_tokens=envs.SGLANG_OPT_CHUNKED_GEMM_AR_MIN_TOKENS.get(),
            num_chunks=envs.SGLANG_OPT_CHUNKED_GEMM_AR_NUM_CHUNKS.get(),
            ratio=envs.SGLANG_OPT_CHUNKED_GEMM_AR_RATIO.get(),
            reserve_sms=envs.SGLANG_OPT_CHUNKED_GEMM_AR_RESERVE_SMS.get(),
        )
    return _KNOBS


def compute_chunk_plan(
    m: int,
    num_chunks: int,
    ratio: float,
    align: int = 256,
    min_rows: int = 512,
) -> Optional[List[Tuple[int, int]]]:
    # Every TP rank derives the plan independently, so it must stay a pure
    # function of its arguments for the collectives to line up.
    if num_chunks < 2 or m < 2 * min_rows:
        return None
    ratio = min(max(ratio, 0.05), 1.0)
    weights = [ratio**i for i in range(num_chunks)]
    total = sum(weights)
    bounds = [0]
    acc = 0.0
    for w in weights[:-1]:
        acc += m * w / total
        b = int(round(acc / align)) * align
        b = max(b, bounds[-1] + min_rows)
        if b > m - min_rows:
            break
        bounds.append(b)
    bounds.append(m)
    if len(bounds) < 3:
        return None
    return list(zip(bounds[:-1], bounds[1:]))


def _get_total_sms(device: torch.device) -> int:
    global _TOTAL_SMS
    if _TOTAL_SMS is None:
        _TOTAL_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _TOTAL_SMS


def try_forward_chunked_gemm_ar(
    layer: RowParallelLinear,
    input_parallel: torch.Tensor,
    bias: Optional[torch.Tensor],
    forward_batch: Optional[ForwardBatch],
) -> Optional[torch.Tensor]:
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    if layer.tp_group is not group:
        return None
    return _try_forward_pipeline(
        layer=layer,
        x=input_parallel,
        bias=bias,
        forward_batch=forward_batch,
        group=group,
    )


def try_forward_chunked_gemm_ar_inline(
    layer: RowParallelLinear,
    input_parallel: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    reduce_on_attn_tp: bool,
) -> Optional[torch.Tensor]:
    """Reduce a ``reduce_results=False`` layer's output in the pipeline; a
    non-None result obliges the caller to tell the downstream sum it is done."""
    if not chunked_gemm_ar_enabled():
        return None
    from sglang.srt.distributed.utils import get_group_rank_size
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.runtime_context import get_parallel

    # Wrappers (e.g. LoRA) add work around the base GEMM that the pipeline skips.
    if not isinstance(layer, RowParallelLinear):
        return None

    tp_rank, tp_size = get_group_rank_size(layer.tp_group)
    # A layer that reduces itself is served from RowParallelLinear.forward.
    if layer.reduce_results or tp_size <= 1 or not layer.input_is_parallel:
        return None
    if layer.skip_bias_add and layer.bias is not None:
        return None
    if layer.use_dp_attention_reduce or layer.use_decode_attn_tp:
        return None

    parallel = get_parallel()
    group = parallel.attn_tp_group if reduce_on_attn_tp else parallel.tp_group
    # Bias is fused into rank 0's GEMM only, so it is added once after the AR.
    bias = None if tp_rank > 0 else layer.bias
    return _try_forward_pipeline(
        layer=layer,
        x=input_parallel,
        bias=bias,
        forward_batch=forward_batch,
        group=group,
    )


def _try_forward_pipeline(
    layer: RowParallelLinear,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    forward_batch: Optional[ForwardBatch],
    group: GroupCoordinator,
) -> Optional[torch.Tensor]:
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or not x.is_cuda:
        return None
    m = x.shape[0]
    knobs = _knobs()
    if m < knobs.min_tokens:
        return None
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None

    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        is_symmetric_memory_enabled,
    )
    from sglang.srt.distributed.utils import get_group_rank_size
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.runtime_context import get_exec, get_forward

    if should_skip_mlp_all_reduce() or get_forward().sp_active:
        return None
    if is_symmetric_memory_enabled():
        return None
    if (
        forward_batch is not None
        and not forward_batch.forward_mode.is_decode_or_idle()
        and get_exec().comm.enable_quant_communications
    ):
        return None

    apply_into = getattr(layer.quant_method, "apply_into", None)
    if apply_into is None:
        return None
    if group.world_size != get_group_rank_size(layer.tp_group)[1]:
        return None

    plan = compute_chunk_plan(m=m, num_chunks=knobs.num_chunks, ratio=knobs.ratio)
    if plan is None:
        return None

    from sglang.srt.layers.deep_gemm_wrapper import configure_deep_gemm_num_sms

    capped_sms = None
    if knobs.reserve_sms > 0:
        total_sms = _get_total_sms(x.device)
        if total_sms - knobs.reserve_sms >= 8:
            capped_sms = total_sms - knobs.reserve_sms

    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "Chunked GEMM+AR overlap engaged: m=%d chunks=%s capped_sms=%s group=%s",
            m,
            [hi - lo for lo, hi in plan],
            capped_sms,
            group.unique_name,
        )

    output = torch.empty((m, layer.output_size), dtype=x.dtype, device=x.device)
    device_group = group.device_group
    works = []
    for i, (lo, hi) in enumerate(plan):
        sm_ctx = configure_deep_gemm_num_sms(capped_sms) if i > 0 else nullcontext()
        with sm_ctx:
            apply_into(layer, x[lo:hi], output[lo:hi], bias=bias)
        # c10d orders this after the chunk's GEMM, on PG-NCCL's own stream.
        works.append(
            torch.distributed.all_reduce(
                output[lo:hi], group=device_group, async_op=True
            )
        )
    for work in works:
        work.wait()
    return output
