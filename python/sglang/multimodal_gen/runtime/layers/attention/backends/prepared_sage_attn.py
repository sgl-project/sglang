# SPDX-License-Identifier: Apache-2.0
"""Dense SageAttention consumer of FlashInfer's prepared Sage2 operands."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen.runtime.distributed import sage_ulysses
from sglang.multimodal_gen.runtime.distributed.sage_ulysses_route import (
    resolve_prepared_sage_route,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from flashinfer.comm.ulysses import UlyssesQKV

logger = init_logger(__name__)


@contextmanager
def _sage_kernel_stream(device):
    # SageAttention's CUDA extension launches on the default stream. Bridge
    # an incoming NCCL/compute stream around that launch. On the default
    # stream this only selects the device; no extra CUDA events are needed.
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    if current != default:
        default.wait_stream(current)
    try:
        with torch.cuda.stream(default):
            yield
    finally:
        if current != default:
            current.wait_stream(default)


@lru_cache(maxsize=1)
def _load_prepared_dense_sage_kernel():
    try:
        from sageattention.sm89_compile import (
            qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf,
        )
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Prepared dense Sage requires SageAttention with the SM120 "
            "sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf "
            "kernel; install a SageAttention build with SM120 support."
        ) from exc
    return qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf


class PreparedDenseSageAttention:
    """Consume NHD INT8 Q/K and permuted FP8 V without requantization.

    This uses SageAttention2's SM89/SM120 per-warp kernel with the FP16
    short-term accumulator and FP32 accumulation buffer. FlashInfer's Sage2
    exchange already supplies its per-warp scales and V range (2.25).
    """

    def __init__(self):
        # Resolve the optional extension before entering QKV collectives.
        self._attention = _load_prepared_dense_sage_kernel()

    def __call__(self, operands: UlyssesQKV, *, softmax_scale: float) -> torch.Tensor:
        if operands.layout != "sage2_sm89_sm120":
            raise ValueError("Dense SM120 Sage requires sage2_sm89_sm120 operands")
        used = operands.used_sequence
        if not 0 < used <= operands.logical_sequence:
            raise ValueError("Dense Sage requires a valid live sequence prefix")
        output = torch.empty_like(operands.q, dtype=operands.input_dtype)
        # NHD Q/K/output views preserve their batch/token strides. In
        # particular, using logical_sequence here would attend to padding.
        # V's permuted padded storage and all six received operands are reused.
        stream = (
            _sage_kernel_stream(operands.q.device)
            if operands.q.is_cuda
            else nullcontext()
        )
        with stream:
            self._attention(
                operands.q[:, :used],
                operands.k[:, :used],
                operands.v,
                output[:, :used],
                operands.q_scale,
                operands.k_scale,
                operands.v_scale,
                0,  # tensor_layout=NHD
                0,  # non-causal H3 attention
                2,  # qk_quant_gran=per_warp
                softmax_scale,
                0,  # return_lse=False; the kernel writes output in place
            )
        if used != operands.logical_sequence:
            output[:, used:].zero_()
        return output


def prepared_dense_sage_attention_forward(
    query,
    key,
    value,
    *,
    impl,
    cu_seqlens_host,
    max_seqlen,
    ulysses_active,
    ring_active,
    transport=None,
):
    """Return rank-local dense Sage output, or decline before QKV exchange."""
    if not impl.quantize_before_a2a or impl.causal or impl.dropout != 0:
        return None
    decision, group = resolve_prepared_sage_route(
        query,
        key,
        value,
        cu_seqlens_host=cu_seqlens_host,
        max_seqlen=max_seqlen,
        ulysses_active=ulysses_active,
        ring_active=ring_active,
        transport=transport,
    )
    if not decision.enabled:
        logger.info_once(
            f"Prepared dense Sage declined before exchange: {decision.reason}"
        )
        return None
    attention = partial(PreparedDenseSageAttention(), softmax_scale=impl.softmax_scale)
    output = sage_ulysses.prepared_sage_ulysses_forward(
        query,
        key,
        value,
        group=group,
        used_sequence=cu_seqlens_host[1],
        attention=attention,
        transport=transport,
    )
    logger.info_once("MiniMax H3 prepared dense Sage active: input a2a=nccl")
    return output
