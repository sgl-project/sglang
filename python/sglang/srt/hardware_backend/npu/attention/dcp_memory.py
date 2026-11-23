"""Bounded DCP runtime headroom for the packed read and gather prefetch."""

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel


def runtime_gather_reserve_bytes(
    model,
    tokens,
    *,
    context_tokens=0,
    prefill_requests=1,
    capture_tokens=(),
    page_size=128,
):
    """Device bytes the DCP extend gather needs beyond the KV pool, so the
    pool sizing leaves them free. 0 unless the packed read is on."""
    if not envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get():
        return 0
    row_bytes = 0
    prefix_row_bytes = 0
    for layer in model.modules():
        # Prepared projection rings are already resident in the free-memory sample.
        projection = getattr(layer, "w_vc_qrep", None)
        if projection is None:
            projection = getattr(layer, "w_vc_stream_shard", None)
        if projection is None:
            projection = getattr(layer, "w_vc", None)
        if projection is not None and getattr(layer, "use_dsa", False):
            hidden = layer.o_proj.output_size
            # Dense/MoE/operand rings plus local FFN/residual/output.
            staging_rows = (5 + 6 / get_parallel().attn_tp_size) * sum(capture_tokens)
            row_bytes = max(
                row_bytes,
                int((2 * tokens + staging_rows) * hidden * projection.element_size()),
            )
            prefix_row_bytes = max(
                prefix_row_bytes,
                (layer.kv_lora_rank + layer.qk_rope_head_dim)
                * projection.element_size(),
            )
    # Packed prefixes can pad one virtual page beyond a request's context.
    if context_tokens:
        context_tokens += page_size * get_parallel().attn_tp_size
    slots = 2 if envs.SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH.get() else 1
    # The generic layout holds both contiguous output and rank-major scratch.
    copies = (
        1
        if envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get() and prefill_requests == 1
        else 2
    )
    prefix_bytes = context_tokens * prefill_requests * prefix_row_bytes * slots * copies
    local_sends = (
        (2 + slots) * context_tokens * prefill_requests * prefix_row_bytes
        + get_parallel().attn_tp_size
        - 1
    ) // get_parallel().attn_tp_size
    # Serial fallback retains a packed output and a contiguous output, but only
    # one bounded generic scratch. Native EP kernels also need non-Torch memory.
    serial_scratch = 0
    if not envs.SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH.get():
        # Packed and generic outputs use separate names and may coexist.
        prefix_bytes = 2 * context_tokens * prefill_requests * prefix_row_bytes
        piece_rows = envs.SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS.get()
        piece_rows = (
            min(context_tokens * prefill_requests, piece_rows)
            if piece_rows > 0
            else context_tokens * prefill_requests
        )
        serial_scratch = (piece_rows + tokens) * prefix_row_bytes
    native_headroom = (1 << 30) if get_parallel().moe_ep_size > 1 else 0
    return row_bytes + prefix_bytes + local_sends + serial_scratch + native_headroom
