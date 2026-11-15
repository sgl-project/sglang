"""DCP runtime headroom ported from glm52-a3-optimizations f204b928a3."""

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel


def _flag(name, default=False):
    flag = getattr(envs, name, None)
    return flag.get() if flag is not None else default


def runtime_gather_reserve_bytes(
    model,
    tokens,
    *,
    context_tokens=0,
    prefill_requests=1,
    capture_tokens=(),
    page_size=128,
):
    if not (
        _flag("SGLANG_NPU_DSA_CP_TOKEN_SHARDED")
        or _flag("SGLANG_NPU_ENABLE_DCP_PACKED_READ")
    ):
        return 0
    row_bytes = 0
    prefix_row_bytes = 0
    decode_row_bytes = 0
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
            latent, rope = layer.kv_lora_rank, layer.qk_rope_head_dim
            decode_row_bytes = max(
                decode_row_bytes, latent + rope * 2 + latent // 128 * 4 + latent * 4 + 8
            )
            prefix_row_bytes = max(
                prefix_row_bytes,
                (layer.kv_lora_rank + layer.qk_rope_head_dim)
                * projection.element_size(),
            )
    # Packed prefixes can pad one virtual page beyond a request's context.
    if context_tokens:
        context_tokens += page_size * get_parallel().attn_tp_size
    slots = 2 if _flag("SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH") else 1
    # The generic layout holds both contiguous output and rank-major scratch.
    copies = (
        1 if _flag("SGLANG_NPU_ENABLE_DCP_PACKED_READ") and prefill_requests == 1 else 2
    )
    prefix_bytes = context_tokens * prefill_requests * prefix_row_bytes * slots * copies
    local_sends = (
        (2 + slots) * context_tokens * prefill_requests * prefix_row_bytes
        + get_parallel().attn_tp_size
        - 1
    ) // get_parallel().attn_tp_size
    decode_scratch = 0
    if _flag("SGLANG_NPU_SFA_KV_8BIT") and prefix_row_bytes and context_tokens:
        rank = get_parallel().attn_tp_size
        local_rows = (context_tokens + tokens + rank - 1) // rank + 128
        decode_scratch = local_rows * prefill_requests * 2 * decode_row_bytes
    # Serial fallback retains a packed output and a contiguous output, but only
    # one bounded generic scratch. Native EP kernels also need non-Torch memory.
    serial_scratch = 0
    if not (
        _flag("SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH")
        or _flag("SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP")
    ):
        # Packed and generic outputs use separate names and may coexist.
        prefix_bytes = 2 * context_tokens * prefill_requests * prefix_row_bytes
        piece_rows = _flag("SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS", 262144)
        piece_rows = (
            min(context_tokens * prefill_requests, piece_rows)
            if piece_rows > 0
            else context_tokens * prefill_requests
        )
        serial_scratch = (piece_rows + tokens) * prefix_row_bytes
    native_headroom = (1 << 30) if getattr(get_parallel(), "moe_ep_size", 1) > 1 else 0
    return (
        row_bytes
        + prefix_bytes
        + local_sends
        + decode_scratch
        + serial_scratch
        + native_headroom
    )
