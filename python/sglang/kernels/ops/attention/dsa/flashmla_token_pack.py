"""Opt-in V32 prefill packing with per-token top-k ownership masks.

Input top-k slots must be unique within each row. The union changes key order,
so this path is disabled for deterministic inference. Q concatenation and tail
padding are fused with union construction in the AOT FlashMLA extension.
"""

import torch


def flashmla_token_pack_available() -> bool:
    from sgl_kernel import flash_mla

    return flash_mla._flashmla_import_error is None and hasattr(
        torch.ops.sgl_kernel, "flashmla_token_pack_prepare"
    )


def flashmla_kv_token_packed(q, q_rope, kv_cache, indices, sm_scale):
    """Return [T, 1, H, 512], or None for unsupported widths/layouts.

    The C++ launcher validates matching shapes, devices and index dtype.
    """
    if (
        q.dtype != torch.bfloat16
        or q.shape[0] == 0
        or not 1 <= q.shape[1] <= 32
        or q.shape[2] != (512 if q_rope is not None else 576)
        or kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or kv_cache.shape[1:] != (64, 1, 656)
        or kv_cache.shape[0] * 64 > 1 << 26
        or not 1 <= indices.shape[1] <= 2048
        or indices.stride(1) != 1
    ):
        return None
    for part in (q, q_rope):
        if part is not None and (
            part.stride(-1) != 1
            or part.stride(0) % 8
            or part.stride(1) % 8
            or part.data_ptr() % 16
        ):
            return None
    packed_q, packed_indices, lengths = (
        torch.ops.sgl_kernel.flashmla_token_pack_prepare(q, q_rope, indices)
    )
    out, _, _, _ = torch.ops.sgl_kernel.sparse_decode_fwd.default(
        packed_q,
        kv_cache,
        packed_indices.unsqueeze(1),
        lengths,
        None,
        None,
        None,
        None,
        None,
        None,
        512,
        sm_scale,
        "V32",
        q.shape[1],
    )
    return out.view(-1, 1, q.shape[1], 512)[: q.shape[0]]
