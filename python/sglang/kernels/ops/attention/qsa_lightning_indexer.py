"""Paged QSA rows using the CANN 9.2.0 image's sgl-kernel-npu interface."""

import torch

# Register the image vendor before any NPU tensors or graph capture.
from sgl_kernel_npu.operators.indexer import (
    lightning_indexer_v2,
    lightning_indexer_v2_metadata,
)


def qsa_lightning_block_indices(q, k_cache, page_table, compressed_lengths):
    """Return 512 logical compressed-group indices per independent query row.

    The caller already limits each row's compressed length to complete visible
    groups. Request duplication for MTP is represented by the row page table.
    Inputs and outputs remain on device, including zero-length padding rows.
    """
    rows, heads, dim = q.shape
    if dim != 128 or q.dtype != torch.bfloat16 or k_cache.dtype != q.dtype:
        raise ValueError("QSA LightningIndexerV2 requires BF16 Q/K and width 128")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("QSA LightningIndexerV2 requires paged [P,S,1,128] keys")
    if page_table.shape[0] != rows or compressed_lengths.shape != (rows,):
        raise ValueError("QSA LightningIndexerV2 requires one page table and length per row")
    if not rows:
        return torch.empty((0, 512), dtype=torch.int32, device=q.device)
    lens = compressed_lengths.to(torch.int32).contiguous()
    # Empty image-vendor rows do not initialize their output. Keep a valid
    # one-key launch during empty graph capture, then mask padded results.
    launch_lens = lens.clamp_min(1)
    table = torch.where(lens[:, None] > 0, page_table, 0).to(torch.int32).contiguous()
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    residual = torch.zeros_like(lens)
    weights = torch.full((rows, heads), 128**-0.5, dtype=torch.float32, device=q.device)
    options = dict(
        cu_seqlens_q=cu_q, seqused_k=launch_lens, cmp_residual_k=residual,
        layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=4,
    )
    # Producer and consumer both come from the image's bundled LI vendor.
    metadata = lightning_indexer_v2_metadata(
        heads, 1, 128, 512, batch_size=rows, max_seqlen_q=1,
        max_seqlen_k=page_table.shape[1] * k_cache.shape[1], **options,
    )
    indices, _ = lightning_indexer_v2(
        q.contiguous(), k_cache, weights, 512,
        block_table=table, metadata=metadata,
        **options,
    )
    ranks = torch.arange(512, device=q.device)
    return torch.where(ranks[None, :] < lens[:, None], indices[:, 0], -1)
