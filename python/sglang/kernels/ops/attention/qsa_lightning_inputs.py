"""A3 LI-v2 request view, prepared once for fixed-width target verification.

Reuse the vLLM NPU plugin TND request layout (fixed commit 24487f4f48ee99cf45d86d1c504e645549b8c1ef,
models/qwen4_exp/lightning_indexer.py), preserving SGLang's FP32 head scale and
row-validity mask. The existing per-row metadata remains available to all
other consumers. No new device kernel or altered sparse selection is used.
"""

import torch
from sgl_kernel_npu.operators.indexer import lightning_indexer_v2
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware


class TargetVerifyLightningInputs:
    def __init__(self, rows, query_width, pages, heads, device):
        if get_hardware().generation is not Generation.ARCH22:
            raise ValueError("Prepared QSA LI without opaque metadata is qualified only on A3")
        if rows <= 0 or query_width not in (1, 2, 3, 4) or rows % query_width:
            raise ValueError("QSA request view requires a complete fixed-width verify layout")
        self.rows, self.width, self.heads = rows, query_width, heads
        requests = rows // query_width
        self.cu_q = torch.arange(0, rows + 1, query_width, dtype=torch.int32, device=device)
        self.lengths = torch.ones(requests, dtype=torch.int32, device=device)
        self.residual = torch.zeros_like(self.lengths)
        self.table = torch.zeros((requests, pages), dtype=torch.int32, device=device)
        self.weights = torch.full((rows, heads), 128**-0.5, dtype=torch.float32, device=device)
        self.ranks = torch.arange(512, device=device)
        self.valid_output = torch.zeros((rows, 512), dtype=torch.bool, device=device)

    def refresh(self, row_lengths, row_compressed_lengths, row_page_table):
        """Refresh in place before model replay; no CPU readback or layer data.

        A real verify request has width consecutive suffix rows. Padded
        requests have all row lengths1 and hence all compressed lengths0.
        Their one-key safe launch is masked completely, as in the old wrapper.
        """
        if row_lengths.shape != (self.rows,) or row_compressed_lengths.shape != (self.rows,):
            raise ValueError("Prepared LI rows do not match the captured verify bucket")
        if row_page_table.shape != (self.rows, self.table.shape[1]):
            raise ValueError("Prepared LI page table does not match the captured bucket")
        final_lengths = row_lengths[self.width - 1 :: self.width]
        compressed = final_lengths // 4
        self.lengths.copy_(compressed.clamp_min(1))
        self.residual.copy_(final_lengths % 4)
        self.table.copy_(torch.where(
            compressed[:, None] > 0,
            row_page_table[self.width - 1 :: self.width],
            0,
        ))
        self.valid_output.copy_(self.ranks[None, :] < row_compressed_lengths[:, None])

    def select(self, q, k_cache):
        if q.shape != (self.rows, self.heads, 128) or q.dtype != torch.bfloat16 or k_cache.dtype != q.dtype:
            raise ValueError("Prepared LI expects the qualified BF16 QSA indexer shape")
        if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
            raise ValueError("Prepared LI requires paged [P,S,1,128] index keys")
        # The ARCH22 indexer reads the sequence tensors directly with metadata=None.
        indices, _ = lightning_indexer_v2(
            q.contiguous(), k_cache, self.weights, 512,
            block_table=self.table, metadata=None,
            cu_seqlens_q=self.cu_q, seqused_k=self.lengths,
            cmp_residual_k=self.residual, layout_q="TND", layout_k="PA_BBND",
            mask_mode=3, cmp_ratio=4,
        )
        return torch.where(self.valid_output, indices[:, 0], -1)
