"""DeepSeek-V4 C4 indexer selection in ascending logical order for TOP.

Deterministic FlashInfer top-k over each row's request-local compressed positions
(ties to the smaller position), sorted ascending, then the native page-table
transform. The trainer selects with ordered_dsa_topk over the same logits.
"""

import torch

from sglang.srt.true_on_policy.dsa.ordered_topk import ordered_dsa_topk


def ordered_topk_transform(
    logits, seq_lens, page_table, out_page_indices, page_size, out_raw_indices=None
):
    raw = ordered_dsa_topk(
        logits, seq_lens.reshape(-1).to(torch.int32), out_page_indices.shape[1]
    )
    valid = raw >= 0
    safe = raw.clamp_min(0).long()
    pages = page_table.gather(1, safe // page_size)
    physical = pages * page_size + safe % page_size
    out_page_indices.copy_(torch.where(valid, physical, -1).to(torch.int32))
    if out_raw_indices is not None:
        out_raw_indices.copy_(raw)
