"""Paged full masks for NGRAM's irregular verification trees."""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_ngram_attention_mask(
    tree_mask,
    prefix_lens,
    out,
    D: tl.constexpr,
    KV_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    query = tl.program_id(1)
    keys = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    prefix = tl.load(prefix_lens + batch)
    draft_key = keys - prefix
    ancestor = tl.load(
        tree_mask + batch * D * D + query * D + draft_key,
        mask=(draft_key >= 0) & (draft_key < D) & (keys < KV_LEN),
        other=0,
    )
    # NGRAM uses True=allowed; Ascend's atten_mask uses True=blocked.
    blocked = (keys >= prefix) & (ancestor == 0)
    tl.store(out + (batch * D + query) * KV_LEN + keys, blocked, keys < KV_LEN)


def build_ngram_attention_mask(
    tree_mask: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_token_num: int,
    padded_kv_len: int,
) -> torch.Tensor:
    """Return [B, 1, D, page-rounded KV length], with padding blocked.

    The BSH FIA path accepts full masks with paged KV. Its last mask dimension
    must cover every slot in the block table, including the unused page tail.
    """
    bs = prefix_lens.numel()
    out = torch.empty(
        (bs, 1, draft_token_num, padded_kv_len),
        dtype=torch.bool,
        device=tree_mask.device,
    )
    if bs:
        _build_ngram_attention_mask[
            (bs, draft_token_num, triton.cdiv(padded_kv_len, 256))
        ](tree_mask, prefix_lens, out, draft_token_num, padded_kv_len, 256)
    return out
