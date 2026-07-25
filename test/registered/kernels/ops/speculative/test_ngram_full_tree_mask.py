import pytest
import torch

from sglang.kernels.ops.speculative.ngram_mask import (
    build_ngram_full_tree_mask,
    build_ngram_full_tree_mask_ref,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bs", [1, 2, 8, 32])
@pytest.mark.parametrize("draft_token_num", [4, 8])
@pytest.mark.parametrize("seq_mode", ["equal", "mixed", "long"])
def test_bit_identical(bs, draft_token_num, seq_mode):
    torch.manual_seed(0)
    D = draft_token_num
    if seq_mode == "equal":
        seq_lens = torch.full((bs,), 128, dtype=torch.int32)
    elif seq_mode == "mixed":
        base = torch.tensor([1, 7, 16, 64, 128, 257, 512, 1023], dtype=torch.int32)
        seq_lens = base[torch.arange(bs) % len(base)]
    else:
        seq_lens = torch.full((bs,), 4096, dtype=torch.int32)

    # random draft-draft mask
    draft_mask = torch.randint(0, 2, (bs, D, D), device="cuda", dtype=torch.bool)
    # lower-triangular-ish variant also covered when bs>=1: force one case
    if bs >= 2:
        draft_mask[1] = torch.tril(torch.ones(D, D, device="cuda", dtype=torch.bool))

    ref = build_ngram_full_tree_mask_ref(draft_mask, seq_lens, D)
    out = build_ngram_full_tree_mask(draft_mask, seq_lens, D)
    assert out.dtype == torch.bool
    assert out.shape == ref.shape
    assert torch.equal(out, ref), (
        f"mismatch bs={bs} D={D} seq_mode={seq_mode} "
        f"diff={(out != ref).sum().item()} / {out.numel()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_batch():
    draft_mask = torch.empty(0, 8, 8, dtype=torch.bool, device="cuda")
    seq_lens = torch.empty(0, dtype=torch.int32)
    out = build_ngram_full_tree_mask(draft_mask, seq_lens, 8)
    assert out.numel() == 0
