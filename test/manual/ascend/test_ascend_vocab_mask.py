import math

import pytest
import torch

from sglang.srt.constrained import xgrammar_backend as xb


def _pack_mask(allowed_ids, vocab_size, batch_size=1):
    nwords = math.ceil(vocab_size / 32)
    m = torch.zeros((batch_size, nwords), dtype=torch.int32)
    for b in range(batch_size):
        for tid in allowed_ids[b]:
            m[b, tid // 32] |= 1 << (tid % 32)
    return m


def _apply_ref_cpu(logits, vocab_mask):
    vocab_size = logits.shape[-1]
    token_ids = torch.arange(vocab_size, device="cpu", dtype=torch.int64)
    word_idx = token_ids // 32
    bit_idx = (token_ids % 32).to(torch.int32)
    words = vocab_mask.cpu()[:, word_idx].to(torch.int32)
    allowed = ((words >> bit_idx) & 1).bool().to(logits.device)
    out = logits.clone()
    out.masked_fill_(~allowed, float("-inf"))
    return out


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU required"
)
def test_mask_blocks_disallowed_token_on_npu():
    device = "npu:0"
    vocab_size = 64

    logits = torch.zeros((1, vocab_size), device=device, dtype=torch.float32)
    logits[0, 16] = 22.125
    logits[0, 5] = 10.0

    allowed = [[5, 6, 7, 8]]
    vocab_mask = _pack_mask(allowed, vocab_size).to(device=device, dtype=torch.int32)

    g = xb.XGrammarGrammar.__new__(xb.XGrammarGrammar)
    out = logits.clone()
    g.apply_vocab_mask(out, vocab_mask)

    assert not torch.isfinite(out[0, 16])
    assert int(torch.argmax(out[0]).item()) != 16


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU required"
)
def test_npu_path_matches_reference_random():
    device = "npu:0"
    B, V = 4, 257
    torch.manual_seed(0)

    logits = torch.randn(B, V, device=device, dtype=torch.float32)

    allowed = []
    for _ in range(B):
        ids = torch.randperm(V)[: V // 4].tolist()
        allowed.append(ids)
    vocab_mask = _pack_mask(allowed, V, B).to(device=device, dtype=torch.int32)

    g = xb.XGrammarGrammar.__new__(xb.XGrammarGrammar)
    out_npu = logits.clone()
    g.apply_vocab_mask(out_npu, vocab_mask)

    out_ref = _apply_ref_cpu(logits, vocab_mask)

    assert torch.equal(torch.isfinite(out_npu), torch.isfinite(out_ref))
    diff = (
        torch.nan_to_num(out_npu - out_ref, nan=0.0, posinf=0.0, neginf=0.0)
        .abs()
        .max()
        .item()
    )
    assert diff < 1e-5
