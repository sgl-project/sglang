"""Correctness tests for the MiniMax-M3 single-stage radix-select decode topk.

The kernel selects, per (head, batch) row, the indices of the ``topk`` largest
block scores among the row's first ``num_blocks = ceil(seq_len / block_size)``
entries, front-packing valid block ids and ``-1``-padding the tail. This mirrors
the consumer ``_gqa_share_sparse_decode_kernel`` contract.
"""

import pytest
import torch

from sglang.jit_kernel.minimax_decode_topk import minimax_decode_topk
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=40, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=40, suite="base-b-kernel-unit-1-gpu-b200")
register_amd_ci(est_time=15, suite="nightly-amd-kernel-1-gpu", nightly=True)


def _ref(score, seq_lens, block_size, topk):
    H, B, S = score.shape
    out = torch.full((H, B, topk), -1, dtype=torch.int32, device=score.device)
    for h in range(H):
        for b in range(B):
            sl = int(seq_lens[b])
            nb = min((sl + block_size - 1) // block_size, S)
            if nb <= topk:
                for i in range(nb):
                    out[h, b, i] = i
                continue
            keff = min(topk, nb)
            _, idx = torch.topk(score[h, b, :nb], keff)
            out[h, b, :keff] = idx.to(torch.int32)
    return out


def _selected_scores_sorted(score, out):
    """Per-row sorted-desc multiset of the scores the kernel selected (tie-robust)."""
    H, B, _ = score.shape
    rows = []
    for h in range(H):
        for b in range(B):
            sel = out[h, b]
            sel = sel[sel >= 0].long()
            assert len(sel.unique()) == len(sel), f"duplicate idx h{h} b{b}: {sel}"
            rows.append(torch.sort(score[h, b, sel], descending=True).values)
    return rows


def _check_contract(out, seq_lens, block_size, topk, S):
    H, B, _ = out.shape
    for h in range(H):
        for b in range(B):
            o = out[h, b]
            nvalid = int((o >= 0).sum())
            # valid entries are front-packed, -1 after
            assert torch.all(o[:nvalid] >= 0) and torch.all(o[nvalid:] == -1), o
            nb = min((int(seq_lens[b]) + block_size - 1) // block_size, S)
            assert nvalid == min(topk, nb)
            assert torch.all(o[:nvalid] < nb)


@pytest.mark.parametrize("dtype_sl", [torch.int32, torch.int64])
@pytest.mark.parametrize("H", [1, 2])
@pytest.mark.parametrize("B", [1, 5, 32])
@pytest.mark.parametrize("topk", [16, 32, 64])
@pytest.mark.parametrize("max_ctx", [4096, 131072, 524288])
def test_decode_topk_distinct(dtype_sl, H, B, topk, max_ctx):
    torch.manual_seed(1234)
    block_size = 128
    S = (max_ctx + block_size - 1) // block_size
    # distinct scores per row -> exact index-set match against torch.topk
    score = torch.empty(H, B, S, dtype=torch.float32, device="cuda")
    for h in range(H):
        for b in range(B):
            score[h, b] = torch.randperm(S, device="cuda").float() + torch.rand(
                1, device="cuda"
            )
    seq_lens = torch.randint(1, max_ctx + 1, (B,), device="cuda", dtype=dtype_sl)

    out = minimax_decode_topk(score, seq_lens, block_size, topk)
    ref = _ref(score, seq_lens, block_size, topk)
    _check_contract(out, seq_lens, block_size, topk, S)
    # exact index-set equality (distinct scores)
    for h in range(H):
        for b in range(B):
            assert set(out[h, b][out[h, b] >= 0].tolist()) == set(
                ref[h, b][ref[h, b] >= 0].tolist()
            )


@pytest.mark.parametrize("kind", ["ties", "negative", "neg_inf_padding", "all_equal"])
def test_decode_topk_adversarial(kind):
    torch.manual_seed(7)
    block_size = 128
    H, B, S, topk = 1, 6, 1024, 16
    if kind == "ties":
        score = torch.randint(0, 4, (H, B, S), device="cuda").float()
    elif kind == "negative":
        score = -torch.rand(H, B, S, device="cuda") * 1000 - 1.0
    elif kind == "neg_inf_padding":
        score = torch.randn(H, B, S, device="cuda")
        score[:, :, ::7] = float("-inf")  # scattered -inf in valid range
    else:  # all_equal
        score = torch.full((H, B, S), 3.14, dtype=torch.float32, device="cuda")
    seq_lens = torch.randint(1, S * block_size, (B,), device="cuda", dtype=torch.int32)

    out = minimax_decode_topk(score, seq_lens, block_size, topk)
    ref = _ref(score, seq_lens, block_size, topk)
    _check_contract(out, seq_lens, block_size, topk, S)
    # tie-robust: the multiset of selected scores must match torch.topk's
    for a, b in zip(
        _selected_scores_sorted(score, out), _selected_scores_sorted(score, ref)
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("seq_len", [1, 128, 129, 2048, 2049])
def test_decode_topk_small_num_blocks(seq_len):
    # num_blocks around / below topk -> naive identity path and boundary.
    block_size = 128
    H, B, S, topk = 1, 1, 64, 16
    score = torch.randn(H, B, S, dtype=torch.float32, device="cuda")
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    out = minimax_decode_topk(score, seq_lens, block_size, topk)
    _check_contract(out, seq_lens, block_size, topk, S)
    nb = min((seq_len + block_size - 1) // block_size, S)
    if nb <= topk:
        assert out[0, 0, :nb].tolist() == list(range(nb))


def test_decode_topk_out_param():
    block_size = 128
    H, B, S, topk = 1, 4, 1024, 16
    score = torch.randn(H, B, S, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((B,), 100000, device="cuda", dtype=torch.int32)
    out = torch.empty((H, B, topk), dtype=torch.int32, device="cuda")
    res = minimax_decode_topk(score, seq_lens, block_size, topk, out=out)
    assert res is out


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
