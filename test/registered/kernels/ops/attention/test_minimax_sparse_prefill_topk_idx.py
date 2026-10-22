"""Score-only prefill indexer: split-KV block scores and top-k equal one split's."""

import sys

import pytest
import torch
import triton

from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HD, BLK, TOPK, PAGE = 128, 128, 16, 128

# (cached prefix, new tokens) per request
CASES = {
    # a few (q tile, head) CTAs per request without splits
    "short_turns": [(80_512, 305), (124_928, 773)],
    "mixed_batch": [(0, 2_000), (5_000, 1), (300_000, 1_536), (777, 129)],
    "full_chunk": [(200_000, 16_384)],
}
DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp8_k": (torch.float8_e4m3fn, torch.bfloat16),  # widening: bf16 q, fp8 index K
    "fp8_qk": (torch.float8_e4m3fn, torch.float8_e4m3fn),  # fp8 attn-GEMM mode
}


def _inputs(reqs, k_dtype, q_dtype, seed=0):
    torch.manual_seed(seed)
    batch = len(reqs)
    seq = [p + n for p, n in reqs]
    width = max(seq)
    pages_per_req = (width + PAGE - 1) // PAGE
    pages = torch.randperm(batch * pages_per_req, device="cuda")
    slots = pages.view(batch, pages_per_req, 1) * PAGE + torch.arange(
        PAGE, device="cuda"
    )
    total_q = sum(n for _, n in reqs)
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.tensor([n for _, n in reqs], device="cuda").cumsum(0)
    return dict(
        q=torch.randn(total_q, 1, HD, dtype=torch.bfloat16, device="cuda").to(q_dtype),
        k_cache=torch.randn(batch * pages_per_req * PAGE, 1, HD, device="cuda").to(
            k_dtype
        ),
        v_cache=None,
        sink=None,
        req_to_token=slots.reshape(batch, -1)[:, :width].to(torch.int32).contiguous(),
        slot_ids=torch.arange(batch, dtype=torch.int64, device="cuda"),
        cu_seqlens=cu_seqlens,
        seq_lens=torch.tensor(seq, dtype=torch.int32, device="cuda"),
        prefix_lens=torch.tensor(
            [p for p, _ in reqs], dtype=torch.int32, device="cuda"
        ),
        max_seqlen_q=max(n for _, n in reqs),
        max_seqlen_k=width,
        block_size_q=1,
        block_size_k=BLK,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        disable_index_value=True,
        page_size=PAGE,
        k_scale=0.75 if k_dtype != torch.bfloat16 else None,
    )


def _run(monkeypatch, kwargs, splits, score_type="max"):
    # The block-score buffer is the call's only float32 -inf torch.full.
    scores = []
    orig_full = torch.full

    def full(size, fill_value, *args, **kw):
        out = orig_full(size, fill_value, *args, **kw)
        if kw.get("dtype") == torch.float32 and fill_value == float("-inf"):
            scores.append(out)
        return out

    with monkeypatch.context() as m:
        m.setattr(torch, "full", full)
        _, topk_idx = flash_prefill_with_topk_index(
            **kwargs, score_type=score_type, num_kv_splits=splits
        )
    assert len(scores) == 1
    return topk_idx, scores[0]


def _check(monkeypatch, kwargs, splits, score_type="max"):
    want_idx, want_score = _run(monkeypatch, kwargs, 1, score_type)
    got_idx, got_score = _run(monkeypatch, kwargs, splits, score_type)
    torch.testing.assert_close(got_score, want_score, rtol=0, atol=0)
    torch.testing.assert_close(got_idx, want_idx, rtol=0, atol=0)


@pytest.mark.parametrize("case", list(CASES))
def test_default_split_matches_one_split(monkeypatch, case):
    _check(monkeypatch, _inputs(CASES[case], *DTYPES["bf16"]), None)


@pytest.mark.parametrize(
    "dtypes,score_type", [("fp8_k", "max"), ("fp8_qk", "max"), ("bf16", "lse")]
)
def test_default_split_fp8_and_lse(monkeypatch, dtypes, score_type):
    _check(
        monkeypatch, _inputs(CASES["short_turns"], *DTYPES[dtypes]), None, score_type
    )


def test_scores_match_reference(monkeypatch):
    reqs = [(300, 77), (0, 200)]
    kwargs = _inputs(reqs, *DTYPES["bf16"])
    _, score = _run(monkeypatch, kwargs, None)
    want = torch.full_like(score, float("-inf"))
    q_start = 0
    for b, (prefix, new) in enumerate(reqs):
        seq = prefix + new
        k = kwargs["k_cache"][kwargs["req_to_token"][b, :seq].long(), 0].float()
        q = kwargs["q"][q_start : q_start + new, 0].float()
        # The kernel scores in log2 units: q . k * sm_scale * log2(e).
        qk = (q[:, None, :] * k[None, :, :]).sum(-1) * (HD**-0.5 * 1.4426950409)
        causal = torch.arange(seq, device="cuda")[None, :] <= (
            prefix + torch.arange(new, device="cuda")[:, None]
        )
        qk = qk.masked_fill(~causal, float("-inf"))
        blocks = triton.cdiv(seq, BLK)
        qk = torch.nn.functional.pad(qk, (0, blocks * BLK - seq), value=float("-inf"))
        want[0, q_start : q_start + new, :blocks] = qk.view(new, blocks, BLK).amax(-1)
        q_start += new
    torch.testing.assert_close(score, want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("splits", [2, 7, 4096])
def test_explicit_split_counts(monkeypatch, splits):
    """Counts that do not divide the K tiles, and the clamp to the K tiles."""
    _check(monkeypatch, _inputs(CASES["mixed_batch"], *DTYPES["bf16"]), splits)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
