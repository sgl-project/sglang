# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Sampled DFLASH selector verify: the fused sparse-q accept must reproduce the dense-q
path draw for draw, and both must sample the speculative-sampling distribution."""

import math
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_sampling,
    verify_target_probs,
)
from sglang.kernels.ops.speculative.selector_sampled_accept import (
    commit_selector_block,
    selector_sampled_accept,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_worker_v2 import _commit_accept
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK = 8
GAMMA = BLOCK - 1
TOPK = 16
# Chi-square goodness-of-fit: fail below this p-value (fixed seeds, so a pass is
# deterministic; under a correct sampler a fresh seed fails with probability 1e-6).
P_VALUE_FLOOR = 1e-6


def _sampling_info(bs, device):
    return SamplingBatchInfo(
        temperatures=torch.ones((bs, 1), dtype=torch.float32, device=device),
        top_ps=torch.ones((bs,), dtype=torch.float32, device=device),
        top_ks=torch.full((bs,), TOP_K_ALL, dtype=torch.int32, device=device),
        min_ps=torch.zeros((bs,), dtype=torch.float32, device=device),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=0,
        device=str(device),
    )


_DRAFT_INPUT = SimpleNamespace(max_top_k=1, uniform_top_k_value=None)


def _dense_path(candidates, logits, cand_ids, q_rows, prefix_lens):
    """The production dense-q path: _selector_sampling_accept + _commit_accept."""
    bs, block = candidates.shape
    dense_q = torch.zeros(
        (bs, block - 1, logits.shape[-1]), dtype=torch.float32, device=logits.device
    )
    dense_q.scatter_(-1, cand_ids, q_rows)
    num_correct, bonus, _ = accept_sampling(
        candidates=candidates,
        target_logits=logits,
        draft_probs=dense_q,
        sampling_info=_sampling_info(bs, logits.device),
        draft_input=_DRAFT_INPUT,
        gamma=block - 1,
        verify_num_draft_tokens=block,
    )
    num_correct, bonus = num_correct.to(torch.int32), bonus.to(torch.int64)
    out_tokens, commit_lens = _commit_accept(candidates, num_correct, bonus)
    return num_correct, commit_lens, bonus, out_tokens, prefix_lens + commit_lens


def _fused_path(candidates, logits, cand_ids, q_rows, prefix_lens):
    """What DFlashWorkerV2._selector_fused_accept runs, minus the TP sync."""
    bs, block = candidates.shape
    target_probs = verify_target_probs(
        target_logits=logits,
        sampling_info=_sampling_info(bs, logits.device),
        draft_input=_DRAFT_INPUT,
        bs=bs,
        verify_num_draft_tokens=block,
    )
    uniforms = torch.rand((bs, block - 1), dtype=torch.float32, device=logits.device)
    uniforms_final = torch.rand((bs,), dtype=torch.float32, device=logits.device)
    result = selector_sampled_accept(
        candidates=candidates,
        target_probs=target_probs,
        candidate_ids=cand_ids,
        q_rows=q_rows,
        uniform_samples=uniforms,
        uniform_samples_final=uniforms_final,
    )
    num_correct, commit_lens, bonus, out_tokens, new_seq_lens = commit_selector_block(
        candidates=candidates, accept_result=result, prefix_lens=prefix_lens
    )
    return num_correct, commit_lens, bonus, out_tokens, new_seq_lens


def _random_round(gen, bs, vocab, mode):
    """Random verify rows and selector drafts; mode steers how far the chain gets."""
    logits = torch.randn((bs * BLOCK, vocab), generator=gen) * 1.5
    tops = torch.randint(0, vocab, (bs * BLOCK,), generator=gen)
    logits[torch.arange(bs * BLOCK), tops] += 14.0
    tops = tops.view(bs, BLOCK)
    cand_ids = torch.empty((bs, GAMMA, TOPK), dtype=torch.int64)
    for b in range(bs):
        for s in range(GAMMA):
            ids = torch.randperm(vocab, generator=gen)[:TOPK]
            if tops[b, s] not in ids:
                ids[0] = tops[b, s]
            cand_ids[b, s] = ids
    scores = torch.randn((bs, GAMMA, TOPK), generator=gen) * 2.0
    q_rows = torch.softmax(scores, dim=-1)
    top_slot = (cand_ids == tops[:, :GAMMA, None]).int().argmax(dim=-1)
    path = torch.randint(0, TOPK, (bs, GAMMA), generator=gen)
    if mode == "random":
        # Mostly the target's top token, so chains of every length occur.
        path = torch.where(torch.rand((bs, GAMMA), generator=gen) < 0.8, top_slot, path)
    elif mode == "accept_all":
        # The target's top token drafted with q far below its p: p / q > 1 at every slot.
        path = top_slot
        q_rows.scatter_(-1, path[..., None], 1e-12)
    elif mode == "greedy_q":
        # One-hot q (greedy rows of the selector) on the drafted candidate.
        q_rows = torch.nn.functional.one_hot(path, TOPK).float()
    candidates = torch.empty((bs, BLOCK), dtype=torch.int64)
    candidates[:, 0] = torch.randint(0, vocab, (bs,), generator=gen)
    candidates[:, 1:] = cand_ids.gather(-1, path[..., None])[..., 0]
    logits = 30.0 * torch.tanh(logits / 30.0)
    prefix_lens = torch.randint(1, 100000, (bs,), generator=gen)
    return [
        t.cuda() for t in (candidates, logits, cand_ids, q_rows.contiguous(), prefix_lens)
    ]


@pytest.mark.parametrize("vocab", [163840, 3 * 4096 + 1000])
@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("mode", ["random", "accept_all", "greedy_q"])
def test_fused_matches_dense_path_bitwise(vocab, bs, mode):
    gen = torch.Generator().manual_seed(vocab + 31 * bs + len(mode))
    outcomes = set()
    for trial in range(12):
        inputs = _random_round(gen, bs, vocab, mode)
        torch.manual_seed(1000 + trial)
        expected = _dense_path(*inputs)
        torch.manual_seed(1000 + trial)
        actual = _fused_path(*inputs)
        for name, e, a in zip(
            ("num_correct", "commit_lens", "bonus", "out_tokens", "new_seq_lens"),
            expected,
            actual,
        ):
            assert e.dtype == a.dtype, name
            assert torch.equal(e, a), f"{name}: dense {e.tolist()} fused {a.tolist()}"
        outcomes.update(expected[0].tolist())
    if mode == "accept_all":
        assert outcomes == {GAMMA}
    elif mode == "random":
        assert len(outcomes) > 1


def _chi_square_p_value(observed, expected):
    """Pearson chi-square over bins with expected >= 5 (the rest pooled), with the
    Wilson-Hilferty normal approximation of the survival function."""
    keep = expected >= 5
    obs = torch.cat([observed[keep], observed[~keep].sum().view(1)])
    exp = torch.cat([expected[keep], expected[~keep].sum().view(1)])
    if exp[-1] < 5:
        obs, exp = obs[:-1], exp[:-1]
        obs[-1] += observed[~keep].sum()
        exp[-1] += expected[~keep].sum()
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = obs.numel() - 1
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2)), stat, dof


def _fixed_case(vocab):
    """Peaked target rows and a selector q that overweights the drafted token, so the
    residual max(p - q, 0) differs sharply from p; some mass sits in the last block."""
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn((BLOCK, vocab), generator=gen, dtype=torch.float64) * 0.5
    a = torch.randint(0, 4096, (BLOCK,), generator=gen)
    b = vocab - 1 - torch.randint(0, 900, (BLOCK,), generator=gen)
    c = 4096 + torch.randint(0, 4096, (BLOCK,), generator=gen)
    rows = torch.arange(BLOCK)
    logits[rows, a] += 10.5
    logits[rows, b] += 9.8
    logits[rows, c] += 8.5
    cand_ids = torch.empty((GAMMA, TOPK), dtype=torch.int64)
    q_rows = torch.empty((GAMMA, TOPK), dtype=torch.float64)
    for s in range(GAMMA):
        others = torch.randperm(vocab, generator=gen)
        others = others[(others != a[s]) & (others != b[s]) & (others != c[s])][: TOPK - 3]
        cand_ids[s] = torch.cat([torch.stack([a[s], b[s], c[s]]), others])
        q_rows[s] = torch.cat(
            [torch.tensor([0.62, 0.08, 0.05]), torch.full((TOPK - 3,), 0.25 / (TOPK - 3))]
        )
    return logits, cand_ids, q_rows


def _exact_distribution(p, cand_ids, q_rows, drafts):
    """P(num_correct = j, bonus = x) of token-wise speculative sampling with fixed
    drafts: prod_{s<=j} min(1, p_{s-1}(d_s) / q_{s-1}(d_s)), then on rejection the
    residual max(p_j - q_j, 0) normalized, or p_GAMMA when every draft is accepted."""
    vocab = p.shape[-1]
    dense_q = torch.zeros((GAMMA, vocab), dtype=torch.float64)
    dense_q.scatter_(-1, cand_ids, q_rows)
    out = torch.zeros((BLOCK, vocab), dtype=torch.float64)
    reach = 1.0
    for j in range(GAMMA):
        d = drafts[j]
        accept = min(1.0, float(p[j, d] / dense_q[j, d]))
        residual = (p[j] - dense_q[j]).clamp_min(0.0)
        out[j] = reach * (1.0 - accept) * residual / residual.sum()
        reach *= accept
    out[GAMMA] = reach * p[GAMMA]
    return out


def _sample_many(path_fn, logits32, cand_ids, q_rows, candidates, rows, calls):
    vocab = logits32.shape[-1]
    counts = torch.zeros((BLOCK, vocab), dtype=torch.float64, device="cuda")
    first = torch.zeros((vocab,), dtype=torch.float64, device="cuda")
    batch_logits = logits32.repeat(rows, 1)
    batch_ids = cand_ids.expand(rows, -1, -1).contiguous()
    batch_q = q_rows.expand(rows, -1, -1).contiguous()
    prefix = torch.zeros((rows,), dtype=torch.int64, device="cuda")
    for _ in range(calls):
        cands = candidates() if callable(candidates) else candidates
        num_correct, _, bonus, out_tokens, _ = path_fn(
            cands, batch_logits, batch_ids, batch_q, prefix
        )
        counts.view(-1).index_add_(
            0,
            num_correct.long() * vocab + bonus,
            torch.ones_like(bonus, dtype=torch.float64),
        )
        first.index_add_(
            0, out_tokens[:, 0], torch.ones_like(bonus, dtype=torch.float64)
        )
    return counts, first


@pytest.mark.parametrize("path", ["fused", "dense"])
def test_sampled_distribution_matches_formula(path):
    """Fixed drafts: the (num_correct, bonus) histogram vs the exact formula."""
    vocab = 3 * 4096 + 1000
    path_fn = _fused_path if path == "fused" else _dense_path
    logits, cand_ids, q_rows = _fixed_case(vocab)
    logits32 = (30.0 * torch.tanh(logits / 30.0)).float()
    p = torch.softmax(logits32.double(), dim=-1)
    drafts = cand_ids[:, 0]
    rows, calls = 4096, 25
    candidates = torch.cat([torch.tensor([5]), drafts]).expand(rows, -1).contiguous()
    torch.manual_seed(20261030)
    counts, _ = _sample_many(
        path_fn,
        logits32.cuda(),
        cand_ids.cuda(),
        q_rows.float().cuda(),
        candidates.cuda(),
        rows,
        calls,
    )
    expected = _exact_distribution(p, cand_ids, q_rows, drafts) * rows * calls
    counts = counts.cpu()
    impossible = counts[expected == 0].sum()
    assert impossible == 0, f"{int(impossible)} samples in zero-probability outcomes"
    p_value, stat, dof = _chi_square_p_value(counts.view(-1), expected.view(-1))
    assert p_value > P_VALUE_FLOOR, f"chi2 {stat:.1f} dof {dof} p {p_value:.3g}"


@pytest.mark.parametrize("path", ["fused", "dense"])
def test_first_token_marginal_is_target(path):
    """Drafts drawn from q: the first committed token must follow p_0 exactly
    (the defining property of speculative sampling)."""
    vocab = 3 * 4096 + 1000
    path_fn = _fused_path if path == "fused" else _dense_path
    logits, cand_ids, q_rows = _fixed_case(vocab)
    logits32 = (30.0 * torch.tanh(logits / 30.0)).float()
    p0 = torch.softmax(logits32.double(), dim=-1)[0]
    rows, calls = 4096, 25
    ids_gpu, q_gpu = cand_ids.cuda(), q_rows.float().cuda()
    gen = torch.Generator(device="cuda").manual_seed(99)

    def draw_candidates():
        picks = torch.multinomial(q_gpu, rows, replacement=True, generator=gen).T
        drafts = ids_gpu[torch.arange(GAMMA, device="cuda"), picks]
        anchor = torch.full((rows, 1), 5, dtype=torch.int64, device="cuda")
        return torch.cat([anchor, drafts], dim=1).contiguous()

    torch.manual_seed(20261031)
    _, first = _sample_many(
        path_fn, logits32.cuda(), ids_gpu, q_gpu, draw_candidates, rows, calls
    )
    expected = p0 * rows * calls
    first = first.cpu()
    p_value, stat, dof = _chi_square_p_value(first, expected)
    assert p_value > P_VALUE_FLOOR, f"chi2 {stat:.1f} dof {dof} p {p_value:.3g}"
