from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

"""Tests for speculative_sampling_block_kernel (block verification).

Paper: https://arxiv.org/abs/2403.10444, Algorithm 3:
h_i = Z_{i+1} / (Z_{i+1} + 1 - p_i), tau = argmax_i {coin_i < h_i}.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_block_speculative_sampling_triton,
    chain_speculative_sampling_triton,
)
from sglang.test.test_utils import CustomTestCase

DEV = "cuda"
# Impossible token id (test vocabs are all <= 4096); kernel predict buffer and
# reference map are pre-filled with it so a missed write is detectable.
_SENTINEL = 999999


def block_verify_reference(
    candidates, target_probs, draft_probs, coins, coin_final, vocab_size
):
    """Vectorized torch reference of Algorithm 3 (arXiv:2403.10444).

    Returns (accept_token_num, predict_map) in the kernel's slot layout:
    slot k < tau holds draft k+1, slot tau holds the final token, untouched
    slots keep the sentinel.
    """
    dev = candidates.device
    bs, S = candidates.shape
    gamma = S - 1

    p = torch.ones(bs, device=dev)
    tau = torch.zeros(bs, dtype=torch.long, device=dev)
    z_res = torch.zeros(bs, device=dev)
    p_res = torch.ones(bs, device=dev)

    for i in range(1, gamma + 1):
        cand_i = candidates[:, i]
        t_i = target_probs[:, i - 1].gather(1, cand_i[:, None]).squeeze(1)
        d_i = draft_probs[:, i - 1].gather(1, cand_i[:, None]).squeeze(1)
        d_ok = (d_i == d_i) & (d_i > 0)
        ratio = torch.where(
            d_ok,
            t_i / torch.where(d_ok, d_i, torch.ones_like(d_i)),
            torch.where(t_i > 0, torch.ones_like(t_i), torch.zeros_like(t_i)),
        )
        p = torch.minimum(p * ratio, torch.ones_like(p))

        if i < gamma:
            z = torch.clamp(
                p[:, None] * target_probs[:, i]
                - torch.nan_to_num(draft_probs[:, i], nan=0.0),
                min=0.0,
            ).sum(-1)
        else:
            z = torch.zeros_like(p)

        denom = z + 1.0 - p
        h_safe = torch.where(
            denom > 0,
            z / torch.where(denom > 0, denom, torch.ones_like(denom)),
            torch.ones_like(denom),
        )
        h = p if i == gamma else h_safe

        upd = coins[:, i - 1] < h
        tau = torch.where(upd, torch.full_like(tau, i), tau)
        z_res = torch.where(upd, z, z_res)
        p_res = torch.where(upd, p, p_res)

    # tau == 0 rows: residual on the root row with scale p_0 = 1.
    z0 = torch.clamp(
        target_probs[:, 0] - torch.nan_to_num(draft_probs[:, 0], nan=0.0), min=0.0
    ).sum(-1)
    z_res = torch.where(tau > 0, z_res, z0)

    all_acc = tau == gamma
    vocab = target_probs.shape[-1]
    row_t = target_probs.gather(1, tau[:, None, None].expand(bs, 1, vocab)).squeeze(1)
    row_d = draft_probs.gather(
        1, torch.clamp(tau, max=gamma - 1)[:, None, None].expand(bs, 1, vocab)
    ).squeeze(1)
    vals = torch.where(
        all_acc[:, None],
        row_t,
        torch.clamp(p_res[:, None] * row_t - torch.nan_to_num(row_d, nan=0.0), min=0.0),
    )
    norm = torch.where(all_acc, row_t.sum(-1), z_res)

    cdf = vals.cumsum(-1)
    thr = (coin_final * norm)[:, None]
    idx = (cdf > thr).float().argmax(-1)
    final = torch.where(
        cdf[:, -1] > thr[:, 0], idx, torch.full_like(idx, vocab_size - 1)
    )

    predict_map = torch.full((bs, S), _SENTINEL, dtype=torch.long, device=dev)
    for i in range(1, gamma + 1):
        predict_map[:, i - 1] = torch.where(
            tau >= i, candidates[:, i], predict_map[:, i - 1]
        )
    predict_map.scatter_(1, tau[:, None], final[:, None])
    return tau.to(torch.int32), predict_map


def _chain_buffers(bs, S, dev):
    predicts = torch.full((bs * S,), _SENTINEL, dtype=torch.int32, device=dev)
    accept_index = torch.full((bs, S), -1, dtype=torch.int32, device=dev)
    accept_token_num = torch.empty(bs, dtype=torch.int32, device=dev)
    retrive_index = torch.arange(bs * S, device=dev, dtype=torch.int32).view(bs, S)
    return predicts, accept_index, accept_token_num, retrive_index


def run_block_kernel(candidates, target_probs, draft_probs, coins, coin_final):
    bs, S = candidates.shape
    dev = candidates.device
    predicts, accept_index, accept_token_num, retrive_index = _chain_buffers(bs, S, dev)
    chain_block_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coin_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return accept_token_num, predicts.view(bs, S).to(torch.long), accept_index


def run_classic_kernel(candidates, target_probs, draft_probs, coins, coin_final):
    bs, S = candidates.shape
    dev = candidates.device
    predicts, accept_index, accept_token_num, retrive_index = _chain_buffers(bs, S, dev)
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coin_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return accept_token_num, predicts.view(bs, S).to(torch.long)


def make_chain_inputs(bs, gamma, vocab, seed, one_hot_frac=0.0):
    """Random chain inputs: softmax target/draft rows, sampled draft tokens.

    one_hot_frac of the draft rows are replaced with exact one-hots at their
    argmax, which routes the kernel through its closed-form Z path while the
    reference still computes the full scan.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    S = gamma + 1

    def rows(scale):
        logits = torch.randn(bs, S, vocab, generator=g, device=DEV) * scale
        return torch.softmax(logits, dim=-1)

    target_probs = rows(2.5)
    draft_probs = rows(2.5)
    if one_hot_frac > 0:
        mask_rows = torch.rand(bs, S, generator=g, device=DEV) < one_hot_frac
        amax = draft_probs.argmax(-1)
        onehot = torch.zeros_like(draft_probs)
        onehot.scatter_(-1, amax[..., None], 1.0)
        draft_probs = torch.where(mask_rows[..., None], onehot, draft_probs)

    draft_flat = torch.nan_to_num(draft_probs, nan=0.0).view(-1, vocab)
    # guard against zero-mass rows after nan cleaning
    bad = draft_flat.sum(-1) == 0
    draft_flat[bad] = 1.0 / vocab
    sample = torch.multinomial(draft_flat, 1, generator=g).view(bs, S)
    candidates = torch.empty(bs, S, dtype=torch.int64, device=DEV)
    candidates[:, 0] = 0  # root slot token unused by the kernel
    candidates[:, 1:] = sample[:, :-1]
    coins = torch.rand(bs, S, generator=g, device=DEV)
    coin_final = torch.rand(bs, generator=g, device=DEV)
    return candidates, target_probs, draft_probs, coins, coin_final


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA (Triton kernels)")
class TestBlockVerificationKernel(CustomTestCase):
    def test_block_matches_reference(self):
        # One-hot coverage is folded in via one_hot_frac: the reference always
        # computes the full vocab scan, so frac=1.0 cross-checks the kernel's
        # closed-form shortcut against it.
        for one_hot_frac in (0.0, 0.5, 1.0):
            for gamma in (2, 4, 8):
                for vocab in (127, 2500):
                    with self.subTest(
                        one_hot_frac=one_hot_frac, gamma=gamma, vocab=vocab
                    ):
                        c, t, d, coins, cf = make_chain_inputs(
                            bs=7,
                            gamma=gamma,
                            vocab=vocab,
                            seed=1234 + gamma * 31 + vocab + int(one_hot_frac * 100),
                            one_hot_frac=one_hot_frac,
                        )
                        tau_k, pred_k, acc_idx_k = run_block_kernel(c, t, d, coins, cf)
                        tau_r, pred_r = block_verify_reference(
                            c, t, d, coins, cf, vocab_size=vocab
                        )
                        torch.testing.assert_close(
                            tau_k.cpu(), tau_r.cpu(), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            pred_k.cpu(), pred_r.cpu(), rtol=0, atol=0
                        )
                        # accepted slots [0..tau] materialized, rest stays -1
                        expected_idx = torch.where(
                            torch.arange(acc_idx_k.shape[1])[None, :]
                            <= tau_r[:, None].cpu(),
                            torch.arange(acc_idx_k.numel(), dtype=torch.int32).view(
                                acc_idx_k.shape
                            ),
                            torch.full_like(acc_idx_k, -1).cpu(),
                        )
                        torch.testing.assert_close(
                            acc_idx_k.cpu(), expected_idx, rtol=0, atol=0
                        )

    def test_zero_coin_strict_reject(self):
        # Guard for the strict `<` accept test: a zero coin must NOT accept a
        # step with h == 0 (which would land tau on a zero-residual row), and
        # must still accept a step with h > 0.
        bs, gamma, vocab = 2, 2, 3
        S = gamma + 1
        t_row = torch.tensor([0.5, 0.5, 0.0], device=DEV)
        d_row = torch.tensor([0.6, 0.3, 0.1], device=DEV)
        target_probs = t_row.expand(bs, S, vocab).contiguous()
        draft_probs = d_row.expand(bs, S, vocab).contiguous()

        candidates = torch.zeros(bs, S, dtype=torch.int64, device=DEV)
        # req 0: drafted token 2 has zero target mass -> p = 0 -> h = 0.
        candidates[0, 1:] = 2
        # req 1: drafted token 1 has r > 1 -> p = 1, Z > 0 -> h = 1.
        candidates[1, 1:] = 1
        coins = torch.zeros(bs, S, device=DEV)
        cf = torch.full((bs,), 0.5, device=DEV)

        tau_k, pred_k, _ = run_block_kernel(
            candidates, target_probs, draft_probs, coins, cf
        )
        self.assertEqual(tau_k[0].item(), 0)
        self.assertGreaterEqual(tau_k[1].item(), 1)
        # zero-coin request still gets a real residual token from the root row
        self.assertTrue((pred_k[0, 0] < vocab).item())

    def test_toy_model_losslessness_and_block_efficiency(self):
        # The paper's Sec. 2 toy model, context-free: M_b = [1/3, 2/3],
        # M_s = [2/3, 1/3]. Analytic for gamma=2: token E[tau] = 10/9,
        # block E[tau] = 11/9. Emitted stream must be i.i.d. M_b.
        bs, gamma, vocab = 20000, 2, 3
        S = gamma + 1
        t_row = torch.tensor([1 / 3, 2 / 3, 0.0], device=DEV)
        d_row = torch.tensor([2 / 3, 1 / 3, 0.0], device=DEV)
        target_probs = t_row.expand(bs, S, vocab).contiguous()
        draft_probs = d_row.expand(bs, S, vocab).contiguous()

        g = torch.Generator(device=DEV).manual_seed(2024)
        candidates = torch.zeros(bs, S, dtype=torch.int64, device=DEV)
        candidates[:, 1] = torch.multinomial(d_row.expand(bs, vocab), 1, generator=g)[
            :, 0
        ]
        candidates[:, 2] = torch.multinomial(d_row.expand(bs, vocab), 1, generator=g)[
            :, 0
        ]
        coins = torch.rand(bs, S, generator=g, device=DEV)
        cf = torch.rand(bs, generator=g, device=DEV)

        tau_classic, _ = run_classic_kernel(
            candidates, target_probs, draft_probs, coins, cf
        )
        tau_block, pred_block, _ = run_block_kernel(
            candidates, target_probs, draft_probs, coins, cf
        )

        mean_classic = tau_classic.float().mean().item()
        mean_block = tau_block.float().mean().item()
        self.assertAlmostEqual(mean_classic, 10 / 9, delta=0.03)
        self.assertAlmostEqual(mean_block, 11 / 9, delta=0.03)
        self.assertGreater(mean_block, mean_classic)

        # marginal losslessness: first emitted token ~ M_b
        first = pred_block[:, 0]
        for tok in (0, 1):
            emp = (first == tok).float().mean().item()
            expect = t_row[tok].item()
            sigma = (expect * (1 - expect) / bs) ** 0.5
            self.assertAlmostEqual(emp, expect, delta=5 * sigma)

    def test_toy_model_two_token_joint(self):
        # Stronger losslessness check: over many iterations of the paper's
        # outer loop, adjacent emitted-token pairs must be i.i.d. M_b x M_b.
        streams, gamma, iters = 8192, 2, 16
        vocab = 3
        S = gamma + 1
        t_row = torch.tensor([1 / 3, 2 / 3, 0.0], device=DEV)
        d_row = torch.tensor([2 / 3, 1 / 3, 0.0], device=DEV)
        target_probs = t_row.expand(streams, S, vocab).contiguous()
        draft_probs = d_row.expand(streams, S, vocab).contiguous()

        g = torch.Generator(device=DEV).manual_seed(7)
        stream = [[] for _ in range(streams)]
        for _ in range(iters):
            candidates = torch.zeros(streams, S, dtype=torch.int64, device=DEV)
            # Draw one token per chain step: multinomial with num_samples > 1
            # defaults to replacement=False, which would corrupt the chain's
            # conditional structure (never (0,0) / (1,1) with a 3-token toy).
            candidates[:, 1] = torch.multinomial(
                d_row.expand(streams, vocab), 1, generator=g
            )[:, 0]
            candidates[:, 2] = torch.multinomial(
                d_row.expand(streams, vocab), 1, generator=g
            )[:, 0]
            coins = torch.rand(streams, S, generator=g, device=DEV)
            cf = torch.rand(streams, generator=g, device=DEV)
            tau_b, pred_b, _ = run_block_kernel(
                candidates, target_probs, draft_probs, coins, cf
            )
            tau_l = tau_b.cpu().tolist()
            pred_l = pred_b.cpu().tolist()
            for b in range(streams):
                stream[b].extend(pred_l[b][: tau_l[b] + 1])

        bigram = torch.zeros(vocab, vocab)
        total = 0
        tok2_seen = 0
        for s in stream:
            tok2_seen += sum(1 for x in s if x == 2)
            for a, b in zip(s, s[1:]):
                bigram[a, b] += 1
                total += 1

        self.assertEqual(tok2_seen, 0)  # zero-mass token must never be emitted
        for a in (0, 1):
            for b in (0, 1):
                expect = t_row[a].item() * t_row[b].item()
                emp = bigram[a, b].item() / total
                sigma = (expect * (1 - expect) / total) ** 0.5
                self.assertAlmostEqual(emp, expect, delta=5 * sigma)


if __name__ == "__main__":
    unittest.main()
