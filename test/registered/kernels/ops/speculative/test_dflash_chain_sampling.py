from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

import unittest

import torch

from sglang.kernels.ops.speculative.dflash_chain_sampling import (
    chain_speculative_sampling_target_only,
)
from sglang.test.test_utils import CustomTestCase


def _reference(
    candidates: torch.Tensor,
    uniform: torch.Tensor,
    uniform_final: torch.Tensor,
    target_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
):
    """Scalar transcription of TreeSpeculativeSamplingTargetOnly.

    Mirrors csrc/speculative/speculative_sampling.cuh restricted to the chain
    topology DFlash produces, deliberately keeping the C++ control flow rather
    than the vectorised form used by the implementation under test.
    """
    bs, draft_len = candidates.shape
    vocab = target_probs.shape[-1]
    predicts = torch.full((bs * draft_len,), -1, dtype=torch.int64)
    accept_num = torch.zeros(bs, dtype=torch.int64)
    draft_probs = torch.zeros_like(target_probs)

    for b in range(bs):
        cur_row = b * draft_len
        coin = float(uniform[b, 0])
        last_slot = b * draft_len
        num_acc = 0
        prob_acc = 0.0
        cur_index = 0
        for _ in range(1, draft_len):
            cur_index = cur_index + 1 if cur_index + 1 < draft_len else -1
            if cur_index == -1:
                break
            token = int(candidates[b, cur_index])
            tps = float(target_probs[cur_row, token])
            prob_acc += tps
            if coin <= prob_acc / threshold_acc or tps >= threshold_single:
                prob_acc = 0.0
                cur_row = b * draft_len + cur_index
                coin = float(uniform[b, cur_index])
                predicts[last_slot] = token
                num_acc += 1
                last_slot = b * draft_len + cur_index
            else:
                draft_probs[cur_row, token] = target_probs[cur_row, token]
                break
        accept_num[b] = num_acc

        resid = target_probs[cur_row].clone().float()
        if num_acc != draft_len - 1:
            resid = torch.clamp(resid - draft_probs[cur_row].float(), min=0.0)
        threshold = float(uniform_final[b]) * float(resid.sum())
        run = 0.0
        sampled = vocab
        last_valid = -1
        for v in range(vocab):
            p = float(resid[v])
            if p > 0:
                run += p
                last_valid = v
                if run > threshold and sampled == vocab:
                    sampled = v
        if sampled == vocab:
            sampled = last_valid if last_valid != -1 else vocab - 1
        predicts[last_slot] = sampled
    return predicts, accept_num, draft_probs


@unittest.skipUnless(torch.cuda.is_available(), "GPU is required for this test.")
class TestDFlashChainSampling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def _run_case(self, bs, draft_len, vocab, threshold_single, threshold_acc, seed):
        device = self.device
        torch.manual_seed(seed)
        logits = torch.randn(bs * draft_len, vocab, device=device) * 2.0
        target_probs = (
            torch.softmax(logits, dim=-1)
            .float()
            .view(bs, draft_len, vocab)
            .contiguous()
        )
        flat = target_probs.reshape(bs * draft_len, vocab)
        candidates = torch.randint(
            0, vocab, (bs, draft_len), device=device, dtype=torch.int64
        )
        # Bias half the candidates towards the target argmax so that both the
        # accept and the reject path are exercised in every case.
        top = flat.argmax(dim=-1).view(bs, draft_len)
        candidates = torch.where(
            torch.rand(bs, draft_len, device=device) < 0.5, top, candidates
        )
        uniform = torch.rand(bs, draft_len, device=device, dtype=torch.float32)
        uniform_final = torch.rand(bs, device=device, dtype=torch.float32)

        predicts = torch.full((bs * draft_len,), -1, dtype=torch.int32, device=device)
        accept_index = torch.empty((bs, draft_len), dtype=torch.int32, device=device)
        accept_num = torch.empty((bs,), dtype=torch.int32, device=device)
        draft_probs = torch.zeros_like(target_probs)
        retrive_index = torch.arange(
            bs * draft_len, device=device, dtype=torch.int64
        ).view(bs, draft_len)
        row_next = torch.arange(1, draft_len + 1, device=device, dtype=torch.int64)
        row_next[-1] = -1
        retrive_next_token = row_next.unsqueeze(0).expand(bs, -1).clone()
        retrive_next_sibling = torch.full(
            (bs, draft_len), -1, device=device, dtype=torch.int64
        )

        chain_speculative_sampling_target_only(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=uniform,
            uniform_samples_for_final_sampling=uniform_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=True,
        )

        ref_predicts, ref_accept, ref_draft = _reference(
            candidates.cpu(),
            uniform.cpu(),
            uniform_final.cpu(),
            flat.cpu(),
            threshold_single,
            threshold_acc,
        )

        self.assertTrue(
            torch.equal(accept_num.cpu().to(torch.int64), ref_accept),
            f"accept_token_num mismatch: {accept_num.tolist()} vs {ref_accept.tolist()}",
        )
        self.assertTrue(
            torch.equal(draft_probs.reshape(bs * draft_len, vocab).cpu(), ref_draft),
            "draft_probs side effect mismatch",
        )
        got = predicts.cpu()
        flat_cpu = flat.cpu().double()
        for b in range(bs):
            num_acc = int(ref_accept[b])
            # Slots past the accepted prefix plus the residual draw are left
            # undefined by the CUDA kernel, so only defined slots are compared.
            for k in range(num_acc):
                slot = b * draft_len + k
                self.assertEqual(
                    int(got[slot]),
                    int(ref_predicts[slot]),
                    f"predicts mismatch at batch {b} slot {k}",
                )
            slot = b * draft_len + num_acc
            if int(got[slot]) == int(ref_predicts[slot]):
                continue
            # The residual draw is an inverse-CDF search. The kernel accumulates
            # in blocked float32, the reference in sequential float64, so a coin
            # landing within accumulation error of a CDF boundary legitimately
            # selects the neighbouring token. Accept that, and only that.
            row = b * draft_len + num_acc
            resid = (flat_cpu[row] - ref_draft[row].double()).clamp_(min=0.0)
            cdf = resid.cumsum(0)
            threshold = float(uniform_final[b]) * float(cdf[-1])
            tok = int(got[slot])
            lo = float(cdf[tok - 1]) if tok > 0 else 0.0
            tol = 1e-6 * float(cdf[-1])
            self.assertTrue(
                float(resid[tok]) > 0
                and lo <= threshold + tol
                and float(cdf[tok]) > threshold - tol,
                f"residual draw at batch {b}: got {tok}, reference "
                f"{int(ref_predicts[slot])}, threshold {threshold:.10g}, "
                f"cdf [{lo:.10g}, {float(cdf[tok]):.10g}]",
            )

    def test_matches_cuda_reference_semantics(self):
        cases = [
            (4, 8, 4096, 0.3, 1.0, 0),
            (8, 8, 4096, 0.1, 1.0, 1),
            (3, 8, 32000, 0.5, 2.0, 2),
            (1, 8, 4096, 0.99, 1.0, 3),  # nearly everything rejected
            (5, 8, 4096, 0.0, 1.0, 4),  # everything accepted via threshold_single
            (2, 4, 8192, 0.3, 1.5, 5),
            (6, 2, 4096, 0.4, 1.0, 6),
            (2, 1, 4096, 0.4, 1.0, 7),  # degenerate: no verify levels
        ]
        for bs, draft_len, vocab, ts, ta, seed in cases:
            with self.subTest(bs=bs, draft_len=draft_len, vocab=vocab, seed=seed):
                self._run_case(bs, draft_len, vocab, ts, ta, seed)


if __name__ == "__main__":
    unittest.main()
