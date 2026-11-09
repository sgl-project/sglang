import types
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    sample_simulated_bonus,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")
register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

DEVICE = get_device()


def _info(temps):
    return types.SimpleNamespace(
        temperatures=torch.tensor(temps, device=DEVICE),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )


class TestSimulatedBonus(unittest.TestCase):
    def test_picks_row_at_correct_len(self):
        torch.manual_seed(0)
        bs, rows, vocab = 3, 6, 1000
        logits = torch.randn(bs * rows, vocab, device=DEVICE)
        correct_len = torch.tensor([0, 5, 2], device=DEVICE, dtype=torch.int32)
        greedy_bonus = torch.zeros(bs, dtype=torch.int64, device=DEVICE)
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=greedy_bonus,
            sampling_info=_info([1e-6] * bs),
            bs=bs,
            verify_num_draft_tokens=rows,
            seed=0,
        )
        ref = logits.view(bs, rows, vocab).argmax(-1)[
            torch.arange(bs), correct_len.long()
        ]
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, ref))

    def test_bf16_full_vocab_low_temperature(self):
        torch.manual_seed(2)
        bs, rows, vocab = 8, 6, 129280
        logits = torch.randn(bs * rows, vocab, device=DEVICE, dtype=torch.bfloat16)
        correct_len = torch.randint(0, rows, (bs,), device=DEVICE, dtype=torch.int64)
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=torch.zeros(bs, dtype=torch.int64, device=DEVICE),
            sampling_info=_info([1e-6] * bs),
            bs=bs,
            verify_num_draft_tokens=rows,
            seed=0,
        )
        picked_rows = logits.float().view(bs, rows, vocab)[
            torch.arange(bs), correct_len
        ]
        # bf16 logits can tie at the row max; any tied token is a valid argmax.
        picked = picked_rows.gather(1, out.view(-1, 1)).squeeze(1)
        self.assertTrue(torch.equal(picked, picked_rows.max(dim=-1).values))

    def test_matches_softmax_distribution(self):
        torch.manual_seed(1)
        vocab, n = 8, 200_000
        base = torch.randn(vocab, device=DEVICE)
        logits = base.repeat(n, 1)
        correct_len = torch.zeros(n, dtype=torch.int32, device=DEVICE)
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=torch.zeros(n, dtype=torch.int64, device=DEVICE),
            sampling_info=_info([0.7] * n),
            bs=n,
            verify_num_draft_tokens=1,
            seed=1,
        )
        freq = torch.bincount(out, minlength=vocab).float() / n
        want = torch.softmax(base / 0.7, dim=-1)
        self.assertLess((freq - want).abs().max().item(), 5e-3)


if __name__ == "__main__":
    unittest.main()
