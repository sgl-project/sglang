import types
import unittest

import torch

from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
    simulated_bonus_sample_graph,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    sample_simulated_bonus,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd")


def _info(temps):
    return types.SimpleNamespace(
        temperatures=torch.tensor(temps, device="cuda"),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )


class TestSimulatedBonus(unittest.TestCase):
    def test_picks_row_at_correct_len(self):
        torch.manual_seed(0)
        bs, rows, vocab = 3, 6, 1000
        logits = torch.randn(bs * rows, vocab, device="cuda")
        correct_len = torch.tensor([0, 5, 2], device="cuda", dtype=torch.int32)
        greedy_bonus = torch.zeros(bs, dtype=torch.int64, device="cuda")
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
        logits = torch.randn(bs * rows, vocab, device="cuda", dtype=torch.bfloat16)
        correct_len = torch.randint(0, rows, (bs,), device="cuda", dtype=torch.int64)
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=torch.zeros(bs, dtype=torch.int64, device="cuda"),
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
        base = torch.randn(vocab, device="cuda")
        logits = base.repeat(n, 1)
        correct_len = torch.zeros(n, dtype=torch.int32, device="cuda")
        out = sample_simulated_bonus(
            target_logits=logits,
            correct_len=correct_len,
            greedy_bonus=torch.zeros(n, dtype=torch.int64, device="cuda"),
            sampling_info=_info([0.7] * n),
            bs=n,
            verify_num_draft_tokens=1,
            seed=1,
        )
        freq = torch.bincount(out, minlength=vocab).float() / n
        want = torch.softmax(base / 0.7, dim=-1)
        self.assertLess((freq - want).abs().max().item(), 5e-3)


def _residual_draw(target, draft, correct_len, rows, t, td, seed, width=None):
    bs = correct_len.shape[0]
    return simulated_bonus_sample_graph(
        target_logits=target,
        correct_len=correct_len,
        temperatures=torch.full((bs,), t, device="cuda"),
        seed=torch.full((1,), seed, dtype=torch.int32, device="cuda"),
        bs=bs,
        rows_per_request=rows,
        draft_logits=draft if width is None else draft[..., :width].contiguous(),
        draft_temperatures=torch.full((bs,), td, device="cuda"),
    )


class TestResidualSimulatedBonus(unittest.TestCase):
    """SGLANG_SIMULATE_ACC_RESIDUAL_BONUS: a stop row draws max(p - q, 0)."""

    def test_residual_matches_torch_reference(self):
        torch.manual_seed(3)
        vocab, rows, n = 16, 4, 200_000
        base_t = torch.randn(vocab, device="cuda")
        base_d = base_t + 0.8 * torch.randn(vocab, device="cuda")
        target = base_t.repeat(n * rows, 1)
        draft = base_d.repeat(n, rows - 1, 1)
        correct_len = torch.ones(n, dtype=torch.int64, device="cuda")
        out = _residual_draw(target, draft, correct_len, rows, 0.9, 0.9, seed=5)
        p = torch.softmax(base_t / 0.9, -1)
        q = torch.softmax(base_d / 0.9, -1)
        want = (p - q).clamp(min=0)
        want = want / want.sum()
        freq = torch.bincount(out, minlength=vocab).float() / n
        self.assertLess((freq - want).abs().max().item(), 5e-3)

    def test_residual_excludes_the_draft_favourite(self):
        # p splits between tokens 2 and 5; the draft put all its mass on 2.
        vocab, rows, bs = 64, 3, 512
        target = torch.full((bs * rows, vocab), -1e4, device="cuda")
        target[:, 2] = 0.0
        target[:, 5] = 0.0
        draft = torch.full((bs, rows - 1, vocab), -1e4, device="cuda")
        draft[:, :, 2] = 0.0
        correct_len = torch.zeros(bs, dtype=torch.int64, device="cuda")
        out = _residual_draw(target, draft, correct_len, rows, 1.0, 1.0, seed=9)
        self.assertTrue(torch.all(out == 5))

    def test_all_accepted_row_draws_from_target(self):
        torch.manual_seed(4)
        vocab, rows, n = 16, 4, 200_000
        base_t = torch.randn(vocab, device="cuda")
        target = base_t.repeat(n * rows, 1)
        draft = torch.randn(vocab, device="cuda").repeat(n, rows - 1, 1)
        correct_len = torch.full((n,), rows - 1, dtype=torch.int64, device="cuda")
        out = _residual_draw(target, draft, correct_len, rows, 0.8, 0.8, seed=7)
        freq = torch.bincount(out, minlength=vocab).float() / n
        want = torch.softmax(base_t / 0.8, -1)
        self.assertLess((freq - want).abs().max().item(), 5e-3)

    def test_argmax_rows_keep_argmax(self):
        torch.manual_seed(8)
        vocab, rows, bs = 1000, 4, 32
        target = torch.randn(bs * rows, vocab, device="cuda")
        draft = torch.randn(bs, rows - 1, vocab, device="cuda")
        correct_len = torch.randint(0, rows, (bs,), device="cuda")
        out = _residual_draw(target, draft, correct_len, rows, 0.0, 1.0, seed=3)
        ref = target.view(bs, rows, vocab).argmax(-1)[torch.arange(bs), correct_len]
        self.assertTrue(torch.equal(out, ref))

    def test_residual_with_empty_vocab_splits(self):
        # 5 vocab blocks pad to 8 splits; the empty splits must not poison the logsumexp.
        torch.manual_seed(12)
        vocab, rows, n = 20480, 3, 4096
        base_t = torch.randn(vocab, device="cuda")
        target = base_t.repeat(n * rows, 1)
        draft = (base_t + torch.randn(vocab, device="cuda")).repeat(n, rows - 1, 1)
        correct_len = torch.ones(n, dtype=torch.int64, device="cuda")
        out = _residual_draw(target, draft, correct_len, rows, 1.0, 1.0, seed=2)
        p = torch.softmax(base_t, -1)
        q = torch.softmax(draft[0, 0], -1)
        self.assertTrue(torch.all((p - q)[out] > 0))

    def test_narrow_draft_vocab_and_graph_capture(self):
        torch.manual_seed(6)
        vocab, rows, bs, width = 129280, 4, 8, 129280 - 280
        target = torch.randn(bs * rows, vocab, device="cuda", dtype=torch.bfloat16)
        draft = torch.randn(bs, rows - 1, vocab, device="cuda", dtype=torch.bfloat16)
        draft = draft[..., :width].contiguous()
        correct_len = torch.randint(0, rows, (bs,), device="cuda")
        temps = torch.full((bs,), 1.0, device="cuda")
        seed = torch.full((1,), 11, dtype=torch.int32, device="cuda")

        def draw():
            return simulated_bonus_sample_graph(
                target_logits=target,
                correct_len=correct_len,
                temperatures=temps,
                seed=seed,
                bs=bs,
                rows_per_request=rows,
                draft_logits=draft,
                draft_temperatures=temps,
            )

        eager = draw()
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            draw()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(graph):
            out = draw()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, eager))
        self.assertTrue(torch.all((out >= 0) & (out < vocab)))


if __name__ == "__main__":
    unittest.main()
