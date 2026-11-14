"""DFlash forced acceptance (SGLANG_SIMULATE_ACC_LEN): the fused CUDA op must write
the same outputs as the eager path, and both must match hand-derived values."""

import itertools
import unittest

import torch

from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

POISON = -7
OUTPUT_NAMES = ("accept_len", "commit_lens", "bonus", "out_tokens", "new_seq_lens")


def _poisoned(shape, dtype, device):
    return torch.full(shape, POISON, dtype=dtype, device=device)


def _outputs(*, bs, block_size, device, strided):
    # Returns (views the op writes, backing buffers compared in full).
    if not strided:
        views = (
            _poisoned((bs,), torch.int32, device),
            _poisoned((bs,), torch.int32, device),
            _poisoned((bs,), torch.int64, device),
            _poisoned((bs, block_size), torch.int64, device),
        )
        return views, views
    # int32 bonus/out_tokens behind non-unit strides, int64 lengths.
    accept_len = _poisoned((bs,), torch.int64, device)
    commit_lens = _poisoned((bs,), torch.int64, device)
    bonus = _poisoned((2 * bs,), torch.int32, device)
    out_tokens = _poisoned((block_size + 2, bs + 1), torch.int32, device)
    views = (
        accept_len,
        commit_lens,
        bonus[::2],
        out_tokens.t()[:bs, 1 : 1 + block_size],
    )
    return views, (accept_len, commit_lens, bonus, out_tokens)


def _apply(*, device, inputs, token_mode, mean, from_logits, strided):
    candidates, target_predict, logits, prefix_lens = (t.to(device) for t in inputs)
    bs, block_size = candidates.shape
    views, backings = _outputs(
        bs=bs, block_size=block_size, device=device, strided=strided
    )
    # The worker passes new_seq_lens only when the greedy Triton accept wrote it.
    new_seq_lens = None if strided else _poisoned((bs,), torch.int64, device)
    returned = apply_dflash_simulated_acceptance(
        candidates=candidates,
        target_predict=None if from_logits else target_predict,
        accept_len=views[0],
        commit_lens=views[1],
        bonus=views[2],
        out_tokens=views[3],
        simulate_acc_len=mean,
        simulate_acc_method="match-expected",
        simulate_acc_token_mode=token_mode,
        next_token_logits=logits if from_logits else None,
        prefix_lens=prefix_lens,
        new_seq_lens=new_seq_lens,
    )
    if returned is None:
        # The caller's own recompute when new_seq_lens comes back None.
        returned = prefix_lens + views[1].to(prefix_lens.dtype)
    return [t.cpu() for t in (*backings, returned)]


class TestDFlashSimulatedAccept(CustomTestCase):
    def test_hand_built(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            candidates = torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]], device=device
            )
            target_predict = torch.tensor(
                [[30, 31, 32, 33], [40, 41, 42, 43]], device=device
            )
            accept_len = _poisoned((2,), torch.int32, device)
            commit_lens = _poisoned((2,), torch.int32, device)
            bonus = _poisoned((2,), torch.int64, device)
            out_tokens = _poisoned((2, 4), torch.int64, device)
            common = dict(
                candidates=candidates,
                accept_len=accept_len,
                commit_lens=commit_lens,
                bonus=bonus,
                out_tokens=out_tokens,
                simulate_acc_method="match-expected",
            )

            apply_dflash_simulated_acceptance(
                **common,
                target_predict=target_predict,
                simulate_acc_len=3.0,
                simulate_acc_token_mode="real-draft-token",
            )
            self.assertEqual(accept_len.tolist(), [2, 2])
            self.assertEqual(commit_lens.tolist(), [3, 3])
            self.assertEqual(bonus.tolist(), [32, 42])
            self.assertEqual(out_tokens.tolist(), [[11, 12, 32, 0], [21, 22, 42, 0]])

            apply_dflash_simulated_acceptance(
                **common,
                target_predict=None,
                simulate_acc_len=2.0,
                simulate_acc_token_mode="fixed",
            )
            self.assertEqual(accept_len.tolist(), [1, 1])
            self.assertEqual(commit_lens.tolist(), [2, 2])
            self.assertEqual(bonus.tolist(), [100, 100])
            self.assertEqual(out_tokens.tolist(), [[100] * 4, [100] * 4])

            # Forced row 1 of each request; each row has a different argmax.
            logits = torch.zeros(8, 5, device=device)
            logits[torch.arange(8), torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])] = 1.0
            apply_dflash_simulated_acceptance(
                **common,
                target_predict=None,
                simulate_acc_len=2.0,
                simulate_acc_token_mode="real-draft-token",
                next_token_logits=logits,
            )
            self.assertEqual(bonus.tolist(), [1, 0])
            self.assertEqual(out_tokens.tolist(), [[11, 1, 0, 0], [21, 0, 0, 0]])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_matches_eager(self):
        gen = torch.Generator().manual_seed(0)
        # An unaligned vocab keeps sampling_argmax on torch.argmax, so only the
        # fused op is JIT-built.
        bs, block_size, vocab = 5, 8, 257
        inputs = (
            torch.randint(0, vocab, (bs, block_size), generator=gen),
            torch.randint(0, vocab, (bs, block_size), generator=gen),
            torch.randn(bs * block_size, vocab, generator=gen),
            torch.randint(1, 4096, (bs,), generator=gen),
        )
        # Integer means make match-expected deterministic: lengths 1, 3 and block_size.
        for token_mode, from_logits, mean, strided in itertools.product(
            ("fixed", "real-draft-token"),
            (False, True),
            (1.0, 3.0, float(block_size)),
            (False, True),
        ):
            case = dict(
                token_mode=token_mode,
                mean=mean,
                from_logits=from_logits,
                strided=strided,
            )
            with self.subTest(**case):
                want = _apply(device="cpu", inputs=inputs, **case)
                got = _apply(device="cuda", inputs=inputs, **case)
                for name, g, w in zip(OUTPUT_NAMES, got, want):
                    self.assertTrue(torch.equal(g, w), f"{name}: {g} != {w}")


if __name__ == "__main__":
    unittest.main()
