"""DFlash forced acceptance (SGLANG_SIMULATE_ACC_LEN) must stay byte-identical to the
eager implementation it replaced: every output, the untouched memory around them,
and the CPU RNG state after the call. Integer outputs, so the tolerance is exact."""

import itertools
import unittest
from typing import Optional

import torch

from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
from sglang.srt.speculative.spec_utils import sample_simulated_acc_len
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

POISON = -7
TOKEN_MODES = ("fixed", "real-draft-token")
METHODS = ("match-expected", "multinomial")
LAYOUTS = ("fresh", "buffer", "strided")
_I32, _I64 = torch.int32, torch.int64
# (candidates, target, accept/commit lengths, bonus, out_tokens); the worker's first.
DTYPE_SETS = tuple(
    dict(zip(("cand", "target", "accept", "bonus", "out"), dtypes))
    for dtypes in (
        (_I64, _I64, _I32, _I64, _I64),
        (_I32, _I32, _I64, _I32, _I32),
        (_I64, _I64, _I32, _I32, _I64),
        (_I32, _I64, _I64, _I64, _I64),
        (_I64, _I64, _I32, _I64, _I32),
    )
)


def _eager_reference(
    *,
    candidates: torch.Tensor,
    target_predict: Optional[torch.Tensor],
    accept_len: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus: torch.Tensor,
    out_tokens: torch.Tensor,
    simulate_acc_len: float,
    simulate_acc_method: str,
    simulate_acc_token_mode: str,
    next_token_logits: Optional[torch.Tensor] = None,
    fixed_token_id: int = 100,
) -> None:
    # Frozen copy of the eager apply_dflash_simulated_acceptance body (main c757c4bb7c).
    bs, block_size = candidates.shape
    if simulate_acc_token_mode == "real-draft-token" and target_predict is None:
        # The sampling-verify branch does not materialize the target argmax.
        target_predict = torch.argmax(next_token_logits, dim=-1).view(bs, block_size)

    # sample_simulated_acc_len clamps to [1, block_size].
    forced_commit_len = sample_simulated_acc_len(
        simulate_acc_len, simulate_acc_method, block_size
    )
    forced_accept_len = forced_commit_len - 1

    accept_len.fill_(forced_accept_len)
    commit_lens.fill_(forced_commit_len)

    if simulate_acc_token_mode != "real-draft-token":
        bonus.fill_(fixed_token_id)
        out_tokens.fill_(fixed_token_id)
        return

    out_tokens.zero_()
    if forced_accept_len > 0:
        out_tokens[:, :forced_accept_len].copy_(candidates[:, 1:forced_commit_len])
    bonus.copy_(target_predict[:, forced_accept_len].to(dtype=bonus.dtype))
    out_tokens[:, forced_accept_len].copy_(bonus.to(dtype=out_tokens.dtype))


def _alloc(*, shape, dtype, device, layout):
    # Returns (view handed to the op, poisoned backing storage compared in full).
    bs = shape[0]
    if layout == "fresh":
        backing = torch.full(shape, POISON, dtype=dtype, device=device)
        return backing, backing
    if layout == "buffer":
        backing = torch.full((bs + 3, *shape[1:]), POISON, dtype=dtype, device=device)
        return backing[:bs], backing
    if len(shape) == 2:
        # Neither stride is the contiguous one: a window of a transposed buffer.
        backing = torch.full((shape[1] + 2, bs + 3), POISON, dtype=dtype, device=device)
        return backing.t()[:bs, 1 : 1 + shape[1]], backing
    backing = torch.full((2 * bs + 3,), POISON, dtype=dtype, device=device)
    return backing[: 2 * bs : 2], backing


def _int_matrix(*, bs, block_size, dtype, device, layout, gen, low, high):
    if layout == "strided":
        # Column stride != 1: the transpose of a [block_size, bs] tensor.
        values = torch.randint(low, high, (block_size, bs), generator=gen).t()
    else:
        values = torch.randint(low, high, (bs, block_size), generator=gen)
    return values.to(dtype).to(device)


def _logits_with_ties(*, rows, vocab, dtype, device, seed):
    # Every row's maximum appears at three random columns, and every fifth row
    # holds two NaNs, so argmax must return the first index.
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=gen, device=device).to(dtype)
    tied = torch.randint(0, vocab, (rows, 3), generator=gen, device=device)
    top = (logits.max(dim=-1).values + 1).unsqueeze(1).expand(rows, 3)
    logits.scatter_(1, tied, top)
    nan_rows = torch.arange(2, rows, 5, device=device)
    nan_cols = torch.randint(
        0, vocab, (nan_rows.numel(), 2), generator=gen, device=device
    )
    logits[nan_rows.unsqueeze(1), nan_cols] = float("nan")
    return logits


class _ForcedAcceptCase:
    def __init__(
        self, *, device, bs, block_size, layout, dtypes, gen, vocab, large_ids
    ):
        self.device = device
        self.bs = bs
        self.block_size = block_size
        self.layout = layout
        self.dtypes = dtypes
        # Ids past int32 check that int64 -> int32 stores wrap the way torch copies do.
        low, high = (2**31 - 8, 2**31 + 8) if large_ids else (0, vocab)
        self.candidates = _int_matrix(
            bs=bs,
            block_size=block_size,
            dtype=dtypes["cand"],
            device=device,
            layout=layout,
            gen=gen,
            low=low,
            high=high,
        )
        self.target_predict = _int_matrix(
            bs=bs,
            block_size=block_size,
            dtype=dtypes["target"],
            device=device,
            layout=layout,
            gen=gen,
            low=low,
            high=high,
        )
        self.prefix_lens = torch.randint(1, 4096, (bs,), generator=gen).to(device)

    def outputs(self):
        bs, block_size = self.bs, self.block_size
        views, backings = zip(
            *(
                _alloc(shape=shape, dtype=dtype, device=self.device, layout=self.layout)
                for shape, dtype in (
                    ((bs,), self.dtypes["accept"]),
                    ((bs,), self.dtypes["accept"]),
                    ((bs,), self.dtypes["bonus"]),
                    ((bs, block_size), self.dtypes["out"]),
                    ((bs,), torch.int64),
                )
            )
        )
        return views, backings


class TestDFlashSimulatedAccept(CustomTestCase):
    def _check_rounds(
        self,
        *,
        case,
        token_mode,
        method,
        mean,
        logits=None,
        rounds=4,
        seed=0,
        seen_lens=None,
        pass_new_seq_lens=True,
    ):
        # Successive rounds on one RNG stream; each round must match the reference
        # in every output and in the CPU RNG state it leaves behind. Without
        # new_seq_lens (the sampling branch) the op must leave that buffer alone.
        torch.manual_seed(seed)
        target_predict = None if logits is not None else case.target_predict
        for _ in range(rounds):
            rng_before = torch.get_rng_state()
            (ref_views, ref_backings) = case.outputs()
            _eager_reference(
                candidates=case.candidates,
                target_predict=target_predict,
                accept_len=ref_views[0],
                commit_lens=ref_views[1],
                bonus=ref_views[2],
                out_tokens=ref_views[3],
                simulate_acc_len=mean,
                simulate_acc_method=method,
                simulate_acc_token_mode=token_mode,
                next_token_logits=logits,
            )
            rng_ref = torch.get_rng_state()

            torch.set_rng_state(rng_before)
            (views, backings) = case.outputs()
            returned = apply_dflash_simulated_acceptance(
                candidates=case.candidates,
                target_predict=target_predict,
                accept_len=views[0],
                commit_lens=views[1],
                bonus=views[2],
                out_tokens=views[3],
                simulate_acc_len=mean,
                simulate_acc_method=method,
                simulate_acc_token_mode=token_mode,
                next_token_logits=logits,
                prefix_lens=case.prefix_lens,
                new_seq_lens=views[4] if pass_new_seq_lens else None,
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_ref))

            for name, got, want in zip(
                ("accept_len", "commit_lens", "bonus", "out_tokens"),
                backings[:4],
                ref_backings[:4],
            ):
                self.assertEqual(got.dtype, want.dtype, name)
                if not torch.equal(got, want):
                    self.fail(f"{name}: got {got.tolist()}, want {want.tolist()}")

            # The caller's own rule when new_seq_lens comes back None.
            ref_new_seq_lens = case.prefix_lens + ref_views[1].to(
                case.prefix_lens.dtype
            )
            if returned is None:
                self.assertTrue(torch.equal(backings[4], ref_backings[4]))
            else:
                self.assertIs(returned, views[4])
                self.assertTrue(torch.equal(returned, ref_new_seq_lens))
                rest = backings[4].clone()
                rest_ref = ref_backings[4].clone()
                if case.layout == "strided":
                    rest[: 2 * case.bs : 2] = POISON
                else:
                    rest[: case.bs] = POISON
                self.assertTrue(torch.equal(rest, rest_ref), "new_seq_lens tail")
            if seen_lens is not None:
                seen_lens.add(int(ref_views[1][0]))

    def _matrix(self, *, device, batch_sizes, block_sizes):
        case_index = 0
        for bs, block_size in itertools.product(batch_sizes, block_sizes):
            seen_lens = set()
            means = (
                0.5,
                1.0,
                block_size / 2 + 0.25,
                block_size - 0.5,
                block_size + 2.0,
            )
            for token_mode, method, from_logits, mean in itertools.product(
                TOKEN_MODES, METHODS, (False, True), means
            ):
                layout = LAYOUTS[case_index % len(LAYOUTS)]
                gen = torch.Generator().manual_seed(case_index)
                case = _ForcedAcceptCase(
                    device=device,
                    bs=bs,
                    block_size=block_size,
                    layout=layout,
                    dtypes=DTYPE_SETS[0],
                    gen=gen,
                    vocab=257,
                    large_ids=False,
                )
                logits = (
                    _logits_with_ties(
                        rows=bs * block_size,
                        vocab=257,
                        dtype=torch.float32,
                        device=device,
                        seed=case_index,
                    )
                    if from_logits
                    else None
                )
                with self.subTest(
                    bs=bs,
                    block_size=block_size,
                    token_mode=token_mode,
                    method=method,
                    from_logits=from_logits,
                    mean=mean,
                    layout=layout,
                    new_seq_lens=case_index % 2 == 0,
                ):
                    self._check_rounds(
                        case=case,
                        token_mode=token_mode,
                        method=method,
                        mean=mean,
                        logits=logits,
                        seed=case_index,
                        seen_lens=seen_lens,
                        pass_new_seq_lens=case_index % 2 == 0,
                    )
                case_index += 1
            # The matrix must reach both clamps and an interior length.
            self.assertIn(1, seen_lens)
            self.assertIn(block_size, seen_lens)
            self.assertTrue(seen_lens - {1, block_size}, (bs, block_size, seen_lens))

    def test_reference_hand_built(self):
        """Pins the frozen reference (and the op) to hand-derived values."""
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            candidates = torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]], device=device
            )
            target_predict = torch.tensor(
                [[30, 31, 32, 33], [40, 41, 42, 43]], device=device
            )
            for fn in (_eager_reference, apply_dflash_simulated_acceptance):
                accept_len = torch.full((2,), POISON, dtype=torch.int32, device=device)
                commit_lens = torch.full((2,), POISON, dtype=torch.int32, device=device)
                bonus = torch.full((2,), POISON, dtype=torch.int64, device=device)
                out_tokens = torch.full(
                    (2, 4), POISON, dtype=torch.int64, device=device
                )
                fn(
                    candidates=candidates,
                    target_predict=target_predict,
                    accept_len=accept_len,
                    commit_lens=commit_lens,
                    bonus=bonus,
                    out_tokens=out_tokens,
                    simulate_acc_len=3.0,
                    simulate_acc_method="match-expected",
                    simulate_acc_token_mode="real-draft-token",
                )
                self.assertEqual(accept_len.tolist(), [2, 2])
                self.assertEqual(commit_lens.tolist(), [3, 3])
                self.assertEqual(bonus.tolist(), [32, 42])
                self.assertEqual(
                    out_tokens.tolist(), [[11, 12, 32, 0], [21, 22, 42, 0]]
                )

                fn(
                    candidates=candidates,
                    target_predict=None,
                    accept_len=accept_len,
                    commit_lens=commit_lens,
                    bonus=bonus,
                    out_tokens=out_tokens,
                    simulate_acc_len=2.0,
                    simulate_acc_method="match-expected",
                    simulate_acc_token_mode="fixed",
                )
                self.assertEqual(accept_len.tolist(), [1, 1])
                self.assertEqual(commit_lens.tolist(), [2, 2])
                self.assertEqual(bonus.tolist(), [100, 100])
                self.assertEqual(out_tokens.tolist(), [[100] * 4, [100] * 4])

                # Forced row 1 of the single request has its maximum 3 at columns 1, 3, 4.
                logits = torch.tensor(
                    [
                        [0.0, 9.0, 0.0, 0.0, 0.0],
                        [0.0, 3.0, 1.0, 3.0, 3.0],
                        [9.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    device=device,
                )
                one_bonus = torch.full((1,), POISON, dtype=torch.int64, device=device)
                one_out = torch.full((1, 3), POISON, dtype=torch.int64, device=device)
                fn(
                    candidates=torch.tensor([[5, 6, 7]], device=device),
                    target_predict=None,
                    accept_len=accept_len[:1],
                    commit_lens=commit_lens[:1],
                    bonus=one_bonus,
                    out_tokens=one_out,
                    simulate_acc_len=2.0,
                    simulate_acc_method="match-expected",
                    simulate_acc_token_mode="real-draft-token",
                    next_token_logits=logits,
                )
                self.assertEqual(one_bonus.tolist(), [1])
                self.assertEqual(one_out.tolist(), [[6, 1, 0]])

    def test_eager_fallback_matches_reference(self):
        """Non-CUDA devices keep the eager ops but read only the forced logits row."""
        self._matrix(device="cpu", batch_sizes=(1, 7), block_sizes=(5, 16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_matches_reference(self):
        self._matrix(
            device="cuda", batch_sizes=(1, 2, 7, 64), block_sizes=(4, 5, 8, 16)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dtypes_and_wide_ids(self):
        case_index = 0
        for dtypes, large_ids, layout, token_mode, block_size in itertools.product(
            DTYPE_SETS, (False, True), LAYOUTS, TOKEN_MODES, (5, 16)
        ):
            gen = torch.Generator().manual_seed(1000 + case_index)
            case = _ForcedAcceptCase(
                device="cuda",
                bs=7,
                block_size=block_size,
                layout=layout,
                dtypes=dtypes,
                gen=gen,
                vocab=257,
                large_ids=large_ids,
            )
            pass_new_seq_lens = case_index % 2 == 0
            with self.subTest(
                dtypes=dtypes,
                large_ids=large_ids,
                layout=layout,
                token_mode=token_mode,
                new_seq_lens=pass_new_seq_lens,
            ):
                self._check_rounds(
                    case=case,
                    token_mode=token_mode,
                    method="multinomial",
                    mean=block_size / 2,
                    rounds=6,
                    seed=case_index,
                    pass_new_seq_lens=pass_new_seq_lens,
                )
            case_index += 1

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_argmax_ties_at_real_vocab(self):
        """The forced-row argmax keeps torch.argmax's choice (first NaN, else the
        first maximum) at real vocabulary sizes and logits dtypes."""
        case_index = 0
        for vocab, bs, dtype in itertools.product(
            (32000, 128256), (1, 7, 64), (torch.float32, torch.bfloat16, torch.float16)
        ):
            gen = torch.Generator().manual_seed(2000 + case_index)
            case = _ForcedAcceptCase(
                device="cuda",
                bs=bs,
                block_size=16,
                layout="buffer",
                dtypes=DTYPE_SETS[0],
                gen=gen,
                vocab=vocab,
                large_ids=False,
            )
            logits = _logits_with_ties(
                rows=bs * 16, vocab=vocab, dtype=dtype, device="cuda", seed=case_index
            )
            with self.subTest(vocab=vocab, bs=bs, dtype=dtype):
                self._check_rounds(
                    case=case,
                    token_mode="real-draft-token",
                    method="multinomial",
                    mean=8.0,
                    logits=logits,
                    rounds=6,
                    seed=case_index,
                    pass_new_seq_lens=case_index % 2 == 0,
                )
            del logits
            case_index += 1


if __name__ == "__main__":
    unittest.main()
