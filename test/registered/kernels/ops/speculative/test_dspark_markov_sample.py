from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
from sglang.kernels.ops.speculative.dspark.markov_topk import markov_walk_topk
from sglang.test.test_utils import CustomTestCase

DEVICE = torch.device("cuda")


def _offset(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int64, device=DEVICE)


def _call(impl, *, base, bias, temps, step_dtype, offset=7, step=2, **kwargs):
    return impl(
        base_logits=base,
        bias=bias,
        temperatures=temps,
        key=12345,
        offset=_offset(offset),
        step=step,
        step_dtype=step_dtype,
        **kwargs,
    )


class TestMarkovSampleStep(CustomTestCase):
    def test_topk_walk_matches_dense_candidate_proposal(self):
        # Non-power-of-two K/rank exercise masked lanes; each row walks four
        # dependent steps. Static cache entries override live overlap scores.
        for dtype, scale, mapped, rank in (
            (torch.bfloat16, 1.0, False, 129),
            (torch.float16, -1.0, True, 65),
            (torch.float32, -0.0, True, 5),
        ):
            with self.subTest(dtype=dtype, scale=scale, mapped=mapped, rank=rank):
                g = torch.Generator().manual_seed(19)
                batch, steps, vocab, target_vocab, k = 3, 4, 7, 11, 3
                w1 = torch.randint(-2, 3, (target_vocab, rank), generator=g).to(dtype)
                w2 = torch.randint(-2, 3, (vocab, rank), generator=g).to(dtype)
                w1[[0, 1, 3, 6]] = 0
                correction = scale * (w1.float() @ w2.float().T)
                biases, ids = correction.topk(k, dim=-1)
                biases[2, 0] += 100  # Distinguish the cache from recomputing W1/W2.
                base = torch.randint(-4, 5, (batch, steps, vocab), generator=g).to(
                    dtype
                )
                base[0, :, ids[2, 0]] = 10  # Cache/live overlap.
                base[1] = -torch.inf
                base[2] = 0  # Ties choose the lowest target ID, not draft ID.
                base[0, -1, 0] = torch.nan
                target_ids = (
                    torch.tensor([8, 5, 9, 1, 7, 3, 6])
                    if mapped
                    else torch.arange(vocab)
                )
                d2t = target_ids - torch.arange(vocab) if mapped else None
                anchor = torch.tensor([2, 4, 6])
                values, base_ids = base.topk(k, dim=-1, sorted=False)
                expected = torch.empty(batch, steps, dtype=torch.int64)
                prev = anchor
                for step in range(steps):
                    scores = base[:, step].float() + correction[prev]
                    scores.scatter_(
                        1,
                        ids[prev],
                        base[:, step].float().gather(1, ids[prev]) + biases[prev],
                    )
                    mask = torch.zeros(batch, vocab, dtype=torch.bool)
                    mask.scatter_(1, base_ids[:, step], True)
                    mask.scatter_(1, ids[prev], True)
                    scores.masked_fill_(~mask, -torch.inf)
                    logits = torch.full((batch, target_vocab), -torch.inf)
                    logits[:, target_ids] = scores
                    prev = logits.argmax(-1)
                    fallback = (
                        target_ids.expand(batch, -1)
                        .masked_fill(~mask, target_vocab)
                        .amin(-1)
                    )
                    prev = torch.where(scores.isfinite().any(-1), prev, fallback)
                    expected[:, step] = prev
                actual = markov_walk_topk(
                    base_logits=base.to(DEVICE),
                    base_values=values.to(DEVICE),
                    base_ids=base_ids.to(DEVICE),
                    markov_ids=ids.to(DEVICE),
                    markov_biases=biases.to(DEVICE),
                    w1=w1.to(DEVICE),
                    w2=w2.to(DEVICE),
                    anchor_tokens=anchor.to(DEVICE),
                    d2t=None if d2t is None else d2t.to(DEVICE),
                    scale=scale,
                )
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_triton_matches_torch_reference(self):
        g = torch.Generator(device=DEVICE).manual_seed(0)
        rows, padded, width = 64, 9000, 8990
        for base_dtype, bias_dtype, step_dtype in [
            (torch.bfloat16, torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.bfloat16, torch.float32),
            (torch.float32, torch.float32, torch.float32),
        ]:
            base = torch.randn(rows, padded, generator=g, device=DEVICE) * 4
            base = base.to(base_dtype)
            bias = torch.randn(rows, width, generator=g, device=DEVICE).to(bias_dtype)
            # Half greedy rows, half sampled at mixed temperatures.
            temps = torch.rand(rows, generator=g, device=DEVICE) * 1.5 + 0.1
            temps[::2] = 0.0
            out = {}
            for name, impl in [
                ("triton", MarkovSampleStep.triton),
                ("torch", MarkovSampleStep.torch),
            ]:
                logits = torch.empty(rows, padded, dtype=step_dtype, device=DEVICE)
                tokens = _call(
                    impl,
                    base=base,
                    bias=bias,
                    temps=temps,
                    step_dtype=step_dtype,
                    vocab_start=100,
                    logits_out=logits,
                )
                out[name] = (tokens, logits)
            (t_tok, t_log), (r_tok, r_log) = out["triton"], out["torch"]
            self.assertTrue(torch.equal(t_log, r_log))
            expected_log = base.float()
            expected_log[:, :width] += bias.float()
            self.assertTrue(torch.equal(t_log, expected_log.to(step_dtype)))
            greedy = temps == 0
            expected_greedy = t_log.float()[:, :width].argmax(dim=1) + 100
            self.assertTrue(torch.equal(t_tok[greedy], expected_greedy[greedy]))
            # Same Philox bits on both sides; only a last-ulp log difference on a
            # near tie could split them.
            agree = (t_tok[~greedy] == r_tok[~greedy]).float().mean().item()
            self.assertGreaterEqual(agree, 0.95)
            self.assertTrue(((t_tok >= 100) & (t_tok < 100 + width)).all())

    def test_greedy_only_variant_matches_mixed_kernel(self):
        g = torch.Generator(device=DEVICE).manual_seed(3)
        base = torch.randn(16, 9000, generator=g, device=DEVICE).to(torch.bfloat16)
        bias = torch.randn(16, 8990, generator=g, device=DEVICE).to(torch.bfloat16)
        temps = torch.zeros(16, device=DEVICE)
        kwargs = dict(base=base, bias=bias, temps=temps, step_dtype=torch.bfloat16)
        mixed = _call(MarkovSampleStep.triton, **kwargs)
        for impl in (MarkovSampleStep.triton, MarkovSampleStep.torch):
            greedy_only = _call(impl, has_sampled_rows=False, **kwargs)
            self.assertTrue(torch.equal(greedy_only, mixed))

    def test_special_values_follow_argmax(self):
        base = torch.zeros(4, 16, device=DEVICE)
        base[0, 3] = float("nan")
        base[0, 9] = float("nan")
        base[1, :] = -float("inf")
        base[2, 5] = 2.0
        base[2, 11] = 2.0
        base[3, 15] = 100.0
        bias = torch.zeros(4, 15, device=DEVICE)
        temps = torch.zeros(4, device=DEVICE)
        for impl in (MarkovSampleStep.triton, MarkovSampleStep.torch):
            tokens = _call(
                impl, base=base, bias=bias, temps=temps, step_dtype=torch.float32
            )
            # Lowest NaN, then lowest index of a tie; column 15 is padding.
            self.assertEqual(tokens.tolist(), [3, 0, 5, 0])

    def test_greedy_tie_across_blocks_takes_the_lower_index(self):
        base = torch.zeros(2, 9000, device=DEVICE)
        base[:, 1000] = base[:, 5000] = 5.0
        bias = torch.zeros(2, 9000, device=DEVICE)
        for impl in (MarkovSampleStep.triton, MarkovSampleStep.torch):
            tokens = _call(
                impl,
                base=base,
                bias=bias,
                temps=torch.zeros(2, device=DEVICE),
                step_dtype=torch.float32,
            )
            self.assertEqual(tokens.tolist(), [1000, 1000])

    def test_strided_rows_match_contiguous(self):
        # The draft passes one step of a [bs, gamma, vocab] block, rows strided.
        g = torch.Generator(device=DEVICE).manual_seed(2)
        block = torch.randn(8, 5, 6000, generator=g, device=DEVICE)
        bias = torch.randn(8, 5990, generator=g, device=DEVICE)
        temps = torch.rand(8, generator=g, device=DEVICE)
        temps[:3] = 0.0
        logits_block = torch.empty(8, 5, 6000, device=DEVICE)
        strided = _call(
            MarkovSampleStep.triton,
            base=block[:, 2],
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
            logits_out=logits_block[:, 2],
        )
        logits = torch.empty(8, 6000, device=DEVICE)
        contiguous = _call(
            MarkovSampleStep.triton,
            base=block[:, 2].contiguous(),
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
            logits_out=logits,
        )
        self.assertTrue(torch.equal(strided, contiguous))
        self.assertTrue(torch.equal(logits_block[:, 2], logits))

    def test_sampling_follows_tempered_softmax(self):
        vocab, rows = 32, 1 << 16
        logits = torch.linspace(-3, 3, vocab, device=DEVICE)
        base = logits.expand(rows, vocab).contiguous()
        bias = torch.zeros(rows, vocab, device=DEVICE)
        for temperature in (0.5, 1.0, 2.0):
            temps = torch.full((rows,), temperature, device=DEVICE)
            tokens = _call(
                MarkovSampleStep.triton,
                base=base,
                bias=bias,
                temps=temps,
                step_dtype=torch.float32,
            )
            freq = torch.bincount(tokens, minlength=vocab).float() / rows
            expected = torch.softmax(logits / temperature, dim=0)
            self.assertLess((freq - expected).abs().max().item(), 0.01)

    def test_noise_changes_with_offset_row_and_step(self):
        rows, vocab = 256, 4096
        base = torch.zeros(rows, vocab, device=DEVICE)
        bias = torch.zeros(rows, vocab, device=DEVICE)
        temps = torch.ones(rows, device=DEVICE)
        first = _call(
            MarkovSampleStep.triton,
            base=base,
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
        )
        self.assertGreater(first.unique().numel(), rows // 2)
        for kwargs in ({"offset": 8}, {"step": 3}):
            other = _call(
                MarkovSampleStep.triton,
                base=base,
                bias=bias,
                temps=temps,
                step_dtype=torch.float32,
                **kwargs,
            )
            self.assertLess((other == first).float().mean().item(), 0.01)

    def test_sharded_vocab_matches_full_vocab(self):
        # Every rank must pick the full vocab's token: the noise is keyed by
        # global token id, so the shards draw what the full vocab draws.
        g = torch.Generator(device=DEVICE).manual_seed(1)
        rows, shard = 32, 5000
        base = torch.randn(rows, 3 * shard, generator=g, device=DEVICE)
        bias = torch.randn(rows, 2 * shard - 7, generator=g, device=DEVICE)
        temps = torch.rand(rows, generator=g, device=DEVICE)
        temps[:8] = 0.0
        full = _call(
            MarkovSampleStep.triton,
            base=base[:, : bias.shape[1]].contiguous(),
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
        )
        # The last shard is empty: all its columns are padding.
        shards = [
            (rank * shard, base[:, rank * shard : (rank + 1) * shard].contiguous())
            for rank in range(3)
        ]
        partials = {}

        def run(rank, gather):
            start, shard_base = shards[rank]
            return _call(
                MarkovSampleStep.triton,
                base=shard_base,
                bias=bias[:, start : start + shard].contiguous(),
                temps=temps,
                step_dtype=torch.float32,
                vocab_start=start,
                gather=gather,
            )

        for rank in range(3):
            run(rank, lambda partial, rank=rank: partials.setdefault(rank, partial))
        stacked = torch.cat([partials[rank] for rank in range(3)])
        for rank in range(3):
            self.assertTrue(torch.equal(run(rank, lambda _: stacked), full))

    def test_kept_sharded_logits_match_full_vocab(self):
        # Kept logits are gathered whole and every rank picks from them: the
        # same tokens and the same logits as the full vocab, both backends.
        g = torch.Generator(device=DEVICE).manual_seed(2)
        rows, shard = 32, 5000
        vocab = 2 * shard - 7
        base = torch.randn(rows, 3 * shard, generator=g, device=DEVICE)
        bias = torch.randn(rows, vocab, generator=g, device=DEVICE)
        temps = torch.rand(rows, generator=g, device=DEVICE)
        temps[:8] = 0.0
        full_logits = torch.empty(rows, vocab, device=DEVICE)
        full = _call(
            MarkovSampleStep.triton,
            base=base[:, :vocab].contiguous(),
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
            logits_out=full_logits,
        )
        for impl in (MarkovSampleStep.triton, MarkovSampleStep.torch):
            locals_ = {}

            def run(rank, gather_logits):
                start = rank * shard
                logits_out = torch.empty(rows, vocab, device=DEVICE)
                tokens = _call(
                    impl,
                    base=base[:, start : start + shard].contiguous(),
                    bias=bias[:, start : start + shard].contiguous(),
                    temps=temps,
                    step_dtype=torch.float32,
                    vocab_start=start,
                    gather=lambda _: self.fail("candidates gathered"),
                    gather_logits=gather_logits,
                    logits_out=logits_out,
                )
                return tokens, logits_out

            for rank in range(3):
                run(
                    rank,
                    lambda local, rank=rank: locals_.setdefault(
                        rank, local.clone()
                    ).repeat(1, 3),
                )
            stacked = torch.cat([locals_[rank] for rank in range(3)], dim=1)
            for rank in range(3):
                tokens, logits_out = run(rank, lambda _: stacked)
                self.assertTrue(torch.equal(tokens, full))
                self.assertTrue(torch.equal(logits_out, full_logits))


if __name__ == "__main__":
    unittest.main()
