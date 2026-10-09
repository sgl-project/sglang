from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
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
        # Two shards of the vocab must pick the same tokens as one full vocab:
        # the noise is keyed by global token id, so every rank draws the same.
        g = torch.Generator(device=DEVICE).manual_seed(1)
        rows, shard = 32, 5000
        base = torch.randn(rows, 2 * shard, generator=g, device=DEVICE)
        bias = torch.randn(rows, 2 * shard - 7, generator=g, device=DEVICE)
        temps = torch.rand(rows, generator=g, device=DEVICE)
        temps[:8] = 0.0
        full = _call(
            MarkovSampleStep.triton,
            base=base,
            bias=bias,
            temps=temps,
            step_dtype=torch.float32,
        )
        pieces = [None, None]

        def gather_from(rank):
            def gather(partial):
                pieces[rank] = partial.clone()
                other = pieces[1 - rank]
                if other is None:
                    other = torch.zeros_like(partial)
                ordered = [partial, other] if rank == 0 else [other, partial]
                return torch.cat(ordered).contiguous()

            return gather

        group = SimpleNamespace(world_size=2)
        for rank in (0, 1):
            start = rank * shard
            sharded = _call(
                MarkovSampleStep.triton,
                base=base[:, start : start + shard].contiguous(),
                bias=bias[:, start : start + shard].contiguous(),
                temps=temps,
                step_dtype=torch.float32,
                vocab_start=start,
                group=group,
                gather=gather_from(rank),
            )
        self.assertTrue(torch.equal(sharded, full))


if __name__ == "__main__":
    unittest.main()
