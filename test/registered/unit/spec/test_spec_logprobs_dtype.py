"""Speculative-decoding logprobs must match the sampler's under temperature.

The sampler divides logits by the temperatures in place, so bf16 logits (a
true-on-policy contract) stay bf16 into log_softmax. compute_spec_logprobs used
an out-of-place division by the fp32 temperatures, which promoted them to fp32,
so accepted-token logprobs differed from non-speculative decoding and the
trainer's bf16 log_softmax.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, STRIDE, VOCAB, TOP_K = 2, 3, 64, 4


def _sampler_logprobs(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    # Sampler.forward: logits.div_(sampling_info.temperatures), then log_softmax
    return torch.log_softmax(logits.clone().div_(temperatures), dim=-1)


class TestSpecLogprobsDtype(CustomTestCase):
    def _check(self, dtype: torch.dtype) -> None:
        generator = torch.Generator().manual_seed(0)
        logits = (torch.randn(BS * STRIDE, VOCAB, generator=generator) * 4).to(dtype)
        temperatures = torch.tensor([[0.7], [0.6]], dtype=torch.float32)
        predict = torch.randint(0, VOCAB, (BS * STRIDE,), generator=generator)
        batch = SimpleNamespace(
            seq_lens=[5] * BS,
            sampling_info=SimpleNamespace(
                is_all_greedy=False, temperatures=temperatures
            ),
            top_logprobs_nums=[TOP_K] * BS,
            token_ids_logprobs=None,
        )
        output = SimpleNamespace(next_token_logits=logits)

        compute_spec_logprobs(batch, output, predict, chain_stride=STRIDE)

        expected = _sampler_logprobs(
            logits, torch.repeat_interleave(temperatures, STRIDE, dim=0)
        )
        self.assertEqual(output.next_token_logprobs.dtype, dtype)
        self.assertTrue(
            torch.equal(
                output.next_token_logprobs.view(-1),
                expected.gather(1, predict.view(-1, 1)).view(-1),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.stack(output.next_token_top_logprobs_val),
                expected.topk(TOP_K, dim=-1).values,
            )
        )

    def test_bf16_logits_stay_bf16_like_the_sampler(self):
        self._check(torch.bfloat16)

    def test_fp32_logits_unchanged(self):
        self._check(torch.float32)


if __name__ == "__main__":
    unittest.main()
