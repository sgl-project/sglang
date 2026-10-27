import unittest

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    build_out_tokens,
    build_out_tokens_triton,
)
from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=90, suite="base-b-test-1-npu-a3")


class TestDSparkTritonSampling(unittest.TestCase):
    def test_build_out_tokens(self):
        for bs, gamma in ((1, 0), (1, 7), (33, 7)):
            for dtype in (torch.int32, torch.int64):
                draft = (torch.arange(bs * gamma * 2).reshape(bs, gamma * 2) + 2**40)[
                    :, ::2
                ]
                lengths = (torch.arange(bs * 2) // 2 % (gamma + 1)).to(dtype)[::2]
                bonus = (torch.arange(bs * 2) - 2**40)[::2]
                kwargs = dict(
                    draft_tokens=draft,
                    correct_len=lengths,
                    bonus=bonus,
                    verify_num_draft_tokens=gamma + 1,
                    gamma=gamma,
                )
                expected = build_out_tokens(**kwargs)
                actual = build_out_tokens_triton(
                    **{
                        k: v.npu() if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items()
                    }
                )
                self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_markov_sample_matches_philox_reference(self):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base = torch.randn(8, 4097, dtype=dtype)
            bias = torch.randn_like(base)
            temperatures = torch.linspace(0.2, 1.3, 8)
            temperatures[::2] = 0
            kwargs = dict(
                base_logits=base,
                bias=bias,
                temperatures=temperatures,
                key=12345,
                offset=torch.tensor([7]),
                step=2,
                step_dtype=dtype,
                logits_out=torch.empty_like(base),
            )
            expected = MarkovSampleStep.torch(**kwargs)
            npu_kwargs = {
                k: v.npu() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            actual = MarkovSampleStep.triton(**npu_kwargs)
            self.assertTrue(torch.equal(actual.cpu(), expected))
            self.assertTrue(
                torch.equal(npu_kwargs["logits_out"].cpu(), kwargs["logits_out"])
            )


if __name__ == "__main__":
    unittest.main()
