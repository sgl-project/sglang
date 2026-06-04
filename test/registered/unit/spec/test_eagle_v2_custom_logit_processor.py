"""Regression test for #26330.

The EAGLE-v2 / SPEC_V2 verify path (``EagleVerifyInputV2Mixin.sample``) must
apply custom logit processors, matching the v1 path (``eagle_info.py``). Before
the fix, the v2 ``sample`` skipped ``apply_custom_logit_processor`` entirely, so
custom processors (e.g. a ``thinking_budget`` limiter) were silently ignored
whenever speculative decoding ran through the v2 path — which is the default
(``SGLANG_ENABLE_SPEC_V2`` defaults to ``True``).

This drives ``sample()`` with a spy custom logit processor and asserts it is
invoked on the verify logits. The processor short-circuits by raising, so the
test never reaches the spec-sampling CUDA kernel and stays CPU-only.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.speculative.eagle_info_v2 import EagleVerifyInputV2Mixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _ProcessorApplied(Exception):
    """Raised by the spy processor to prove it was invoked."""


class _SpyLogitProcessor(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        raise _ProcessorApplied()


class _FakeSamplingInfo:
    """Minimal SamplingBatchInfo stand-in carrying one custom logit processor."""

    def __init__(self, bs: int):
        self._bs = bs
        self.has_custom_logit_processor = True
        self.custom_logit_processor = {
            "spy": (_SpyLogitProcessor(), torch.ones(bs, dtype=torch.bool))
        }
        self.custom_params = [{} for _ in range(bs)]

    def __len__(self) -> int:
        return self._bs


class TestEagleV2AppliesCustomLogitProcessor(unittest.TestCase):
    def test_v2_verify_applies_custom_logit_processor(self):
        bs, draft_token_num, vocab = 2, 3, 16

        batch = SimpleNamespace(
            device="cpu",
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=torch.zeros(bs),
            sampling_info=_FakeSamplingInfo(bs),
        )
        logits_output = SimpleNamespace(
            # v2 verify logits shape: (bs * draft_token_num, vocab)
            next_token_logits=torch.randn(bs * draft_token_num, vocab),
        )
        verify_input = SimpleNamespace(draft_token_num=draft_token_num)

        # With the fix, sample() applies the custom logit processor (which raises
        # to signal it ran). Without the fix, the processor is never called and
        # this assertion fails.
        with self.assertRaises(_ProcessorApplied):
            EagleVerifyInputV2Mixin.sample(verify_input, batch, logits_output)


if __name__ == "__main__":
    unittest.main()
