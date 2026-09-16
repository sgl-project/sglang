"""CPU contract tests for GLM's token-count absorb-weight selection."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
    _use_glm_fp8_small_batch_absorb,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlmHybridAbsorbDispatch(CustomTestCase):
    def test_hybrid_uses_fp8_below_bf16_crossover(self) -> None:
        attention = SimpleNamespace(use_glm_bf16_prefill_fp8_decode=True)

        for num_tokens, expected in (
            (1, True),
            (4, True),
            (511, True),
            (512, False),
            (8192, False),
        ):
            with self.subTest(num_tokens=num_tokens):
                self.assertIs(
                    _use_glm_fp8_small_batch_absorb(attention, num_tokens),
                    expected,
                )

    def test_default_off_uses_primary_absorb_weights(self) -> None:
        attention = SimpleNamespace(use_glm_bf16_prefill_fp8_decode=False)

        self.assertFalse(_use_glm_fp8_small_batch_absorb(attention, 1))


if __name__ == "__main__":
    unittest.main()
