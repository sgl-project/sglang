"""Regression test: --load-format=dummy must not crash for KimiVL.

KimiVLForConditionalGeneration wraps DeepseekV2ForCausalLM (MLA). The inner
language model's post_load_weights() splits kv_b_proj into w_kc/w_vc. Without
a standalone post_load_weights() on the wrapper, DummyModelLoader's
utils.post_load_weights() was a no-op, leaving w_kc=None and causing
AttributeError on the first forward pass.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci, run_bench_one_batch

register_cuda_ci(
    est_time=120,
    stage="base-b",
    runner_config="1-gpu",
)


class TestDummyKimiVL(CustomTestCase):

    def test_dummy_kimi_vl_a3b(self):
        _, output_throughput, _ = run_bench_one_batch(
            None,
            [
                "--model",
                "moonshotai/Kimi-VL-A3B-Instruct",
                "--trust-remote-code",
                "--batch-size",
                "1",
                "--tp",
                "1",
                "--load-format",
                "dummy",
                "--json-model-override-args",
                '{"num_hidden_layers": 2}',
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 0)


if __name__ == "__main__":
    unittest.main()
