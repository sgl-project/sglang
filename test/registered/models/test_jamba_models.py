import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=132, suite="stage-b-test-large-2-gpu")


class TestJambaBF16(GSM8KMixin, DefaultServerBase):
    model = "ai21labs/AI21-Jamba2-3B"
    gsm8k_accuracy_thres = 0.74
    other_args = ["--max-mamba-cache-size", "256"]

if __name__ == "__main__":
    unittest.main()
