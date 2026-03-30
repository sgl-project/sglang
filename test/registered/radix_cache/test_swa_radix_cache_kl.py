import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "openai/gpt-oss-20b"

register_cuda_ci(est_time=100, suite="stage-b-test-1-gpu-large")


class TestSWARadixCacheKL(KLDivergenceMixin, DefaultServerBase):
    model = MODEL
    kl_div_thres = 0.002
    kl_div_decode_max_new_tokens = 2048
    other_args = [
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.70",
        "--disable-piecewise-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
