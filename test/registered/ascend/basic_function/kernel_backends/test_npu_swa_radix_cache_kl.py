import unittest

from sglang.test.ascend.test_ascend_utils import GPT_OSS_120B_BF16_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestSWARadixCacheKL(KLDivergenceMixin, DefaultServerBase):
    """
    Enable radix caching for SWA models on the NPU.
    Maintain consistency with the baseline regarding the probability distribution mechanism for text generation,
    thereby ensuring no loss in performance on the server side.
    """

    model = GPT_OSS_120B_BF16_WEIGHTS_PATH
    kl_div_thres = 0.02
    kl_div_decode_max_new_tokens = 2048
    other_args = [
        "--tp-size",
        "8",
        "--mem-fraction-static",
        "0.7",
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
