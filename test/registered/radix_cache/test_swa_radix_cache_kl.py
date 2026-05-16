import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "openai/gpt-oss-20b"

register_cuda_ci(est_time=151, stage="base-b", runner_config="1-gpu-large")


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
        # Pin attention reduction order. Without this the ragged-prefill and
        # paged-decode kernels accumulate bf16 partial sums in different
        # orders, producing ~0.01-0.05 logprob drift per token between the
        # decode-time output_logprob and the prefill-time input_logprob even
        # for the same token at the same context. PR #25429 (temperature
        # 1 -> 0.0 in kl_test_utils) removed the sampling-noise averaging
        # that previously kept this drift under the 0.002 threshold.
        # KL test should measure cache consistency, not fp noise.
        "--enable-deterministic-inference",
    ]


if __name__ == "__main__":
    unittest.main()
