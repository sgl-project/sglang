"""test_disaggregation_dsv4.py with --dsv4-attn-backend trtllm on one B300 node."""

import unittest

import test_disaggregation_dsv4 as dsv4_pd

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-c", runner_config="8-gpu-b300")


class TestDisaggregationDSV4Trtllm(dsv4_pd.TestDisaggregationDSV4):
    gsm8k_backend = "sgl_eval"
    model_repo = "deepseek-ai/DeepSeek-V4-Flash"
    server_env = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256"}
    extra_server_args = [
        "--dsv4-attn-backend",
        "trtllm",
        "--chunked-prefill-size",
        "4096",
    ]


if __name__ == "__main__":
    unittest.main()
