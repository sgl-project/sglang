import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_cuda_ci(est_time=220, stage="extra-b", runner_config="4-gpu-b200")


class TestGptOss4GpuMxfp4CP(BaseTestGptOss):
    def _check_streaming_responses_api_request(self, model):
        super()._check_streaming_responses_api_request(model)
        server_info = requests.get(
            f"{DEFAULT_URL_FOR_TEST}/server_info", timeout=30
        ).json()
        self.assertEqual(
            server_info["cuda_graph_config"]["prefill"]["backend"], "breakable"
        )

    def test_mxfp4_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--enable-prefill-cp",
                "--attn-cp-size",
                "4",
                "--cp-strategy",
                "zigzag",
                "--cuda-graph-backend-prefill",
                "breakable",
                "--cuda-graph-max-bs-decode",
                "200",
            ],
        )


if __name__ == "__main__":
    unittest.main()
