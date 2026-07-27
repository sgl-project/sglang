import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuModelConfigParser(CustomTestCase):
    """Testcase: verify model-config-parser parameter on Mistral model

    [Test Category] Parameter
    [Test Target] --model-config-parser
    """

    model = MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]

    @classmethod
    def _launch_and_test(cls, parser_value: str):
        """Launch server with the given parser value, send a request, assert 200+Paris."""
        process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.base_args + ["--model-config-parser", parser_value],
        )
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"
            assert (
                "Paris" in response.text
            ), f"Expected 'Paris' in response, got: {response.text[:200]}"
        finally:
            kill_process_tree(process.pid)

    def test_model_config_parser_auto(self):
        """auto mode routes to mistral parser on Mistral model"""
        self._launch_and_test("auto")

    def test_model_config_parser_hf(self):
        """explicit hf parser overrides auto detection on Mistral model"""
        self._launch_and_test("hf")


if __name__ == "__main__":
    unittest.main()
