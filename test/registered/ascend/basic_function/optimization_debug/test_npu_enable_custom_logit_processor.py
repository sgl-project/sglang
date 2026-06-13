import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class DeterministicLogitProcessor(CustomLogitProcessor):
    """A dummy logit processor that changes the logits to always
    sample the given token id.
    """

    def __call__(self, logits, custom_param_list):
        # Check that the number of logits matches the number of custom parameters
        assert logits.shape[0] == len(custom_param_list)
        key = "token_id"

        for i, param_dict in enumerate(custom_param_list):
            # Mask all other tokens
            logits[i, :] = -float("inf")
            # Assign highest probability to the specified token
            logits[i, param_dict[key]] = 0.0
        return logits


class TestEnableCustomLogitProcessor(CustomTestCase):
    """
    Testcase：When the service startup configuration --enable-custom-logit-processor is enabled and requests include
    "custom_logit_processor" and "custom_params," custom functions can control the generated text content.

    [Test Category] Parameter
    [Test Target] --enable-custom-logit-processor
    """

    model = QWEN3_32B_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "4",
        "--enable-custom-logit-processor",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_enable_custom_logit_processor(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        text1 = "The capital of France is"
        res = "apple"
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        token_id = tokenizer(res, return_tensors="pt")["input_ids"][0].tolist()[0]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "custom_params": {"token_id": token_id},
                },
                "custom_logit_processor": DeterministicLogitProcessor().to_str(),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], res)

        text2 = "Today is"
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text2,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "custom_params": {"token_id": token_id},
                },
                "custom_logit_processor": DeterministicLogitProcessor().to_str(),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], res)


if __name__ == "__main__":
    unittest.main()
