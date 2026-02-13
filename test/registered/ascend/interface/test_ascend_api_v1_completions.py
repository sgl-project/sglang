import json
import logging
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestEnableThinking(CustomTestCase):
    """Testcase: The test is to verify whether the functions of each parameter of the v1/completions interface are normal.

    [Test Category] Interface
    [Test Target] v1/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
            "--reasoning-parser",
            "qwen3",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            2,
            "--enable-return-hidden-states",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        cls.additional_chat_kwargs = {}

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_model_parameters_model(self):
        # Test model parameter; configured model returns correct name, unconfigured defaults to "default", reasoning works
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"model": self.model, "prompt": "who are you?"},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], self.model)

    def test_model_parameters_prompt(self):
        # Test prompt parameter, The input has str, list[int], list[str], and list[list[int]], reasoning works
        # The input is in str format
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?"},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        # The input is in list[int] format
        list_int = [1, 2, 3, 4]
        response1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": list_int},
        )
        logging.info(f"response1.json:{response1.json()}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")

        # The input is in list[str] format
        list_str = ["who is you", "hello world", "ABChello"]
        response2 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": list_str},
        )
        logging.info(f"response2.json:{response2.json()}")
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")

        # The input is in list[list[int]] format
        list_list_int = [[14990], [1350, 445, 14990, 1879, 899], [14623, 525, 498, 30]]
        response3 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": list_list_int},
        )
        logging.info(f"response3.json:{response3.json()}")
        self.assertEqual(response3.status_code, 200, f"Failed with: {response3.text}")

    def test_model_parameters_max_tokens(self):
        # Test max_completion_tokens parameter; setting to 1 token forces immediate truncation, verify finish_reason is "length"
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "max_tokens": 1},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        logging.info(
            f"response.json_choices:{response.json()['choices'][0]['finish_reason']}"
        )
        # Assertion output includes length
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")

    def test_model_parameters_stream(self):
        # Test stream parameter; verify streaming response contains both reasoning_content and normal content chunks
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "stream": True},
        )
        logging.info(f"response.text:{response.text}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        # Decompose the response and determine if the output format is stream
        has_text = False
        logging.info("\n=== Stream With Reasoning ===")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:

                        if "text" in data["choices"][0]:
                            has_text = True
        self.assertTrue(
            has_text,
            "The text is a stream response",
        )

    def test_model_parameters_temperature(self):
        # Test temperature parameter; temperature=0 yields identical outputs across requests, temperature=2 yields varied outputs
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 0},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        response1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 0},
        )
        logging.info(f"response1.json:{response1.json()}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        # Assert that the configuration temperature is the same and the output response is the same
        self.assertEqual(
            response.json()["choices"][0]["text"],
            response1.json()["choices"][0]["text"],
        )

        response2 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 2},
        )
        logging.info(f"response2.json:{response2.json()}")
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")

        response3 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 2},
        )
        logging.info(f"response3.json:{response3.json()}")
        self.assertEqual(response3.status_code, 200, f"Failed with: {response3.text}")
        self.assertNotEqual(
            response2.json()["choices"][0]["text"],
            response3.json()["choices"][0]["text"],
        )

    def test_model_parameters_hidden_states(self):
        # Test return_hidden_states parameter; verify hidden_states field appears when enabled
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "return_hidden_states": True},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertIn("hidden_states", response.json()["choices"][0])

    def test_model_parameters_top_k(self):
        # Test top_k parameter; with k=20, outputs vary between identical requests due to token sampling
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "top_k": 20},
        )
        logging.info(f"response.json:{response.json()}")
        logging.info(f"response.text:{response.text}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        response1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "top_k": 20},
        )
        logging.info(f"response1.json:{response1.json()}")
        logging.info(f"response1.text:{response1.text}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        self.assertNotEqual(
            response.json()["choices"][0]["text"],
            response1.json()["choices"][0]["text"],
        )

    def test_model_parameters_stop_token_ids(self):
        # Test stop_token_ids parameter; verify response stops at specified token ID (13 is a period) and matched_stop field is correct
        list_ids = [13]
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": "who are you?",
                "stop_token_ids": list_ids,
                "max_tokens": 1024,
            },
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["choices"][0]["matched_stop"], 13)

    def test_model_parameters_rid(self):
        # Test rid parameter; verify response ID matches the requested rid value '10086'
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "rid": "10086"},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["id"], "10086")


if __name__ == "__main__":
    unittest.main()
