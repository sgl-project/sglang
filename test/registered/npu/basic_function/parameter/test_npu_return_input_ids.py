import json
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuReturnInputIds(CustomTestCase):
    """Testcase: Verify the server-level `--return-input-ids` parameter makes
    every chat completion response return the prompt (input) token ids in the
    response-level `sglext` extension.

    [Test Category] Parameter
    [Test Target] --return-input-ids
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--return-input-ids",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion_returns_input_ids_in_sglext(self):
        messages = [{"role": "user", "content": "The capital of France is"}]
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 16,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertIn("sglext", data, "sglext extension missing in response")
        input_ids = data["sglext"]["input_ids"]
        self.assertIsInstance(input_ids, list)
        self.assertGreater(len(input_ids), 0)
        self.assertTrue(all(isinstance(i, int) for i in input_ids))

        # The returned input ids must exactly match the prompt token ids.
        self.assertEqual(input_ids, self._expected_input_ids(messages))

    def _expected_input_ids(self, messages):
        expected = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(expected, list):
            expected = expected["input_ids"]
        if hasattr(expected, "tolist"):
            expected = expected.tolist()
        return expected

    def test_chat_completion_stream_returns_input_ids_in_sglext(self):
        messages = [{"role": "user", "content": "The capital of France is"}]
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 16,
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(response.status_code, 200)

        input_ids = None
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            sglext = chunk.get("sglext")
            if sglext and sglext.get("input_ids") is not None:
                input_ids = sglext["input_ids"]

        self.assertIsNotNone(input_ids, "streaming response missing sglext.input_ids")
        self.assertIsInstance(input_ids, list)
        self.assertGreater(len(input_ids), 0)
        self.assertTrue(all(isinstance(i, int) for i in input_ids))
        self.assertEqual(input_ids, self._expected_input_ids(messages))


if __name__ == "__main__":
    unittest.main()