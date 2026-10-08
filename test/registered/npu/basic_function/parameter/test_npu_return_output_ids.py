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


class TestNpuReturnOutputIds(CustomTestCase):
    """Testcase: Verify the server-level `--return-output-ids` parameter makes
    every chat completion response return the sampled output token ids in the
    response-level `sglext` extension.

    [Test Category] Parameter
    [Test Target] --return-output-ids
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--return-output-ids",
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

    def test_chat_completion_returns_output_ids_in_sglext(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "temperature": 0,
                "max_tokens": 16,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertIn("sglext", data, "sglext extension missing in response")
        output_ids = data["sglext"]["output_ids"]
        self.assertIsInstance(output_ids, list)
        self.assertEqual(len(output_ids), 1)  # n == 1
        self.assertIsInstance(output_ids[0], list)
        self.assertGreater(len(output_ids[0]), 0)
        self.assertTrue(all(isinstance(i, int) for i in output_ids[0]))

        # The returned output ids must decode back to the generated content.
        generated_text = data["choices"][0]["message"]["content"]
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.assertEqual(decoded.strip(), generated_text.strip())

    def test_chat_completion_stream_returns_output_ids_in_sglext(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(response.status_code, 200)

        output_ids = None
        generated_text = ""
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
            if sglext and sglext.get("output_ids") is not None:
                output_ids = sglext["output_ids"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    generated_text += content

        self.assertIsNotNone(output_ids, "streaming response missing sglext.output_ids")
        self.assertIsInstance(output_ids, list)
        self.assertEqual(len(output_ids), 1)  # n == 1
        self.assertIsInstance(output_ids[0], list)
        self.assertGreater(len(output_ids[0]), 0)
        self.assertTrue(all(isinstance(i, int) for i in output_ids[0]))
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.assertEqual(decoded.strip(), generated_text.strip())

    def test_chat_completion_returns_output_ids_n_greater_than_one(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "temperature": 0.8,
                "max_tokens": 16,
                "n": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertIn("sglext", data, "sglext extension missing in response")
        output_ids = data["sglext"]["output_ids"]
        self.assertIsInstance(output_ids, list)
        self.assertEqual(len(output_ids), 2)  # n == 2
        self.assertEqual(len(data["choices"]), 2)

        for ids, choice in zip(output_ids, data["choices"]):
            self.assertIsInstance(ids, list)
            self.assertGreater(len(ids), 0)
            self.assertTrue(all(isinstance(i, int) for i in ids))
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded.strip(), choice["message"]["content"].strip())


if __name__ == "__main__":
    unittest.main()
