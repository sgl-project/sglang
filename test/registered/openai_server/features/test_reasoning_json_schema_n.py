import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")


class TestReasoningJsonSchemaParallelSampling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-0.6B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--reasoning-parser",
                "qwen3",
            ],
        )
        cls.base_url += "/v1"
        cls.json_schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "judge_result": {"type": "boolean"},
            },
            "required": ["reason", "judge_result"],
            "additionalProperties": False,
        }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_schema_parallel_sampling_keeps_content(self):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Judge if this is positive: 'Great product!'",
                    }
                ],
                "temperature": 0.6,
                "max_tokens": 256,
                "n": 2,
                "chat_template_kwargs": {"enable_thinking": True},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_result",
                        "schema": self.json_schema,
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        self.assertEqual(len(data["choices"]), 2)
        for choice in data["choices"]:
            message = choice["message"]
            self.assertIsNotNone(message["content"])
            self.assertIsInstance(message["content"], str)
            payload = json.loads(message["content"])
            self.assertIsInstance(payload["reason"], str)
            self.assertIsInstance(payload["judge_result"], bool)


if __name__ == "__main__":
    unittest.main()
