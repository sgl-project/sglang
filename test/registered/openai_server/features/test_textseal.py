import json
import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TestTextSealEndpoints(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(
            {
                "providers": {
                    "textseal": {
                        "enabled": True,
                        "key_a": "741852963",
                        "key_b": "963852741",
                        "ngram": 2,
                        "mixing_probability": 0.5,
                    }
                }
            },
            cls.config_file,
        )
        cls.config_file.close()
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-cuda-graph",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--watermark-config",
                cls.config_file.name,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.unlink(cls.config_file.name)

    def _request(self, endpoint, payload, *, stream, header=True):
        payload = {**payload, "stream": stream, "temperature": 0.8, "max_tokens": 8}
        headers = {"X-SGLang-Watermark": "textseal"} if header else {}
        response = requests.post(
            DEFAULT_URL_FOR_TEST + endpoint,
            json=payload,
            headers=headers,
            stream=stream,
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        if stream:
            chunks = [line for line in response.iter_lines() if line]
            self.assertTrue(any(line.startswith(b"data:") for line in chunks))
        else:
            self.assertTrue(response.json()["choices"])

    def test_chat_and_completions_streaming_and_non_streaming(self):
        cases = [
            (
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Say hello."}],
                },
            ),
            (
                "/v1/completions",
                {
                    "model": MODEL,
                    "prompt": "The capital of France is",
                },
            ),
        ]
        for endpoint, payload in cases:
            for stream in (False, True):
                with self.subTest(endpoint=endpoint, stream=stream):
                    self._request(endpoint, payload, stream=stream)

    def test_stable_header_errors(self):
        payload = {
            "model": MODEL,
            "prompt": "Hello",
            "temperature": 0.8,
            "max_tokens": 2,
        }
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/v1/completions",
            json=payload,
            headers={"X-SGLang-Watermark": "textseal;key_a=do-not-log"},
            timeout=60,
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["type"], "watermark_invalid_request")
        self.assertNotIn("do-not-log", response.text)


if __name__ == "__main__":
    unittest.main()
