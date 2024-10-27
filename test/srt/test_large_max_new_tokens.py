"""
python3 -m unittest test_large_max_new_tokens.TestLargeMaxNewTokens.test_chat_completion
"""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestLargeMaxNewTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.stdout = open("stdout.txt", "w")
        cls.stderr = open("stderr.txt", "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=("--max-total-token", "1024", "--context-len", "8192"),
            env={"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256", **os.environ},
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid, include_self=True)
        cls.stdout.close()
        cls.stderr.close()
        os.remove("stdout.txt")
        os.remove("stderr.txt")

    def run_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Please repeat the world 'hello' for 10000 times.",
                },
            ],
            temperature=0,
        )
        return response

    def test_chat_completion(self):
        num_requests = 4

        futures = []
        with ThreadPoolExecutor(num_requests) as executor:
            # Send multiple requests
            for i in range(num_requests):
                futures.append(executor.submit(self.run_chat_completion))

            # Ensure that they are running concurrently
            pt = 0
            while pt >= 0:
                lines = open("stderr.txt").readlines()
                for line in lines[pt:]:
                    print(line, end="", flush=True)
                    if f"#running-req: {num_requests}" in line:
                        all_requests_running = True
                        pt = -1
                        break
                    pt += 1

        assert all_requests_running


if __name__ == "__main__":
    unittest.main()
