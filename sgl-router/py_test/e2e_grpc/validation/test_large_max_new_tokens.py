"""
python3 -m unittest openai_server.validation.test_large_max_new_tokens.TestLargeMaxNewTokens.test_chat_completion
"""

import os
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_workers_and_router
from util import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    get_tokenizer,
    kill_process_tree,
)


class TestLargeMaxNewTokens(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            worker_args=(
                "--max-total-token",
                "1536",
                "--context-len",
                "8192",
                "--decode-log-interval",
                "2",
            ),
            num_workers=1,
            tp_size=2,
            env={"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256", **os.environ},
            stdout=cls.stdout,
            stderr=cls.stderr,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        # Cleanup router and workers
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

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
        all_requests_running = False

        futures = []
        with ThreadPoolExecutor(num_requests) as executor:
            # Send multiple requests
            for i in range(num_requests):
                futures.append(executor.submit(self.run_chat_completion))

            # Ensure that they are running concurrently
            pt = 0
            while pt >= 0:
                time.sleep(5)
                # Flush stderr to ensure logs are written
                self.stderr.flush()
                lines = open(STDERR_FILENAME).readlines()
                for line in lines[pt:]:
                    if f"#running-req: {num_requests}" in line:
                        all_requests_running = True
                        pt = -1
                        break
                    pt += 1

        assert all_requests_running


if __name__ == "__main__":
    unittest.main()
