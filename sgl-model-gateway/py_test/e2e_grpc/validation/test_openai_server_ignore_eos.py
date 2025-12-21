"""
gRPC Router E2E Test - Test Openai Server Ignore Eos

This test file is REUSED from test/srt/openai_server/validation/test_openai_server_ignore_eos.py
with minimal changes:
    num_workers=2,
- Swap popen_launch_server() → popen_launch_workers_and_router()
- Update teardown to cleanup router + workers
- All test logic and assertions remain identical

Run with:
    pytest py_test/e2e_grpc/e2e_grpc/validation/test_openai_server_ignore_eos.py -v
"""

import sys
from pathlib import Path

import openai

_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_workers_and_router
from util import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_tokenizer,
    kill_process_tree,
)


class TestOpenAIServerIgnoreEOS(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # CHANGE: Launch gRPC router with integrated workers (single command)
        cls.model = DEFAULT_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            num_workers=1,
            tp_size=2,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        # Cleanup router and workers
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def test_ignore_eos(self):
        """
        Test that ignore_eos=True allows generation to continue beyond EOS token
        and reach the max_tokens limit.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        max_tokens = 200

        response_default = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": False},
        )

        response_ignore_eos = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )

        default_tokens = len(
            self.tokenizer.encode(response_default.choices[0].message.content)
        )
        ignore_eos_tokens = len(
            self.tokenizer.encode(response_ignore_eos.choices[0].message.content)
        )

        # Check if ignore_eos resulted in more tokens or exactly max_tokens
        # The ignore_eos response should either:
        # 1. Have more tokens than the default response (if default stopped at EOS before max_tokens)
        # 2. Have exactly max_tokens (if it reached the max_tokens limit)
        self.assertTrue(
            ignore_eos_tokens > default_tokens or ignore_eos_tokens >= max_tokens,
            f"ignore_eos did not generate more tokens: {ignore_eos_tokens} vs {default_tokens}",
        )

        self.assertEqual(
            response_ignore_eos.choices[0].finish_reason,
            "length",
            f"Expected finish_reason='length' for ignore_eos=True, got {response_ignore_eos.choices[0].finish_reason}",
        )
