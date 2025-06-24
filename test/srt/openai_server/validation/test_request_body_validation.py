import unittest
import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import encode_image_base64

DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN = "Qwen/Qwen2.5-7B-Instruct"


class TestRequestBodyValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start server with auto truncate disabled
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            device="auto",
            other_args=("--max-payload-size", "1024"),  # 1KB
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_body_size_larger_than_maximum_allowed_payload_size_limit(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        text = "hello " * 200
        with self.assertRaises(openai.APIStatusError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                messages=[{"role": "user", "content": text}],
                temperature=0,
            )

        self.assertIn("Payload Too Large", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
