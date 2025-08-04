import unittest

from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestGLM4_1VThinkingServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "THUDM/GLM-4.1V-9B-Thinking"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.70",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"

    # TODO: support video
    # def test_video_chat_completion(self):
    #     self._test_video_chat_completion()


if __name__ == "__main__":
    del TestOpenAIVisionServer  # avoid duplicate test
    unittest.main()
