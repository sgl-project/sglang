import json
import unittest

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import popen_launch_server


class TestOpenAIVisionServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = "liuhaotian/llava-v1.6-vicuna-7b"
        cls.base_url = "http://localhost:8157"
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=300, api_key=cls.api_key,
            other_args=[
                "--chat-template", "vicuna_v1.1",
                "--tokenizer-path", "llava-hf/llava-1.5-7b-hf",
                "--log-requests",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://github.com/sgl-project/sglang/blob/main/assets/logo.png?raw=true"
                            },
                        },
                        {"type": "text", "text": "Describe this image"},
                    ],
                },
            ],
            temperature=0,
            max_tokens=32,
        )

        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # t = TestOpenAIVisionServer()
    # t.setUpClass()
    # t.test_chat_completion()
    # t.tearDownClass()
