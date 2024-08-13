import json
import unittest

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server


class TestOpenAIVisionServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = "liuhaotian/llava-v1.6-vicuna-7b"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            api_key=cls.api_key,
            other_args=[
                "--chat-template",
                "vicuna_v1.1",
                "--tokenizer-path",
                "llava-hf/llava-1.5-7b-hf",
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
                                "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a very short sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert "car" in text or "taxi" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_regex(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        regex = (
            r"""\{\n"""
            + r"""   "color": "[\w]+",\n"""
            + r"""   "number_of_cars": [\d]+\n"""
            + r"""\}"""
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in the JSON format.",
                        },
                    ],
                },
            ],
            temperature=0,
            extra_body={"regex": regex},
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise
        assert isinstance(js_obj["color"], str)
        assert isinstance(js_obj["number_of_cars"], int)


if __name__ == "__main__":
    unittest.main()
