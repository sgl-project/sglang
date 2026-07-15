import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_DISTILL_QWEN_7B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)


class TestReasoningContentAPI(CustomTestCase):
    """
    Testcase：Verify the correctness of reasoning content API under both streaming and non-streaming, and separate
    reasoning is set to  true or false

    [Test Category] Parameter
    [Test Target] --reasoning-parser
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_DISTILL_QWEN_7B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                "deepseek-r1",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_streaming_separate_reasoning_false(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
        # the reasoning_content of chunk should not have a value when separate reasoning is false
        assert len(reasoning_content) == 0
        # the content of chunk should have a value when streaming is true
        assert len(content) > 0

    def test_streaming_separate_reasoning_true(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        # the reasoning_content of chunk should have a value when separate reasoning is true
        assert len(reasoning_content) > 0
        # the content of chunk should have a value when streaming is true
        assert len(content) > 0

    def test_streaming_separate_reasoning_true_stream_reasoning_false(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": True, "stream_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        first_chunk = False
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                if not first_chunk:
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if not first_chunk:
                assert (
                    # Only the first reasoning_content of chunk should have a value when stream reasoning is false
                    not chunk.choices[0].delta.reasoning_content
                    or len(chunk.choices[0].delta.reasoning_content) == 0
                )
        # the reasoning_content of chunk should have a value when separate reasoning is true
        assert len(reasoning_content) > 0
        # the content of chunk should have a value when streaming is true
        assert len(content) > 0

    def test_nonstreaming_separate_reasoning_false(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        # Response should not have content when separate reasoning is false
        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        # Response should have content when nonstreaming
        assert len(response.choices[0].message.content) > 0

    def test_nonstreaming_separate_reasoning_true(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        # Response should have content when separate reasoning is true
        assert len(response.choices[0].message.reasoning_content) > 0
        # Response should have content when nonstreaming
        assert len(response.choices[0].message.content) > 0


class TestReasoningContentWithoutParser(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_DISTILL_QWEN_7B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[],  # No reasoning parser
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_streaming_separate_reasoning_false(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true_stream_reasoning_false(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": True, "stream_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        first_chunk = False
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                if not first_chunk:
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if not first_chunk:
                assert (
                    not chunk.choices[0].delta.reasoning_content
                    or len(chunk.choices[0].delta.reasoning_content) == 0
                )
        assert not reasoning_content or len(reasoning_content) == 0
        assert len(content) > 0

    def test_nonstreaming_separate_reasoning_false(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        assert len(response.choices[0].message.content) > 0

    def test_nonstreaming_separate_reasoning_true(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        assert len(response.choices[0].message.content) > 0


if __name__ == "__main__":
    unittest.main()
