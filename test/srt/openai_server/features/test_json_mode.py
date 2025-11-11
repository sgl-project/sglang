"""
python3 -m unittest openai_server.features.test_json_mode.TestJSONModeOutlines.test_json_mode_response
python3 -m unittest openai_server.features.test_json_mode.TestJSONModeOutlines.test_json_mode_with_streaming
python3 -m unittest.openai_server.features.test_json_mode.TestJSONModeXGrammar.test_json_mode_response
python3 -m unittest.openai_server.features.test_json_mode.TestJSONModeXGrammar.test_json_mode_with_streaming
python3 -m unittest.openai_server.features.test_json_mode.TestJSONModeLLGuidance.test_json_mode_response
python3 -m unittest.openai_server.features.test_json_mode.TestJSONModeLLGuidance.test_json_mode_with_streaming
"""

import json
import unittest

import openai

from sglang.test.test_utils import (
    kill_process_tree,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def setup_class(cls, backend):
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST

    other_args = [
        "--max-running-requests",
        "10",
        "--grammar-backend",
        backend,
    ]

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")


class TestJSONModeOutlines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, "outlines")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_mode_response(self):
        """Test that response_format json_object (also known as "json mode") produces valid JSON."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Give me a JSON object with two keys a and b."}
            ],
            response_format={"type": "json_object"},
        )
        js_obj = json.loads(completion.choices[0].message.content)
        self.assertIsInstance(js_obj, dict)

    def test_json_mode_with_streaming(self):
        """Test that streaming in json mode still yields valid JSON output."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Give me a JSON object with two keys a and b."}
            ],
            response_format={"type": "json_object"},
            stream=True,
        )
        content = ""
        for event in stream:
            if event.choices[0].delta and event.choices[0].delta.content:
                content += event.choices[0].delta.content
        js_obj = json.loads(content)
        self.assertIsInstance(js_obj, dict)


class TestJSONModeXGrammar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, "xgrammar")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_mode_response(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Give me a JSON object with two keys a and b."}
            ],
            response_format={"type": "json_object"},
        )
        js_obj = json.loads(completion.choices[0].message.content)
        self.assertIsInstance(js_obj, dict)

    def test_json_mode_with_streaming(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Give me a JSON object with two keys a and b."}
            ],
            response_format={"type": "json_object"},
            stream=True,
        )
        content = ""
        for event in stream:
            if event.choices[0].delta and event.choices[0].delta.content:
                content += event.choices[0].delta.content
        js_obj = json.loads(content)
        self.assertIsInstance(js_obj, dict)


class TestJSONModeLLGuidance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, "llguidance")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_mode_response(self):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Give me a JSON object with two keys a and b."}
            ],
            response_format={"type": "json_object"},
        )
        js_obj = json.loads(completion.choices[0].message.content)
        self.assertIsInstance(js_obj, dict)

    def test_json_mode_with_streaming(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Give me a JSON object with two keys a and b."}
            ],
            response_format={"type": "json_object"},
            stream=True,
        )
        content = ""
        for event in stream:
            if event.choices[0].delta and event.choices[0].delta.content:
                content += event.choices[0].delta.content
        js_obj = json.loads(content)
        self.assertIsInstance(js_obj, dict)


if __name__ == "__main__":
    unittest.main()
