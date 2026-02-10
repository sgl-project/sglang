import json
import unittest
import logging

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestJSONModeMixin:
    """Mixin class containing JSON mode test methods"""

    def test_json_mode_response(self):
        """Test that response_format json_object (also known as "json mode") produces valid JSON, even without a system prompt that mentions JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content

        logger.info("JSON mode response (%d characters): %s", len(text), text)

        # Verify the response is valid JSON
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            self.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")

        # Verify it's actually an object (dict)
        self.assertIsInstance(js_obj, dict, f"Response is not a JSON object: {text}")

    def test_json_mode_with_streaming(self):
        """Test that streaming with json_object response (also known as "json mode") format works correctly, even without a system prompt that mentions JSON."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream=True,
        )

        # Collect all chunks
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_response = "".join(chunks)

        logger.info("Concatenated streamed JSON response (%d characters): %s", len(full_response), full_response)

        # Verify the combined response is valid JSON
        try:
            js_obj = json.loads(full_response)
        except json.JSONDecodeError as e:
            self.fail(
                f"Streamed response is not valid JSON. Error: {e}. Response: {full_response}"
            )

        self.assertIsInstance(js_obj, dict)


class ServerWithGrammarBackend(CustomTestCase):
    """Testcase: Base class for tests requiring a grammar backend server.

    [Test Category] Parameter
    [Test Target] --grammar-backend
    """

    backend = "xgrammar"

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestJSONModeXGrammar(ServerWithGrammarBackend, TestJSONModeMixin):
    """Testcase: Verify JSON mode functionality with xgrammar grammar backend (non-streaming and streaming).

    [Test Category] Parameter
    [Test Target] --grammar-backend
    """
    backend = "xgrammar"


class TestJSONModeOutlines(ServerWithGrammarBackend, TestJSONModeMixin):
    """Testcase: Verify JSON mode functionality with outlines grammar backend (non-streaming and streaming).

    [Test Target] --grammar-backend
    """

    backend = "outlines"


if __name__ == "__main__":
    unittest.main()
