"""Verify server chat-template defaults with Llama-3.2-1B-Instruct on one NPU.

Run from the repository root (the model path override is optional in Ascend CI):
    SGLANG_TEST_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
    ASCEND_RT_VISIBLE_DEVICES=0 \
    python3 test/registered/npu/basic_function/parameter/test_npu_default_chat_template_kwargs.py -v

Use the model's native date_string, tools_in_user_message and custom_tools
variables. Compare the actual inference prompt token IDs with an independently
rendered tokenizer prompt, rather than relying on generated answer wording.
"""

import json
import os
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=180, suite="full-1-npu-a3", nightly=True)

SERVER_DEFAULTS = {
    "date_string": "01 Jan 2000",
    "tools_in_user_message": True,
}
CUSTOM_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


class TestNpuDefaultChatTemplateKwargs(CustomTestCase):
    """[Test Category] Parameter; [Test Target] --default-chat-template-kwargs."""

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "SGLANG_TEST_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        )
        cls.base_url = os.environ.get("SGLANG_TEST_BASE_URL", DEFAULT_URL_FOR_TEST)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.messages = [{"role": "user", "content": "What is the weather in Paris?"}]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="npu",
            other_args=[
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tp-size",
                "1",
                "--mem-fraction-static",
                "0.2",
                "--context-length",
                "2048",
                "--default-chat-template-kwargs",
                json.dumps(SERVER_DEFAULTS),
            ],
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def _check_prompt(self, request_fields, expected_kwargs, stream):
        expected_ids = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **expected_kwargs,
        )
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0,
            "max_tokens": 4,
            "stream": stream,
            "return_input_ids_in_sglext": True,
            "return_output_ids_in_sglext": True,
            **request_fields,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=stream,
            timeout=120,
        ) as response:
            self.assertEqual(
                response.status_code, 200, response.text if not response.ok else ""
            )
            if stream:
                input_ids = None
                output_ids = None
                usage = None
                finish_reason = None
                done = False
                for line in response.iter_lines():
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    self.assertNotIn("error", chunk, chunk)
                    extension = chunk.get("sglext") or {}
                    if "input_ids" in extension:
                        input_ids = extension["input_ids"]
                    if "output_ids" in extension:
                        output_ids = extension["output_ids"]
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                self.assertTrue(done, "The stream did not finish with [DONE]")
            else:
                data = response.json()
                self.assertNotIn("error", data, data)
                self.assertTrue(data["choices"], data)
                input_ids = data.get("sglext", {}).get("input_ids")
                output_ids = data.get("sglext", {}).get("output_ids")
                usage = data.get("usage")
                finish_reason = data["choices"][0]["finish_reason"]

        self.assertEqual(input_ids, expected_ids, "Server rendered the wrong prompt")
        self.assertTrue(output_ids, "The NPU did not generate any tokens")
        # sglext.output_ids has one token list per choice, even with n=1.
        self.assertEqual(len(output_ids), 1)
        self.assertTrue(output_ids[0], "The only choice contains no generated tokens")
        self.assertIn(finish_reason, ("stop", "length"))
        self.assertIsNotNone(usage)
        self.assertEqual(usage["prompt_tokens"], len(expected_ids))
        self.assertEqual(usage["completion_tokens"], len(output_ids[0]))
        self.assertGreater(usage["completion_tokens"], 0)
        print(
            f"Verified stream={stream}, prompt_tokens={len(input_ids)}, "
            f"completion_tokens={usage['completion_tokens']}, "
            f"date_string={expected_kwargs['date_string']!r}, "
            f"tools_in_user_message={expected_kwargs['tools_in_user_message']}"
        )

    def _check_both_modes(self, request_fields, expected_kwargs):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self._check_prompt(request_fields, expected_kwargs, stream)

    def test_defaults_when_kwargs_omitted(self):
        self._check_both_modes({}, SERVER_DEFAULTS)

    def test_defaults_when_kwargs_null(self):
        self._check_both_modes({"chat_template_kwargs": None}, SERVER_DEFAULTS)

    def test_defaults_when_kwargs_empty(self):
        self._check_both_modes({"chat_template_kwargs": {}}, SERVER_DEFAULTS)

    def test_request_overrides_default_date(self):
        self._check_both_modes(
            {"chat_template_kwargs": {"date_string": "02 Feb 2001"}},
            {"date_string": "02 Feb 2001", "tools_in_user_message": True},
        )

    def test_empty_string_overrides_default(self):
        self._check_both_modes(
            {"chat_template_kwargs": {"date_string": ""}},
            {"date_string": "", "tools_in_user_message": True},
        )

    def test_new_request_key_preserves_defaults(self):
        self._check_both_modes(
            {"chat_template_kwargs": {"custom_tools": CUSTOM_TOOLS}},
            {
                "date_string": "01 Jan 2000",
                "tools_in_user_message": True,
                "custom_tools": CUSTOM_TOOLS,
            },
        )

    def test_false_overrides_default_and_preserves_other_keys(self):
        self._check_both_modes(
            {
                "chat_template_kwargs": {
                    "tools_in_user_message": False,
                    "custom_tools": CUSTOM_TOOLS,
                }
            },
            {
                "date_string": "01 Jan 2000",
                "tools_in_user_message": False,
                "custom_tools": CUSTOM_TOOLS,
            },
        )

    def test_request_overrides_do_not_leak(self):
        overrides = {
            "date_string": "03 Mar 2002",
            "tools_in_user_message": False,
            "custom_tools": CUSTOM_TOOLS,
        }
        for stream in (False, True):
            with self.subTest(stream=stream):
                self._check_prompt(
                    {"chat_template_kwargs": overrides}, overrides, stream
                )
                self._check_prompt({}, SERVER_DEFAULTS, stream)
                self._check_prompt(
                    {"chat_template_kwargs": {"custom_tools": CUSTOM_TOOLS}},
                    {
                        "date_string": "01 Jan 2000",
                        "tools_in_user_message": True,
                        "custom_tools": CUSTOM_TOOLS,
                    },
                    stream,
                )


if __name__ == "__main__":
    unittest.main()
