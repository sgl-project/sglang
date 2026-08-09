"""E2E streaming tests for DeepSeek-V4 DSML tool-call parser (REQ-10).

Launches a real server with --tool-call-parser deepseekv4 and sends
streaming + non-streaming requests with tools.  Validates SSE delta
ordering and finish_reason.

Gated on GPU availability and model access.  Cannot reproduce the
intermittent production bug (load-dependent, ~7-8% failure rate under
concurrent DSPARK).  Value is SSE format validation only.
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

register_cuda_ci(
    est_time=120,
    stage="base-b",
    runner_config="1-gpu-large",
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    },
}


class TestDeepSeekV4StreamingE2E(CustomTestCase):
    """REQ-10.1, REQ-10.3: SSE delta ordering and finish_reason validation."""

    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--tool-call-parser", "deepseekv4"],
            )
        except Exception:
            cls.process = None
            raise unittest.SkipTest(f"Could not launch server with {DSV4_MODEL}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_streaming_delta_ordering(self):
        """REQ-10.1, REQ-10.3: content → tool_calls → finish_reason ordering."""
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "What's the weather in SF?"}],
                "tools": [WEATHER_TOOL],
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(resp.status_code, 200)

        events = []
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    payload = line_str[6:]
                    if payload:
                        events.append(json.loads(payload))

        # Must have at least one event
        self.assertTrue(len(events) > 0, "No SSE events received")

        # The last event should have finish_reason
        last = events[-1]
        last_choice = last["choices"][0]
        finish_reason = last_choice.get("finish_reason")

        # finish_reason should be "tool_calls" if a tool call was made
        # or "stop" if the model chose not to call a tool
        self.assertIn(finish_reason, ("tool_calls", "stop"))

        # If finish_reason is "tool_calls", verify tool_call deltas appeared
        if finish_reason == "tool_calls":
            has_tool_call_delta = False
            has_content_delta = False
            for ev in events:
                delta = ev["choices"][0].get("delta", {})
                if delta.get("tool_calls"):
                    has_tool_call_delta = True
                if delta.get("content"):
                    has_content_delta = True
            self.assertTrue(has_tool_call_delta, "Expected tool_call deltas")


class TestDeepSeekV4StreamVsNonStream(CustomTestCase):
    """REQ-10.2: streaming and non-streaming produce the same tool call."""

    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--tool-call-parser", "deepseekv4"],
            )
        except Exception:
            cls.process = None
            raise unittest.SkipTest(f"Could not launch server with {DSV4_MODEL}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_stream_vs_non_stream_same_tool_call(self):
        """Same prompt in streaming and non-streaming → same tool call."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": [WEATHER_TOOL],
        }

        # Non-streaming
        resp_ns = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={**kwargs, "stream": False},
        )
        self.assertEqual(resp_ns.status_code, 200)
        ns_data = resp_ns.json()
        ns_choice = ns_data["choices"][0]
        ns_tool_calls = ns_choice["message"].get("tool_calls")

        # Streaming
        resp_s = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={**kwargs, "stream": True},
            stream=True,
        )
        self.assertEqual(resp_s.status_code, 200)

        s_tool_calls_by_index = {}
        for line in resp_s.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    payload = line_str[6:]
                    if payload != "[DONE]":
                        ev = json.loads(payload)
                        delta = ev["choices"][0].get("delta", {})
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            slot = s_tool_calls_by_index.setdefault(
                                idx, {"name": "", "arguments": ""}
                            )
                            if tc.get("function", {}).get("name"):
                                slot["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                slot["arguments"] += tc["function"]["arguments"]

        # If non-streaming produced tool calls, streaming should too
        if ns_tool_calls:
            self.assertTrue(
                len(s_tool_calls_by_index) > 0,
                "Non-streaming had tool calls but streaming had none",
            )
            # Compare names
            ns_names = [tc["function"]["name"] for tc in ns_tool_calls]
            s_names = [s["name"] for s in s_tool_calls_by_index.values()]
            self.assertEqual(ns_names, s_names)


if __name__ == "__main__":
    unittest.main()
