"""
E2E tests for NPU pause/resume generation — tool call.
"""

import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=80, suite="full-1-npu-a3", nightly=True)


class TestNpuToolCallWithPauseResume(CustomTestCase):
    """E2E: tool call generation survives pause/resume correctly.

    [Test Category] RL Pause/Resume + Tool Call
    [Test Target] POST /v1/chat/completions (with tools),
                  POST /pause_generation, POST /continue_generation
    """

    REQUEST_TIMEOUT = 180

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather conditions including temperature, humidity, wind speed and precipitation probability for a specified city. Returns structured weather data that can be used for further analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city to query weather for, e.g. Paris, London, Tokyo",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                            "description": "Temperature unit system: metric for Celsius, imperial for Fahrenheit",
                        },
                        "forecast_days": {
                            "type": "integer",
                            "description": "Number of days to forecast, from 1 to 7",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
    ]

    SYSTEM_MESSAGE = (
        "You are a helpful travel planning assistant with tool calling capabilities. "
        "You have access to a weather function. When a user asks a travel-related question, "
        "call the appropriate function with correct parameters. "
        "Respond in JSON format with the function name and parameters."
    )

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",  # required for NPU
                "--tool-call-parser",
                "llama3",  # Llama model tool call parser
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ── helpers ──────────────────────────────────────────────────

    @property
    def _chat_url(self):
        return self.base_url + "/v1/chat/completions"

    def _chat(self, messages, max_tokens=256, temperature=0, tools=None):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            body["tools"] = tools
        return requests.post(
            self._chat_url,
            json=body,
            timeout=self.REQUEST_TIMEOUT,
        )

    def _pause(self, mode: str = "in_place"):
        return requests.post(
            self.base_url + "/pause_generation",
            json={"mode": mode},
            timeout=30,
        )

    def _continue(self, torch_empty_cache: bool = True):
        return requests.post(
            self.base_url + "/continue_generation",
            json={"torch_empty_cache": torch_empty_cache},
            timeout=30,
        )

    # ── tool call generation survives pause ──

    def test_tool_call_pause_during_generation(self):
        """
        Tool call generation survives an engine pause/resume cycle
        and produces valid tool call JSON.

        Submits a long generation alongside the tool call request so that
        pause always catches a running request (tool calls are inherently
        short and may finish before the pause signal arrives on fast NPU).

        RL scenario: while the model is generating a tool call, the RL
        trainer pauses the engine to synchronize weights.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Long companion request as backup — guarantees something is
            # running at pause time even if tool call finishes early
            long_future = executor.submit(
                self._chat,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Write a comprehensive and extremely detailed "
                            "essay about the history of artificial "
                            "intelligence from the 1950s to present day:"
                        ),
                    },
                ],
                max_tokens=1024,
            )

            # Submit tool call.
            tool_future = executor.submit(
                self._chat,
                messages=messages,
                max_tokens=256,
                tools=self.TOOLS,
            )

            time.sleep(0.1)

            self._pause("in_place").raise_for_status()

            try:
                # Record which requests were paused (not yet done before pause)
                long_was_paused = not long_future.done()
                tool_was_paused = not tool_future.done()
                print(
                    f"[pause_state_1] long_was_paused={long_was_paused}, "
                    f"tool_was_paused={tool_was_paused}"
                )

                self.assertTrue(
                    long_was_paused and tool_was_paused,
                    "Tool call or long essay requests finished before pause took effect.",
                )

                # Verify no request completes during the pause window;
                time.sleep(10)
                long_was_paused = not long_future.done()
                tool_was_paused = not tool_future.done()
                print(
                    f"[pause_state_2] long_was_paused={long_was_paused}, "
                    f"tool_was_paused={tool_was_paused}"
                )
                self.assertTrue(
                    long_was_paused and tool_was_paused,
                    "Tool call or long essay requests finished while pause took effect ",
                )
            finally:
                t_continue = time.time()
                self._continue().raise_for_status()

            # Both requests should complete successfully after resume
            tool_resp = tool_future.result(timeout=self.REQUEST_TIMEOUT)
            t_tool_done = time.time()
            self.assertEqual(tool_resp.status_code, 200)
            print(
                f"[tool_call_pause_during_generation] continue->tool_done="
                f"{t_tool_done - t_continue:.3f}s"
            )

            long_resp = long_future.result(timeout=self.REQUEST_TIMEOUT)
            self.assertEqual(long_resp.status_code, 200)
            long_body = long_resp.json()
            long_text = long_body["choices"][0]["message"]["content"]
            self.assertGreater(
                len(long_text),
                0,
                "Long essay request should produce non-empty output after resume",
            )

        # ── Verify tool call format ──
        body = tool_resp.json()
        choice = body["choices"][0]

        # The model should have produced a tool call
        tool_calls = choice["message"].get("tool_calls")
        self.assertIsNotNone(
            tool_calls, "Expected tool_calls in response after pause/resume"
        )
        self.assertGreater(
            len(tool_calls), 0, "tool_calls should be non-empty after pause/resume"
        )

        tool_call = tool_calls[0]
        self.assertEqual(
            tool_call["function"]["name"],
            "get_weather",
            f"Expected function name 'get_weather', "
            f"got '{tool_call['function']['name']}'",
        )

        # Arguments must be valid JSON
        args_str = tool_call["function"]["arguments"]
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            self.fail(
                f"Tool call arguments are not valid JSON after pause/resume: "
                f"'{args_str}'"
            )
        self.assertIn("city", args, f"Expected 'city' in tool call arguments: {args}")

    def test_tool_call_multiturn_with_pause(self):
        """
        T10 extended: Full multi-turn tool call flow with pause between turns.

        Turn1: user asks a question → model responds with tool_call
               [pause to simulate external tool execution]
        Turn2: tool result provided → model produces final answer
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        # Turn 1 — get tool call
        resp = self._chat(messages, max_tokens=256, tools=self.TOOLS)
        self.assertEqual(resp.status_code, 200)
        turn1 = resp.json()
        tool_calls = turn1["choices"][0]["message"].get("tool_calls")
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        tool_call = tool_calls[0]
        tool_call_id = tool_call.get("id", "call_001")

        # Simulate RL environment pause — execute tool externally
        self._pause("in_place").raise_for_status()
        time.sleep(1)

        # Build tool result (normally done by RL environment)
        messages.append(turn1["choices"][0]["message"])
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "Sunny, 22°C",
            }
        )

        self._continue().raise_for_status()

        # Turn 2 — final answer using tool result.
        resp = self._chat(messages, max_tokens=256)
        self.assertEqual(resp.status_code, 200)
        turn2 = resp.json()
        turn2_text = turn2["choices"][0]["message"]["content"]

        self.assertGreater(
            len(turn2_text),
            0,
            "Final answer should be non-empty after multi-turn tool call",
        )
        # The answer should reference the weather data (tool_result="Sunny, 22°C")
        self.assertIn(
            "sunny",
            turn2_text.lower(),
            f"Final answer doesn't reference weather data: " f"'{turn2_text[:200]}'",
        )


if __name__ == "__main__":
    unittest.main()
