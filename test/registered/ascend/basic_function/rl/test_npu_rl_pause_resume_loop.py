"""
E2E tests for NPU — pause/resume + tool call + multi-turn robustness.
"""

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

register_npu_ci(est_time=60, suite="full-1-npu-a3", nightly=True)


class TestNpuRlPauseResume(CustomTestCase):
    """E2E: RL pause/resume robustness with multi-turn + tool call + retract.

    [Test Category] RL Pause/Resume
    [Test Target] POST /v1/chat/completions (with tools),
                  POST /pause_generation (in_place + retract),
                  POST /continue_generation
    """

    REQUEST_TIMEOUT = 240

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather conditions including temperature, humidity and precipitation for a specified city. Returns structured weather data that can be used for further analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city to query weather for, e.g. Paris, London, Tokyo",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation or unit conversion. Supports arithmetic operations and temperature conversion (Celsius to Fahrenheit, Fahrenheit to Celsius).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression or conversion to evaluate, e.g. '15°C to Fahrenheit' or '32 * 1.8 + 32'",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    SYSTEM_MESSAGE = (
        "You are a helpful assistant with tool calling capabilities. "
        "You have access to a weather function for checking weather "
        "and a calculate function for math and unit conversions. "
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
                "ascend",
                "--tool-call-parser",
                "llama3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ── helpers ──────────────────────────────────────────────────

    @property
    def _chat_url(self):
        return self.base_url + "/v1/chat/completions"

    def _chat(
        self,
        messages,
        max_tokens=256,
        temperature=0,
        tools=None,
        ignore_eos: bool = False,
    ):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            body["tools"] = tools
        if ignore_eos:
            body["ignore_eos"] = True
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

    def _assert_valid_response(self, resp, expect_tool_calls=False):
        """Assert a chat completions response is well-formed."""
        self.assertEqual(resp.status_code, 200, f"Request failed: {resp.text[:500]}")
        body = resp.json()
        choice = body["choices"][0]
        if expect_tool_calls:
            tool_calls = choice["message"].get("tool_calls")
            self.assertIsNotNone(tool_calls, "Expected tool_calls")
            self.assertGreater(len(tool_calls), 0, "Empty tool_calls")
            return body, tool_calls
        else:
            content = choice["message"].get("content", "")
            return body, content

    # ── full RL loop ──

    def test_full_rl_loop_in_place(self):
        """
        Simulate a complete RL training step with 3 in_place
        pause/resume cycles interleaved with multi-turn tool call flow.

        Cycle 1 (rollout): user → tool_call → pause → resume
        Cycle 2 (tool result): tool result → answer → pause → resume
        Cycle 3 (follow-up): user follow-up → answer → pause → resume
        Cycle 4 (final): verify engine still healthy with another query
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ]

        # ── Cycle 1: First rollout → tool call ──
        # Pass full tool set to verify model selects the right tool.
        resp = self._chat(messages, max_tokens=256, tools=self.TOOLS)
        body, tool_calls = self._assert_valid_response(resp, expect_tool_calls=True)

        # Pause → simulate tool execution by RL environment
        self._pause("in_place").raise_for_status()
        try:
            time.sleep(1)

            tool_call = tool_calls[0]
            tool_call_id = tool_call.get("id", "call_001")
            messages.append(body["choices"][0]["message"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Cloudy, 15°C",
                }
            )
        finally:
            self._continue().raise_for_status()

        # ── Cycle 2: Process tool result → weather answer ──
        resp = self._chat(messages, max_tokens=128)
        _, turn2_text = self._assert_valid_response(resp, expect_tool_calls=False)
        self.assertGreater(len(turn2_text), 0)

        # Pause → simulate reward computation by RL trainer
        self._pause("in_place").raise_for_status()
        try:
            time.sleep(1)

            messages.append({"role": "assistant", "content": turn2_text})
            messages.append(
                {
                    "role": "user",
                    "content": "How warm is that in Fahrenheit?",
                }
            )
        finally:
            self._continue().raise_for_status()

        # ── Cycle 3: Follow-up → temperature conversion tool call ──
        # Pass full tool set to verify model selects calculate for conversion.
        resp = self._chat(messages, max_tokens=256, tools=self.TOOLS)
        body3, tool_calls3 = self._assert_valid_response(resp, expect_tool_calls=True)
        self.assertEqual(
            tool_calls3[0]["function"]["name"],
            "calculate",
            f"Expected calculate tool call for temperature conversion, "
            f"got '{tool_calls3[0]['function']['name']}'",
        )

        # Final pause/resume — simulate tool execution + weight sync
        self._pause("in_place").raise_for_status()
        try:
            time.sleep(1)

            tool_call3 = tool_calls3[0]
            tool_call_id3 = tool_call3.get("id", "call_002")
            messages.append(body3["choices"][0]["message"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id3,
                    "content": "15°C = 59°F",
                }
            )
        finally:
            self._continue().raise_for_status()

        # ── Cycle 4: Verify engine still serving — model consumes
        # tool result and produces final answer.
        resp = self._chat(messages, max_tokens=128)
        _, final_text = self._assert_valid_response(resp, expect_tool_calls=False)
        self.assertGreater(len(final_text), 0)
        print(f"[Cycle4 final] {final_text[:500]}")

    def test_full_rl_loop_retract(self):
        """
        Full RL loop using retract mode — the more realistic scenario
        where KV cache is flushed between training steps.

        Uses a single long-generation request with retract:
          Start long generation → retract (flush KV cache)
          → resume (recompute prefix) → retract → resume → retract → resume
          → verify final output is valid.

        Also verifies the engine can handle new requests after the loop.
        """
        prompt = (
            "Write a comprehensive overview of artificial intelligence "
            "including its history, key technologies like machine learning "
            "and deep learning, current applications, and future prospects:"
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: self._chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=512,
                    ignore_eos=True,  # force full generation between retracts
                )
            )

            # Multiple retract cycles simulating RL training steps.
            # Each retract flushes KV cache; resume recomputes prefix
            # and continues generating.
            for cycle in range(3):
                time.sleep(1.0)  # ensure some decode tokens are generated
                self._pause("retract").raise_for_status()
                try:
                    # Simulate RL trainer doing work
                    time.sleep(0.5)
                finally:
                    self._continue().raise_for_status()
                print(
                    f"[Retract cycle {cycle + 1}] paused & resumed "
                    f"(KV cache flushed, prefix recomputed)"
                )

            # Ensure engine is recovered even if future.result() fails.
            # If the request hung and we don't recover, subsequent tests
            # would fail too.
            try:
                resp = future.result(timeout=self.REQUEST_TIMEOUT * 2)
            except Exception:
                # Attempt to force-resume in case engine is still paused
                try:
                    self._continue()
                except Exception:
                    pass
                raise

        self.assertEqual(
            resp.status_code,
            200,
            f"Full RL loop failed after 3 retract cycles: {resp.text[:500]}",
        )
        body = resp.json()
        choice = body["choices"][0]
        # ignore_eos=True → finish_reason is "length" (max_tokens reached)
        self.assertEqual(
            choice["finish_reason"],
            "length",
            f"Request should finish with 'length' after 3 retract cycles, "
            f"got finish_reason={choice['finish_reason']}",
        )
        final_text = choice["message"]["content"]
        self.assertGreater(
            len(final_text),
            100,
            f"Output too short after 3 retract cycles: {len(final_text)} chars",
        )
        print(f"[Retract final] output length: {len(final_text)} chars")

        # Engine must still accept new requests after the retract loop
        resp = self._chat(
            [{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=32,
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"Post-retract health check failed: {resp.text[:300]}",
        )
        self.assertGreater(
            len(resp.json()["choices"][0]["message"]["content"]),
            0,
            "Post-retract request returned empty response",
        )

    def test_engine_health_after_sleep_cycles(self):
        """
        Verify engine remains healthy after multiple pause/resume cycles.
        Checks /health and /health_generate endpoints before, during, and after.
        """
        # Baseline: engine is healthy
        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

        # Run several pause/resume cycles
        for mode in ["in_place", "retract", "in_place"]:
            self._pause(mode).raise_for_status()
            time.sleep(0.5)
            self._continue().raise_for_status()

        # After cycles: engine still healthy
        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code,
            200,
            f"Engine unhealthy after sleep cycles: {health.text}",
        )

        # Health-generate should also respond
        health_gen = requests.get(self.base_url + "/health_generate", timeout=10)
        self.assertEqual(health_gen.status_code, 200)

        # Functional test: engine can still generate
        resp = self._chat(
            [{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
        )
        self.assertEqual(resp.status_code, 200)
        final = resp.json()["choices"][0]["message"]["content"]
        self.assertGreater(len(final), 0)


if __name__ == "__main__":
    unittest.main()
