"""
E2E tests for NPU pause/resume generation — multi-turn conversatio.
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


class TestNpuMultiTurnWithPauseResume(CustomTestCase):
    """E2E: multi-turn chat works correctly when pause/resume is inserted
    between conversation turns.

    [Test Category] RL Pause/Resume + Multi-Turn
    [Test Target] POST /v1/chat/completions, POST /pause_generation,
                  POST /continue_generation
    """

    REQUEST_TIMEOUT = 180

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
                "ascend",  # Required for NPU
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ── helpers ──────────────────────────────────────────────────

    @property
    def _chat_url(self):
        return self.base_url + "/v1/chat/completions"

    def _chat(self, messages, max_tokens=128, temperature=0):
        """Send a chat completions request."""
        return requests.post(
            self._chat_url,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
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

    def test_multi_turn_with_pause_between_turns(self):
        """
        T7: Two-turn chat with in_place pause between turns, plus a
        companion long-generation request that proves pause actually
        blocks in-flight work.

        Turn1: tell the model a fact (short, finishes quickly).
        [pause → verify companion is blocked → resume]
        Turn2: ask the model to recall the fact.
        """
        messages = [
            {"role": "user", "content": "My cat's name is Whiskers."},
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Long companion request — proves pause blocks in-flight work
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

            # Turn 1 — short chat, expected to finish before companion
            turn1_future = executor.submit(self._chat, messages, max_tokens=64)

            # Wait for Turn 1 to complete
            turn1_resp = turn1_future.result(timeout=self.REQUEST_TIMEOUT)
            self.assertEqual(
                turn1_resp.status_code, 200, f"Turn 1 failed: {turn1_resp.text}"
            )
            turn1 = turn1_resp.json()
            turn1_text = turn1["choices"][0]["message"]["content"]
            self.assertGreater(len(turn1_text), 0)

            # Companion should still be running — pause while it's in-flight
            self.assertTrue(
                not long_future.done(),
                "Companion finished before pause; it wasn't long enough",
            )

            self._pause("in_place").raise_for_status()

            # Verify companion does NOT complete during pause
            try:
                time.sleep(5)
                self.assertTrue(
                    not long_future.done(),
                    "Companion completed during pause — pause may not have taken effect",
                )
            finally:
                self._continue().raise_for_status()

            # Companion should complete successfully after resume
            long_resp = long_future.result(timeout=self.REQUEST_TIMEOUT)
            self.assertEqual(long_resp.status_code, 200)
            long_text = long_resp.json()["choices"][0]["message"]["content"]
            self.assertGreater(
                len(long_text),
                0,
                "Companion should produce non-empty output after resume",
            )

        # Build conversation history for turn 2
        messages.append({"role": "assistant", "content": turn1_text})

        # Turn 2 — verify context is preserved
        messages.append({"role": "user", "content": "What is my cat's name?"})
        resp = self._chat(messages, max_tokens=64)
        self.assertEqual(resp.status_code, 200, f"Turn 2 failed: {resp.text}")
        turn2 = resp.json()
        turn2_text = turn2["choices"][0]["message"]["content"]

        self.assertGreater(len(turn2_text), 0)
        # The model should recall the cat's name from turn 1
        self.assertIn(
            "Whiskers",
            turn2_text,
            f"Model failed to recall context after pause/resume. "
            f"Turn 1: '{turn1_text[:100]}'. Turn 2: '{turn2_text[:100]}'",
        )

    def test_multi_turn_pause_during_long_generation(self):
        """
        Multi-turn chat where a long generation is paused mid-way,
        then resumed, followed by a follow-up chat turn.

        This simulates an RL rollout scenario where the engine is paused
        mid-generation, and after a weight update, the environment sends
        the next turn.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    "Write a 10-sentence paragraph explaining what "
                    "artificial intelligence is, covering its history, "
                    "key techniques, and future outlook:"
                ),
            },
        ]

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._chat, messages, max_tokens=512)

            time.sleep(0.5)
            self._pause("in_place").raise_for_status()

            # Verify paused; resume even if assertion fails to prevent
            # executor.shutdown() from hanging on a paused future.
            try:
                time.sleep(5)
                self.assertFalse(future.done())
            finally:
                self._continue().raise_for_status()

            resp = future.result(timeout=self.REQUEST_TIMEOUT)

        self.assertEqual(resp.status_code, 200)
        turn1 = resp.json()
        turn1_text = turn1["choices"][0]["message"]["content"]
        self.assertEqual(
            turn1["choices"][0]["finish_reason"],
            "stop",
            "Long generation should finish normally after pause/resume",
        )
        self.assertGreater(len(turn1_text), 100)

        # Follow-up turn — verify semantic correctness
        messages.append({"role": "assistant", "content": turn1_text})
        messages.append({"role": "user", "content": "Summarize that in one sentence."})

        resp = self._chat(messages, max_tokens=128)
        self.assertEqual(resp.status_code, 200)
        turn2_text = resp.json()["choices"][0]["message"]["content"]
        self.assertGreater(len(turn2_text), 0)
        # The summary should reference AI — proves context is preserved
        self.assertIn(
            "intelligence",
            turn2_text.lower(),
            f"Follow-up summary doesn't reference AI. "
            f"Turn 1: '{turn1_text[:100]}'. Turn 2: '{turn2_text[:100]}'",
        )


if __name__ == "__main__":
    unittest.main()
