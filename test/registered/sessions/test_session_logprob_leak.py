"""
Test for token leak in streaming sessions with return_logprob enabled.

When logprob_start_len=0, init_next_round_input truncates the prefix match
key to length 0, which bypasses the session slot's committed KV and orphans
allocated tokens. This test verifies that sessions with logprobs enabled
do not leak tokens.

Usage:
    python3 -m pytest test_session_logprob_leak.py -xvs
"""

import time
import unittest
from typing import Any

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")

NUM_TURNS = 5
NUM_ROUNDS = 30

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Sphinx of black quartz judge my vow.",
    "The five boxing wizards jump quickly.",
]


def _generate(base_url, input_ids, **kwargs) -> dict:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": kwargs.get("max_new_tokens", 8),
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if kwargs.get("return_logprob"):
        payload["return_logprob"] = True
    if kwargs.get("logprob_start_len") is not None:
        payload["logprob_start_len"] = kwargs["logprob_start_len"]
    if kwargs.get("session_params"):
        payload["session_params"] = kwargs["session_params"]
    resp = requests.post(base_url + "/generate", json=payload, timeout=120)
    assert resp.status_code == 200, f"Generate failed: {resp.text}"
    return resp.json()


def _run_one_session(base_url, tokenizer, **gen_kwargs):
    """Open session → N turns → close."""
    resp = requests.post(
        base_url + "/open_session",
        json={"capacity_of_str_len": 50000, "streaming": True},
    )
    assert resp.status_code == 200
    session_id = resp.json()

    rid = None
    for turn in range(NUM_TURNS):
        turn_ids = tokenizer.encode(f"Turn {turn}: {PROMPTS[turn % len(PROMPTS)]}")
        result = _generate(
            base_url,
            turn_ids,
            session_params={"id": session_id, "rid": rid},
            **gen_kwargs,
        )
        rid = result["meta_info"]["id"]

    requests.post(base_url + "/close_session", json={"session_id": session_id})


def _assert_no_leak(base_url, tokenizer, **gen_kwargs):
    """Run many session rounds and verify server stays healthy."""
    requests.post(base_url + "/flush_cache")
    for _ in range(NUM_ROUNDS):
        _run_one_session(base_url, tokenizer, **gen_kwargs)
    time.sleep(3)
    assert (
        requests.get(base_url + "/health").status_code == 200
    ), "Server unhealthy — likely a token memory leak."


class TestSessionLogprobLeak(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-streaming-session"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def _tokenizer(self):
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        return get_tokenizer(self.model)

    def test_session_without_logprob(self) -> None:
        """Streaming sessions without logprobs must not leak tokens."""
        _assert_no_leak(self.base_url, self._tokenizer())

    def test_session_with_output_logprob(self) -> None:
        """Streaming sessions with output logprobs must not leak tokens."""
        _assert_no_leak(self.base_url, self._tokenizer(), return_logprob=True)

    def test_session_with_input_logprob(self) -> None:
        """Streaming sessions with logprob_start_len=0 must not leak tokens."""
        _assert_no_leak(
            self.base_url,
            self._tokenizer(),
            return_logprob=True,
            logprob_start_len=0,
        )


if __name__ == "__main__":
    unittest.main()
