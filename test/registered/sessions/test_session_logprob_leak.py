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
from typing import Any, Optional

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

NUM_SESSIONS = 4
NUM_TURNS = 5
GEN_LEN = 16

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Jackdaws love my big sphinx of quartz. "
    "A wizard's job is to vex chumps quickly in fog. "
    "We promptly judged antique ivory buckles for the next prize. "
) * 20


def _generate(
    base_url: str,
    input_ids: list[int],
    return_logprob: bool = False,
    logprob_start_len: Optional[int] = None,
    max_new_tokens: int = GEN_LEN,
    session_params: Optional[dict[str, Any]] = None,
) -> dict:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if return_logprob:
        payload["return_logprob"] = True
    if logprob_start_len is not None:
        payload["logprob_start_len"] = logprob_start_len
    if session_params:
        payload["session_params"] = session_params
    resp = requests.post(base_url + "/generate", json=payload, timeout=120)
    assert resp.status_code == 200, f"Generate failed: {resp.text}"
    return resp.json()


def _extract_logprobs(raw: list) -> list[float]:
    return [entry[0] if entry[0] is not None else 0.0 for entry in raw]


def _run_sessions(
    base_url: str,
    tokenizer: Any,
    return_logprob: bool,
    logprob_start_len: Optional[int] = None,
) -> None:
    """Run sessions, check health, and optionally compare logprobs."""
    has_input_logprobs = logprob_start_len is not None and logprob_start_len >= 0
    requests.post(base_url + "/flush_cache")

    # Open sessions
    session_ids = []
    for s in range(NUM_SESSIONS):
        resp = requests.post(
            base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        assert resp.status_code == 200
        session_ids.append(resp.json())

    # Per-session tracking
    all_ids: list[list[int]] = [[] for _ in range(NUM_SESSIONS)]
    per_turn_output_logprobs: list[list[tuple[int, list[float]]]] = [
        [] for _ in range(NUM_SESSIONS)
    ]
    all_logprobs: list[list[float]] = [[] for _ in range(NUM_SESSIONS)]
    skip_positions: list[set[int]] = [set() for _ in range(NUM_SESSIONS)]

    rids: list[Optional[str]] = [None] * NUM_SESSIONS
    for turn in range(NUM_TURNS):
        for s in range(NUM_SESSIONS):
            offset = (s * NUM_TURNS + turn) * 200
            text = f"Session {s} turn {turn}: {FILLER[offset : offset + 500]}"
            turn_ids = tokenizer.encode(text)
            output_start = len(all_ids[s]) + len(turn_ids)
            all_ids[s].extend(turn_ids)

            result = _generate(
                base_url,
                turn_ids,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                session_params={"id": session_ids[s], "rid": rids[s]},
            )
            rids[s] = result["meta_info"]["id"]
            all_ids[s].extend(result["output_ids"])

            if return_logprob:
                out_lp = _extract_logprobs(
                    result["meta_info"]["output_token_logprobs"]
                )
                if has_input_logprobs:
                    in_lp = _extract_logprobs(
                        result["meta_info"]["input_token_logprobs"]
                    )
                    if turn > 0:
                        skip_positions[s].add(len(all_logprobs[s]))
                    all_logprobs[s].extend(in_lp)
                    all_logprobs[s].extend(out_lp)
                else:
                    per_turn_output_logprobs[s].append((output_start, out_lp))

    # Close sessions
    for sid in session_ids:
        resp = requests.post(base_url + "/close_session", json={"session_id": sid})
        assert resp.status_code == 200

    # Flush state
    for i in range(3):
        ids = tokenizer.encode(f"Flush request {i}: final cleanup.")
        requests.post(
            base_url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
        )

    # Health check
    time.sleep(5)
    health = requests.get(base_url + "/health")
    assert health.status_code == 200, (
        "Server unhealthy after streaming sessions with logprobs — "
        "likely a token memory leak."
    )

    # Logprob comparison against non-session baseline
    if not return_logprob:
        return

    requests.post(base_url + "/flush_cache")
    for s in range(NUM_SESSIONS):
        ns_result = _generate(
            base_url,
            all_ids[s],
            return_logprob=True,
            logprob_start_len=0,
            max_new_tokens=1,
        )
        ns_input_lp = _extract_logprobs(
            ns_result["meta_info"]["input_token_logprobs"]
        )

        if has_input_logprobs:
            assert len(ns_input_lp) == len(all_logprobs[s]), (
                f"session {s}: length mismatch "
                f"ns={len(ns_input_lp)}, session={len(all_logprobs[s])}"
            )
            for i, (e, a) in enumerate(zip(ns_input_lp, all_logprobs[s])):
                if i in skip_positions[s]:
                    continue
                assert e == a, (
                    f"session {s} token {i}: "
                    f"expected={e:.6f}, actual={a:.6f}, diff={abs(e - a):.6f}"
                )
        else:
            for turn_idx, (start, out_lp) in enumerate(
                per_turn_output_logprobs[s]
            ):
                segment = ns_input_lp[start : start + len(out_lp)]
                for i, (e, a) in enumerate(zip(segment, out_lp)):
                    assert e == a, (
                        f"session {s} turn {turn_idx} token {i}: "
                        f"expected={e:.6f}, actual={a:.6f}"
                    )


class TestSessionLogprobLeak(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-streaming-session",
                "--chunked-prefill-size",
                "512",
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_session_without_logprob(self) -> None:
        """Streaming sessions without logprobs must not leak tokens."""
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model)
        _run_sessions(self.base_url, tokenizer, return_logprob=False)

    def test_session_with_output_logprob(self) -> None:
        """Streaming sessions with output logprobs must not leak tokens."""
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model)
        _run_sessions(self.base_url, tokenizer, return_logprob=True)

    def test_session_with_input_logprob(self) -> None:
        """Streaming sessions with logprob_start_len=0 must not leak tokens."""
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model)
        _run_sessions(
            self.base_url, tokenizer, return_logprob=True, logprob_start_len=0
        )


if __name__ == "__main__":
    unittest.main()
