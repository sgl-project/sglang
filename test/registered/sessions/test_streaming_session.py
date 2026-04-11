"""
Streaming session tests: KV cache mechanics, logprob leak, chunked prefill leak.

All tests share a single server (DEFAULT_SMALL_MODEL) with streaming sessions
and chunked prefill enabled.

Usage:
    python -m pytest test_streaming_session.py -xvs
    python -m unittest test_streaming_session.TestStreamingSession
"""

import asyncio
import time
import unittest
from typing import Any, Optional

import aiohttp
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=67, suite="stage-b-test-1-gpu-large")

# ---------------------------------------------------------------------------
# Logprob leak constants
# ---------------------------------------------------------------------------

LOGPROB_NUM_TURNS = 5
LOGPROB_NUM_ROUNDS = 30

LOGPROB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Sphinx of black quartz judge my vow.",
    "The five boxing wizards jump quickly.",
]

# ---------------------------------------------------------------------------
# Chunked prefill leak constants
# ---------------------------------------------------------------------------

LEAK_NUM_SESSIONS = 4
LEAK_NUM_TURNS = 5
LEAK_GEN_LEN = 16

# Filler text to trigger chunked prefill (200+ tokens per turn)
LEAK_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Jackdaws love my big sphinx of quartz. "
    "A wizard's job is to vex chumps quickly in fog. "
    "We promptly judged antique ivory buckles for the next prize. "
) * 20


# ---------------------------------------------------------------------------
# Logprob leak helpers
# ---------------------------------------------------------------------------


def _logprob_generate(base_url, input_ids, **kwargs) -> dict:
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


def _logprob_run_one_session(base_url, tokenizer, **gen_kwargs):
    """Open session -> N turns -> close."""
    resp = requests.post(
        base_url + "/open_session",
        json={"capacity_of_str_len": 50000, "streaming": True},
    )
    assert resp.status_code == 200
    session_id = resp.json()

    rid = None
    for turn in range(LOGPROB_NUM_TURNS):
        turn_ids = tokenizer.encode(
            f"Turn {turn}: {LOGPROB_PROMPTS[turn % len(LOGPROB_PROMPTS)]}"
        )
        result = _logprob_generate(
            base_url,
            turn_ids,
            session_params={"id": session_id, "rid": rid},
            **gen_kwargs,
        )
        rid = result["meta_info"]["id"]

    requests.post(base_url + "/close_session", json={"session_id": session_id})


def _logprob_assert_no_leak(base_url, tokenizer, **gen_kwargs):
    """Run many session rounds and verify server stays healthy."""
    requests.post(base_url + "/flush_cache")
    for _ in range(LOGPROB_NUM_ROUNDS):
        _logprob_run_one_session(base_url, tokenizer, **gen_kwargs)
    time.sleep(3)
    assert (
        requests.get(base_url + "/health").status_code == 200
    ), "Server unhealthy — likely a token memory leak."


# ---------------------------------------------------------------------------
# Chunked prefill leak helpers
# ---------------------------------------------------------------------------


async def _leak_async_generate(
    base_url: str,
    session: aiohttp.ClientSession,
    input_ids: list[int],
    session_params: Optional[dict[str, Any]] = None,
) -> Any:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": LEAK_GEN_LEN,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if session_params:
        payload["session_params"] = session_params
    timeout = aiohttp.ClientTimeout(total=300)
    async with session.post(
        base_url + "/generate", json=payload, timeout=timeout
    ) as resp:
        assert resp.status == 200, f"Generate failed: {await resp.text()}"
        return await resp.json()


async def _leak_run_all(base_url: str, tokenizer: Any) -> None:
    """Fire all requests per turn simultaneously to create mixed batches."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        # Open all sessions
        sids = []
        for s in range(LEAK_NUM_SESSIONS):
            async with http.post(
                base_url + "/open_session",
                json={"capacity_of_str_len": 50000, "streaming": True},
            ) as resp:
                sids.append(await resp.json())

        # For each turn, fire ALL streaming + non-streaming requests at once
        for turn in range(LEAK_NUM_TURNS):
            tasks = []
            # Streaming requests for all sessions
            for s in range(LEAK_NUM_SESSIONS):
                offset = (s * LEAK_NUM_TURNS + turn) * 200
                text = f"Session {s} turn {turn}: {LEAK_FILLER[offset : offset + 1500]}"
                ids = tokenizer.encode(text)
                tasks.append(
                    _leak_async_generate(
                        base_url,
                        http,
                        ids,
                        session_params={"id": sids[s], "rid": None},
                    )
                )

            # Non-streaming requests interleaved
            for ns in range(LEAK_NUM_SESSIONS // 2):
                text = f"Non-streaming {ns} turn {turn}: {LEAK_FILLER[ns * 100 : ns * 100 + 500]}"
                ids = tokenizer.encode(text)
                tasks.append(_leak_async_generate(base_url, http, ids))

            # Fire all at once — creates mixed batch of streaming + non-streaming
            await asyncio.gather(*tasks)

        # Close all sessions
        for sid in sids:
            async with http.post(
                base_url + "/close_session", json={"session_id": sid}
            ) as resp:
                assert resp.status == 200


# ===================================================================
# Test class
# ===================================================================


class TestStreamingSession(CustomTestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------
    # KV cache mechanics
    # ------------------------------------------------------------------

    def test_kv_cache_inheritance(self, gen_len=12):
        """Verify KV inheritance, radix cache insertion, and flush reclamation."""
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
        ]
        chunks_ids = [self.tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == self.tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # === Part 1: streaming session — check KV inheritance ===
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "streaming": True},
        ).json()
        rid = None

        prev_kv_len = 0
        for turn_idx, chunk_ids in enumerate(chunks_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session_params": {"id": session_id, "rid": rid},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": gen_len,
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
            ).json()
            rid = response["meta_info"]["id"]
            cached = response["meta_info"]["cached_tokens"]
            prompt_tokens = response["meta_info"]["prompt_tokens"]
            completion_tokens = response["meta_info"]["completion_tokens"]

            if turn_idx == 0:
                # Turn 1 should have no cache hit (cache was flushed).
                self.assertEqual(
                    cached, 0, "Turn 1 should have 0 cached tokens (clean start)"
                )
            else:
                # Turns 2+ inherit KV from the previous turn (via inherit_kv_states,
                # not radix tree matching). cached_tokens reflects the inherited prefix.
                self.assertEqual(
                    cached,
                    prev_kv_len,
                    f"Turn {turn_idx + 1}: should inherit {prev_kv_len} KV tokens from previous turn",
                )
            prev_kv_len = prompt_tokens + completion_tokens

        # Close the session before checking cache/memory state.
        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(ret.status_code, 200)

        # === Cache verification (after close, before flush) ===

        # Turn 1's prompt was inserted to the cache.
        verify_resp = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        ).json()
        self.assertGreater(
            verify_resp["meta_info"]["cached_tokens"],
            0,
            "Turn 1's prompt should be cached in the radix tree",
        )

        # Turn 2's prompt tokens should NOT be in cache.
        # The tree should only contain turn 1's extent (prompt + output from
        # cache_unfinished_req during decode). Turn 2's prompt starts fresh tokens
        # that were never inserted.
        verify_resp2 = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[1],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        ).json()
        self.assertEqual(
            verify_resp2["meta_info"]["cached_tokens"],
            0,
            "Turn 2's prompt should not be in cache (no insertion for turns 2+)",
        )

        # === Flush reclamation ===

        requests.post(self.base_url + "/flush_cache")
        verify_resp3 = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        ).json()
        self.assertEqual(
            verify_resp3["meta_info"]["cached_tokens"],
            0,
            "After session close + flush, cache should be fully reclaimed",
        )

    # ------------------------------------------------------------------
    # Logprob leak tests
    # ------------------------------------------------------------------

    def test_leak_logprob_none(self) -> None:
        """Streaming sessions without logprobs must not leak tokens."""
        _logprob_assert_no_leak(self.base_url, self.tokenizer)

    def test_leak_logprob_output(self) -> None:
        """Streaming sessions with output logprobs must not leak tokens."""
        _logprob_assert_no_leak(self.base_url, self.tokenizer, return_logprob=True)

    def test_leak_logprob_input(self) -> None:
        """Streaming sessions with logprob_start_len=0 must not leak tokens."""
        _logprob_assert_no_leak(
            self.base_url,
            self.tokenizer,
            return_logprob=True,
            logprob_start_len=0,
        )

    # ------------------------------------------------------------------
    # Chunked prefill leak test
    # ------------------------------------------------------------------

    def test_leak_chunked_prefill(self) -> None:
        """Concurrent multi-turn streaming sessions then idle health check."""
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_leak_run_all(self.base_url, self.tokenizer))

        # Run a few non-streaming requests to flush state
        for i in range(3):
            ids = self.tokenizer.encode(f"Flush request {i}: final cleanup.")
            requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
            )

        # Wait for server to go idle and run memory check
        time.sleep(5)
        health = requests.get(self.base_url + "/health")
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after streaming session close — "
            "likely a token memory leak from streaming session lifecycle.",
        )


if __name__ == "__main__":
    unittest.main()
