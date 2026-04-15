"""
Streaming session tests: KV cache mechanics, logprob leak, chunked prefill leak.

All tests share a single server (DEFAULT_SMALL_MODEL) with streaming sessions
and chunked prefill enabled.

Usage:
    python -m pytest test_streaming_session.py -xvs
    python -m unittest test_streaming_session.TestStreamingSession
"""

import asyncio
import json
import time
import unittest
from typing import Any, Optional

import aiohttp
import requests

from sglang.srt.environ import envs
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
# Abort-heavy chunked prefill leak repro constants
# ---------------------------------------------------------------------------

ABORT_REPRO_CONTEXT_LEN = 512
ABORT_REPRO_PAGE_SIZE = 16
ABORT_REPRO_GEN_LEN = 4
ABORT_REPRO_SESSIONS = 4
ABORT_REPRO_WARMUP_TURNS = 1
ABORT_REPRO_ROUNDS = 8
ABORT_REPRO_STREAM_TOKENS = 16
ABORT_REPRO_ABORT_TOKENS = 600
ABORT_REPRO_NON_STREAMING_TOKENS = 16
ABORT_REPRO_CHUNKED_PREFILL_SIZE = 128


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


def _make_token_sized_ids(
    tokenizer: Any, prefix: str, min_tokens: int, max_tokens: Optional[int] = None
) -> list[int]:
    text = prefix
    chunk = " pack quartz wizard sphinx zebra fox " * 16
    token_ids = tokenizer.encode(text)
    while len(token_ids) < min_tokens:
        text += chunk
        token_ids = tokenizer.encode(text)
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    return token_ids


async def _abort_repro_generate(
    base_url: str,
    session: aiohttp.ClientSession,
    input_ids: list[int],
    max_new_tokens: int,
    session_params: Optional[dict[str, Any]] = None,
    expect_abort: bool = False,
) -> Optional[dict[str, Any]]:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if session_params:
        payload["session_params"] = session_params

    async with session.post(base_url + "/generate", json=payload) as resp:
        text = await resp.text()
        if expect_abort:
            if resp.status == 200:
                data = json.loads(text)
                finish_reason = data.get("meta_info", {}).get("finish_reason", {})
                assert finish_reason.get("type") == "abort", text
                assert "maximum allowed length" in finish_reason.get(
                    "message", ""
                ) or "context length" in finish_reason.get("message", ""), text
                return data
            assert resp.status == 400, text
            assert "maximum allowed length" in text or "context length" in text, text
            return None

        assert resp.status == 200, text
        data = json.loads(text)
        finish_reason = data.get("meta_info", {}).get("finish_reason", {})
        assert finish_reason.get("type") != "abort", text
        return data


async def _abort_repro_run_all(base_url: str, tokenizer: Any) -> None:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        session_ids = []
        for _ in range(ABORT_REPRO_SESSIONS):
            async with http.post(
                base_url + "/open_session",
                json={"capacity_of_str_len": 50000, "streaming": True},
            ) as resp:
                assert resp.status == 200, await resp.text()
                session_ids.append(await resp.json())

        try:
            for warmup_turn in range(ABORT_REPRO_WARMUP_TURNS):
                warmup_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[warmup={warmup_turn} session={session_idx}]",
                        min_tokens=ABORT_REPRO_STREAM_TOKENS,
                        max_tokens=ABORT_REPRO_STREAM_TOKENS + 8,
                    )
                    warmup_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                        )
                    )
                await asyncio.gather(*warmup_tasks)

            for round_idx in range(ABORT_REPRO_ROUNDS):
                mixed_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} ok session={session_idx}]",
                        min_tokens=ABORT_REPRO_STREAM_TOKENS,
                        max_tokens=ABORT_REPRO_STREAM_TOKENS + 8,
                    )
                    mixed_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                        )
                    )

                for ns_idx in range(2):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} ns={ns_idx}]",
                        min_tokens=ABORT_REPRO_NON_STREAMING_TOKENS,
                        max_tokens=ABORT_REPRO_NON_STREAMING_TOKENS + 8,
                    )
                    mixed_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                        )
                    )
                await asyncio.gather(*mixed_tasks)

                abort_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} abort session={session_idx}]",
                        min_tokens=ABORT_REPRO_ABORT_TOKENS,
                    )
                    abort_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                            expect_abort=True,
                        )
                    )
                await asyncio.gather(*abort_tasks)

                recovery_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} recover session={session_idx}]",
                        min_tokens=ABORT_REPRO_NON_STREAMING_TOKENS,
                        max_tokens=ABORT_REPRO_NON_STREAMING_TOKENS + 8,
                    )
                    recovery_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                        )
                    )
                recovery_results = await asyncio.gather(*recovery_tasks)
                for result in recovery_results:
                    assert result is not None
                    assert result["meta_info"]["cached_tokens"] > 0, result

                health = requests.get(base_url + "/health", timeout=10)
                if health.status_code != 200:
                    raise RuntimeError(
                        f"server unhealthy after round={round_idx}: "
                        f"{health.status_code} {health.text}"
                    )
        finally:
            for session_id in session_ids:
                async with http.post(
                    base_url + "/close_session", json={"session_id": session_id}
                ) as resp:
                    assert resp.status == 200, await resp.text()


# ===================================================================
# Test class
# ===================================================================


class TestStreamingSession(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
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

    def test_nth_mid_abort_recovery(self) -> None:
        """Abort a running streaming session request (nth turn) via the
        abort API. Session rolls back to last successful turn."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            # Turn 1: normal generate to create slot.
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")
            resp_1 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_1,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_1.status_code, 200, resp_1.text)
            data_1 = resp_1.json()
            turn_1_total = (
                data_1["meta_info"]["prompt_tokens"]
                + data_1["meta_info"]["completion_tokens"]
            )

            # Turn 2: long generate, then abort mid-decode.
            ids_2 = self.tokenizer.encode(" Continue the story in great detail.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 100000,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()
            time.sleep(0.5)
            abort_resp = requests.post(
                self.base_url + "/abort_request",
                json={"rid": "", "abort_all": True},
                timeout=10,
            )
            self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
            t.join(timeout=30)

            self.assertIsNotNone(result[0], "Turn 2 should have returned")
            data_2 = result[0].json()
            self.assertEqual(
                data_2["meta_info"]["finish_reason"]["type"],
                "abort",
                "Turn 2 should be aborted, not finished normally",
            )

            # Turn 3: recovery. Rolls back to turn 1.
            ids_3 = self.tokenizer.encode(" What happens next?")
            for attempt in range(20):
                resp_3 = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_3,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                if resp_3.status_code == 200:
                    break
                time.sleep(0.5)
            self.assertEqual(resp_3.status_code, 200, resp_3.text)
            data_3 = resp_3.json()
            # prompt_tokens = turn_1_total + append (BOS stripped).
            bos = 1 if ids_3[0] == self.tokenizer.bos_token_id else 0
            expected_prompt_3 = turn_1_total + len(ids_3) - bos
            self.assertEqual(
                data_3["meta_info"]["prompt_tokens"],
                expected_prompt_3,
                "prompt_tokens must equal turn_1_total + append (no stale abort context)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

    def test_first_mid_abort_recovery(self) -> None:
        """Abort the very first request on a streaming session mid-decode.
        No slot exists yet (ephemeral slot created and nuked).
        Verify the session is still usable afterward."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_1,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 100000,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()
            time.sleep(0.5)
            abort_resp = requests.post(
                self.base_url + "/abort_request",
                json={"rid": "", "abort_all": True},
                timeout=10,
            )
            self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
            t.join(timeout=30)

            self.assertIsNotNone(result[0], "Turn 1 should have returned")
            data_1 = result[0].json()
            self.assertEqual(
                data_1["meta_info"]["finish_reason"]["type"],
                "abort",
                "Turn 1 should be aborted, not finished normally",
            )

            # Turn 2: recovery. No inherited context (req_nodes empty).
            ids_2 = self.tokenizer.encode("Tell me a short joke.")
            for attempt in range(20):
                resp_2 = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                if resp_2.status_code == 200:
                    break
                time.sleep(0.5)
            self.assertEqual(resp_2.status_code, 200, resp_2.text)
            data_2 = resp_2.json()
            self.assertEqual(
                data_2["meta_info"]["prompt_tokens"],
                len(ids_2),
                "prompt_tokens must equal turn 2 input only (no inherited context)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

    def test_preabort_recovery(self) -> None:
        """Pre-aborted request (unsupported offset) does not corrupt session.
        The slot is preserved, and the next turn inherits correctly."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            # Turn 1: normal generate to create slot.
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")
            resp_1 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_1,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_1.status_code, 200, resp_1.text)
            data_1 = resp_1.json()
            turn_1_total = (
                data_1["meta_info"]["prompt_tokens"]
                + data_1["meta_info"]["completion_tokens"]
            )

            # Turn 2: pre-aborted via unsupported offset parameter.
            ids_2 = self.tokenizer.encode(" This should be rejected.")
            resp_2 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_2,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "session_params": {
                        "id": session_id,
                        "rid": None,
                        "offset": 1,
                    },
                },
                timeout=30,
            )
            self.assertIn(resp_2.status_code, (200, 400), resp_2.text)

            # Turn 3: normal append. Slot should be intact from turn 1.
            ids_3 = self.tokenizer.encode(" What happens next?")
            resp_3 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_3,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_3.status_code, 200, resp_3.text)
            data_3 = resp_3.json()
            bos = 1 if ids_3[0] == self.tokenizer.bos_token_id else 0
            expected_prompt_3 = turn_1_total + len(ids_3) - bos
            self.assertEqual(
                data_3["meta_info"]["prompt_tokens"],
                expected_prompt_3,
                "prompt_tokens must equal turn_1_total + append (slot preserved)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)


class TestStreamingSessionMixedChunk(TestStreamingSession):
    """Streaming session with --enable-mixed-chunk."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "512",
                    "--enable-mixed-chunk",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


@unittest.skip("streaming session + retract has a token leak — tracked separately")
class TestStreamingSessionRetract(TestStreamingSession):
    """Streaming session under retract decode pressure."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "128",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


@unittest.skip("streaming session + retract has a token leak — tracked separately")
class TestStreamingSessionRetractMixedChunk(TestStreamingSession):
    """Streaming session under retract decode with --enable-mixed-chunk."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "128",
                    "--enable-mixed-chunk",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionAbortLeakRepro(CustomTestCase):
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
                str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
                "--context-length",
                str(ABORT_REPRO_CONTEXT_LEN),
                "--page-size",
                str(ABORT_REPRO_PAGE_SIZE),
                "--max-running-requests",
                "32",
                "--log-level",
                "info",
            ],
        )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_abort_heavy_chunked_prefill_does_not_leak(self) -> None:
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_abort_repro_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-session cleanup request {i}.")
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200, response.text)

        time.sleep(5)
        self.assertIsNone(
            self.process.poll(),
            "Server crashed during abort-heavy streaming session repro.",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after abort-heavy streaming session cleanup.",
        )


if __name__ == "__main__":
    unittest.main()
