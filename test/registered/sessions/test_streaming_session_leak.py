"""
Test for token leak in streaming sessions with chunked prefill.

Runs concurrent multi-turn streaming sessions interleaved with non-streaming
requests (to create mixed batches), closes all sessions, waits for idle,
and checks server health.

Usage:
    python3 -m pytest test_streaming_session_leak.py -xvs
"""

import asyncio
import time
import unittest
from typing import Any, Optional

import aiohttp
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

# Filler text to trigger chunked prefill (200+ tokens per turn)
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


async def _async_generate(
    base_url: str,
    session: aiohttp.ClientSession,
    input_ids: list[int],
    session_params: Optional[dict[str, Any]] = None,
) -> Any:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": GEN_LEN,
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


async def _run_all(base_url: str, tokenizer: Any) -> None:
    """Fire all requests per turn simultaneously to create mixed batches."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        # Open all sessions
        sids = []
        for s in range(NUM_SESSIONS):
            async with http.post(
                base_url + "/open_session",
                json={"capacity_of_str_len": 50000, "streaming": True},
            ) as resp:
                sids.append(await resp.json())

        # For each turn, fire ALL streaming + non-streaming requests at once
        for turn in range(NUM_TURNS):
            tasks = []
            # Streaming requests for all sessions
            for s in range(NUM_SESSIONS):
                offset = (s * NUM_TURNS + turn) * 200
                text = f"Session {s} turn {turn}: {FILLER[offset : offset + 1500]}"
                ids = tokenizer.encode(text)
                tasks.append(
                    _async_generate(
                        base_url,
                        http,
                        ids,
                        session_params={"id": sids[s], "rid": None},
                    )
                )

            # Non-streaming requests interleaved
            for ns in range(NUM_SESSIONS // 2):
                text = f"Non-streaming {ns} turn {turn}: {FILLER[ns * 100 : ns * 100 + 500]}"
                ids = tokenizer.encode(text)
                tasks.append(_async_generate(base_url, http, ids))

            # Fire all at once — creates mixed batch of streaming + non-streaming
            await asyncio.gather(*tasks)

        # Close all sessions
        for sid in sids:
            async with http.post(
                base_url + "/close_session", json={"session_id": sid}
            ) as resp:
                assert resp.status == 200


class TestStreamingSessionLeak(CustomTestCase):
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

    def test_streaming_session_no_leak(self) -> None:
        """Concurrent multi-turn streaming sessions then idle health check."""
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model)
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_run_all(self.base_url, tokenizer))

        # Run a few non-streaming requests to flush state
        for i in range(3):
            ids = tokenizer.encode(f"Flush request {i}: final cleanup.")
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
