"""Streaming-session test fixture.

`TestStreamingSession` is the base class for all streaming-session tests
(default config — Llama-3.1-8B, no spec). Variants in
test_streaming_session.py and test_streaming_session_extra.py inherit
it and only override `setUpClass`.

Also exports:
- ABORT_REPRO_* constants used by the basic file's abort-leak repro.
- _abort_repro_run_all coroutine reused by the basic file.

Lives under sglang.test.server_fixtures so siblings under test/registered
can `import` it without sys.path hacks.
"""

import asyncio
import json
from typing import Any, Optional

import aiohttp
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

LOGPROB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Sphinx of black quartz judge my vow.",
    "The five boxing wizards jump quickly.",
]

# Long enough to trigger chunked prefill at 200+ tokens per slice.
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

ABORT_REPRO_CONTEXT_LEN = 512
ABORT_REPRO_PAGE_SIZE = 256
ABORT_REPRO_GEN_LEN = 4
ABORT_REPRO_SESSIONS = 4
ABORT_REPRO_WARMUP_TURNS = 1
ABORT_REPRO_ROUNDS = 8
ABORT_REPRO_STREAM_TOKENS = 16
ABORT_REPRO_ABORT_TOKENS = 600
ABORT_REPRO_NON_STREAMING_TOKENS = 16
ABORT_REPRO_CHUNKED_PREFILL_SIZE = 4096

CONCURRENT_LOGPROB_SESSIONS = 6
CONCURRENT_LOGPROB_TURNS = 5
CONCURRENT_LOGPROB_ROUNDS = 10

STRESS_NUM_SESSIONS = 8
STRESS_NUM_NON_STREAMING = 4
STRESS_NUM_TURNS = 6
STRESS_GEN_LEN = 16


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


async def _async_generate(
    base_url: str,
    session: aiohttp.ClientSession,
    input_ids: list[int],
    max_new_tokens: int = 8,
    session_params: Optional[dict[str, Any]] = None,
    return_logprob: bool = False,
    logprob_start_len: Optional[int] = None,
) -> dict[str, Any]:
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
    if return_logprob:
        payload["return_logprob"] = True
    if logprob_start_len is not None:
        payload["logprob_start_len"] = logprob_start_len
    timeout = aiohttp.ClientTimeout(total=300)
    async with session.post(
        base_url + "/generate", json=payload, timeout=timeout
    ) as resp:
        assert resp.status == 200, f"Generate failed: {await resp.text()}"
        return await resp.json()


async def _concurrent_logprob_run(base_url: str, tokenizer: Any, **gen_kwargs) -> None:
    """N sessions per round, all requests fired simultaneously per turn so
    the running batch has real concurrency (retract can actually kick one).
    """
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        for _ in range(CONCURRENT_LOGPROB_ROUNDS):
            sids: list[str] = []
            for _ in range(CONCURRENT_LOGPROB_SESSIONS):
                async with http.post(
                    base_url + "/open_session",
                    json={"capacity_of_str_len": 50000, "streaming": True},
                ) as resp:
                    assert resp.status == 200
                    sids.append(await resp.json())

            rids: list[Optional[str]] = [None] * CONCURRENT_LOGPROB_SESSIONS
            for turn in range(CONCURRENT_LOGPROB_TURNS):
                tasks = []
                for s in range(CONCURRENT_LOGPROB_SESSIONS):
                    text = (
                        f"S{s} T{turn}: "
                        f"{LOGPROB_PROMPTS[turn % len(LOGPROB_PROMPTS)]}"
                    )
                    ids = tokenizer.encode(text)
                    tasks.append(
                        _async_generate(
                            base_url,
                            http,
                            ids,
                            session_params={"id": sids[s], "rid": rids[s]},
                            **gen_kwargs,
                        )
                    )
                results = await asyncio.gather(*tasks)
                for s in range(CONCURRENT_LOGPROB_SESSIONS):
                    rids[s] = results[s]["meta_info"]["id"]

            for sid in sids:
                async with http.post(
                    base_url + "/close_session", json={"session_id": sid}
                ) as resp:
                    assert resp.status == 200


async def _stress_run_all(base_url: str, tokenizer: Any) -> None:
    """Streaming + non-streaming mixed batches under retract pressure.
    Long prompts (~200+ tokens) trigger chunked prefill so retract can
    interrupt mid-extend.
    """
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        sids: list[str] = []
        for _ in range(STRESS_NUM_SESSIONS):
            async with http.post(
                base_url + "/open_session",
                json={"capacity_of_str_len": 50000, "streaming": True},
            ) as resp:
                assert resp.status == 200
                sids.append(await resp.json())

        rids: list[Optional[str]] = [None] * STRESS_NUM_SESSIONS
        for turn in range(STRESS_NUM_TURNS):
            tasks = []
            # Streaming requests — long prompts to trigger chunked prefill.
            for s in range(STRESS_NUM_SESSIONS):
                offset = (s * STRESS_NUM_TURNS + turn) * 200
                text = (
                    f"Session {s} turn {turn}: " f"{LEAK_FILLER[offset : offset + 800]}"
                )
                ids = tokenizer.encode(text)
                tasks.append(
                    _async_generate(
                        base_url,
                        http,
                        ids,
                        max_new_tokens=STRESS_GEN_LEN,
                        session_params={"id": sids[s], "rid": rids[s]},
                    )
                )

            # Non-streaming requests interleaved.
            for ns in range(STRESS_NUM_NON_STREAMING):
                text = (
                    f"Non-streaming {ns} turn {turn}: "
                    f"{LEAK_FILLER[ns * 100 : ns * 100 + 400]}"
                )
                ids = tokenizer.encode(text)
                tasks.append(
                    _async_generate(
                        base_url,
                        http,
                        ids,
                        max_new_tokens=STRESS_GEN_LEN,
                    )
                )

            results = await asyncio.gather(*tasks)
            for s in range(STRESS_NUM_SESSIONS):
                rids[s] = results[s]["meta_info"]["id"]

        for sid in sids:
            async with http.post(
                base_url + "/close_session", json={"session_id": sid}
            ) as resp:
                assert resp.status == 200


class StreamingSessionServerBase(CustomTestCase):
    """Minimal streaming-session server fixture.

    Subclasses override class attrs to customize launch:
    - `model`: defaults to the small model.
    - `extra_args`: appended after `--enable-streaming-session` (set
      `--chunked-prefill-size`, `--page-size`, spec args, etc. here).
    - `env_overrides`: list of `(env_attr_name, value)` tuples; each is
      pushed onto the `setUpClass` context stack so the env override is
      live during `popen_launch_server` and torn down on
      `tearDownClass`-time. `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=2`
      is always applied on top of these.
    """

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    extra_args: list = []
    env_overrides: list = []

    @classmethod
    def setUpClass(cls):
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2)
            )
            for name, val in cls.env_overrides:
                stack.enter_context(getattr(envs, name).override(val))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--enable-streaming-session"] + list(cls.extra_args),
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
