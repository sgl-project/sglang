"""Unit tests for `--warmups all`, context bounds, and multi-worker coordination."""

from __future__ import annotations

import asyncio
import glob
import multiprocessing
import os
import shutil
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sglang.srt.entrypoints import warmup as warmup_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(
        self,
        *,
        context_len: int = 65536,
        max_req_input_len: int | None = None,
        reserved: int = 0,
    ):
        self.requests, self.flush_count = [], 0
        self.is_generation, self.skip_tokenizer_init = True, False
        self.context_len, self.num_reserved_tokens = context_len, reserved
        self.max_req_input_len = (
            context_len - 6 if max_req_input_len is None else max_req_input_len
        )
        self.model_config = SimpleNamespace(vocab_size=151936, context_len=context_len)
        self.server_args = SimpleNamespace(
            tokenizer_worker_num=4,
            port=18999,
            grammar_backend="xgrammar",
            context_length=context_len,
            chunked_prefill_size=4096,
            max_prefill_tokens=16384,
        )
        self.startup_time = {"scheduler_init": 123.45}
        self.dynamic_batch_tokenizer = SimpleNamespace(
            encode=AsyncMock(return_value=[1, 2])
        )

    async def generate_request(self, req, request=None):
        n_in = len(req.input_ids) + self.num_reserved_tokens
        max_new = int((req.sampling_params or {}).get("max_new_tokens", 0) or 0)
        if n_in >= self.context_len or n_in + max_new > self.context_len:
            raise ValueError(f"{n_in=} + {max_new=} exceeds {self.context_len=}")
        if (
            self.max_req_input_len is not None
            and len(req.input_ids) >= self.max_req_input_len
        ):
            raise ValueError(f"{len(req.input_ids)=} >= {self.max_req_input_len=}")
        self.requests.append(req)
        yield {"text": "ok"}

    async def flush_cache(self, timeout_s=None):
        self.flush_count += 1
        return SimpleNamespace(success=True, message="")


def test_warmup_all_covers_all_paths_and_flushes_once():
    tm = _FakeTokenizerManager(context_len=65536, max_req_input_len=40960)
    chat = SimpleNamespace(
        _convert_to_internal_request=MagicMock(return_value=(None, None))
    )
    asyncio.run(
        warmup_mod.execute_warmups(
            "null", ["all"], tm, app_state=SimpleNamespace(openai_serving_chat=chat)
        )
    )

    lengths = [len(r.input_ids) for r in tm.requests if r.input_ids]
    assert max(lengths) == 32768 and {3, 21, 85}.issubset(lengths)
    assert any(r.return_logprob and r.logprob_start_len == 0 for r in tm.requests)
    assert any(
        (r.sampling_params or {}).get("repetition_penalty") == 1.05 for r in tm.requests
    )
    assert any((r.sampling_params or {}).get("min_p") == 0.05 for r in tm.requests)
    assert any("json_schema" in (r.sampling_params or {}) for r in tm.requests)
    assert any("regex" in (r.sampling_params or {}) for r in tm.requests)
    tm.dynamic_batch_tokenizer.encode.assert_awaited_once()
    chat._convert_to_internal_request.assert_called_once()
    assert tm.flush_count == 1


@pytest.mark.parametrize(
    ("ctx", "unset_max_req", "res"),
    [
        (4096, True, 0),
        (512, False, 0),
        (64, False, 0),
        (24, False, 0),
        (8208, False, 48),
    ],
)
@pytest.mark.parametrize("name", ["all", "prefill_shapes"])
def test_warmups_never_exceed_small_context_length(ctx, unset_max_req, res, name):
    tm = _FakeTokenizerManager(context_len=ctx, reserved=res)
    if unset_max_req:
        tm.max_req_input_len = None
    asyncio.run(warmup_mod.execute_warmups("null", [name], tm))
    assert all(
        len(r.input_ids)
        + res
        + int((r.sampling_params or {}).get("max_new_tokens", 0) or 0)
        <= ctx
        for r in tm.requests
    )


def test_warmup_all_deduplicates_gpu_work_across_multi_workers():
    shared = {"run_id": uuid.uuid4().hex}
    tm_0 = _FakeTokenizerManager(context_len=512, max_req_input_len=500)
    tm_1 = _FakeTokenizerManager(context_len=512, max_req_input_len=500)
    tm_0.startup_time = tm_1.startup_time = shared
    assert warmup_mod.sync_multi_worker_ready_barrier(tm_0) is True

    with patch.object(
        multiprocessing,
        "current_process",
        return_value=SimpleNamespace(_parent_pid=os.getpid()),
    ):
        prefix, _, _ = warmup_mod._multi_worker_info(tm_0)
        try:
            asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm_0))
            asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm_1))
        finally:
            for p in glob.glob(f"{prefix}_*"):
                shutil.rmtree(p, ignore_errors=True)
                if os.path.exists(p):
                    os.remove(p)

    assert len(tm_0.requests) > 0 and tm_0.flush_count == 1
    assert len(tm_1.requests) == 0 and tm_1.flush_count == 0
    tm_0.dynamic_batch_tokenizer.encode.assert_awaited_once()
    tm_1.dynamic_batch_tokenizer.encode.assert_awaited_once()
