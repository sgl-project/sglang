"""Unit tests for the SGLang startup warmup suite and multi-worker coordination."""

from __future__ import annotations

import asyncio
import glob
import multiprocessing
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.entrypoints import warmup as warmup_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self, *, max_req_input_len: int = 4096, vocab_size: int = 151936):
        self.requests = []
        self.flush_count = 0
        self.is_generation = True
        self.skip_tokenizer_init = False
        self.max_req_input_len = max_req_input_len
        self.model_config = SimpleNamespace(vocab_size=vocab_size)
        self.server_args = SimpleNamespace(
            tokenizer_worker_num=4,
            port=18999,
            grammar_backend="xgrammar",
        )
        self.startup_time = {"scheduler_init": 123.45}
        self.dynamic_batch_tokenizer = SimpleNamespace(
            encode=AsyncMock(return_value=[1, 2])
        )

    async def generate_request(self, req, request=None):
        self.requests.append(req)
        yield {"text": "ok"}

    async def flush_cache(self, timeout_s=None):
        self.flush_count += 1
        return SimpleNamespace(success=True, message="")


def test_all_suite_runs_every_stage_and_flushes_cache_once():
    tm = _FakeTokenizerManager(max_req_input_len=40960)
    serving_chat = SimpleNamespace(
        _convert_to_internal_request=MagicMock(return_value=(None, None))
    )
    app_state = SimpleNamespace(openai_serving_chat=serving_chat)

    asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm, app_state=app_state))

    assert "guard" not in warmup_mod._warmup_registry
    max_input_len = max(len(r.input_ids) for r in tm.requests if r.input_ids)
    assert max_input_len == 32768
    # Logprobs (output & prompt), penalties, min_p, json_schema, regex all covered.
    assert any(r.return_logprob and r.logprob_start_len == 0 for r in tm.requests)
    assert any(
        (r.sampling_params or {}).get("repetition_penalty") == 1.05 for r in tm.requests
    )
    assert any((r.sampling_params or {}).get("min_p") == 0.05 for r in tm.requests)
    assert any("json_schema" in (r.sampling_params or {}) for r in tm.requests)
    assert any("regex" in (r.sampling_params or {}) for r in tm.requests)
    # Local worker warmup invoked both dynamic batch tokenizer and chat template.
    tm.dynamic_batch_tokenizer.encode.assert_awaited_once()
    serving_chat._convert_to_internal_request.assert_called_once()
    # Deferred flush_cache executed once after all synthetic GPU warmups finished.
    assert tm.flush_count == 1


def test_once_per_server_deduplicates_across_multi_tokenizer_workers():
    shared_startup = {"run_id": uuid.uuid4().hex}
    tm_worker_0 = _FakeTokenizerManager(max_req_input_len=1024)
    tm_worker_1 = _FakeTokenizerManager(max_req_input_len=1024)
    tm_worker_0.startup_time = shared_startup
    tm_worker_1.startup_time = shared_startup
    fake_proc = SimpleNamespace(_parent_pid=987654321)

    with patch.object(multiprocessing, "current_process", return_value=fake_proc):
        coord = warmup_mod._multi_worker_Coordination_key(tm_worker_0)
        assert coord is not None
        prefix, _ = coord
        try:
            asyncio.run(
                warmup_mod.execute_warmups(
                    "null", ["sampling", "tokenizer_worker"], tm_worker_0
                )
            )
            asyncio.run(
                warmup_mod.execute_warmups(
                    "null", ["sampling", "tokenizer_worker"], tm_worker_1
                )
            )
        finally:
            for p in glob.glob(f"{prefix}_*"):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # Worker 0 ran the 6 GPU sampling requests + 1 cache flush; Worker 1 skipped GPU
    # sampling completely, while STILL warming its own local dynamic_batch_tokenizer!
    assert len(tm_worker_0.requests) == 6
    assert tm_worker_0.flush_count == 1
    assert len(tm_worker_1.requests) == 0
    assert tm_worker_1.flush_count == 0
    tm_worker_0.dynamic_batch_tokenizer.encode.assert_awaited_once()
    tm_worker_1.dynamic_batch_tokenizer.encode.assert_awaited_once()


def test_prefill_shapes_respects_max_req_input_len():
    tm = _FakeTokenizerManager(max_req_input_len=500)
    asyncio.run(warmup_mod.execute_warmups("null", ["prefill_shapes"], tm))

    lengths = [len(r.input_ids) for r in tm.requests]
    assert max(lengths) <= 500 - 16
    # Short & odd shapes and prefix-hit extend shapes are included.
    assert 3 in lengths
    assert 21 in lengths
    assert 64 + 21 in lengths
    assert tm.flush_count == 1
