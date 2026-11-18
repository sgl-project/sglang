"""Unit tests for the SGLang `--warmups all` suite and multi-worker coordination."""

from __future__ import annotations

import asyncio
import glob
import multiprocessing
import os
import shutil
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


def test_warmup_all_covers_all_paths_and_flushes_once():
    tm = _FakeTokenizerManager(max_req_input_len=40960)
    serving_chat = SimpleNamespace(
        _convert_to_internal_request=MagicMock(return_value=(None, None))
    )
    app_state = SimpleNamespace(openai_serving_chat=serving_chat)

    asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm, app_state=app_state))

    assert "guard" not in warmup_mod._warmup_registry
    lengths = [len(r.input_ids) for r in tm.requests if r.input_ids]
    assert max(lengths) == 32768
    assert 3 in lengths and 21 in lengths and (64 + 21) in lengths
    assert any(r.return_logprob and r.logprob_start_len == 0 for r in tm.requests)
    assert any(
        (r.sampling_params or {}).get("repetition_penalty") == 1.05 for r in tm.requests
    )
    assert any((r.sampling_params or {}).get("min_p") == 0.05 for r in tm.requests)
    assert any("json_schema" in (r.sampling_params or {}) for r in tm.requests)
    assert any("regex" in (r.sampling_params or {}) for r in tm.requests)
    tm.dynamic_batch_tokenizer.encode.assert_awaited_once()
    serving_chat._convert_to_internal_request.assert_called_once()
    assert tm.flush_count == 1


def test_warmup_all_deduplicates_gpu_work_across_multi_workers():
    shared_startup = {"run_id": uuid.uuid4().hex}
    tm_0 = _FakeTokenizerManager(max_req_input_len=500)
    tm_1 = _FakeTokenizerManager(max_req_input_len=500)
    tm_0.startup_time = shared_startup
    tm_1.startup_time = shared_startup
    assert warmup_mod.sync_multi_worker_ready_barrier(tm_0) is True
    fake_proc = SimpleNamespace(_parent_pid=os.getpid())

    with patch.object(multiprocessing, "current_process", return_value=fake_proc):
        info = warmup_mod._multi_worker_info(tm_0)
        assert info is not None
        prefix, _, _ = info
        try:
            asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm_0))
            asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm_1))
        finally:
            for p in glob.glob(f"{prefix}_*"):
                shutil.rmtree(p, ignore_errors=True)
                if os.path.exists(p):
                    os.remove(p)

    # Worker 0 ran GPU warmups (bounded by max_req_input_len=500) + 1 cache flush;
    # Worker 1 skipped GPU requests & flush, while still warming its local tokenizer!
    assert len(tm_0.requests) > 0
    assert max(len(r.input_ids) for r in tm_0.requests) <= 500 - 16
    assert tm_0.flush_count == 1
    assert len(tm_1.requests) == 0
    assert tm_1.flush_count == 0
    tm_0.dynamic_batch_tokenizer.encode.assert_awaited_once()
    tm_1.dynamic_batch_tokenizer.encode.assert_awaited_once()
