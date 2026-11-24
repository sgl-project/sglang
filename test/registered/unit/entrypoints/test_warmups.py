"""Unit tests for `--warmups all` and context-length bounded prefill warmups."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
        vocab_size: int = 151936,
        is_generation: bool = True,
    ):
        self.requests = []
        self.flush_count = 0
        self.is_generation = is_generation
        self.skip_tokenizer_init = False
        self.context_len = context_len
        self.num_reserved_tokens = reserved
        self.max_req_input_len = (
            context_len - 6 if max_req_input_len is None else max_req_input_len
        )
        self.model_config = SimpleNamespace(
            vocab_size=vocab_size, context_len=context_len
        )
        self.server_args = SimpleNamespace(
            grammar_backend="xgrammar",
            context_length=context_len,
            chunked_prefill_size=4096,
            max_prefill_tokens=16384,
        )
        self.tokenizer = SimpleNamespace(
            encode=MagicMock(return_value=[1, 2]),
            chat_template="{{ messages }}",
            apply_chat_template=MagicMock(return_value="Warmup."),
        )
        self.async_dynamic_batch_tokenizer = SimpleNamespace(
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
        if any(not (0 <= tok < self.model_config.vocab_size) for tok in req.input_ids):
            raise ValueError("token id out of vocab range")
        if (
            req.return_logprob
            and req.top_logprobs_num is not None
            and req.top_logprobs_num > self.model_config.vocab_size
        ):
            raise ValueError("top_logprobs_num exceeds vocab_size")
        self.requests.append(req)
        yield {"text": "ok"}

    async def flush_cache(self, timeout_s=None):
        self.flush_count += 1
        return SimpleNamespace(success=True, message="")


def test_warmup_all_covers_all_paths_and_flushes_once():
    tm = _FakeTokenizerManager(context_len=65536, max_req_input_len=40960)
    asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm))

    lengths = [len(r.input_ids) for r in tm.requests if r.input_ids]
    assert max(lengths) == 32768
    assert {16, 32, 80, 128}.issubset(lengths)
    assert any(r.return_logprob and r.logprob_start_len == 0 for r in tm.requests)
    assert any(
        (r.sampling_params or {}).get("repetition_penalty") == 1.05 for r in tm.requests
    )
    assert any((r.sampling_params or {}).get("min_p") == 0.05 for r in tm.requests)
    assert any("json_schema" in (r.sampling_params or {}) for r in tm.requests)
    assert any("regex" in (r.sampling_params or {}) for r in tm.requests)
    tm.async_dynamic_batch_tokenizer.encode.assert_awaited_once()
    tm.tokenizer.apply_chat_template.assert_called_once()
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


def test_warmup_all_handles_tiny_vocab_and_non_generation_models():
    tiny_vocab_tm = _FakeTokenizerManager(context_len=128, vocab_size=3)
    asyncio.run(warmup_mod.execute_warmups("null", ["all"], tiny_vocab_tm))
    assert len(tiny_vocab_tm.requests) > 0

    embed_tm = _FakeTokenizerManager(context_len=512, is_generation=False)
    asyncio.run(warmup_mod.execute_warmups("null", ["all"], embed_tm))
    assert len(embed_tm.requests) == 0
    assert embed_tm.flush_count == 0
    embed_tm.async_dynamic_batch_tokenizer.encode.assert_awaited_once()


def test_warmup_all_respects_max_prefill_size_env(monkeypatch):
    monkeypatch.setenv("SGLANG_WARMUP_MAX_PREFILL_SIZE", "256")
    tm = _FakeTokenizerManager(context_len=65536, max_req_input_len=40960)
    asyncio.run(warmup_mod.execute_warmups("null", ["all"], tm))
    assert max(len(r.input_ids) for r in tm.requests) == 256
