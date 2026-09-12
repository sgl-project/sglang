"""Unit tests for AsyncDynamicbatchTokenizer dispatch & future resolution — no server.

`_process_dynamic_batch` picks one of three encode strategies and resolves each
request's future. These tests pin every branch and the resolution semantics that a
refactor must not regress:

Dispatch (`_build_encode_fn`):
  - slow (`is_fast=False`) tokenizer, no kwargs -> ``encode()`` per prompt (skips the
    slow ``__call__`` -> ``tokenize()`` chain), single & concurrently-batched, ids
    byte-for-byte equal to the stock ``__call__``;
  - missing ``is_fast`` attribute is treated as slow;
  - fast (`is_fast=True`) tokenizer stays on the batched ``__call__`` path, and the
    per-prompt split preserves every key (``attention_mask``, ``token_type_ids``);
  - non-empty kwargs fall back to ``__call__`` (semantics preserved);
  - heterogeneous kwargs run per item and emit the batching-disabled warning;
  - a single fast request with kwargs runs per item without warning.

Resolution (`_set_results` / `_set_exception`):
  - a future cancelled by a disconnected client is skipped while its siblings still
    resolve;
  - a tokenizer error is fanned out to every pending future;
  - a pre-cancelled future is left cancelled when the batch errors.

Minimal char-level tokenizers stand in for tiktoken/HF (Fakes; the fast one also
spies on which path ran), so no model download / GPU. Two driving seams are used on
purpose: ``_encode_*`` exercise the real public ``encode()`` (coalescing + executor),
while ``_process_batch`` drives ``_process_dynamic_batch`` directly with crafted
futures for deterministic control over batching and cancellation. These are
behavior-based example tests.
"""

import asyncio
import logging
import unittest

from transformers import PreTrainedTokenizer

from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.managers.async_dynamic_batch_tokenizer"


class _MiniSlowTokenizer(PreTrainedTokenizer):
    """Minimal slow tokenizer (is_fast == False).

    ``encode(text)`` returns ids straight from a "fast backend" (no tokenize()),
    mirroring Kimi's tiktoken-backed ``encode()``. The stock ``__call__`` path goes
    through ``_tokenize`` (counted); for char-level input both produce the same ids
    (no structural special tokens), so the routed ``encode()`` output is byte-for-byte
    equal to the stock ``__call__`` output.
    """

    def __init__(self, **kw):
        self.slow_tokenize_calls = 0
        super().__init__(**kw)

    @property
    def vocab_size(self):
        return 256

    def get_vocab(self):
        return {chr(c): c for c in range(256)}

    def _tokenize(self, text, **kw):
        self.slow_tokenize_calls += 1  # slow HF path marker
        return list(text)

    def _convert_token_to_id(self, tok):
        return ord(tok) if len(tok) == 1 else 0

    def _convert_id_to_token(self, idx):
        return chr(idx)

    def encode(self, text, **kwargs):
        if kwargs:  # defer to the (slow) base impl, matching real slow tokenizers
            return super().encode(text, **kwargs)
        return [ord(c) for c in text]


class _MiniFastTokenizer:
    """Minimal duck-typed fast tokenizer (is_fast == True), records which path ran.

    ``__call__`` returns ``attention_mask`` alongside ``input_ids`` (and
    ``token_type_ids`` when asked) so tests can assert the batched-split path keeps
    every key.
    """

    is_fast = True

    def __init__(self):
        self.call_calls = 0
        self.encode_calls = 0

    def __call__(self, prompts, **kw):
        self.call_calls += 1
        single = isinstance(prompts, str)
        items = [prompts] if single else list(prompts)
        ids = [[ord(c) for c in p] for p in items]
        out = {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}
        if kw.get("return_token_type_ids"):
            out["token_type_ids"] = [[0] * len(x) for x in ids]
        if single:
            out = {k: v[0] for k, v in out.items()}
        return out

    def encode(self, text, **kw):
        self.encode_calls += 1
        return [ord(c) for c in text]


class _NoIsFastTokenizer:
    """Duck-typed tokenizer with NO ``is_fast`` attribute (getattr default -> slow)."""

    def __call__(self, prompts, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("missing-is_fast tokenizer should route to encode()")

    def encode(self, text, **kw):
        return [ord(c) for c in text]


class _FailingTokenizer:
    """Fast tokenizer whose ``__call__`` always raises (tests exception fan-out)."""

    is_fast = True
    error = ValueError("tokenizer boom")

    def __call__(self, prompts, **kw):
        raise self.error


_PROMPT = "hello, 世界 def f(x): return x*2"


def _run(coro):
    return asyncio.run(coro)


def _stop(adbt):
    """The one structure-sensitive touch in this file: cancel the private background
    batcher task so the coalescing loop doesn't outlive the test. Isolated here (not
    inlined at each call site) so a rename lands in one place and the test bodies stay
    structure-insensitive."""
    task = getattr(adbt, "_batcher_task", None)
    if task is not None:
        task.cancel()


async def _encode_one(adbt, prompt, **kwargs):
    try:
        return await adbt.encode(prompt, **kwargs)
    finally:
        _stop(adbt)


async def _encode_concurrent(adbt, prompts):
    try:
        return await asyncio.gather(*(adbt.encode(p) for p in prompts))
    finally:
        _stop(adbt)


async def _process_batch(tok, prompts, kwargs_list, cancel=()):
    """Drive ``_process_dynamic_batch`` directly with crafted futures for
    deterministic control over batching and cancellation. Returns the futures."""
    adbt = AsyncDynamicbatchTokenizer(tok)
    loop = asyncio.get_running_loop()
    futures = [loop.create_future() for _ in prompts]
    for i in cancel:
        futures[i].cancel()
    await adbt._process_dynamic_batch(prompts, kwargs_list, futures)
    await asyncio.sleep(0)  # let cancellations settle
    return futures


class TestAsyncDynamicBatchTokenizerDispatch(CustomTestCase):
    # ---- slow / encode routing -------------------------------------------------
    def test_single_non_fast_uses_encode(self):
        tok = _MiniSlowTokenizer()
        self.assertFalse(tok.is_fast)
        stock_ids = tok(_PROMPT)["input_ids"]

        tok.slow_tokenize_calls = 0
        out = _run(_encode_one(AsyncDynamicbatchTokenizer(tok), _PROMPT))

        self.assertEqual(out["input_ids"], stock_ids)  # byte-for-byte equal
        self.assertEqual(tok.slow_tokenize_calls, 0)  # slow path skipped

    def test_concurrent_batch_non_fast_uses_encode(self):
        tok = _MiniSlowTokenizer()
        prompts = [_PROMPT, _PROMPT + "!", "abc", "中文字符"]
        stock = [tok(p)["input_ids"] for p in prompts]

        tok.slow_tokenize_calls = 0
        outs = _run(_encode_concurrent(AsyncDynamicbatchTokenizer(tok), prompts))

        self.assertEqual([o["input_ids"] for o in outs], stock)
        self.assertEqual(tok.slow_tokenize_calls, 0)

    def test_missing_is_fast_attr_treated_as_slow(self):
        # getattr(tokenizer, "is_fast", False) -> False, so it must route to encode()
        # (the _NoIsFastTokenizer.__call__ asserts if reached).
        outs = _run(
            _encode_concurrent(
                AsyncDynamicbatchTokenizer(_NoIsFastTokenizer()), ["ab", "cde"]
            )
        )
        self.assertEqual([o["input_ids"] for o in outs], [[97, 98], [99, 100, 101]])

    # ---- fast / __call__ routing ----------------------------------------------
    def test_fast_tokenizer_stays_on_call_path(self):
        tok = _MiniFastTokenizer()
        outs = _run(
            _encode_concurrent(AsyncDynamicbatchTokenizer(tok), ["abc", "de", "fghi"])
        )
        self.assertTrue(all("input_ids" in o for o in outs))
        self.assertEqual(tok.encode_calls, 0)  # encode() not used for fast tokenizers
        self.assertGreater(tok.call_calls, 0)  # original __call__ path used

    def test_batched_call_preserves_all_keys(self):
        # Coalesced fast-tokenizer requests run one batched __call__; the per-prompt
        # split must keep every key (not just input_ids), e.g. attention_mask.
        tok = _MiniFastTokenizer()
        prompts = ["abc", "de", "fghi"]
        outs = _run(_encode_concurrent(AsyncDynamicbatchTokenizer(tok), prompts))
        for prompt, out in zip(prompts, outs):
            self.assertEqual(out["input_ids"], [ord(c) for c in prompt])
            self.assertEqual(out["attention_mask"], [1] * len(prompt))

    def test_token_type_ids_preserved_in_batched_call(self):
        # Uniform return_token_type_ids -> one batched __call__; the split keeps
        # token_type_ids per prompt.
        tok = _MiniFastTokenizer()
        prompts = ["abc", "defg"]
        futs = _run(_process_batch(tok, prompts, [{"return_token_type_ids": True}] * 2))
        for prompt, fut in zip(prompts, futs):
            res = fut.result()
            self.assertEqual(res["input_ids"], [ord(c) for c in prompt])
            self.assertEqual(res["token_type_ids"], [0] * len(prompt))

    # ---- kwargs handling -------------------------------------------------------
    def test_kwargs_present_falls_back_to_call(self):
        tok = _MiniSlowTokenizer()
        tok.slow_tokenize_calls = 0
        out = _run(
            _encode_one(
                AsyncDynamicbatchTokenizer(tok), _PROMPT, add_special_tokens=False
            )
        )
        self.assertIn("input_ids", out)
        self.assertGreater(tok.slow_tokenize_calls, 0)  # fell back to slow __call__

    def test_heterogeneous_kwargs_run_per_item_and_warn(self):
        # Differing kwargs across a coalesced batch disable batching: each request is
        # tokenized with its own kwargs, and a warning is emitted.
        tok = _MiniFastTokenizer()
        prompts = ["abc", "de"]
        kwargs_list = [{"return_token_type_ids": True}, {}]
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            futs = _run(_process_batch(tok, prompts, kwargs_list))
        self.assertTrue(any("batching disabled" in m.lower() for m in cm.output))
        self.assertIn("token_type_ids", futs[0].result())  # honored its kwarg
        self.assertNotIn("token_type_ids", futs[1].result())  # honored empty kwargs

    def test_single_fast_with_kwargs_runs_per_item_without_warning(self):
        tok = _MiniFastTokenizer()
        logger = logging.getLogger(_LOGGER)
        with self.assertNoLogs(logger, level="WARNING"):
            futs = _run(_process_batch(tok, ["abc"], [{"return_token_type_ids": True}]))
        self.assertIn("token_type_ids", futs[0].result())


class TestAsyncDynamicBatchTokenizerResolution(CustomTestCase):
    def test_cancelled_future_skipped_others_resolved(self):
        # A client disconnect cancels its future; _set_results must skip it (no
        # "invalid state") while siblings still resolve.
        tok = _MiniFastTokenizer()
        futs = _run(
            _process_batch(tok, ["abc", "de", "fghi"], [{}, {}, {}], cancel=[1])
        )
        self.assertTrue(futs[1].cancelled())
        self.assertEqual(futs[0].result()["input_ids"], [97, 98, 99])
        self.assertEqual(futs[2].result()["input_ids"], [102, 103, 104, 105])

    def test_tokenizer_error_fans_out_to_all_pending(self):
        # A tokenizer error is delivered to every pending future (not just the first).
        tok = _FailingTokenizer()
        futs = _run(_process_batch(tok, ["a", "b", "c"], [{}, {}, {}]))
        for fut in futs:
            self.assertIsInstance(fut.exception(), ValueError)
            self.assertEqual(str(fut.exception()), "tokenizer boom")

    def test_error_leaves_precancelled_future_cancelled(self):
        # On batch error, a future already cancelled by the client stays cancelled;
        # only the still-pending ones receive the exception.
        tok = _FailingTokenizer()
        futs = _run(_process_batch(tok, ["a", "b", "c"], [{}, {}, {}], cancel=[0]))
        self.assertTrue(futs[0].cancelled())
        self.assertIsInstance(futs[1].exception(), ValueError)
        self.assertIsInstance(futs[2].exception(), ValueError)


if __name__ == "__main__":
    unittest.main()
