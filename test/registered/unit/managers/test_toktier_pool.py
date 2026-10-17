"""Verification of the toktier worker pool behind --tokenizer-backend=toktier:
with TOKTIER_SGLANG_SCHED_WORKERS > 0, TokenizerManager._tokenize_texts awaits
plain text encodes from the pool, and every id it returns equals a from-scratch
encode by the Hugging Face tokenizer. Inputs the pool does not take fall through
to the existing path.

Requires toktier >= 0.3.0 (which ships the adapter and the pool); skipped
otherwise. No server is started; the pool's worker processes are.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import AutoTokenizer

from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _toktier_session_scope,
    _wrap_tokenizer_with_toktier,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

try:
    from toktier.integrations import sglang as toktier_sglang

    HAS_TOKTIER_ADAPTER = True
except ImportError:
    HAS_TOKTIER_ADAPTER = False

MODEL = "Qwen/Qwen3-0.6B"

TRANSCRIPT = (
    "<|im_start|>user\nSummarize the following.\n"
    + "The quick brown fox jumps over the lazy dog. " * 300
    + "<|im_end|>\n<|im_start|>assistant\n"
)
NEXT_TURN = (
    "Sure. The fox is quick and the dog is lazy.<|im_end|>\n"
    "<|im_start|>user\nShorter.<|im_end|>\n<|im_start|>assistant\n"
)


def _manager(tokenizer):
    """A bare TokenizerManager with the fields _tokenize_texts reads."""
    set_global_server_args_for_tokenizer(
        ServerArgs(model_path="dummy", tokenizer_backend="toktier")
    )
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.tokenizer = tokenizer
    manager.async_dynamic_batch_tokenizer = None
    manager.model_config = SimpleNamespace(is_embedding_gemma=False)
    return manager


def _pool_wrapper(session_key=None):
    """A wrapper with its own pool of two workers, in the given session-key
    mode (None: the variable unset, so the adapter's default applies)."""
    with patch.dict(os.environ, {"TOKTIER_SGLANG_SCHED_WORKERS": "2"}):
        if session_key is None:
            os.environ.pop("TOKTIER_SGLANG_SESSION_KEY", None)
        else:
            os.environ["TOKTIER_SGLANG_SESSION_KEY"] = session_key
        return _wrap_tokenizer_with_toktier(AutoTokenizer.from_pretrained(MODEL))


def _two_turns(manager):
    """Two turns of one session through _tokenize_texts, the second
    extending the first."""
    request = SimpleNamespace(session_id="session-pool", cache_salt=None, rid="r1")

    async def run():
        with _toktier_session_scope(request):
            first, _ = await manager._tokenize_texts(TRANSCRIPT)
            second, _ = await manager._tokenize_texts(TRANSCRIPT + NEXT_TURN)
        return first, second

    return asyncio.run(run())


@unittest.skipUnless(HAS_TOKTIER_ADAPTER, "toktier package (>= 0.3.0) not installed")
class TestToktierPool(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # One pool of two worker processes for the whole class.
        cls.hf = AutoTokenizer.from_pretrained(MODEL)
        with patch.dict(os.environ, {"TOKTIER_SGLANG_SCHED_WORKERS": "2"}):
            cls.wrapped = _wrap_tokenizer_with_toktier(
                AutoTokenizer.from_pretrained(MODEL)
            )
        cls.manager = _manager(cls.wrapped)

    @classmethod
    def tearDownClass(cls):
        cls.wrapped.close()

    def test_pool_is_running(self):
        self.assertIsNotNone(self.wrapped.scheduler)

    def test_single_text_is_answered_by_the_pool(self):
        for text in ("Hello, world!", TRANSCRIPT):
            ids, token_type_ids = asyncio.run(self.manager._tokenize_texts(text))
            self.assertEqual(ids, self.hf.encode(text))
            self.assertIsNone(token_type_ids)

    def test_batch_of_texts_is_answered_by_the_pool(self):
        texts = ["Hello, world!", TRANSCRIPT, TRANSCRIPT + NEXT_TURN]
        rows, token_type_ids = asyncio.run(self.manager._tokenize_texts(texts))
        self.assertEqual(rows, [self.hf.encode(t) for t in texts])
        self.assertIsNone(token_type_ids)

    def test_session_append_through_the_pool_matches_reference(self):
        # The bound session_id is read in "contextvar" mode, and a keyed
        # request is served on its home worker's session path; the default
        # "auto" mode looks sessions up by content instead.
        wrapped = _pool_wrapper("contextvar")
        self.addCleanup(wrapped.close)
        first, second = _two_turns(_manager(wrapped))
        self.assertEqual(first, self.hf.encode(TRANSCRIPT))
        self.assertEqual(second, self.hf.encode(TRANSCRIPT + NEXT_TURN))
        stats = wrapped.stats()
        self.assertEqual(stats["how"], {"sched_session": 2})
        # Both turns went to the same worker, the session's home.
        self.assertEqual(list(stats["scheduler"]["tasks_by_worker"].values()), [2])

    def test_default_mode_is_content_lookup_on_the_pool(self):
        # Without TOKTIER_SGLANG_SESSION_KEY the bound session_id is not read;
        # both turns are served by content lookup on the worker chosen from
        # the text's start.
        wrapped = _pool_wrapper()
        self.addCleanup(wrapped.close)
        first, second = _two_turns(_manager(wrapped))
        self.assertEqual(first, self.hf.encode(TRANSCRIPT))
        self.assertEqual(second, self.hf.encode(TRANSCRIPT + NEXT_TURN))
        stats = wrapped.stats()
        self.assertEqual(stats["how"], {"sched_auto": 2})
        self.assertEqual(list(stats["scheduler"]["tasks_by_worker"].values()), [2])

    def test_inputs_the_pool_does_not_take_fall_through(self):
        pool = toktier_sglang.tokenize_texts_with_pool

        async def run():
            cross = await pool(self.manager, "Hello, world!", is_cross_encoder=True)
            empty = await pool(self.manager, "")
            pairs = await pool(self.manager, [("a", "b")])
            plain = await pool(_manager(self.hf), "Hello")
            return cross, empty, pairs, plain

        self.assertEqual(asyncio.run(run()), (None, None, None, None))


if __name__ == "__main__":
    unittest.main()
