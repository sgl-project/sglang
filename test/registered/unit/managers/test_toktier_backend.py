"""Verification that --tokenizer-backend=toktier wraps the tokenizer the
TokenizerManager loads, and that every encode path of that wrapper returns
the ids a from-scratch encode by the Hugging Face tokenizer returns.

Requires toktier >= 0.3.0 (which ships the SGLang adapter); skipped otherwise.
No server is started.
"""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from transformers import AutoTokenizer

from sglang.srt.managers.tokenizer_manager import (
    _toktier_session_scope,
    _wrap_tokenizer_with_toktier,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

try:
    from toktier.integrations import sglang as toktier_sglang

    HAS_TOKTIER_ADAPTER = True
except ImportError:
    HAS_TOKTIER_ADAPTER = False

# A byte-level BPE tokenizer toktier serves itself, and a WordPiece tokenizer
# outside its families, which the wrapper must pass through.
BPE_MODEL = "Qwen/Qwen3-0.6B"
WORDPIECE_MODEL = "bert-base-uncased"

TRANSCRIPT = (
    "<|im_start|>user\nSummarize the following.\n"
    + "The quick brown fox jumps over the lazy dog. " * 300
    + "<|im_end|>\n<|im_start|>assistant\n"
)
NEXT_TURN = (
    "Sure. The fox is quick and the dog is lazy.<|im_end|>\n"
    "<|im_start|>user\nShorter.<|im_end|>\n<|im_start|>assistant\n"
)


def _set_backend(backend):
    set_global_server_args_for_tokenizer(
        ServerArgs(model_path="dummy", tokenizer_backend=backend)
    )


@unittest.skipUnless(HAS_TOKTIER_ADAPTER, "toktier package (>= 0.3.0) not installed")
class TestToktierBackend(CustomTestCase):
    def setUp(self):
        _set_backend("toktier")
        self.hf = AutoTokenizer.from_pretrained(BPE_MODEL)
        self.wrapped = _wrap_tokenizer_with_toktier(self.hf)

    def test_wrapper_is_applied_once(self):
        self.assertIsInstance(self.wrapped, toktier_sglang.SGLangToktierTokenizer)
        self.assertIs(_wrap_tokenizer_with_toktier(self.wrapped), self.wrapped)

    def test_full_encode_matches_reference(self):
        for text in ("Hello, world!", TRANSCRIPT, TRANSCRIPT + NEXT_TURN):
            self.assertEqual(self.wrapped.encode(text), self.hf.encode(text))
            self.assertEqual(
                self.wrapped(text)["input_ids"], self.hf(text)["input_ids"]
            )

    def test_session_append_matches_reference(self):
        # The bound session_id is read in "contextvar" mode; the default
        # "auto" mode looks sessions up by content instead.
        with patch.dict(os.environ, {"TOKTIER_SGLANG_SESSION_KEY": "contextvar"}):
            wrapped = _wrap_tokenizer_with_toktier(
                AutoTokenizer.from_pretrained(BPE_MODEL)
            )
        request = SimpleNamespace(session_id="session-1", cache_salt=None, rid="r1")
        with _toktier_session_scope(request):
            first = wrapped.encode(TRANSCRIPT)
            second = wrapped.encode(TRANSCRIPT + NEXT_TURN)
        self.assertEqual(first, self.hf.encode(TRANSCRIPT))
        self.assertEqual(second, self.hf.encode(TRANSCRIPT + NEXT_TURN))
        stats = wrapped.stats()
        self.assertEqual(stats["how"], {"session": 2})
        # The second turn extends the first, so it is served as an append.
        self.assertEqual(stats["store"]["session_appends"], 1)

    def test_default_mode_is_content_lookup(self):
        # Without TOKTIER_SGLANG_SESSION_KEY the bound session_id is not read;
        # a turn that extends a stored text is still served as an append.
        with patch.dict(os.environ):
            os.environ.pop("TOKTIER_SGLANG_SESSION_KEY", None)
            wrapped = _wrap_tokenizer_with_toktier(
                AutoTokenizer.from_pretrained(BPE_MODEL)
            )
        request = SimpleNamespace(session_id="session-1", cache_salt=None, rid="r1")
        with _toktier_session_scope(request):
            first = wrapped.encode(TRANSCRIPT)
            second = wrapped.encode(TRANSCRIPT + NEXT_TURN)
        self.assertEqual(first, self.hf.encode(TRANSCRIPT))
        self.assertEqual(second, self.hf.encode(TRANSCRIPT + NEXT_TURN))
        stats = wrapped.stats()
        self.assertEqual(stats["how"], {"auto": 2})
        self.assertEqual(stats["store"]["auto_appends"], 1)

    def test_session_scope_binds_session_id_then_cache_salt(self):
        key = toktier_sglang.SESSION_KEY
        scope = _toktier_session_scope
        with scope(SimpleNamespace(session_id="s", cache_salt="salt")):
            self.assertEqual(key.get(None), "s")
        with scope(SimpleNamespace(session_id=None, cache_salt="salt")):
            self.assertEqual(key.get(None), "salt")
        # Server-side continuation sends only the new turn, so no key.
        with scope(SimpleNamespace(session_id="s", session_params={"id": "x"})):
            self.assertIsNone(key.get(None))
        self.assertIsNone(key.get(None))

    def test_chunked_full_encode_matches_reference(self):
        # In "contextvar" mode a request without a key is stateless, so one
        # of at least CHUNKED_MIN_CHARS characters takes the chunked route.
        chunked = {
            "TOKTIER_SGLANG_SESSION_KEY": "contextvar",
            "TOKTIER_SGLANG_CHUNKED_MIN_CHARS": "4096",
            "TOKTIER_SGLANG_CHUNKED_K": "4",
        }
        with patch.dict(os.environ, chunked):
            wrapped = _wrap_tokenizer_with_toktier(
                AutoTokenizer.from_pretrained(BPE_MODEL)
            )
        text = (TRANSCRIPT + NEXT_TURN) * 4
        self.assertEqual(wrapped.encode(text), self.hf.encode(text))
        stats = wrapped.stats()
        self.assertEqual(stats["how"], {"chunked": 1})
        self.assertEqual(stats["chunked"]["applied"], 1)

    def test_unmodeled_call_shapes_are_delegated(self):
        first, second = "first sentence", "second sentence"
        self.assertEqual(
            self.wrapped(first, second)["input_ids"],
            self.hf(first, second)["input_ids"],
        )
        self.assertEqual(
            self.wrapped(TRANSCRIPT, truncation=True, max_length=16)["input_ids"],
            self.hf(TRANSCRIPT, truncation=True, max_length=16)["input_ids"],
        )
        self.assertEqual(
            self.wrapped(first, return_token_type_ids=True)["token_type_ids"],
            self.hf(first, return_token_type_ids=True)["token_type_ids"],
        )
        self.assertEqual(self.wrapped.decode(self.hf.encode(first)), first)

    def test_tokenizer_outside_toktier_families_passes_through(self):
        hf = AutoTokenizer.from_pretrained(WORDPIECE_MODEL)
        wrapped = _wrap_tokenizer_with_toktier(hf)
        for text in ("Hello, world!", TRANSCRIPT):
            self.assertEqual(wrapped.encode(text), hf.encode(text))
            self.assertEqual(wrapped(text)["input_ids"], hf(text)["input_ids"])

    def test_other_backends_get_a_noop_scope(self):
        _set_backend("huggingface")
        self.assertIsInstance(
            _toktier_session_scope(SimpleNamespace(session_id="s")), nullcontext
        )


if __name__ == "__main__":
    unittest.main()
