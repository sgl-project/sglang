"""Unit tests for reasoning_redirect_registry.build_redirect_config.

Uses fake in-memory tokenizers so no real model has to be downloaded.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest

from sglang.srt.parser.reasoning_redirect_registry import (
    ReasoningRedirectConfig,
    build_redirect_config,
    is_supported_reasoning_parser,
)
from sglang.test.test_utils import CustomTestCase


class _FakeTokenizer:
    """Tiny fake tokenizer that maps a fixed vocabulary string -> token id.

    Strings that are not in the vocabulary encode to multiple "char" tokens so
    the single-token check fails naturally. We store eos_token_id as either an
    int or a list to mimic real HF tokenizers (some configs use a list).
    """

    def __init__(self, vocab, eos_token_id):
        self._vocab = dict(vocab)
        self.eos_token_id = eos_token_id

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        # Fall back to per-character encoding so non-vocab strings become
        # multi-token sequences. Specific id values don't matter — we only care
        # that len > 1.
        return [10_000 + i for i, _ in enumerate(text)]


class TestIsSupportedReasoningParser(CustomTestCase):
    def test_known_parsers(self):
        for name in ("qwen3", "QWEN3", "deepseek-r1", "kimi_k2", "glm45"):
            self.assertTrue(is_supported_reasoning_parser(name), name)

    def test_unknown_parser(self):
        self.assertFalse(is_supported_reasoning_parser("totally_random_parser"))

    def test_none(self):
        self.assertFalse(is_supported_reasoning_parser(None))

    def test_empty_string(self):
        self.assertFalse(is_supported_reasoning_parser(""))


class TestBuildRedirectConfigQwen3(CustomTestCase):
    def setUp(self):
        # Qwen3-style: <think>=151667, </think>=151668, <|im_end|>=151645
        self.tokenizer = _FakeTokenizer(
            vocab={
                "<think>": 151667,
                "</think>": 151668,
                "<|im_end|>": 151645,
            },
            eos_token_id=151645,  # Qwen3 EOS == im_end
        )

    def test_happy_path(self):
        cfg = build_redirect_config("qwen3", self.tokenizer)
        self.assertIsInstance(cfg, ReasoningRedirectConfig)
        self.assertEqual(cfg.think_end_token_id, 151668)
        self.assertEqual(cfg.think_start_token_id, 151667)
        self.assertIn(151645, cfg.redirect_eos_token_ids)
        # think_end never lands in the EOS redirect set
        self.assertNotIn(151668, cfg.redirect_eos_token_ids)
        self.assertFalse(cfg.force_reasoning)

    def test_extra_chat_end_token_ids_are_added(self):
        cfg = build_redirect_config(
            "qwen3", self.tokenizer, extra_chat_end_token_ids=[999, 1000]
        )
        self.assertIn(999, cfg.redirect_eos_token_ids)
        self.assertIn(1000, cfg.redirect_eos_token_ids)


class TestBuildRedirectConfigDeepSeekR1(CustomTestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer(
            vocab={
                "<think>": 128001,
                "</think>": 128002,
            },
            eos_token_id=128009,
        )

    def test_force_reasoning_is_set(self):
        cfg = build_redirect_config("deepseek-r1", self.tokenizer)
        self.assertIsNotNone(cfg)
        self.assertTrue(cfg.force_reasoning)
        self.assertEqual(cfg.think_end_token_id, 128002)
        self.assertIn(128009, cfg.redirect_eos_token_ids)


class TestMultiTokenThinkEndDowngrade(CustomTestCase):
    def test_multi_token_think_end_returns_none(self):
        # think_end is NOT in the vocab → encode falls back to char-tokens
        # → len > 1 → registry should refuse to enable path-1.
        tok = _FakeTokenizer(vocab={"<|im_end|>": 200}, eos_token_id=200)
        cfg = build_redirect_config("qwen3", tok)
        self.assertIsNone(cfg)


class TestUnknownParserReturnsNone(CustomTestCase):
    def test_unknown(self):
        tok = _FakeTokenizer(
            vocab={"<think>": 1, "</think>": 2}, eos_token_id=3
        )
        self.assertIsNone(build_redirect_config("not-a-parser", tok))

    def test_none_parser(self):
        tok = _FakeTokenizer(
            vocab={"<think>": 1, "</think>": 2}, eos_token_id=3
        )
        self.assertIsNone(build_redirect_config(None, tok))


class TestNoEosTokensReturnsNone(CustomTestCase):
    def test_no_eos_at_all(self):
        # Nothing maps to a usable EOS id → registry refuses (would otherwise
        # have nothing to redirect away from).
        # NOTE: tokenizer.eos_token_id is None and "<|im_end|>" is missing.
        tok = _FakeTokenizer(
            vocab={"<think>": 1, "</think>": 2}, eos_token_id=None
        )
        cfg = build_redirect_config("qwen3", tok)
        self.assertIsNone(cfg)


class TestEosIdAsListIsAccepted(CustomTestCase):
    def test_list_eos(self):
        tok = _FakeTokenizer(
            vocab={"<think>": 1, "</think>": 2, "<|im_end|>": 3},
            eos_token_id=[3, 4, 5],
        )
        cfg = build_redirect_config("qwen3", tok)
        self.assertIsNotNone(cfg)
        for v in (3, 4, 5):
            self.assertIn(v, cfg.redirect_eos_token_ids)


class TestThinkEndExcludedFromEosSet(CustomTestCase):
    def test_extra_chat_end_token_ids_does_not_re_add_think_end(self):
        # Even if the caller mistakenly passes think_end_id into the extra
        # set, we must drop it.
        tok = _FakeTokenizer(
            vocab={"<think>": 1, "</think>": 2, "<|im_end|>": 3}, eos_token_id=3
        )
        cfg = build_redirect_config(
            "qwen3", tok, extra_chat_end_token_ids=[2, 4]
        )
        self.assertIsNotNone(cfg)
        self.assertNotIn(2, cfg.redirect_eos_token_ids)
        self.assertIn(4, cfg.redirect_eos_token_ids)


if __name__ == "__main__":
    unittest.main()
