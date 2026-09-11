import random
import unittest
from contextlib import contextmanager

from transformers import AutoTokenizer

from sglang.srt.utils.patch_tokenizer import (
    _EncodePieceFastPathPatcher,
    _SpecialTokensCachePatcher,
    decode_without_hf_kwargs,
    unpatch_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=16, suite="stage-b-test-cpu-intel")


class TestPatchTokenizerEndToEndTest(unittest.TestCase):
    def test_patched_produces_same_results_as_raw(self):
        tokenizer = _load_tokenizer()
        test_texts = self._generate_test_texts(tokenizer)
        raw_results = self._run_tokenizer_ops(tokenizer, test_texts)

        _SpecialTokensCachePatcher.patch(tokenizer)
        patched_results = self._run_tokenizer_ops(tokenizer, test_texts)
        unpatch_tokenizer(tokenizer)

        self.assertEqual(raw_results, patched_results)

    @classmethod
    def _generate_test_texts(cls, tokenizer):
        special_tokens = tokenizer.all_special_tokens
        return [
            "Hello, world!",
            "This is a longer sentence with multiple words.",
            "Numbers 12345 and symbols !@#$%",
            "    leading and trailing spaces    ",
            "\n\nMultiple\n\nNewlines\n\n",
            *[f"Text with {tok} inside" for tok in special_tokens],
            " ".join(special_tokens),
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=100)
                for _ in range(5)
            ],
            *[
                cls._random_text_from_tokens(tokenizer, num_tokens=1000)
                for _ in range(3)
            ],
        ]

    @classmethod
    def _random_text_from_tokens(cls, tokenizer, num_tokens):
        token_ids = [
            random.randint(0, tokenizer.vocab_size - 1) for _ in range(num_tokens)
        ]
        return tokenizer.decode(token_ids)

    @classmethod
    def _run_tokenizer_ops(cls, tokenizer, texts):
        encode_results = [tokenizer.encode(t) for t in texts]
        batch_encode_results = tokenizer(texts)["input_ids"]
        return {
            "encode": encode_results,
            "batch_encode": batch_encode_results,
            "decode": [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in encode_results
            ],
            "batch_decode": tokenizer.batch_decode(
                encode_results, skip_special_tokens=True
            ),
            "special_tokens": tokenizer.all_special_tokens,
            "special_ids": tokenizer.all_special_ids,
        }


class TestPatchTokenizerUnitTest(unittest.TestCase):
    def test_patch_unpatch_restores_original(self):
        tokenizer = _load_tokenizer()
        cls = type(tokenizer)

        original_ids = _get_class_attr_ids(cls)

        _SpecialTokensCachePatcher.patch(tokenizer)
        self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))

        patched_ids = _get_class_attr_ids(cls)
        changed_attrs = [
            name
            for name in original_ids
            if name in patched_ids and patched_ids[name] != original_ids[name]
        ]
        self.assertGreater(len(changed_attrs), 0, "Patch should change some attributes")

        unpatch_tokenizer(tokenizer)
        self.assertFalse(getattr(cls, "_sglang_special_tokens_patched", False))

        restored_ids = _get_class_attr_ids(cls)
        for name in original_ids:
            if name.startswith("_sglang") or name.startswith("_original"):
                continue
            self.assertEqual(
                restored_ids.get(name),
                original_ids[name],
                f"Attribute {name} should be restored to original",
            )

    def test_patch_caches_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            tokens1 = tokenizer.all_special_tokens
            ids1 = tokenizer.all_special_ids
            tokens2 = tokenizer.all_special_tokens
            ids2 = tokenizer.all_special_ids

            self.assertIs(tokens1, tokens2)
            self.assertIs(ids1, ids2)

    def test_patch_blocks_add_special_tokens(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.assertIn(
                "Cannot modify special tokens after patch", str(ctx.exception)
            )

    def test_patch_blocks_add_tokens_with_special_flag(self):
        with _patched_tokenizer() as tokenizer:
            with self.assertRaises(AssertionError) as ctx:
                tokenizer.add_tokens(["<new>"], special_tokens=True)
            self.assertIn("Cannot add special tokens after patch", str(ctx.exception))

            tokenizer.add_tokens(["<regular>"], special_tokens=False)

    def test_unpatch_clears_cache(self):
        with _patched_tokenizer() as tokenizer:
            _ = tokenizer.all_special_tokens
            _ = tokenizer.all_special_ids
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_tokens"))
            self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_ids"))

        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_ids"))

    def test_double_patch_is_idempotent(self):
        tokenizer = _load_tokenizer()
        _SpecialTokensCachePatcher.patch(tokenizer)
        _SpecialTokensCachePatcher.patch(tokenizer)

        self.assertTrue(
            getattr(type(tokenizer), "_sglang_special_tokens_patched", False)
        )

        unpatch_tokenizer(tokenizer)

    def test_decode_without_hf_kwargs_uses_native_decode(self):
        tokenizer = _FakeDecodeTokenizer()

        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], True),
            "ab",
        )
        self.assertEqual(
            decode_without_hf_kwargs(tokenizer, [1, 99, 2], False),
            "a<special>b",
        )
        self.assertEqual(tokenizer.decode_calls, [[1, 2], [1, 99, 2]])


class TestEncodePieceFastPathPatcher(CustomTestCase):
    """The fast path must be a pure shortcut: every segment shape it handles
    (or declines) has to encode to the same ids as the original method."""

    def test_encode_piece_matches_original_on_every_segment_shape(self):
        tokenizer = _load_tokenizer()
        specials = list(tokenizer.special_tokens)
        rng = random.Random(0)
        random_texts = [
            _random_text_from_tokens(tokenizer, num_tokens=n, rng=rng)
            for n in (1, 7, 100, 1000)
        ]
        segments = [
            # exactly one special token -> table lookup
            *[(tok, True) for tok in specials],
            # special token with a suffix / two specials -> original path
            *[(tok + "x", True) for tok in specials[:8]],
            (specials[0] + specials[1], True),
            # plain text -> encode_ordinary
            *[(text, False) for text in random_texts],
            ("", False),
            ("", True),
            (" " * 40, False),
            # special literal inside plain text -> original path
            *[(f"user wrote {tok} literally", False) for tok in specials[:8]],
            # longer than MAX_NO_WHITESPACES_CHARS -> original splitter path
            ("b" * 30_000 + " tail", False),
            ("b" * 30_000 + " tail", True),
        ]
        original = type(tokenizer)._encode_text_piece
        _EncodePieceFastPathPatcher.patch(tokenizer)
        try:
            for text, allow_special in segments:
                self.assertEqual(
                    original(tokenizer, text, allow_special),
                    tokenizer._encode_text_piece(text, allow_special),
                    (text[:40], allow_special),
                )
        finally:
            _EncodePieceFastPathPatcher.unpatch(tokenizer)

    def test_chat_template_ids_unchanged_for_tool_call_conversation(self):
        # Kimi-K3's encoding_k3 renders one segment per control token and per
        # tool-call attribute; this is the shape the fast path exists for.
        tokenizer = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K3", trust_remote_code=True
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "deploy it"},
        ]
        for i in range(300):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": '{"command": "ls -la /tmp/%d"}' % i,
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": f"total 0\n<|im_end|> literal in tool output {i}",
                }
            )
        kwargs = dict(tools=tools, tokenize=True, add_generation_prompt=True)
        expected = tokenizer.apply_chat_template(messages, **kwargs)

        _EncodePieceFastPathPatcher.patch(tokenizer)
        try:
            self.assertEqual(
                expected, tokenizer.apply_chat_template(messages, **kwargs)
            )
        finally:
            _EncodePieceFastPathPatcher.unpatch(tokenizer)

    def test_unpatch_restores_encode_piece(self):
        tokenizer = _load_tokenizer()
        cls = type(tokenizer)
        original = cls._encode_text_piece

        _EncodePieceFastPathPatcher.patch(tokenizer)
        self.assertIsNot(cls._encode_text_piece, original)
        _EncodePieceFastPathPatcher.patch(tokenizer)  # idempotent
        _EncodePieceFastPathPatcher.unpatch(tokenizer)

        self.assertIs(cls._encode_text_piece, original)
        self.assertFalse(hasattr(cls, "_original_encode_text_piece"))
        self.assertFalse(hasattr(tokenizer, "_sglang_special_literal_regex"))


def _random_text_from_tokens(tokenizer, num_tokens, rng):
    token_ids = [rng.randint(0, tokenizer.vocab_size - 1) for _ in range(num_tokens)]
    return tokenizer.decode(token_ids)


def _get_class_attr_ids(cls):
    return {
        n: id(v.fget if isinstance(v, property) else v) for n, v in vars(cls).items()
    }


def _load_tokenizer():
    # The slowness is mainly observed in Kimi
    return AutoTokenizer.from_pretrained(
        "nvidia/Kimi-K2-Thinking-NVFP4", trust_remote_code=True
    )


@contextmanager
def _patched_tokenizer():
    tokenizer = _load_tokenizer()
    _SpecialTokensCachePatcher.patch(tokenizer)
    try:
        yield tokenizer
    finally:
        unpatch_tokenizer(tokenizer)


class _FakeDecodeTokenizer:
    all_special_ids_set = {99}

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        token_ids = list(token_ids)
        self.decode_calls.append(token_ids)
        token_text = {1: "a", 2: "b", 99: "<special>"}
        return "".join(token_text[token_id] for token_id in token_ids)


if __name__ == "__main__":
    unittest.main()
