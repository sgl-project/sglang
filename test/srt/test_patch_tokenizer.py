import random
import string
import unittest
from contextlib import contextmanager

from transformers import AutoTokenizer

from sglang.srt.utils.patch_tokenizer import (
    _SpecialTokensCachePatcher,
    unpatch_tokenizer,
)


class TestPatchTokenizerEndToEndTest(unittest.TestCase):
    def test_patched_produces_same_results_as_raw(self):
        tokenizer = _load_tokenizer()
        test_texts = _generate_test_texts(tokenizer)
        raw_results = _run_tokenizer_ops(tokenizer, test_texts)

        _SpecialTokensCachePatcher.patch(tokenizer)
        patched_results = _run_tokenizer_ops(tokenizer, test_texts)
        unpatch_tokenizer(tokenizer)

        self.assertEqual(raw_results, patched_results)


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


def _generate_test_texts(tokenizer):
    special_tokens = tokenizer.all_special_tokens
    return [
        # Basic texts
        "Hello, world!",
        "This is a longer sentence with multiple words.",
        "Numbers 12345 and symbols !@#$%",
        "中文测试 mixed with English",
        "    leading and trailing spaces    ",
        "\n\nMultiple\n\nNewlines\n\n",
        # Each special token individually
        *[f"Text with {tok} inside" for tok in special_tokens],
        # All special tokens combined
        " ".join(special_tokens),
        # Random generated texts
        *[_random_text(length=100) for _ in range(5)],
        *[_random_text(length=1000) for _ in range(3)],
    ]


def _random_text(length):
    chars = string.ascii_letters + string.digits + " \n\t中文日本語한국어"
    return "".join(random.choice(chars) for _ in range(length))


def _run_tokenizer_ops(tokenizer, texts):
    encode_results = [tokenizer.encode(t) for t in texts]
    batch_encode_results = tokenizer(texts)["input_ids"]
    return {
        "encode": encode_results,
        "batch_encode": batch_encode_results,
        "decode": [tokenizer.decode(ids, skip_special_tokens=True) for ids in encode_results],
        "batch_decode": tokenizer.batch_decode(encode_results, skip_special_tokens=True),
        "special_tokens": tokenizer.all_special_tokens,
        "special_ids": tokenizer.all_special_ids,
    }


def _get_class_attr_ids(cls):
    return {n: id(v.fget if isinstance(v, property) else v) for n, v in vars(cls).items()}


def _load_tokenizer():
    # The slowness is mainly observed in Kimi
    return AutoTokenizer.from_pretrained("nvidia/Kimi-K2-Thinking-NVFP4", trust_remote_code=True)


@contextmanager
def _patched_tokenizer():
    tokenizer = _load_tokenizer()
    _SpecialTokensCachePatcher.patch(tokenizer)
    try:
        yield tokenizer
    finally:
        unpatch_tokenizer(tokenizer)


if __name__ == "__main__":
    unittest.main()
