import random
import unittest
from contextlib import contextmanager

from transformers import AutoTokenizer

from sglang.srt.utils.patch_tokenizer import (
    _SpecialTokensCachePatcher,
    unpatch_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


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


if __name__ == "__main__":
    unittest.main()
