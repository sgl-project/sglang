import unittest
from contextlib import contextmanager

from transformers import AutoTokenizer

from sglang.srt.utils.patch_tokenizer import (
    _SpecialTokensCachePatcher,
    unpatch_tokenizer,
)


class TestPatchTokenizer(unittest.TestCase):
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

    def test_patched_produces_same_results_as_raw(self):
        """End-to-end test: patched tokenizer should produce identical results"""
        test_texts = [
            "Hello, world!",
            "This is a longer sentence with multiple words.",
            "Special chars: <|im_start|> <|im_end|> [BOS] [EOS]",
            "Numbers 12345 and symbols !@#$%",
            "中文测试 mixed with English",
            "    leading and trailing spaces    ",
            "\n\nMultiple\n\nNewlines\n\n",
        ]

        # Get results from raw tokenizer
        raw_tokenizer = _load_tokenizer()
        raw_encode_results = [raw_tokenizer.encode(t) for t in test_texts]
        raw_decode_results = [
            raw_tokenizer.decode(ids, skip_special_tokens=True)
            for ids in raw_encode_results
        ]
        raw_batch_decode_results = raw_tokenizer.batch_decode(
            raw_encode_results, skip_special_tokens=True
        )
        raw_special_tokens = raw_tokenizer.all_special_tokens
        raw_special_ids = raw_tokenizer.all_special_ids

        # Get results from patched tokenizer
        _SpecialTokensCachePatcher.patch(raw_tokenizer)
        patched_encode_results = [raw_tokenizer.encode(t) for t in test_texts]
        patched_decode_results = [
            raw_tokenizer.decode(ids, skip_special_tokens=True)
            for ids in patched_encode_results
        ]
        patched_batch_decode_results = raw_tokenizer.batch_decode(
            patched_encode_results, skip_special_tokens=True
        )
        patched_special_tokens = raw_tokenizer.all_special_tokens
        patched_special_ids = raw_tokenizer.all_special_ids

        unpatch_tokenizer(raw_tokenizer)

        # Compare results
        self.assertEqual(raw_encode_results, patched_encode_results)
        self.assertEqual(raw_decode_results, patched_decode_results)
        self.assertEqual(raw_batch_decode_results, patched_batch_decode_results)
        self.assertEqual(raw_special_tokens, patched_special_tokens)
        self.assertEqual(raw_special_ids, patched_special_ids)


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
