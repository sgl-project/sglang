import unittest

from transformers import AutoTokenizer

from sglang.srt.utils.patch_tokenizer import (
    _patch_special_tokens_cache,
    unpatch_tokenizer,
)


class TestPatchTokenizer(unittest.TestCase):
    def test_patch_unpatch_restores_original(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls = type(tokenizer)

        original_ids = {
            "all_special_tokens": id(cls.all_special_tokens.fget),
            "all_special_ids": id(cls.all_special_ids.fget),
            "add_special_tokens": id(cls.add_special_tokens),
            "add_tokens": id(cls.add_tokens),
        }

        _patch_special_tokens_cache(tokenizer)
        self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))

        unpatch_tokenizer(tokenizer)
        self.assertFalse(getattr(cls, "_sglang_special_tokens_patched", False))

        self.assertEqual(
            id(cls.all_special_tokens.fget), original_ids["all_special_tokens"]
        )
        self.assertEqual(id(cls.all_special_ids.fget), original_ids["all_special_ids"])
        self.assertEqual(id(cls.add_special_tokens), original_ids["add_special_tokens"])
        self.assertEqual(id(cls.add_tokens), original_ids["add_tokens"])

    def test_patch_caches_special_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _patch_special_tokens_cache(tokenizer)

        tokens1 = tokenizer.all_special_tokens
        ids1 = tokenizer.all_special_ids
        tokens2 = tokenizer.all_special_tokens
        ids2 = tokenizer.all_special_ids

        self.assertIs(tokens1, tokens2)
        self.assertIs(ids1, ids2)

        unpatch_tokenizer(tokenizer)

    def test_patch_blocks_add_special_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _patch_special_tokens_cache(tokenizer)

        with self.assertRaises(AssertionError) as ctx:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.assertIn("Cannot modify special tokens after patch", str(ctx.exception))

        unpatch_tokenizer(tokenizer)

    def test_patch_blocks_add_tokens_with_special_flag(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _patch_special_tokens_cache(tokenizer)

        with self.assertRaises(AssertionError) as ctx:
            tokenizer.add_tokens(["<new>"], special_tokens=True)
        self.assertIn("Cannot add special tokens after patch", str(ctx.exception))

        tokenizer.add_tokens(["<regular>"], special_tokens=False)

        unpatch_tokenizer(tokenizer)

    def test_unpatch_clears_cache(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _patch_special_tokens_cache(tokenizer)

        _ = tokenizer.all_special_tokens
        _ = tokenizer.all_special_ids
        self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_ids"))

        unpatch_tokenizer(tokenizer)
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_ids"))

    def test_double_patch_is_idempotent(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _patch_special_tokens_cache(tokenizer)
        _patch_special_tokens_cache(tokenizer)

        self.assertTrue(
            getattr(type(tokenizer), "_sglang_special_tokens_patched", False)
        )

        unpatch_tokenizer(tokenizer)

    def test_unpatch_without_patch_is_noop(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        result = unpatch_tokenizer(tokenizer)
        self.assertIs(result, tokenizer)


if __name__ == "__main__":
    unittest.main()

