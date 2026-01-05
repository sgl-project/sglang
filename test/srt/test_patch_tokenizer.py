import unittest

from transformers import AutoTokenizer

from sglang.srt.utils.patch_tokenizer import (
    _SpecialTokensCachePatcher,
    unpatch_tokenizer,
)


PATCHED_ATTRS = ["all_special_tokens", "all_special_ids", "add_special_tokens", "add_tokens"]


def _get_attr_id(cls, name):
    attr = getattr(cls, name)
    if isinstance(attr, property):
        return id(attr.fget)
    return id(attr)


def _get_all_patched_attr_ids(cls):
    return {name: _get_attr_id(cls, name) for name in PATCHED_ATTRS}


class TestPatchTokenizer(unittest.TestCase):
    def test_patch_unpatch_restores_original(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls = type(tokenizer)

        original_ids = _get_all_patched_attr_ids(cls)

        _SpecialTokensCachePatcher.patch(tokenizer)
        self.assertTrue(getattr(cls, "_sglang_special_tokens_patched", False))

        patched_ids = _get_all_patched_attr_ids(cls)
        for name in PATCHED_ATTRS:
            self.assertNotEqual(
                patched_ids[name],
                original_ids[name],
                f"{name} should be patched (id should change)",
            )

        unpatch_tokenizer(tokenizer)
        self.assertFalse(getattr(cls, "_sglang_special_tokens_patched", False))

        restored_ids = _get_all_patched_attr_ids(cls)
        for name in PATCHED_ATTRS:
            self.assertEqual(
                restored_ids[name],
                original_ids[name],
                f"{name} should be restored to original",
            )

    def test_patch_caches_special_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _SpecialTokensCachePatcher.patch(tokenizer)

        tokens1 = tokenizer.all_special_tokens
        ids1 = tokenizer.all_special_ids
        tokens2 = tokenizer.all_special_tokens
        ids2 = tokenizer.all_special_ids

        self.assertIs(tokens1, tokens2)
        self.assertIs(ids1, ids2)

        unpatch_tokenizer(tokenizer)

    def test_patch_blocks_add_special_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _SpecialTokensCachePatcher.patch(tokenizer)

        with self.assertRaises(AssertionError) as ctx:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.assertIn("Cannot call add_special_tokens after patch", str(ctx.exception))

        unpatch_tokenizer(tokenizer)

    def test_patch_blocks_add_tokens_with_special_flag(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _SpecialTokensCachePatcher.patch(tokenizer)

        with self.assertRaises(AssertionError) as ctx:
            tokenizer.add_tokens(["<new>"], special_tokens=True)
        self.assertIn("Cannot call add_tokens", str(ctx.exception))

        tokenizer.add_tokens(["<regular>"], special_tokens=False)

        unpatch_tokenizer(tokenizer)

    def test_unpatch_clears_cache(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _SpecialTokensCachePatcher.patch(tokenizer)

        _ = tokenizer.all_special_tokens
        _ = tokenizer.all_special_ids
        self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertTrue(hasattr(tokenizer, "_sglang_cached_special_ids"))

        unpatch_tokenizer(tokenizer)
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_tokens"))
        self.assertFalse(hasattr(tokenizer, "_sglang_cached_special_ids"))

    def test_double_patch_is_idempotent(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _SpecialTokensCachePatcher.patch(tokenizer)
        _SpecialTokensCachePatcher.patch(tokenizer)

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

