"""
Unit tests for sglang.srt.constrained.utils.

Test Coverage:
- is_legacy_structural_tag: legacy format detection, new format detection,
  missing fields, edge cases with assertion errors.

Usage:
    python -m pytest test_utils.py -v
"""

import unittest

import torch

from sglang.srt.constrained.utils import (
    is_dense_bool_mask_allowed_token,
    is_legacy_structural_tag,
    is_packed_bitmask_allowed_token,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


class TestIsLegacyStructuralTag(unittest.TestCase):
    """Test is_legacy_structural_tag function."""

    def test_legacy_format_returns_true(self):
        obj = {
            "structures": [{"begin": "<tool>", "end": "</tool>"}],
            "triggers": ["<tool>"],
        }
        self.assertTrue(is_legacy_structural_tag(obj))

    def test_legacy_format_empty_lists(self):
        obj = {"structures": [], "triggers": []}
        self.assertTrue(is_legacy_structural_tag(obj))

    def test_new_format_returns_false(self):
        obj = {"format": {"type": "json_schema", "schema": {}}}
        self.assertFalse(is_legacy_structural_tag(obj))

    def test_new_format_empty_format(self):
        obj = {"format": {}}
        self.assertFalse(is_legacy_structural_tag(obj))

    def test_legacy_missing_triggers_raises(self):
        """Legacy format requires both 'structures' and 'triggers'."""
        obj = {"structures": [{"begin": "<tool>", "end": "</tool>"}]}
        with self.assertRaises(AssertionError):
            is_legacy_structural_tag(obj)

    def test_new_format_missing_format_raises(self):
        """New format (no 'structures') requires 'format' key."""
        obj = {"other_key": "value"}
        with self.assertRaises(AssertionError):
            is_legacy_structural_tag(obj)

    def test_empty_dict_raises(self):
        with self.assertRaises(AssertionError):
            is_legacy_structural_tag({})

    def test_structures_none_uses_new_format_path(self):
        """Explicitly None 'structures' should fall to new format check."""
        obj = {"structures": None, "format": {"type": "json_schema"}}
        self.assertFalse(is_legacy_structural_tag(obj))

    def test_both_keys_present_legacy_wins(self):
        """When both 'structures' and 'format' present, 'structures' takes priority."""
        obj = {
            "structures": [{"begin": "<tool>"}],
            "triggers": ["<tool>"],
            "format": {"type": "json_schema"},
        }
        self.assertTrue(is_legacy_structural_tag(obj))


class TestVocabMaskAllowedTokenUtils(unittest.TestCase):
    def test_packed_bitmask_allowed_token(self):
        vocab_mask = torch.zeros(2, dtype=torch.int32)
        vocab_mask[0] = 1 << 7
        vocab_mask[1] = 1 << 2

        self.assertTrue(is_packed_bitmask_allowed_token(vocab_mask, 7, vocab_size=64))
        self.assertFalse(is_packed_bitmask_allowed_token(vocab_mask, 8, vocab_size=64))
        self.assertTrue(is_packed_bitmask_allowed_token(vocab_mask, 34, vocab_size=64))
        self.assertFalse(is_packed_bitmask_allowed_token(vocab_mask, 34, vocab_size=34))
        self.assertFalse(is_packed_bitmask_allowed_token(vocab_mask, 64, vocab_size=64))

    def test_dense_bool_mask_allowed_token(self):
        # Dense bool masks use False for allowed tokens and True for masked tokens.
        vocab_mask = torch.tensor([True, False, True, False], dtype=torch.bool)

        self.assertTrue(is_dense_bool_mask_allowed_token(vocab_mask, 1, vocab_size=4))
        self.assertFalse(is_dense_bool_mask_allowed_token(vocab_mask, 0, vocab_size=4))
        self.assertFalse(is_dense_bool_mask_allowed_token(vocab_mask, 4, vocab_size=4))
        self.assertFalse(is_dense_bool_mask_allowed_token(vocab_mask, 3, vocab_size=3))


if __name__ == "__main__":
    unittest.main()
