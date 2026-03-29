"""
Unit tests for sglang.srt.constrained.utils.

Test Coverage:
- is_legacy_structural_tag: legacy format detection, new format detection,
  missing fields, edge cases with assertion errors.

Usage:
    python -m pytest test_utils.py -v
"""

import unittest

from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "stage-a-cpu-only")


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


if __name__ == "__main__":
    unittest.main()
