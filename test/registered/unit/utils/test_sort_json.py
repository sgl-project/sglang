"""
Unit-tests for sglang.srt.utils.common.sort_json (no pytest).
Run with:
    python -m unittest test_sort_json -v
from this directory.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest

from sglang.srt.utils.common import sort_json
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class SortJsonTestCase(unittest.TestCase):
    def test_sorts_nested_dict_keys(self):
        result = sort_json({"b": {"d": 1, "c": 2}, "a": 3})
        self.assertEqual(list(result.keys()), ["a", "b"])
        self.assertEqual(list(result["b"].keys()), ["c", "d"])

    def test_preserves_list_order(self):
        value = {"required": ["zebra", "apple"], "items": [3, 1, 2]}
        result = sort_json(value)
        self.assertEqual(result["required"], ["zebra", "apple"])
        self.assertEqual(result["items"], [3, 1, 2])

    def test_sorts_dicts_inside_lists(self):
        result = sort_json([{"b": 1, "a": 2}])
        self.assertEqual(list(result[0].keys()), ["a", "b"])

    def test_mixed_type_keys_do_not_crash(self):
        self.assertEqual(sort_json({1: "x", "a": "y"}), {"1": "x", "a": "y"})

    def test_passthrough_scalars(self):
        self.assertEqual(sort_json("s"), "s")
        self.assertEqual(sort_json(3.14), 3.14)
        self.assertIsNone(sort_json(None))

    def test_does_not_mutate_input(self):
        value = {"b": 1, "a": 2}
        sort_json(value)
        self.assertEqual(list(value.keys()), ["b", "a"])


if __name__ == "__main__":
    unittest.main()
