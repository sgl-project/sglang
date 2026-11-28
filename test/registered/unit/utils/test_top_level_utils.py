import unittest
from unittest import mock

import sglang.utils as sglang_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import info_once, is_same_type, trim_overlap

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestInfoOnce(CustomTestCase):
    def setUp(self):
        info_once.cache_clear()
        self.addCleanup(info_once.cache_clear)

    def test_logs_each_distinct_message_once(self):
        with mock.patch.object(sglang_utils.logger, "info") as mock_info:
            info_once("first message")
            info_once("first message")
            info_once("second message")
            info_once("second message")
            info_once("first message")

        self.assertEqual(
            [call.args[0] for call in mock_info.call_args_list],
            ["first message", "second message"],
        )


class TestIsSameType(CustomTestCase):
    def test_empty_and_singleton(self):
        self.assertTrue(is_same_type([]))
        self.assertTrue(is_same_type([42]))

    def test_homogeneous_values(self):
        self.assertTrue(is_same_type([1, 2, 3]))
        self.assertTrue(is_same_type([True, False]))
        self.assertTrue(is_same_type(["a", "b"]))

    def test_bool_and_int_symmetry(self):
        self.assertFalse(is_same_type([True, 1]))
        self.assertFalse(is_same_type([1, True]))

    def test_subclass_and_base_class_symmetry(self):
        class Base:
            pass

        class Sub(Base):
            pass

        self.assertFalse(is_same_type([Base(), Sub()]))
        self.assertFalse(is_same_type([Sub(), Base()]))


class TestTrimOverlap(CustomTestCase):
    def test_basic_overlap_cases(self):
        self.assertEqual(trim_overlap("hello world", "world!"), "!")
        self.assertEqual(trim_overlap("hello world", "no overlap"), "no overlap")
        self.assertEqual(trim_overlap("", "chunk"), "chunk")
        self.assertEqual(trim_overlap("existing", ""), "")
        self.assertEqual(trim_overlap("abc", "abc"), "")
        self.assertEqual(trim_overlap("abc", "abcdef"), "def")

    def test_repeated_prefix_picks_largest_overlap(self):
        self.assertEqual(trim_overlap("xxababab", "abababcd"), "cd")
        self.assertEqual(trim_overlap("aaaa", "aaab"), "b")


if __name__ == "__main__":
    unittest.main()
