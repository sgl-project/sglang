import itertools
import random
import unittest

from sglang.srt.function_call.utils import _find_common_prefix
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _reference_common_prefix(left: str, right: str) -> str:
    length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        length += 1
    return left[:length]


class TestFindCommonPrefix(unittest.TestCase):
    def test_edge_cases(self):
        cases = [
            ("", ""),
            ("", "abc"),
            ("abc", ""),
            ("abc", "abc"),
            ("abc", "abcdef"),
            ("abcdef", "abc"),
            ("abc", "xyz"),
            ("abx", "aby"),
            ('{"city": "San', '{"city": "San Francisco"}'),
            ("你好，世界", "你好，工具调用"),
            ("😀tool", "😀token"),
        ]

        for left, right in cases:
            with self.subTest(left=left, right=right):
                expected = _reference_common_prefix(left, right)
                self.assertEqual(_find_common_prefix(left, right), expected)
                self.assertEqual(_find_common_prefix(right, left), expected)

    def test_exhaustive_short_strings(self):
        values = [
            "".join(chars)
            for length in range(5)
            for chars in itertools.product("ab", repeat=length)
        ]

        for left in values:
            for right in values:
                self.assertEqual(
                    _find_common_prefix(left, right),
                    _reference_common_prefix(left, right),
                )

    def test_random_strings(self):
        random_generator = random.Random(0)
        alphabet = "ab<>｜{}😀"

        for _ in range(1000):
            left = "".join(
                random_generator.choices(alphabet, k=random_generator.randrange(128))
            )
            right = "".join(
                random_generator.choices(alphabet, k=random_generator.randrange(128))
            )
            self.assertEqual(
                _find_common_prefix(left, right),
                _reference_common_prefix(left, right),
            )


if __name__ == "__main__":
    unittest.main()
