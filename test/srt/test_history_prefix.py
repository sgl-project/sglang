import unittest

from sglang.srt.mem_cache.unified_radix_cache import history_prefix
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHistoryPrefix(unittest.TestCase):
    def test_matches_concat_then_slice(self):
        origin, output = [1, 2, 3, 4], [5, 6, 7]
        for length in range(0, len(origin) + len(output) + 2):
            self.assertEqual(
                history_prefix(origin, output, length),
                (origin + output)[:length],
                length,
            )

    def test_prefix_inside_the_prompt_does_not_touch_the_output(self):
        origin, output = [1, 2, 3, 4], [5, 6, 7]
        self.assertEqual(history_prefix(origin, output, 2), [1, 2])
        self.assertEqual(history_prefix(origin, output, 4), [1, 2, 3, 4])

    def test_keeps_the_sequence_type(self):
        from array import array

        origin, output = array("q", [1, 2, 3]), array("q", [4, 5])
        got = history_prefix(origin, output, 4)
        self.assertIsInstance(got, array)
        self.assertEqual(list(got), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
