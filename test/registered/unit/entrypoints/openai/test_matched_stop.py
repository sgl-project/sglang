import unittest

from sglang.srt.sampling.sampling_params import MAX_LEN, get_max_seq_length
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestRegexPatternMaxLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regex_str_to_max_len = {
            "((ab|cd(e|f){2}){3,5}g|hij)*k": MAX_LEN,
            # - '*' -> infinite tokens need to be stored
            "abc*?k": MAX_LEN,
            # - '*?' -> infinite tokens still need to be stored even if lazy matching used
            "^spec(foo|at)$": 7,
            # - '^' and '$' don't add any characters to the max length
            # "spec" -> 4
            # "(foo|at)" -> max(3, 2) = 3
            # Whole regex = 7
            "(a(bca|de(fg|hi){2,3})j){2}kl": 22,
            # - Innermost alt: "fg" vs "hi" -> 2
            # - Repeat {2,3}: max = 3 * 2 = 6
            # - Inner group "de(...)": 2 (for "de") + 6 = 8.
            # - "bca" or "de(...)" -> max(3, 8) = 8
            # - Whole group: "a" (1) + group (8) + "j"(1) = 10
            # - Repeat {2} -> 20
            # - Add "kl"(2) -> 22
            "(foo(bar|baz(qux){1,2}))|(x(yz){5,10})": 21,
            # Branch 1:
            #   "foo"(3) + max("bar"(3), "baz"(3)+"qux"{2} = 3 + 6 = 9) = 3 + 9 = 12
            # Branch 2:
            #   "x"(1) + "yz"{10} = 1 + 20 =21
            # Whole regex = max(12, 21) = 21
            "(((a|bc){1,3}(d(e|f){2}|gh){2,4})|(ijk|lmp(no|p){3})){5}": 90,
            # Branch A:
            #   (a|bc){1,3} -> max = 3 * 2 = 6
            #   Inside: d(e|f){2} = 1 + 2 * 1 = 3 vs gh = 2 -> max = 3
            #   Repeat {2,4} -> 4 * 3 = 12
            #   Branch A total = 18
            # Branch B:
            #   "ijk"(3) vs "lmp(no|p){3}" = 3 + 3 * max(2, 1) = 3 + 6 = 9 -> max = 9
            #   Branch B total = 9
            # Whole outer alt = max(18, 9) = 18
            # Repeat {5} -> 90
        }

    def test_get_max_length(self):
        for regex_str, max_len in self.regex_str_to_max_len.items():
            if max_len == MAX_LEN:
                self.assertGreaterEqual(get_max_seq_length(regex_str), MAX_LEN)
            else:
                self.assertEqual(get_max_seq_length(regex_str), max_len)


class TestNegatedCharClassMaxLength(unittest.TestCase):
    """A single-character negated class must get a finite bound (sglang#30932).

    `[^a]` matches exactly one character, but was bounded at MAX_LEN, which
    downstream becomes the per-decode-step re-decode tail window.
    """

    FINITE_CASES = {
        "[^a]": 1,
        "[^\n]": 1,
        # Two or more excluded characters take a different parse path; it stayed
        # correct throughout and must not regress.
        "[^ab]": 1,
        "ab[^,]cd": 5,
        "[^a][^a]": 2,
        "[^a]|b": 1,
        "[^a]{3}": 3,
    }

    # Controls: a fix that merely weakened the unbounded-repeat branch would make
    # these finite too.
    UNBOUNDED_CASES = ["[^a]+", "a+", '"[^"]*"']

    def test_finite_negated_class_bounds(self):
        for regex_str, expected in self.FINITE_CASES.items():
            with self.subTest(regex_str=regex_str):
                self.assertEqual(get_max_seq_length(regex_str), expected)

    def test_unbounded_patterns_stay_unbounded(self):
        for regex_str in self.UNBOUNDED_CASES:
            with self.subTest(regex_str=regex_str):
                self.assertGreaterEqual(get_max_seq_length(regex_str), MAX_LEN)

    def test_no_unhandled_token_warning(self):
        """Guards the failure mode itself: an opcode falling to the catch-all else.

        The bound would still be sound but silently MAX_LEN, so only the log line
        distinguishes "unbounded" from "unsupported".
        """
        for regex_str in [*self.FINITE_CASES, *self.UNBOUNDED_CASES]:
            with self.subTest(regex_str=regex_str):
                with self.assertNoLogs(
                    "sglang.srt.sampling.sampling_params", level="WARNING"
                ):
                    get_max_seq_length(regex_str)


if __name__ == "__main__":
    unittest.main()
