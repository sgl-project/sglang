import unittest

from sglang.srt.sampling.sampling_params import MAX_LEN, get_max_seq_length
from sglang.srt.utils import kill_process_tree
from sglang.test.kit_matched_stop import MatchedStopMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestMatchedStop(CustomTestCase, MatchedStopMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=["--max-running-requests", "10"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestRegexPatternMaxLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regex_str_to_max_len = {
            "((ab|cd(e|f){2}){3,5}g|hij)*k": MAX_LEN,
            # - '*' → infinite tokens need to be stored
            "abc*?k": MAX_LEN,
            # - '*?' → infinite tokens still need to be stored even if lazy matching used
            "^spec(foo|at)$": 7,
            # - '^' and '$' don't add any characters to the max length
            # "spec" → 4
            # "(foo|at)" → max(3, 2) = 3
            # Whole regex = 7
            "(a(bca|de(fg|hi){2,3})j){2}kl": 22,
            # - Innermost alt: "fg" vs "hi" → 2
            # - Repeat {2,3}: max = 3 * 2 = 6
            # - Inner group "de(...)": 2 (for "de") + 6 = 8.
            # - "bca" or "de(...)" → max(3, 8) = 8
            # - Whole group: "a" (1) + group (8) + "j"(1) = 10
            # - Repeat {2} → 20
            # - Add "kl"(2) → 22
            "(foo(bar|baz(qux){1,2}))|(x(yz){5,10})": 21,
            # Branch 1:
            #   "foo"(3) + max("bar"(3), "baz"(3)+"qux"{2} = 3 + 6 = 9) = 3 + 9 = 12
            # Branch 2:
            #   "x"(1) + "yz"{10} = 1 + 20 =21
            # Whole regex = max(12, 21) = 21
            "(((a|bc){1,3}(d(e|f){2}|gh){2,4})|(ijk|lmp(no|p){3})){5}": 90,
            # Branch A:
            #   (a|bc){1,3} → max = 3 * 2 = 6
            #   Inside: d(e|f){2} = 1 + 2 * 1 = 3 vs gh = 2 → max = 3
            #   Repeat {2,4} → 4 * 3 = 12
            #   Branch A total = 18
            # Branch B:
            #   "ijk"(3) vs "lmp(no|p){3}" = 3 + 3 * max(2, 1) = 3 + 6 = 9 → max = 9
            #   Branch B total = 9
            # Whole outer alt = max(18, 9) = 18
            # Repeat {5} → 90
        }

    def test_get_max_length(self):
        for regex_str, max_len in self.regex_str_to_max_len.items():
            if max_len == MAX_LEN:
                self.assertGreaterEqual(get_max_seq_length(regex_str), MAX_LEN)
            else:
                self.assertEqual(get_max_seq_length(regex_str), max_len)


if __name__ == "__main__":
    unittest.main()
