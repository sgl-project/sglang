import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _resolve_numa(binding_str, rank):
    """Emulates the inline parsing logic in hiradix_cache.py."""
    if not binding_str or not binding_str.strip():
        return None
    numa = None
    for entry in binding_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        r, n = entry.split(":", 1)
        if int(r.strip()) == rank:
            numa = int(n.strip())
            break
    return numa


class TestHiCacheNumaBinding(unittest.TestCase):

    def test_resolve_multi_rank(self):
        self.assertEqual(_resolve_numa("0:7,1:5,2:5,3:7", 0), 7)
        self.assertEqual(_resolve_numa("0:7,1:5,2:5,3:7", 1), 5)
        self.assertEqual(_resolve_numa("0:7,1:5,2:5,3:7", 2), 5)
        self.assertEqual(_resolve_numa("0:7,1:5,2:5,3:7", 3), 7)
        self.assertIsNone(_resolve_numa("0:7,1:5", 99))

    def test_single_rank(self):
        self.assertEqual(_resolve_numa("0:7", 0), 7)

    def test_none_input(self):
        self.assertIsNone(_resolve_numa(None, 0))

    def test_empty_string(self):
        self.assertIsNone(_resolve_numa("", 0))

    def test_whitespace_only(self):
        self.assertIsNone(_resolve_numa("   ", 0))

    def test_whitespace_handling(self):
        self.assertEqual(_resolve_numa(" 0 : 7 , 1 : 5 ", 0), 7)
        self.assertEqual(_resolve_numa(" 0 : 7 , 1 : 5 ", 1), 5)

    def test_trailing_comma(self):
        self.assertEqual(_resolve_numa("0:7,", 0), 7)

    def test_empty_entry_between_commas(self):
        self.assertEqual(_resolve_numa("0:7,,1:5", 0), 7)
        self.assertEqual(_resolve_numa("0:7,,1:5", 1), 5)

    def test_non_integer_raises(self):
        with self.assertRaises(ValueError):
            _resolve_numa("a:7", 0)
        with self.assertRaises(ValueError):
            _resolve_numa("0:b", 0)

    def test_missing_colon_raises(self):
        # Use a tp_rank that doesn't match "1:5" so it reaches the malformed entry
        with self.assertRaises(ValueError):
            _resolve_numa("1:5,0", 0)


if __name__ == "__main__":
    unittest.main()
