import pathlib
import unittest

from audit_test_value import audit_source, candidates


class TestAuditTestValue(unittest.TestCase):
    def audit(self, source: str):
        return audit_source(pathlib.Path("test/registered/unit/test_sample.py"), source)

    def test_signal_b_requires_real_assertions(self):
        row = self.audit(
            "def test_only_call_tracking(mock):\n" "    mock.run.assert_called_once()\n"
        )
        self.assertIsNotNone(row)
        self.assertFalse(row.signal_b)

    def test_signal_b_flags_call_dominated_file(self):
        row = self.audit(
            "def test_call_dominated(mock):\n"
            "    mock.run.assert_called_once()\n"
            "    assert mock.result == 1\n"
        )
        self.assertIsNotNone(row)
        self.assertTrue(row.signal_b)

    def test_call_args_is_a_call_tracking_assertion(self):
        row = self.audit(
            "def test_call_args(mock):\n" "    assert mock.run.call_args.args == (1,)\n"
        )
        self.assertIsNotNone(row)
        self.assertEqual(row.call_assertions, 1)
        self.assertEqual(row.real_assertions, 1)

    def test_signal_c_uses_a_minimum_mock_reference_floor(self):
        row = self.audit(
            "from sglang.srt.foo import Foo\n"
            "def test_small_mock():\n"
            "    value = Mock()\n"
            "    assert value is not None\n"
        )
        self.assertIsNotNone(row)
        self.assertFalse(row.signal_c)

    def test_signal_c_flags_mock_to_real_import_ratio(self):
        row = self.audit(
            "from sglang.srt.foo import Foo\n"
            "def test_mock_heavy():\n"
            + "\n".join(f"    value_{index} = Mock()" for index in range(10))
            + "\n    assert Foo is not None\n"
        )
        self.assertIsNotNone(row)
        self.assertTrue(row.signal_c)

    def test_candidates_are_sorted_and_exclude_clean_rows(self):
        candidate = self.audit(
            "def test_candidate(mock):\n"
            "    mock.run.assert_called_once()\n"
            "    assert mock.result == 1\n"
        )
        clean = self.audit("def test_clean():\n    assert 1 + 1 == 2\n")
        self.assertEqual(candidates([clean, candidate]), [candidate])


if __name__ == "__main__":
    unittest.main()
