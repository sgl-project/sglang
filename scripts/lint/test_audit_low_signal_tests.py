import unittest

from audit_low_signal_tests import audit_source


def _source(body: str, padding: int = 0) -> str:
    return body + ("\n# padding" * padding) + "\n"


class TestAuditLowSignalTests(unittest.TestCase):
    def test_mock_proxy_is_selected(self):
        row = audit_source(
            "test/registered/unit/demo/test_proxy.py",
            _source("""
from unittest.mock import MagicMock

def test_proxy():
    target = MagicMock()
    target()
    target.assert_called_once()
    assert target is not None
"""),
        )
        self.assertIn("mock-proxy", row.reasons)

    def test_source_contract_is_selected(self):
        row = audit_source(
            "test/registered/unit/demo/test_api_contract.py",
            _source("""
def test_contract():
    assert True
"""),
        )
        self.assertIn("structural-contract", row.reasons)

    def test_numerical_parity_is_not_sparse_behavior(self):
        row = audit_source(
            "test/registered/unit/demo/test_math.py",
            _source(
                """
def test_math():
    torch.testing.assert_close(actual, expected)
""",
                padding=120,
            ),
            added="2026-08-01",
        )
        self.assertNotIn("sparse-behavior", row.reasons)
        self.assertNotIn("rapid-sparse-growth", row.reasons)

    def test_recent_large_sparse_file_is_selected(self):
        row = audit_source(
            "test/registered/unit/demo/test_recent.py",
            _source(
                """
def test_recent():
    assert value
""",
                padding=120,
            ),
            added="2026-08-01",
        )
        self.assertIn("rapid-sparse-growth", row.reasons)

    def test_small_behavior_test_is_retained(self):
        row = audit_source(
            "test/registered/unit/demo/test_small.py",
            _source("""
def test_small():
    assert compute() == 42
"""),
            added="2026-08-01",
        )
        self.assertFalse(row.selected)

    def test_dependent_wrapper_is_selected_without_test_functions(self):
        row = audit_source(
            "test/registered/unit/mem_cache/" "test_rust_unified_radix_cache_bench.py",
            "def load_tests(loader, tests, pattern):\n    return tests\n",
        )
        self.assertIn("dependent-wrapper", row.reasons)


if __name__ == "__main__":
    unittest.main()
