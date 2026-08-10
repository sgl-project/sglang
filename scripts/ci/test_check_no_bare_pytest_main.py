import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_no_bare_pytest_main import find_bare_pytest_main


class TestFindBarePytestMain(unittest.TestCase):
    def check_source(self, source: str) -> int | None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "example.py"
            path.write_text(source, encoding="utf-8")
            return find_bare_pytest_main(path)

    def test_rejects_discarded_result(self):
        source = """
if __name__ == "__main__":
    pytest.main([__file__])
"""
        self.assertEqual(self.check_source(source), 3)

    def test_rejects_discarded_result_with_whitespace(self):
        source = """
if __name__ == "__main__":
    pytest . main([__file__])
"""
        self.assertEqual(self.check_source(source), 3)

    def test_accepts_propagated_result(self):
        source = """
if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
"""
        self.assertIsNone(self.check_source(source))

    def test_accepts_assigned_result(self):
        source = """
if "__main__" == __name__:
    exit_code = pytest.main([__file__])
"""
        self.assertIsNone(self.check_source(source))

    def test_ignores_call_outside_main_guard(self):
        self.assertIsNone(self.check_source("pytest.main([__file__])\n"))


if __name__ == "__main__":
    unittest.main()
