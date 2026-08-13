import pathlib
import tempfile
import unittest

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

    def test_rejects_assigned_result(self):
        source = """
if "__main__" == __name__:
    exit_code = pytest.main([__file__])
"""
        self.assertEqual(self.check_source(source), 3)

    def test_accepts_assigned_result_that_is_later_exited(self):
        source = """
if __name__ == "__main__":
    exit_code = pytest.main([__file__])
    sys.exit(exit_code)
"""
        self.assertIsNone(self.check_source(source))

    def test_accepts_assigned_result_that_is_raised(self):
        source = """
if __name__ == "__main__":
    exit_code = pytest.main([__file__])
    raise SystemExit(exit_code)
"""
        self.assertIsNone(self.check_source(source))

    def test_rejects_nested_discarded_result(self):
        source = """
if __name__ == "__main__":
    if enabled:
        pytest.main([__file__])
"""
        self.assertEqual(self.check_source(source), 4)

    def test_accepts_raised_system_exit(self):
        source = """
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
"""
        self.assertIsNone(self.check_source(source))

    def test_rejects_unraised_system_exit(self):
        source = """
if __name__ == "__main__":
    error = SystemExit(pytest.main([__file__]))
"""
        self.assertEqual(self.check_source(source), 3)

    def test_ignores_call_outside_main_guard(self):
        self.assertIsNone(self.check_source("pytest.main([__file__])\n"))


if __name__ == "__main__":
    unittest.main()
