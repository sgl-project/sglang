import importlib.util
import pathlib
import unittest

_CI_REGISTER = pathlib.Path(__file__).resolve().parents[2] / (
    "python/sglang/test/ci/ci_register.py"
)
_SPEC = importlib.util.spec_from_file_location("ci_register", _CI_REGISTER)
ci_register = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_register)


class TestRegisteredTestFileFilter(unittest.TestCase):
    def test_registered_utils_file_remains_collectable(self):
        self.assertTrue(
            ci_register.is_registered_test_file("unit/entrypoints/openai/utils.py")
        )

    def test_non_test_helpers_are_excluded(self):
        self.assertFalse(ci_register.is_registered_test_file("cpu/utils.py"))
        self.assertFalse(ci_register.is_registered_test_file("unit/layers/utils.py"))

    def test_standard_test_file_is_collectable(self):
        self.assertTrue(ci_register.is_registered_test_file("unit/test_example.py"))


if __name__ == "__main__":
    unittest.main()
