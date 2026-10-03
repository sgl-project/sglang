import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


def _load_module():
    module_name = "slash_command_handler_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    requests_stub = types.ModuleType("requests")
    sys.modules.setdefault("requests", requests_stub)

    github_stub = types.ModuleType("github")
    github_stub.Auth = object
    github_stub.Github = object
    sys.modules.setdefault("github", github_stub)

    runner_configs_stub = types.ModuleType("runner_configs")
    runner_configs_stub.load = lambda: {}
    sys.modules.setdefault("runner_configs", runner_configs_stub)

    module_path = pathlib.Path(__file__).with_name("slash_command_handler.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


slash_command_handler = _load_module()


class ValidateRegisteredTestMainEntriesTest(unittest.TestCase):
    def test_reports_missing_main_once_per_registered_file(self):
        fake_ci_register = mock.Mock()
        fake_ci_register.collect_tests.side_effect = ValueError(
            'test/registered/foo/test_missing_main.py: missing `if __name__ == "__main__":` entry.'
        )
        resolved_specs = [
            {
                "spec": "registered/foo/test_missing_main.py",
                "test_command": "registered/foo/test_missing_main.py TestCase.test_one",
                "mode": "cuda",
            },
            {
                "spec": "registered/foo/test_missing_main.py::TestCase.test_two",
                "test_command": "registered/foo/test_missing_main.py TestCase.test_two",
                "mode": "cuda",
            },
            {
                "spec": "python/sglang/multimodal_gen/test/unit/test_mm.py",
                "test_command": "python/sglang/multimodal_gen/test/unit/test_mm.py::test_mm",
                "mode": "multimodal_gen",
            },
        ]

        with mock.patch.object(
            slash_command_handler,
            "_load_ci_register_module",
            return_value=fake_ci_register,
        ):
            failures = slash_command_handler._validate_registered_test_main_entries(
                resolved_specs
            )

        self.assertEqual(
            failures,
            [
                {
                    "spec": "registered/foo/test_missing_main.py",
                    "error": 'test/registered/foo/test_missing_main.py: missing `if __name__ == "__main__":` entry.',
                }
            ],
        )
        fake_ci_register.collect_tests.assert_called_once_with(
            ["test/registered/foo/test_missing_main.py"], sanity_check=True
        )

    def test_allows_valid_registered_and_multimodal_specs(self):
        fake_ci_register = mock.Mock()
        resolved_specs = [
            {
                "spec": "registered/foo/test_ok.py",
                "test_command": "registered/foo/test_ok.py",
                "mode": "cpu",
            },
            {
                "spec": "python/sglang/multimodal_gen/test/unit/test_mm.py",
                "test_command": "python/sglang/multimodal_gen/test/unit/test_mm.py::test_mm",
                "mode": "multimodal_gen",
            },
        ]

        with mock.patch.object(
            slash_command_handler,
            "_load_ci_register_module",
            return_value=fake_ci_register,
        ):
            failures = slash_command_handler._validate_registered_test_main_entries(
                resolved_specs
            )

        self.assertEqual(failures, [])
        fake_ci_register.collect_tests.assert_called_once_with(
            ["test/registered/foo/test_ok.py"], sanity_check=True
        )


if __name__ == "__main__":
    unittest.main()
