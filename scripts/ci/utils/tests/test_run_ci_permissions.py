import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
HANDLER_PATH = ROOT / "scripts/ci/utils/slash_command_handler.py"

requests_stub = types.ModuleType("requests")
github_stub = types.ModuleType("github")


class _Auth:
    @staticmethod
    def Token(token):
        return token


github_stub.Auth = _Auth
github_stub.Github = object
sys.modules.setdefault("requests", requests_stub)
sys.modules.setdefault("github", github_stub)
sys.modules.setdefault("runner_configs", types.ModuleType("runner_configs"))

spec = importlib.util.spec_from_file_location("slash_command_handler", HANDLER_PATH)
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


class ResolveTagRunCiPermissionsTest(unittest.TestCase):
    def test_explicit_permission_skips_repository_lookup(self):
        repo = Mock()

        result = handler.resolve_tag_run_ci_permissions(
            repo, "configured-user", {"can_tag_run_ci_label": True}
        )

        self.assertTrue(result["can_tag_run_ci_label"])
        repo.get_collaborator_permission.assert_not_called()

    def test_write_or_higher_repository_role_is_authorized(self):
        for permission in ("write", "maintain", "admin"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "maintainer", None
                )

                self.assertTrue(result["can_tag_run_ci_label"])

    def test_lower_repository_role_is_not_authorized(self):
        for permission in ("none", "read", "triage"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "external-user", None
                )

                self.assertNotIn("can_tag_run_ci_label", result)

    def test_lookup_failure_fails_closed(self):
        repo = Mock()
        repo.get_collaborator_permission.side_effect = RuntimeError("API unavailable")

        result = handler.resolve_tag_run_ci_permissions(repo, "unknown-user", None)

        self.assertNotIn("can_tag_run_ci_label", result)


if __name__ == "__main__":
    unittest.main()
