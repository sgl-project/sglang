import json
import unittest
from pathlib import Path

from sglang.srt.entrypoints.http_server import app
from sglang.srt.utils.auth import (
    AuthLevel,
    _get_auth_level_from_app_and_scope,
    _iter_leaf_routes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

AUTH_LEVELS_MANIFEST = (
    Path(__file__).parents[4]
    / "rust/sglang-server/src/api_server/http/auth_levels.json"
)


# Registered only under DUMPER_SERVER_PORT=reuse; the Rust server does not serve it.
ENV_GATED_ROUTES = {"/dumper/{method}"}


def build_auth_levels_manifest() -> dict:
    """Every route of the app and the auth level the middleware resolves for it.

    The Rust server reads `auth_levels.json`; regenerate it with
    ``json.dump(build_auth_levels_manifest(), f, indent=2, sort_keys=True)``
    plus a trailing newline.
    """
    levels = {}
    for prefix, route in _iter_leaf_routes(app.router.routes):
        path = prefix + route.path
        if path in ENV_GATED_ROUTES:
            continue
        kind = "websocket" if path == "/v1/realtime" else "http"
        method = next(iter(getattr(route, "methods", None) or ["GET"]))
        scope = {"type": kind, "method": method, "path": path, "root_path": ""}
        level = _get_auth_level_from_app_and_scope(app, scope).value
        assert levels.setdefault(path, level) == level, path
    return levels


class TestAuthLevelsManifest(CustomTestCase):
    def test_manifest_matches_the_app(self):
        with open(AUTH_LEVELS_MANIFEST) as f:
            self.assertEqual(
                json.load(f),
                build_auth_levels_manifest(),
                "regenerate auth_levels.json with build_auth_levels_manifest()",
            )

    def test_admin_routes_take_no_path_parameters(self):
        # The Rust server finds an admin route by its exact path.
        for path, level in build_auth_levels_manifest().items():
            if level != AuthLevel.NORMAL:
                self.assertNotIn("{", path)

    def test_api_key_middleware_is_outermost_on_the_module_level_app(self):
        self.assertEqual(app.user_middleware[0].cls.__name__, "_ApiKeyASGIMiddleware")


if __name__ == "__main__":
    unittest.main()
