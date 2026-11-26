import unittest

from sglang.srt.entrypoints.http_server import app
from sglang.srt.utils.auth import (
    AUTH_LEVELS,
    AuthLevel,
    _iter_leaf_routes,
    decide_request_auth,
    undeclared_routes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# Declared routes neither imported app has: the dumper's (registered under
# DUMPER_SERVER_PORT=reuse), the metrics mount (added at launch) and the Rust
# server's routes under it, and the Rust server's relay.
SERVED_ELSEWHERE = {
    "/dumper/{method}",
    "/metrics",
    "/metrics/",
    "/metrics/{path:path}",
    "/internal/peer",
}

# Routes only the diffusion server serves; its app needs the diffusion extra, so
# python/sglang/multimodal_gen/test/unit/test_http_server_auth.py checks that
# app declares every route it serves.
SERVED_BY_DIFFUSION = {
    "/get_weights_checksum",
    "/liveness",
    "/models",
    "/openpi/policy",
    "/rollout/generate",
    "/stats",
    "/update_weights_from_tensor_checker",
    "/v1/actions/generations",
    "/v1/actions/metadata",
    "/v1/actions/realtime",
    "/v1/images/edits",
    "/v1/images/generations",
    "/v1/images/{image_id}/content",
    "/v1/list_loras",
    "/v1/merge_lora_weights",
    "/v1/meshes",
    "/v1/meshes/{mesh_id}",
    "/v1/meshes/{mesh_id}/content",
    "/v1/model_info",
    "/v1/realtime_video/generate",
    "/v1/set_lora",
    "/v1/unmerge_lora_weights",
    "/v1/videos",
    "/v1/videos/{video_id}",
    "/v1/videos/{video_id}/content",
}


class TestAuthLevels(CustomTestCase):
    """An undeclared route is NORMAL at runtime, so sglang's own must be declared."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.disaggregation.encoder.http_server import app as encoder_app

        cls.apps = (app, encoder_app)

    def test_every_route_declares_its_level(self):
        for served in self.apps:
            self.assertEqual(undeclared_routes(served), [])

    def test_every_declared_route_is_served(self):
        served = {
            prefix + route.path
            for served_app in self.apps
            for prefix, route in _iter_leaf_routes(served_app.routes)
        }
        self.assertLessEqual(
            set(AUTH_LEVELS) - served, SERVED_ELSEWHERE | SERVED_BY_DIFFUSION
        )

    def test_api_key_middleware_is_outermost_on_the_module_level_app(self):
        self.assertEqual(app.user_middleware[0].cls.__name__, "_ApiKeyASGIMiddleware")


class TestHiCacheClearRequiresAdminApiKey(CustomTestCase):
    """POST /hicache/storage-backend/clear erases the persistent
    hierarchical cache backend, so it is admin_force like the attach/detach
    endpoints for the same backend (issue #44191): an ordinary API key must
    not unlock it, and it is refused outright when the server was started
    without --admin-api-key.
    """

    PATH = "/hicache/storage-backend/clear"

    def _decide(self, authorization_header, api_key, admin_api_key):
        return decide_request_auth(
            method="POST",
            path=self.PATH,
            authorization_header=authorization_header,
            api_key=api_key,
            admin_api_key=admin_api_key,
            auth_level=AUTH_LEVELS[self.PATH],
        )

    def test_clear_is_admin_force(self):
        self.assertEqual(AUTH_LEVELS[self.PATH], AuthLevel.ADMIN_FORCE)

    def test_clear_refused_without_admin_api_key_configured(self):
        decision = self._decide("Bearer user-key", "user-key", None)
        self.assertFalse(decision.allowed)

    def test_clear_refused_with_only_api_key(self):
        decision = self._decide("Bearer user-key", "user-key", "admin-key")
        self.assertFalse(decision.allowed)

    def test_clear_allowed_with_admin_api_key(self):
        decision = self._decide("Bearer admin-key", "user-key", "admin-key")
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
