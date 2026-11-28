import unittest

from sglang.srt.entrypoints.http_server import app
from sglang.srt.utils.auth import AUTH_LEVELS, _iter_leaf_routes, undeclared_routes
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


if __name__ == "__main__":
    unittest.main()
