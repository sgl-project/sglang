import re
import unittest
from pathlib import Path

from sglang.srt.entrypoints.http_server import app
from sglang.srt.utils.auth import (
    AuthLevel,
    _get_auth_level_from_app_and_scope,
    _iter_leaf_routes,
    decide_request_auth,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

ADMIN_PATHS = [
    "/load_lora_adapter",
    "/load_lora_adapter_from_tensors",
    "/unload_lora_adapter",
    "/start_profile",
    "/stop_profile",
    "/set_trace_level",
    "/freeze_gc",
    "/checkpoint/prepare",
    "/checkpoint/restore",
]

ROUTE_AUTH_LEVELS = {
    AuthLevel.NORMAL: {
        "/",
        "/api/chat",
        "/api/generate",
        "/api/show",
        "/api/tags",
        "/classify",
        "/close_session",
        "/detokenize",
        "/docs",
        "/docs/oauth2-redirect",
        "/encode",
        "/generate",
        "/get_hicache_l3_cache_layout",
        "/get_model_info",
        "/get_server_info",
        "/health",
        "/health_generate",
        "/invocations",
        "/model_info",
        "/open_session",
        "/openapi.json",
        "/parse_function_call",
        "/ping",
        "/ready",
        "/redoc",
        "/separate_reasoning",
        "/server_info",
        "/tokenize",
        "/v1/audio/transcriptions",
        "/v1/chat/completions",
        "/v1/classify",
        "/v1/completions",
        "/v1/decisions",
        "/v1/detokenize",
        "/v1/embeddings",
        "/v1/loads",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/models",
        "/v1/models/{model:path}",
        "/v1/realtime",
        "/v1/rerank",
        "/v1/responses",
        "/v1/responses/{response_id}",
        "/v1/responses/{response_id}/cancel",
        "/v1/score",
        "/v1/systemone",
        "/v1/tokenize",
        "/vertex_generate",
    },
    AuthLevel.ADMIN_OPTIONAL: {
        "/abort_request",
        "/add_external_corpus",
        "/begin_weight_update",
        "/configure_logging",
        "/continue_generation",
        "/destroy_weights_update_group",
        "/dump_expert_distribution_record",
        "/dumper/{method}",
        "/end_weight_update",
        "/flush_cache",
        "/freeze_gc",
        "/get_remote_instance_transfer_engine_info",
        "/get_weights_by_name",
        "/hicache/storage-backend",
        "/hicache/storage-backend/clear",
        "/init_weights_send_group_for_remote_instance",
        "/init_weights_update_group",
        "/is_scaling_elastic_ep",
        "/list_external_corpora",
        "/load_lora_adapter",
        "/load_lora_adapter_from_tensors",
        "/pause_generation",
        "/pd_role_switch",
        "/plugin_weight_update",
        "/release_memory_occupation",
        "/remote_instance_transfer_engine_info",
        "/remove_external_corpus",
        "/resume_memory_occupation",
        "/scale_elastic_ep",
        "/send_weights_to_remote_instance",
        "/set_internal_state",
        "/set_trace_level",
        "/slow_down",
        "/start_expert_distribution_record",
        "/start_profile",
        "/stop_expert_distribution_record",
        "/stop_profile",
        "/unload_lora_adapter",
        "/update_weight_version",
        "/update_weights_from_disk",
        "/update_weights_from_distributed",
        "/update_weights_from_ipc",
        "/update_weights_from_tensor",
        "/weights_checker",
    },
}
RUST_AUTH = Path(__file__).parents[4] / "rust/sglang-server/src/api_server/http/auth.rs"


class TestEveryRouteDeclaresItsAuthLevel(CustomTestCase):
    def test_routes_match_the_declared_table(self):
        declared = {
            path: level for level, paths in ROUTE_AUTH_LEVELS.items() for path in paths
        }
        for prefix, route in _iter_leaf_routes(app.router.routes):
            path = prefix + route.path
            kind = "websocket" if path == "/v1/realtime" else "http"
            method = next(iter(getattr(route, "methods", None) or ["GET"]))
            scope = {"type": kind, "method": method, "path": path, "root_path": ""}
            with self.subTest(path=path):
                self.assertEqual(
                    declared.get(path), _get_auth_level_from_app_and_scope(app, scope)
                )
        self.assertLessEqual(
            {"/scale_elastic_ep", "/v1/loads", "/start_profile"},
            {prefix + route.path for prefix, route in _iter_leaf_routes(app.routes)},
        )

    def test_rust_server_tables_agree(self):
        source = RUST_AUTH.read_text()
        for name, level in (("ADMIN_OPTIONAL_PATHS", AuthLevel.ADMIN_OPTIONAL),):
            body = re.search(rf"const {name}: &\[&str\] = &\[(.*?)\];", source, re.S)
            paths = set(re.findall(r'"(/[^"]*)"', body.group(1)))
            self.assertTrue(paths)
            self.assertLessEqual(paths, ROUTE_AUTH_LEVELS[level])


class TestAdminEndpointAuthLevels(CustomTestCase):
    def _decide(self, path: str, authorization_header: str | None):
        scope = {"type": "http", "method": "POST", "path": path, "root_path": ""}
        return decide_request_auth(
            method="POST",
            path=path,
            authorization_header=authorization_header,
            api_key=None,
            admin_api_key="admin-secret",
            auth_level=_get_auth_level_from_app_and_scope(app, scope),
        )

    def test_state_mutating_endpoints_require_admin_api_key(self):
        """A route missing @auth_level resolves to NORMAL, which a server started
        with only --admin-api-key leaves open to unauthenticated requests."""
        for path in ADMIN_PATHS:
            with self.subTest(path=path):
                self.assertFalse(self._decide(path, None).allowed)
                self.assertTrue(self._decide(path, "Bearer admin-secret").allowed)

    def test_api_key_middleware_is_outermost_on_the_module_level_app(self):
        self.assertEqual(app.user_middleware[0].cls.__name__, "_ApiKeyASGIMiddleware")


if __name__ == "__main__":
    unittest.main()
