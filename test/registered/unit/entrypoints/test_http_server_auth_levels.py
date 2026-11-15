import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from sglang.srt.disaggregation.encoder import http_server as encoder_http
from sglang.srt.disaggregation.encoder.receiver import EncoderBootstrapServer
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,
)
from sglang.srt.entrypoints.http_server import app
from sglang.srt.model_executor.model_runner_components import (
    remote_instance_weight_transporter as transporter,
)
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


class TestAuthLevels(CustomTestCase):
    """An undeclared route is NORMAL at runtime, so sglang's own must be declared."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.disaggregation.encoder.http_server import app as encoder_app

        with patch("threading.Thread.start"):
            engine = EngineInfoBootstrapServer("127.0.0.1", 0)
            encoder = EncoderBootstrapServer("127.0.0.1", 0, health_check_interval=0)
        cls.apps = (app, encoder_app, engine._server.config.app, encoder.app)

    def test_every_route_declares_its_level(self):
        for served in self.apps:
            self.assertEqual(undeclared_routes(served), [])

    def test_every_declared_route_is_served(self):
        served = {
            prefix + route.path
            for served_app in self.apps
            for prefix, route in _iter_leaf_routes(served_app.routes)
        }
        self.assertLessEqual(set(AUTH_LEVELS) - served, SERVED_ELSEWHERE)

    def test_api_key_middleware_is_outermost_on_the_module_level_app(self):
        self.assertEqual(app.user_middleware[0].cls.__name__, "_ApiKeyASGIMiddleware")


class TestBootstrapAuth(CustomTestCase):
    """Bootstrap state must only be readable or mutable with the admin key."""

    admin_key = "bootstrap-admin"
    headers = {"Authorization": f"Bearer {admin_key}"}
    metadata = {
        "tp_rank": 0,
        "transfer_engine_info": {
            "session_id": "seed-session",
            "weights_info_dict": {"weight": [1, 2, 3]},
        },
    }

    def make_servers(self, key):
        # Exercise the real ASGI apps without opening ports or starting threads.
        with patch("threading.Thread.start"):
            engine = EngineInfoBootstrapServer("127.0.0.1", 0, admin_api_key=key)
            encoder = EncoderBootstrapServer(
                "127.0.0.1",
                0,
                urls=["http://encoder:30000"],
                health_check_interval=0,
                admin_api_key=key,
            )
        return engine, encoder

    def test_unauthorized_requests_preserve_registration_state(self):
        for key, status in ((None, 403), (self.admin_key, 401)):
            engine, encoder = self.make_servers(key)
            engine.transfer_engine_info[0] = ("original", {})
            with (
                TestClient(engine._server.config.app) as ec,
                TestClient(encoder.app) as cc,
            ):
                requests = (
                    (ec, "PUT", "/register_transfer_engine_info", self.metadata),
                    (ec, "GET", "/get_transfer_engine_info?rank=0", None),
                    (
                        cc,
                        "POST",
                        "/register_encoder_url",
                        {"url": "http://new-encoder:30000"},
                    ),
                    (
                        cc,
                        "DELETE",
                        "/unregister_encoder_url",
                        {"url": "http://encoder:30000"},
                    ),
                    (cc, "GET", "/list_encoder_urls", None),
                )
                headers_to_reject = [{}, {"Authorization": "Bearer user-key"}]
                if key is None:
                    headers_to_reject.append(self.headers)
                for headers in headers_to_reject:
                    for client, method, path, body in requests:
                        with self.subTest(
                            key_configured=bool(key), path=path, headers=headers
                        ):
                            response = client.request(
                                method, path, json=body, headers=headers
                            )
                            self.assertEqual(response.status_code, status)
                self.assertEqual(ec.get("/health").status_code, 200)
                self.assertEqual(cc.get("/health").status_code, 200)
            with self.subTest(key_configured=bool(key), state="engine"):
                self.assertEqual(engine.get_transfer_engine_info(0), ("original", {}))
            with self.subTest(key_configured=bool(key), state="encoder"):
                self.assertEqual(encoder.list_urls(), ["http://encoder:30000"])

    def test_authorized_registration_round_trip(self):
        engine, encoder = self.make_servers(self.admin_key)
        with TestClient(engine._server.config.app) as ec, TestClient(encoder.app) as cc:
            self.assertEqual(
                ec.put(
                    "/register_transfer_engine_info",
                    json=self.metadata,
                    headers=self.headers,
                ).status_code,
                200,
            )
            self.assertEqual(
                ec.get("/get_transfer_engine_info?rank=0", headers=self.headers).json(),
                {
                    "rank": 0,
                    "remote_instance_transfer_engine_info": [
                        "seed-session",
                        {"weight": [1, 2, 3]},
                    ],
                },
            )
            self.assertEqual(
                cc.post(
                    "/register_encoder_url",
                    json={"url": "http://new-encoder:30000"},
                    headers=self.headers,
                ).status_code,
                200,
            )
            self.assertEqual(
                cc.request(
                    "DELETE",
                    "/unregister_encoder_url",
                    json={"url": "http://encoder:30000"},
                    headers=self.headers,
                ).status_code,
                200,
            )
            self.assertEqual(
                cc.get("/list_encoder_urls", headers=self.headers).json(),
                {"encoder_urls": ["http://new-encoder:30000"]},
            )

    def test_encoder_client_registers_and_unregisters_with_key(self):
        _, encoder = self.make_servers(self.admin_key)
        with (
            TestClient(encoder.app) as client,
            patch.object(
                encoder_http,
                "get_serving",
                return_value=SimpleNamespace(
                    host="127.0.0.1",
                    port=30001,
                    ssl_certfile=None,
                    admin_api_key=self.admin_key,
                ),
            ),
            patch.object(
                encoder_http,
                "get_disagg",
                return_value=SimpleNamespace(
                    encoder_register_urls=["http://testserver"]
                ),
            ),
            patch.object(encoder_http.http_requests, "post", side_effect=client.post),
            patch.object(
                encoder_http.http_requests,
                "delete",
                side_effect=lambda url, **kw: client.request("DELETE", url, **kw),
            ),
            patch("threading.Thread.start", lambda thread: thread.run()),
            patch.object(encoder_http.time, "sleep"),
        ):
            encoder_http._register_encoder_url_with_bootstrap()
            self.assertIn("http://127.0.0.1:30001", encoder.list_urls())
            encoder_http._unregister_encoder_url_from_bootstrap()
            self.assertEqual(encoder.list_urls(), ["http://encoder:30000"])

    def test_weight_registration_client_and_main_server_proxy_use_key(self):
        engine, _ = self.make_servers(self.admin_key)
        worker = object.__new__(transporter.RemoteInstanceWeightTransporter)
        worker.tp_rank = 0
        worker.session_id = "seed-session"
        worker.weight_info = {"weight": [1, 2, 3]}
        with (
            TestClient(engine._server.config.app) as client,
            patch.object(
                transporter,
                "get_parallel",
                return_value=SimpleNamespace(dist_init_addr=None),
            ),
            patch.object(
                transporter,
                "get_model",
                return_value=SimpleNamespace(engine_info_bootstrap_port=30000),
            ),
            patch.object(
                transporter,
                "get_serving",
                return_value=SimpleNamespace(admin_api_key=self.admin_key),
            ),
            patch("requests.put", side_effect=client.put),
        ):
            worker._register_to_engine_info_bootstrap()
            self.assertEqual(
                engine.get_transfer_engine_info(0),
                (worker.session_id, worker.weight_info),
            )
            with (
                patch.object(
                    http_server,
                    "get_serving",
                    return_value=SimpleNamespace(admin_api_key=self.admin_key),
                ),
                patch.object(
                    http_server._global_state,
                    "tokenizer_manager",
                    SimpleNamespace(
                        server_args=SimpleNamespace(
                            engine_info_bootstrap_url="http://testserver"
                        )
                    ),
                ),
                patch("requests.get", side_effect=client.get),
            ):
                self.assertEqual(
                    asyncio.run(http_server.remote_instance_transfer_engine_info(0)),
                    {
                        "rank": 0,
                        "remote_instance_transfer_engine_info": [
                            worker.session_id,
                            worker.weight_info,
                        ],
                    },
                )


if __name__ == "__main__":
    unittest.main()
