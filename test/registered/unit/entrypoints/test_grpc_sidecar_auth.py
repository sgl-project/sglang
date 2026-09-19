import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp.test_utils import TestClient, TestServer

from sglang.srt.entrypoints import grpc_server
from sglang.srt.managers.io_struct import ProfileReqType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGrpcSidecarAuth(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, api_key=None, admin_api_key=None):
        app = grpc_server._create_sidecar_app(
            api_key=api_key, admin_api_key=admin_api_key
        )
        manager = SimpleNamespace(
            send_communicator_req=AsyncMock(
                return_value=[SimpleNamespace(success=True)]
            )
        )
        grpc_server._add_admin_routes(app, manager)
        grpc_server._add_metrics_routes(app)
        client = TestClient(TestServer(app))
        self.addAsyncCleanup(client.close)
        await client.start_server()
        return client, manager

    async def test_profiling_auth_matrix(self):
        for api_key, admin_key, accepted_key in (
            (None, None, None),
            ("api-secret", None, "api-secret"),
            (None, "admin-secret", "admin-secret"),
            ("api-secret", "admin-secret", "admin-secret"),
        ):
            client, manager = await self.make_client(api_key, admin_key)
            for path, req_type in (
                ("/start_profile", ProfileReqType.START_PROFILE),
                ("/stop_profile", ProfileReqType.STOP_PROFILE),
            ):
                for authorization in (
                    None,
                    "Bearer wrong-secret",
                    "Basic admin-secret",
                    "Bearer api-secret",
                    "Bearer admin-secret",
                ):
                    with self.subTest(
                        api_key=api_key,
                        admin_key=admin_key,
                        path=path,
                        authorization=authorization,
                    ):
                        manager.send_communicator_req.reset_mock()
                        headers = (
                            {"Authorization": authorization} if authorization else {}
                        )
                        response = await client.post(path, json={}, headers=headers)
                        allowed = accepted_key is None or authorization == (
                            f"Bearer {accepted_key}"
                        )
                        self.assertEqual(response.status, 200 if allowed else 401)
                        if allowed:
                            await response.read()
                            manager.send_communicator_req.assert_awaited_once()
                            args, kwargs = manager.send_communicator_req.call_args
                            self.assertEqual(args[0].req_type, req_type)
                            self.assertEqual(args[1], "profile_communicator")
                            self.assertEqual(kwargs, {"timeout": 600.0})
                        else:
                            self.assertEqual(
                                await response.json(), {"error": "Unauthorized"}
                            )
                            manager.send_communicator_req.assert_not_awaited()

    async def test_auth_runs_before_profile_body_parsing(self):
        client, manager = await self.make_client(admin_api_key="admin-secret")
        response = await client.post("/start_profile", data="not-json")
        self.assertEqual(response.status, 401)
        await response.read()
        manager.send_communicator_req.assert_not_awaited()

        response = await client.post(
            "/start_profile",
            data="not-json",
            headers={"Authorization": "Bearer admin-secret"},
        )
        self.assertEqual(response.status, 400)
        await response.read()
        manager.send_communicator_req.assert_not_awaited()

    async def test_metrics_remain_public(self):
        client, manager = await self.make_client("api-secret", "admin-secret")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"PROMETHEUS_MULTIPROC_DIR": directory}),
        ):
            for headers in ({}, {"Authorization": "Bearer wrong-secret"}):
                response = await client.get("/metrics", headers=headers)
                self.assertEqual(response.status, 200)
                self.assertIn("openmetrics", response.headers["Content-Type"])
                self.assertIn("# EOF", await response.text())
        manager.send_communicator_req.assert_not_awaited()

    async def test_serve_grpc_passes_configured_keys_to_sidecar(self):
        manager = SimpleNamespace(
            send_communicator_req=AsyncMock(
                return_value=[SimpleNamespace(success=True)]
            )
        )
        cfg = SimpleNamespace(
            api_key="api-secret",
            admin_api_key="admin-secret",
            enable_metrics=False,
            host="127.0.0.1",
            port=30000,
            smg_http_sidecar_port=30001,
        )
        runner = SimpleNamespace(cleanup=AsyncMock())
        clients = []

        async def start_sidecar(host, port, app):
            self.assertEqual((host, port), (cfg.host, cfg.smg_http_sidecar_port))
            client = TestClient(TestServer(app))
            self.addAsyncCleanup(client.close)
            await client.start_server()
            clients.append(client)
            return runner

        async def serve(server_args, model_info, *, on_request_manager_ready):
            await on_request_manager_ready(manager, server_args, {})
            client = clients[0]
            for token, status in (
                (None, 401),
                ("api-secret", 401),
                ("admin-secret", 200),
            ):
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                response = await client.post("/stop_profile", headers=headers)
                self.assertEqual(response.status, status)
                await response.read()

        modules = {
            name: ModuleType(name)
            for name in (
                "smg_grpc_servicer",
                "smg_grpc_servicer.sglang",
                "smg_grpc_servicer.sglang.server",
            )
        }
        modules["smg_grpc_servicer.sglang.server"].serve_grpc = serve
        with (
            patch.dict(sys.modules, modules),
            patch.object(grpc_server, "resolving_view", return_value=cfg),
            patch.object(
                grpc_server, "_start_sidecar_server", side_effect=start_sidecar
            ),
        ):
            await grpc_server.serve_grpc(cfg)
        manager.send_communicator_req.assert_awaited_once()
        runner.cleanup.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
