"""Unit tests for srt/entrypoints/http_server.py build_app / init_app_state.

These tests exercise the embedding factory (issue #31356): building a fresh
FastAPI app bound to an in-process engine, without launching a server or
loading model weights.
"""

import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _configure_single_tokenizer_app,
    _setup_and_run_http_server,
    build_app,
    get_global_state,
    init_app_state,
    set_global_state,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_server_args(**overrides):
    """Minimal stand-in for the fields build_app reads."""
    fields = dict(
        tokenizer_worker_num=1,
        api_key=None,
        admin_api_key=None,
        enable_metrics=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _fake_engine(server_args):
    tokenizer_manager = MagicMock(name="tokenizer_manager")
    tokenizer_manager.server_args = server_args
    return SimpleNamespace(
        server_args=server_args,
        tokenizer_manager=tokenizer_manager,
        template_manager=MagicMock(name="template_manager"),
        _scheduler_init_result=SimpleNamespace(
            scheduler_infos=[{"max_req_input_len": 128}]
        ),
    )


def _route_keys(app):
    """(path, sorted methods) for every API route on the app."""
    keys = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        methods = getattr(route, "methods", None)
        keys.add((path, tuple(sorted(methods)) if methods else None))
    return keys


class HttpServerAppFactoryTestBase(CustomTestCase):
    def setUp(self):
        # init_app_state publishes to the process-global state; restore it so
        # tests do not leak into each other. Start each test with no engine
        # bound, because init_app_state refuses to replace a different engine.
        saved_global_state = get_global_state()
        self.addCleanup(set_global_state, saved_global_state)
        set_global_state(None)


class TestBuildApp(HttpServerAppFactoryTestBase):
    def test_returns_fresh_app_with_module_routes(self):
        new_app = build_app(_fake_server_args())

        self.assertIsNot(new_app, http_server.app)
        self.assertIsNot(new_app.state, http_server.app.state)
        # Every route of the module-level app must be served by the fresh app
        # (the fresh app additionally has its own doc routes).
        missing = _route_keys(http_server.app) - _route_keys(new_app)
        self.assertEqual(missing, set())

    def test_app_serves_requests_without_engine_state(self):
        client = TestClient(build_app(_fake_server_args()))
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "SGLang is running")

    def test_apps_are_independent(self):
        app_a = build_app(_fake_server_args())
        app_b = build_app(_fake_server_args())
        self.assertIsNot(app_a, app_b)

        # A route registered on one app must not appear on the other, nor on
        # the module-level app.
        @app_a.get("/___factory_test_only_a")
        def _only_a():
            return "a"

        self.assertEqual(
            TestClient(app_a).get("/___factory_test_only_a").status_code, 200
        )
        self.assertEqual(
            TestClient(app_b).get("/___factory_test_only_a").status_code, 404
        )
        self.assertNotIn(
            "/___factory_test_only_a",
            {getattr(r, "path", None) for r in http_server.app.routes},
        )

        # Snapshot semantics: a route registered on the module-level app after
        # build_app must not leak into an already-built app.
        @http_server.app.get("/___factory_test_late_module_route")
        def _late():
            return "late"

        late_route = http_server.app.router.routes[-1]
        self.addCleanup(http_server.app.router.routes.remove, late_route)
        self.assertEqual(
            TestClient(app_a).get("/___factory_test_late_module_route").status_code,
            404,
        )

    def test_rejects_multi_tokenizer_mode(self):
        with self.assertRaises(ValueError):
            build_app(_fake_server_args(tokenizer_worker_num=2))

    def test_api_key_middleware_plumbed_from_server_args(self):
        client = TestClient(build_app(_fake_server_args(api_key="secret")))
        self.assertEqual(client.get("/ping").status_code, 401)
        self.assertEqual(
            client.get("/ping", headers={"Authorization": "Bearer wrong"}).status_code,
            401,
        )
        self.assertEqual(
            client.get("/ping", headers={"Authorization": "Bearer secret"}).status_code,
            200,
        )

        # And no auth middleware when no keys are configured.
        client_open = TestClient(build_app(_fake_server_args()))
        self.assertEqual(client_open.get("/ping").status_code, 200)


class TestInitAppState(HttpServerAppFactoryTestBase):
    def test_binds_engine_to_state_and_publishes_global_state(self):
        server_args = _fake_server_args()
        engine = _fake_engine(server_args)
        app = build_app(server_args)

        self.assertIsNone(app.state.global_state)
        init_app_state(engine, app.state)

        self.assertIs(
            app.state.global_state.tokenizer_manager, engine.tokenizer_manager
        )
        self.assertIs(app.state.global_state.template_manager, engine.template_manager)
        self.assertEqual(
            app.state.global_state.scheduler_info, {"max_req_input_len": 128}
        )
        self.assertIs(app.state.server_args, server_args)
        # Native endpoints still read the process-global state; init_app_state
        # must publish there as well.
        self.assertIs(get_global_state().tokenizer_manager, engine.tokenizer_manager)

    def test_second_app_state_stays_independent(self):
        server_args = _fake_server_args()
        app_a = build_app(server_args)
        app_b = build_app(server_args)
        init_app_state(_fake_engine(server_args), app_a.state)
        self.assertIsNotNone(app_a.state.global_state)
        self.assertIsNone(app_b.state.global_state)

    def test_lifespan_creates_serving_handlers_from_bound_engine(self):
        """The lifespan must consume the per-app binding from init_app_state.

        Guards against regressing the lifespan to read only the process-global
        state: a conflicting engine is published globally after init_app_state,
        and the handlers must still bind the app's own engine.
        """
        # ServerArgs(model_path="dummy") is the sanctioned lightweight fixture;
        # skip_server_warmup avoids the HTTP self-warmup in the warmup thread.
        server_args = ServerArgs(model_path="dummy", skip_server_warmup=True)
        engine = _fake_engine(server_args)
        app = build_app(server_args)
        init_app_state(engine, app.state)

        other_engine = _fake_engine(server_args)
        set_global_state(
            http_server._GlobalState(
                tokenizer_manager=other_engine.tokenizer_manager,
                template_manager=other_engine.template_manager,
                scheduler_info={"max_req_input_len": 128},
            )
        )

        with TestClient(app) as client:
            self.assertEqual(client.get("/ping").status_code, 200)
            self.assertIs(
                app.state.openai_serving_completion.tokenizer_manager,
                engine.tokenizer_manager,
            )
            self.assertIs(
                app.state.openai_serving_tokenize.tokenizer_manager,
                engine.tokenizer_manager,
            )


class TestFactoryAppWarmup(HttpServerAppFactoryTestBase):
    def test_no_http_warmup_by_default(self):
        """Bug regression: a factory built app used to run the standalone
        warmup thread, which sends HTTP requests carrying server_args.api_key
        to server_args.host and port (an address the embedding host usually
        does not listen on) and kills the whole process when they fail.

        By default the factory app must send no warmup HTTP requests, must
        never kill the process, and must still mark the engine as ready.
        """
        server_args = ServerArgs(model_path="dummy")
        self.assertFalse(server_args.skip_server_warmup)
        engine = _fake_engine(server_args)
        app = build_app(server_args)
        init_app_state(engine, app.state)

        with (
            patch.object(http_server.time, "sleep"),
            patch.object(
                http_server.requests,
                "get",
                side_effect=requests.exceptions.ConnectionError(),
            ) as http_get,
            patch.object(http_server, "kill_process_tree") as kill_stub,
        ):
            with TestClient(app) as client:
                self.assertEqual(client.get("/ping").status_code, 200)

        http_get.assert_not_called()
        kill_stub.assert_not_called()
        self.assertEqual(engine.tokenizer_manager.server_status, ServerStatus.Up)

    def test_http_warmup_runs_when_opted_in(self):
        """build_app(server_args, warmup=True) must run the HTTP warmup, and
        must disable the kill on failure behavior of the standalone path."""
        server_args = ServerArgs(model_path="dummy")
        engine = _fake_engine(server_args)
        app = build_app(server_args, warmup=True)
        init_app_state(engine, app.state)

        with patch.object(
            http_server, "_execute_server_warmup", return_value=True
        ) as warmup_stub:
            with TestClient(app):
                pass

        warmup_stub.assert_called_once_with(server_args, kill_process_on_failure=False)

    def test_warmup_failure_can_skip_process_kill(self):
        """_execute_server_warmup(kill_process_on_failure=False) must report
        failure without killing the process tree of the caller."""
        server_args = ServerArgs(model_path="dummy")
        with (
            patch.object(http_server.time, "sleep"),
            patch.object(
                http_server.requests,
                "get",
                side_effect=requests.exceptions.ConnectionError(),
            ),
            patch.object(http_server, "kill_process_tree") as kill_stub,
        ):
            result = http_server._execute_server_warmup(
                server_args, kill_process_on_failure=False
            )
        self.assertFalse(result)
        kill_stub.assert_not_called()


class TestFactoryAppMetrics(HttpServerAppFactoryTestBase):
    def test_build_app_twice_with_metrics_enabled(self):
        """Bug regression: the second build_app call with metrics enabled used
        to crash, because the HTTP tracking collectors were registered in the
        process wide prometheus registry once per call under fixed names."""
        build_app(_fake_server_args(enable_metrics=True))
        build_app(_fake_server_args(enable_metrics=True))

    def test_metrics_endpoint_served_by_built_app_not_module_app(self):
        """The lifespan must add the /metrics route to the app it is starting.

        Guards the fix where the lifespan used to add /metrics to the module
        level app even when it was starting a factory built app, so the built
        app returned 404 for /metrics and the module level app grew a stray
        route.
        """
        prometheus_dir = tempfile.TemporaryDirectory()
        self.addCleanup(prometheus_dir.cleanup)
        saved_env = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

        def _restore_env():
            if saved_env is None:
                os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            else:
                os.environ["PROMETHEUS_MULTIPROC_DIR"] = saved_env

        self.addCleanup(_restore_env)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_dir.name

        server_args = ServerArgs(
            model_path="dummy", skip_server_warmup=True, enable_metrics=True
        )
        engine = _fake_engine(server_args)
        app = build_app(server_args)
        init_app_state(engine, app.state)

        module_paths_before = {getattr(r, "path", None) for r in http_server.app.routes}
        self.assertNotIn("/metrics", module_paths_before)

        with TestClient(app) as client:
            self.assertEqual(client.get("/metrics").status_code, 200)

        module_paths_after = {getattr(r, "path", None) for r in http_server.app.routes}
        self.assertNotIn("/metrics", module_paths_after)


class TestFactoryAppMountedAuth(HttpServerAppFactoryTestBase):
    def test_health_exempt_from_auth_under_mount_prefix(self):
        """Bug regression: the probe exemption in the api key middleware
        matched the full request path, so an app mounted under a prefix
        returned 401 for /health and /metrics and probes needed the api key.
        """
        factory_app = build_app(_fake_server_args(api_key="secret"))
        host = FastAPI()
        host.mount("/sglang", factory_app)
        client = TestClient(host, raise_server_exceptions=False)

        # Probe endpoints must pass authentication without a key. They may
        # still fail later for other reasons (no engine is bound here), so
        # only the auth outcome is asserted.
        self.assertNotIn(client.get("/sglang/health").status_code, (401, 403))
        self.assertNotIn(client.get("/sglang/health_generate").status_code, (401, 403))

        # Every other endpoint still requires the key under the prefix.
        self.assertEqual(client.get("/sglang/ping").status_code, 401)
        self.assertEqual(
            client.get(
                "/sglang/ping", headers={"Authorization": "Bearer secret"}
            ).status_code,
            200,
        )

        # A non mounted app keeps the same behavior.
        direct = TestClient(factory_app, raise_server_exceptions=False)
        self.assertNotIn(direct.get("/health").status_code, (401, 403))
        self.assertEqual(direct.get("/ping").status_code, 401)


class TestEngineBindingGuards(HttpServerAppFactoryTestBase):
    def test_binding_a_second_engine_raises(self):
        """Bug regression: a second init_app_state call used to silently
        repoint the native endpoints of an already bound app at the other
        engine while its chat and completion handlers kept the first engine.
        """
        server_args = _fake_server_args()
        app_a = build_app(server_args)
        app_b = build_app(server_args)
        engine_a = _fake_engine(server_args)
        engine_b = _fake_engine(server_args)

        init_app_state(engine_a, app_a.state)
        with self.assertRaises(RuntimeError):
            init_app_state(engine_b, app_b.state)

        # Binding the same engine again is allowed.
        init_app_state(engine_a, app_a.state)

    def test_lifespan_raises_when_factory_app_was_never_bound(self):
        """Bug regression: a factory built app that was never passed to
        init_app_state used to silently fall back to the process global state
        and bind whatever engine another caller had published."""
        server_args = ServerArgs(model_path="dummy", skip_server_warmup=True)
        app = build_app(server_args)

        # A foreign engine is published globally, as another server in the
        # same process would do.
        other_engine = _fake_engine(server_args)
        set_global_state(
            http_server._GlobalState(
                tokenizer_manager=other_engine.tokenizer_manager,
                template_manager=other_engine.template_manager,
                scheduler_info={"max_req_input_len": 128},
            )
        )

        with self.assertRaises(RuntimeError):
            with TestClient(app):
                pass

    def test_init_app_state_server_args_must_match_engine(self):
        """The optional server_args parameter of init_app_state must match the
        arguments the engine was built with; a mismatch would silently imply a
        configuration hook that does not exist."""
        server_args = _fake_server_args()
        engine = _fake_engine(server_args)
        app = build_app(server_args)

        with self.assertRaises(ValueError):
            init_app_state(
                engine,
                app.state,
                server_args=_fake_server_args(api_key="different"),
            )

        init_app_state(engine, app.state, server_args=server_args)
        self.assertIs(app.state.server_args, server_args)


class TestLaunchServerSharesFactoryPath(HttpServerAppFactoryTestBase):
    def test_launch_server_and_build_app_use_same_configuration_helper(self):
        """launch_server and build_app must configure apps through one code path."""
        helper_name = _configure_single_tokenizer_app.__name__
        self.assertIn(helper_name, inspect.getsource(_setup_and_run_http_server))
        self.assertIn(helper_name, inspect.getsource(build_app))


if __name__ == "__main__":
    unittest.main()
