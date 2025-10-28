"""
Unit tests for startup sequence logic in sglang_router.

These tests focus on testing the startup sequence logic in isolation,
including router initialization, configuration validation, and startup flow.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from sglang_router.launch_router import RouterArgs, launch_router
from sglang_router.router import policy_from_str


# Local helper mirroring the router logger setup used in production
def setup_logger():
    logger = logging.getLogger("router")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[Router (Python)] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


from sglang_router_rs import PolicyType


class TestSetupLogger:
    """Test logger setup functionality."""

    def test_setup_logger_returns_logger(self):
        """Test that setup_logger returns a logger instance."""
        logger = setup_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "router"
        assert logger.level == logging.INFO

    def test_setup_logger_has_handler(self):
        """Test that setup_logger configures a handler."""
        logger = setup_logger()

        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)

    def test_setup_logger_has_formatter(self):
        """Test that setup_logger configures a formatter."""
        logger = setup_logger()

        handler = logger.handlers[0]
        formatter = handler.formatter

        assert formatter is not None
        assert "[Router (Python)]" in formatter._fmt

    def test_setup_logger_multiple_calls(self):
        """Test that multiple calls to setup_logger work correctly."""
        logger1 = setup_logger()
        logger2 = setup_logger()

        # Should return the same logger instance
        assert logger1 is logger2


class TestPolicyFromStr:
    """Test policy string to enum conversion in startup context."""

    def test_policy_conversion_in_startup(self):
        """Test policy conversion during startup sequence."""
        # Test all valid policies
        policies = ["random", "round_robin", "cache_aware", "power_of_two"]
        expected_enums = [
            PolicyType.Random,
            PolicyType.RoundRobin,
            PolicyType.CacheAware,
            PolicyType.PowerOfTwo,
        ]

        for policy_str, expected_enum in zip(policies, expected_enums):
            result = policy_from_str(policy_str)
            assert result == expected_enum

    def test_invalid_policy_in_startup(self):
        """Test handling of invalid policy during startup."""
        with pytest.raises(KeyError):
            policy_from_str("invalid_policy")


class TestRouterInitialization:
    """Test router initialization logic."""

    def test_router_initialization_basic(self):
        """Test basic router initialization."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"],
            policy="cache_aware",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}

            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                # capture needed fields from RouterArgs
                captured_args.update(
                    dict(
                        host=router_args.host,
                        port=router_args.port,
                        worker_urls=router_args.worker_urls,
                        policy=policy_from_str(router_args.policy),
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify Router.from_args was called and captured fields match
            router_mod.from_args.assert_called_once()
            assert captured_args["host"] == "127.0.0.1"
            assert captured_args["port"] == 30000
            assert captured_args["worker_urls"] == ["http://worker1:8000"]
            assert captured_args["policy"] == PolicyType.CacheAware

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_pd_mode(self):
        """Test router initialization in PD mode."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000)],
            decode_urls=["http://decode1:8001"],
            policy="power_of_two",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        pd_disaggregation=router_args.pd_disaggregation,
                        prefill_urls=router_args.prefill_urls,
                        decode_urls=router_args.decode_urls,
                        policy=policy_from_str(router_args.policy),
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify Router.from_args was called with PD parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["pd_disaggregation"] is True
            assert captured_args["prefill_urls"] == [("http://prefill1:8000", 9000)]
            assert captured_args["decode_urls"] == ["http://decode1:8001"]
            assert captured_args["policy"] == PolicyType.PowerOfTwo

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_service_discovery(self):
        """Test router initialization with service discovery."""
        args = RouterArgs(
            service_discovery=True,
            selector={"app": "worker", "env": "prod"},
            service_discovery_port=8080,
            service_discovery_namespace="default",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        service_discovery=router_args.service_discovery,
                        selector=router_args.selector,
                        service_discovery_port=router_args.service_discovery_port,
                        service_discovery_namespace=router_args.service_discovery_namespace,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify Router.from_args was called with service discovery parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["service_discovery"] is True
            assert captured_args["selector"] == {"app": "worker", "env": "prod"}
            assert captured_args["service_discovery_port"] == 8080
            assert captured_args["service_discovery_namespace"] == "default"

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_retry_config(self):
        """Test router initialization with retry configuration."""
        args = RouterArgs(
            retry_max_retries=3,
            retry_initial_backoff_ms=100,
            retry_max_backoff_ms=10000,
            retry_backoff_multiplier=2.0,
            retry_jitter_factor=0.1,
            disable_retries=False,
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        retry_max_retries=router_args.retry_max_retries,
                        retry_initial_backoff_ms=router_args.retry_initial_backoff_ms,
                        retry_max_backoff_ms=router_args.retry_max_backoff_ms,
                        retry_backoff_multiplier=router_args.retry_backoff_multiplier,
                        retry_jitter_factor=router_args.retry_jitter_factor,
                        disable_retries=router_args.disable_retries,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify router was created with retry parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["retry_max_retries"] == 3
            assert captured_args["retry_initial_backoff_ms"] == 100
            assert captured_args["retry_max_backoff_ms"] == 10000
            assert captured_args["retry_backoff_multiplier"] == 2.0
            assert captured_args["retry_jitter_factor"] == 0.1
            assert captured_args["disable_retries"] is False

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_circuit_breaker_config(self):
        """Test router initialization with circuit breaker configuration."""
        args = RouterArgs(
            cb_failure_threshold=5,
            cb_success_threshold=2,
            cb_timeout_duration_secs=30,
            cb_window_duration_secs=60,
            disable_circuit_breaker=False,
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        cb_failure_threshold=router_args.cb_failure_threshold,
                        cb_success_threshold=router_args.cb_success_threshold,
                        cb_timeout_duration_secs=router_args.cb_timeout_duration_secs,
                        cb_window_duration_secs=router_args.cb_window_duration_secs,
                        disable_circuit_breaker=router_args.disable_circuit_breaker,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify router was created with circuit breaker parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["cb_failure_threshold"] == 5
            assert captured_args["cb_success_threshold"] == 2
            assert captured_args["cb_timeout_duration_secs"] == 30
            assert captured_args["cb_window_duration_secs"] == 60
            assert captured_args["disable_circuit_breaker"] is False

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_rate_limiting_config(self):
        """Test router initialization with rate limiting configuration."""
        args = RouterArgs(
            max_concurrent_requests=512,
            queue_size=200,
            queue_timeout_secs=120,
            rate_limit_tokens_per_second=100,
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        max_concurrent_requests=router_args.max_concurrent_requests,
                        queue_size=router_args.queue_size,
                        queue_timeout_secs=router_args.queue_timeout_secs,
                        rate_limit_tokens_per_second=router_args.rate_limit_tokens_per_second,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify router was created with rate limiting parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["max_concurrent_requests"] == 512
            assert captured_args["queue_size"] == 200
            assert captured_args["queue_timeout_secs"] == 120
            assert captured_args["rate_limit_tokens_per_second"] == 100

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_health_check_config(self):
        """Test router initialization with health check configuration."""
        args = RouterArgs(
            health_failure_threshold=2,
            health_success_threshold=1,
            health_check_timeout_secs=3,
            health_check_interval_secs=30,
            health_check_endpoint="/healthz",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        health_failure_threshold=router_args.health_failure_threshold,
                        health_success_threshold=router_args.health_success_threshold,
                        health_check_timeout_secs=router_args.health_check_timeout_secs,
                        health_check_interval_secs=router_args.health_check_interval_secs,
                        health_check_endpoint=router_args.health_check_endpoint,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify router was created with health check parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["health_failure_threshold"] == 2
            assert captured_args["health_success_threshold"] == 1
            assert captured_args["health_check_timeout_secs"] == 3
            assert captured_args["health_check_interval_secs"] == 30
            assert captured_args["health_check_endpoint"] == "/healthz"

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_prometheus_config(self):
        """Test router initialization with Prometheus configuration."""
        args = RouterArgs(prometheus_port=29000, prometheus_host="127.0.0.1")

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        prometheus_port=router_args.prometheus_port,
                        prometheus_host=router_args.prometheus_host,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify router was created with Prometheus parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["prometheus_port"] == 29000
            assert captured_args["prometheus_host"] == "127.0.0.1"

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_cors_config(self):
        """Test router initialization with CORS configuration."""
        args = RouterArgs(
            cors_allowed_origins=["http://localhost:3000", "https://example.com"]
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(cors_allowed_origins=router_args.cors_allowed_origins)
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify router was created with CORS parameters
            router_mod.from_args.assert_called_once()
            assert captured_args["cors_allowed_origins"] == [
                "http://localhost:3000",
                "https://example.com",
            ]

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_tokenizer_config(self):
        """Test router initialization with tokenizer configuration."""
        # Note: model_path and tokenizer_path are not available in current RouterArgs
        pytest.skip("Tokenizer configuration not available in current implementation")


class TestStartupValidation:
    """Test startup validation logic."""

    def test_pd_mode_validation_during_startup(self):
        """Test PD mode validation during startup."""
        # PD mode without URLs should fail
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=False,
        )

        with pytest.raises(
            ValueError, match="PD disaggregation mode requires --prefill"
        ):
            launch_router(args)

    def test_pd_mode_with_service_discovery_validation(self):
        """Test PD mode with service discovery validation during startup."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=True,
        )

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()

            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            result = launch_router(args)

            # Should create router instance
            router_mod.from_args.assert_called_once()

    def test_policy_warning_during_startup(self):
        """Test policy warning during startup in PD mode."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy="power_of_two",
            decode_policy="round_robin",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # The policy messages are emitted by router_args logger
            with patch("sglang_router.router_args.logger") as mock_logger:
                result = launch_router(args)

                # Should log warning about policy usage
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]
                assert (
                    "Both --prefill-policy and --decode-policy are specified"
                    in warning_call
                )

                # Should create router instance
                router_mod.from_args.assert_called_once()

    def test_policy_info_during_startup(self):
        """Test policy info logging during startup in PD mode."""
        # Test with only prefill policy specified
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy="power_of_two",
            decode_policy=None,
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # The policy messages are emitted by router_args logger
            with patch("sglang_router.router_args.logger") as mock_logger:
                result = launch_router(args)

                # Should log info about policy usage
                mock_logger.info.assert_called_once()
                info_call = mock_logger.info.call_args[0][0]
                assert "Using --prefill-policy 'power_of_two'" in info_call
                assert "and --policy 'cache_aware'" in info_call

                # Should create router instance
                router_mod.from_args.assert_called_once()

    def test_policy_info_decode_only_during_startup(self):
        """Test policy info logging during startup with only decode policy specified."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy=None,
            decode_policy="round_robin",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # The policy messages are emitted by router_args logger
            with patch("sglang_router.router_args.logger") as mock_logger:
                result = launch_router(args)

                # Should log info about policy usage
                mock_logger.info.assert_called_once()
                info_call = mock_logger.info.call_args[0][0]
                assert "Using --policy 'cache_aware'" in info_call
                assert "and --decode-policy 'round_robin'" in info_call

                # Should create router instance
                router_mod.from_args.assert_called_once()


class TestStartupErrorHandling:
    """Test startup error handling logic."""

    def test_router_creation_error_handling(self):
        """Test error handling when router creation fails."""
        args = RouterArgs(
            host="127.0.0.1", port=30000, worker_urls=["http://worker1:8000"]
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            # Simulate router creation failure in from_args
            router_mod.from_args = MagicMock(
                side_effect=Exception("Router creation failed")
            )

            with patch("sglang_router.launch_router.logger") as mock_logger:
                with pytest.raises(Exception, match="Router creation failed"):
                    launch_router(args)

                # Should log error
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert "Error starting router: Router creation failed" in error_call

    def test_router_start_error_handling(self):
        """Test error handling when router start fails."""
        args = RouterArgs(
            host="127.0.0.1", port=30000, worker_urls=["http://worker1:8000"]
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # Simulate router start failure
            mock_router_instance.start.side_effect = Exception("Router start failed")

            with patch("sglang_router.launch_router.logger") as mock_logger:
                with pytest.raises(Exception, match="Router start failed"):
                    launch_router(args)

                # Should log error
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert "Error starting router: Router start failed" in error_call


# --- Added unit tests for Router wrapper and launch_server helpers ---


def _install_sglang_stubs(monkeypatch):
    """Install lightweight stubs for sglang.srt to avoid heavy deps during unit tests."""
    import sys
    import types

    sglang_mod = types.ModuleType("sglang")
    srt_mod = types.ModuleType("sglang.srt")
    entry_mod = types.ModuleType("sglang.srt.entrypoints")
    http_server_mod = types.ModuleType("sglang.srt.entrypoints.http_server")
    server_args_mod = types.ModuleType("sglang.srt.server_args")
    utils_mod = types.ModuleType("sglang.srt.utils")

    def launch_server(_args):
        return None

    class ServerArgs:
        # Minimal fields used by launch_server_process
        def __init__(self):
            self.port = 0
            self.base_gpu_id = 0
            self.dp_size = 1
            self.tp_size = 1

        @staticmethod
        def add_cli_args(_parser):
            return None

        @staticmethod
        def from_cli_args(_args):
            sa = ServerArgs()
            if hasattr(_args, "dp_size"):
                sa.dp_size = _args.dp_size
            if hasattr(_args, "tp_size"):
                sa.tp_size = _args.tp_size
            if hasattr(_args, "host"):
                sa.host = _args.host
            else:
                sa.host = "127.0.0.1"
            return sa

    def is_port_available(_port: int) -> bool:
        return True

    http_server_mod.launch_server = launch_server
    server_args_mod.ServerArgs = ServerArgs
    utils_mod.is_port_available = is_port_available

    # Also stub external deps imported at module top-level
    def _dummy_get(*_a, **_k):
        raise NotImplementedError

    requests_stub = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(RequestException=Exception), get=_dummy_get
    )
    setproctitle_stub = types.SimpleNamespace(setproctitle=lambda *_a, **_k: None)

    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    monkeypatch.setitem(sys.modules, "setproctitle", setproctitle_stub)

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entry_mod)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.http_server", http_server_mod
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)


def test_router_defaults_and_start(monkeypatch):
    """Router wrapper: defaults normalization and start() call.

    Mocks the Rust-backed _Router to avoid native deps.
    """
    from sglang_router import router as router_mod

    captured = {}

    class FakeRouter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(router_mod, "_Router", FakeRouter, raising=True)

    from sglang_router.router_args import RouterArgs as _RouterArgs

    Router = router_mod.Router
    args = _RouterArgs(
        worker_urls=["http://w1:8000"],
        policy="round_robin",
        selector=None,
        prefill_selector=None,
        decode_selector=None,
        cors_allowed_origins=None,
    )

    r = Router.from_args(args)

    # Defaults preserved/normalized by Router.from_args
    assert captured["selector"] is None
    assert captured["prefill_selector"] is None
    assert captured["decode_selector"] is None
    assert captured["cors_allowed_origins"] is None
    assert captured["worker_urls"] == ["http://w1:8000"]
    from sglang_router_rs import PolicyType

    assert captured["policy"] == PolicyType.RoundRobin

    r.start()
    assert captured.get("started") is True


def test_find_available_ports_and_wait_health(monkeypatch):
    """launch_server helpers: port finding and health waiting with transient error."""
    _install_sglang_stubs(monkeypatch)
    import importlib

    ls = importlib.import_module("sglang_router.launch_server")

    # Deterministic increments
    monkeypatch.setattr(ls.random, "randint", lambda a, b: 100)
    ports = ls.find_available_ports(30000, 3)
    assert ports == [30000, 30100, 30200]

    calls = {"n": 0}

    class Ok:
        status_code = 200

    def fake_get(_url, timeout=5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ls.requests.exceptions.RequestException("boom")
        return Ok()

    monkeypatch.setattr(ls.requests, "get", fake_get)
    monkeypatch.setattr(ls.time, "sleep", lambda _s: None)
    base = {"t": 0.0}
    monkeypatch.setattr(
        ls.time,
        "perf_counter",
        lambda: (base.__setitem__("t", base["t"] + 0.1) or base["t"]),
    )

    assert ls.wait_for_server_health("127.0.0.1", 12345, timeout=1)


def test_launch_server_process_and_cleanup(monkeypatch):
    """launch_server: process creation args and cleanup SIGTERM/SIGKILL logic."""
    _install_sglang_stubs(monkeypatch)
    import importlib

    ls = importlib.import_module("sglang_router.launch_server")

    created = {}

    class FakeProcess:
        def __init__(self, target, args):
            created["target"] = target
            created["args"] = args
            self.pid = 4242
            self._alive = True

        def start(self):
            created["started"] = True

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(ls.mp, "Process", FakeProcess)

    import sys as _sys

    SA = _sys.modules["sglang.srt.server_args"].ServerArgs
    sa = SA()
    sa.tp_size = 2

    proc = ls.launch_server_process(sa, worker_port=31001, dp_id=3)
    assert created.get("started") is True
    targ, targ_args = created["target"], created["args"]
    assert targ is ls.run_server
    passed_sa = targ_args[0]
    assert passed_sa.port == 31001
    assert passed_sa.base_gpu_id == 3 * 2
    assert passed_sa.dp_size == 1

    # cleanup_processes
    p1 = FakeProcess(target=None, args=())
    p1._alive = False
    p2 = FakeProcess(target=None, args=())
    p2._alive = True

    calls = []

    def fake_killpg(pid, sig):
        calls.append((pid, sig))

    monkeypatch.setattr(ls.os, "killpg", fake_killpg)

    ls.cleanup_processes([p1, p2])

    import signal as _sig

    assert (p1.pid, _sig.SIGTERM) in calls and (p2.pid, _sig.SIGTERM) in calls
    assert (p2.pid, _sig.SIGKILL) in calls

    def test_validation_error_handling(self):
        """Test error handling when validation fails."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=False,
        )

        with patch("sglang_router.launch_router.logger") as mock_logger:

            with pytest.raises(
                ValueError, match="PD disaggregation mode requires --prefill"
            ):
                launch_router(args)

            # Should log error for validation failures
            mock_logger.error.assert_called_once()


class TestStartupFlow:
    """Test complete startup flow."""

    def test_complete_startup_flow_basic(self):
        """Test complete startup flow for basic configuration."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000", "http://worker2:8000"],
            policy="cache_aware",
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.5,
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            result = launch_router(args)

            # Verify complete flow
            router_mod.from_args.assert_called_once()
            mock_router_instance.start.assert_called_once()

    def test_complete_startup_flow_pd_mode(self):
        """Test complete startup flow for PD mode configuration."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[
                ("http://prefill1:8000", 9000),
                ("http://prefill2:8000", None),
            ],
            decode_urls=["http://decode1:8001", "http://decode2:8001"],
            policy="power_of_two",
            prefill_policy="cache_aware",
            decode_policy="round_robin",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            with patch("sglang_router.router_args.logger") as mock_logger:
                result = launch_router(args)

                # Verify complete flow
                router_mod.from_args.assert_called_once()
                mock_router_instance.start.assert_called_once()

                # Verify policy warning was logged
                mock_logger.warning.assert_called_once()

    def test_complete_startup_flow_with_all_features(self):
        """Test complete startup flow with all features enabled."""
        args = RouterArgs(
            host="0.0.0.0",
            port=30001,
            worker_urls=["http://worker1:8000"],
            policy="round_robin",
            service_discovery=True,
            selector={"app": "worker"},
            service_discovery_port=8080,
            service_discovery_namespace="default",
            dp_aware=True,
            api_key="test-key",
            log_dir="/tmp/logs",
            log_level="debug",
            prometheus_port=29000,
            prometheus_host="0.0.0.0",
            request_id_headers=["x-request-id", "x-trace-id"],
            request_timeout_secs=1200,
            max_concurrent_requests=512,
            queue_size=200,
            queue_timeout_secs=120,
            rate_limit_tokens_per_second=100,
            cors_allowed_origins=["http://localhost:3000"],
            retry_max_retries=3,
            retry_initial_backoff_ms=100,
            retry_max_backoff_ms=10000,
            retry_backoff_multiplier=2.0,
            retry_jitter_factor=0.1,
            cb_failure_threshold=5,
            cb_success_threshold=2,
            cb_timeout_duration_secs=30,
            cb_window_duration_secs=60,
            health_failure_threshold=2,
            health_success_threshold=1,
            health_check_timeout_secs=3,
            health_check_interval_secs=30,
            health_check_endpoint="/healthz",
        )

        with patch("sglang_router.launch_router.Router") as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        host=router_args.host,
                        port=router_args.port,
                        worker_urls=router_args.worker_urls,
                        policy=policy_from_str(router_args.policy),
                        service_discovery=router_args.service_discovery,
                        selector=router_args.selector,
                        service_discovery_port=router_args.service_discovery_port,
                        service_discovery_namespace=router_args.service_discovery_namespace,
                        dp_aware=router_args.dp_aware,
                        api_key=router_args.api_key,
                        log_dir=router_args.log_dir,
                        log_level=router_args.log_level,
                        prometheus_port=router_args.prometheus_port,
                        prometheus_host=router_args.prometheus_host,
                        request_id_headers=router_args.request_id_headers,
                        request_timeout_secs=router_args.request_timeout_secs,
                        max_concurrent_requests=router_args.max_concurrent_requests,
                        queue_size=router_args.queue_size,
                        queue_timeout_secs=router_args.queue_timeout_secs,
                        rate_limit_tokens_per_second=router_args.rate_limit_tokens_per_second,
                        cors_allowed_origins=router_args.cors_allowed_origins,
                        retry_max_retries=router_args.retry_max_retries,
                        retry_initial_backoff_ms=router_args.retry_initial_backoff_ms,
                        retry_max_backoff_ms=router_args.retry_max_backoff_ms,
                        retry_backoff_multiplier=router_args.retry_backoff_multiplier,
                        retry_jitter_factor=router_args.retry_jitter_factor,
                        cb_failure_threshold=router_args.cb_failure_threshold,
                        cb_success_threshold=router_args.cb_success_threshold,
                        cb_timeout_duration_secs=router_args.cb_timeout_duration_secs,
                        cb_window_duration_secs=router_args.cb_window_duration_secs,
                        health_failure_threshold=router_args.health_failure_threshold,
                        health_success_threshold=router_args.health_success_threshold,
                        health_check_timeout_secs=router_args.health_check_timeout_secs,
                        health_check_interval_secs=router_args.health_check_interval_secs,
                        health_check_endpoint=router_args.health_check_endpoint,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            result = launch_router(args)

            # Verify complete flow
            router_mod.from_args.assert_called_once()
            mock_router_instance.start.assert_called_once()

            # Verify key parameters were propagated into RouterArgs
            assert captured_args["host"] == "0.0.0.0"
            assert captured_args["port"] == 30001
            assert captured_args["worker_urls"] == ["http://worker1:8000"]
            assert captured_args["policy"] == PolicyType.RoundRobin
            assert captured_args["service_discovery"] is True
            assert captured_args["selector"] == {"app": "worker"}
            assert captured_args["service_discovery_port"] == 8080
            assert captured_args["service_discovery_namespace"] == "default"
            assert captured_args["dp_aware"] is True
            assert captured_args["api_key"] == "test-key"
            assert captured_args["log_dir"] == "/tmp/logs"
            assert captured_args["log_level"] == "debug"
            assert captured_args["prometheus_port"] == 29000
            assert captured_args["prometheus_host"] == "0.0.0.0"
            assert captured_args["request_id_headers"] == ["x-request-id", "x-trace-id"]
            assert captured_args["request_timeout_secs"] == 1200
            assert captured_args["max_concurrent_requests"] == 512
            assert captured_args["queue_size"] == 200
            assert captured_args["queue_timeout_secs"] == 120
            assert captured_args["rate_limit_tokens_per_second"] == 100
            assert captured_args["cors_allowed_origins"] == ["http://localhost:3000"]
            assert captured_args["retry_max_retries"] == 3
            assert captured_args["retry_initial_backoff_ms"] == 100
            assert captured_args["retry_max_backoff_ms"] == 10000
            assert captured_args["retry_backoff_multiplier"] == 2.0
            assert captured_args["retry_jitter_factor"] == 0.1
            assert captured_args["cb_failure_threshold"] == 5
            assert captured_args["cb_success_threshold"] == 2
            assert captured_args["cb_timeout_duration_secs"] == 30
            assert captured_args["cb_window_duration_secs"] == 60
            assert captured_args["health_failure_threshold"] == 2
            assert captured_args["health_success_threshold"] == 1
            assert captured_args["health_check_timeout_secs"] == 3
            assert captured_args["health_check_interval_secs"] == 30
            assert captured_args["health_check_endpoint"] == "/healthz"
