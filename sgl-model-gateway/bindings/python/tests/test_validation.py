"""
Unit tests for validation logic in sglang_router.

These tests focus on testing the validation logic in isolation,
including parameter validation, URL validation, and configuration validation.
"""

from unittest.mock import MagicMock, patch

import pytest
from sglang_router.launch_router import RouterArgs, launch_router


class TestURLValidation:
    """Test URL validation logic."""

    def test_valid_worker_urls(self):
        """Test validation of valid worker URLs."""
        valid_urls = [
            "http://worker1:8000",
            "https://worker2:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://192.168.1.100:8000",
            "http://worker.example.com:8000",
        ]

        for url in valid_urls:
            args = RouterArgs(worker_urls=[url])
            # Should not raise any validation errors
            assert url in args.worker_urls

    def test_valid_prefill_urls(self):
        """Test validation of valid prefill URLs."""
        valid_prefill_urls = [
            ("http://prefill1:8000", 9000),
            ("https://prefill2:8000", None),
            ("http://localhost:8000", 9000),
            ("http://127.0.0.1:8000", None),
        ]

        for url, bootstrap_port in valid_prefill_urls:
            args = RouterArgs(prefill_urls=[(url, bootstrap_port)])
            # Should not raise any validation errors
            assert (url, bootstrap_port) in args.prefill_urls

    def test_valid_decode_urls(self):
        """Test validation of valid decode URLs."""
        valid_decode_urls = [
            "http://decode1:8001",
            "https://decode2:8001",
            "http://localhost:8001",
            "http://127.0.0.1:8001",
        ]

        for url in valid_decode_urls:
            args = RouterArgs(decode_urls=[url])
            # Should not raise any validation errors
            assert url in args.decode_urls

    def test_malformed_urls(self):
        """Test handling of malformed URLs."""
        # Note: The current implementation doesn't validate URL format
        # This test documents the current behavior
        malformed_urls = [
            "not-a-url",
            "ftp://worker1:8000",  # Wrong protocol
            "http://",  # Missing host
            ":8000",  # Missing protocol and host
            "http://worker1",  # Missing port
        ]

        for url in malformed_urls:
            args = RouterArgs(worker_urls=[url])
            # Currently, malformed URLs are accepted
            # This might be something to improve in the future
            assert url in args.worker_urls


class TestPortValidation:
    """Test port validation logic."""

    def test_valid_ports(self):
        """Test validation of valid port numbers."""
        valid_ports = [1, 80, 8000, 30000, 65535]

        for port in valid_ports:
            args = RouterArgs(port=port)
            assert args.port == port

    def test_invalid_ports(self):
        """Test handling of invalid port numbers."""
        # Note: The current implementation doesn't validate port ranges
        # This test documents the current behavior
        invalid_ports = [0, -1, 65536, 70000]

        for port in invalid_ports:
            args = RouterArgs(port=port)
            # Currently, invalid ports are accepted
            # This might be something to improve in the future
            assert args.port == port

    def test_bootstrap_port_validation(self):
        """Test validation of bootstrap ports in PD mode."""
        valid_bootstrap_ports = [1, 80, 9000, 30000, 65535, None]

        for bootstrap_port in valid_bootstrap_ports:
            args = RouterArgs(prefill_urls=[("http://prefill1:8000", bootstrap_port)])
            assert args.prefill_urls[0][1] == bootstrap_port


class TestParameterValidation:
    """Test parameter validation logic."""

    def test_cache_threshold_validation(self):
        """Test cache threshold parameter validation."""
        # Valid cache thresholds
        valid_thresholds = [0.0, 0.1, 0.5, 0.9, 1.0]

        for threshold in valid_thresholds:
            args = RouterArgs(cache_threshold=threshold)
            assert args.cache_threshold == threshold

    def test_balance_threshold_validation(self):
        """Test load balancing threshold parameter validation."""
        # Valid absolute thresholds
        valid_abs_thresholds = [0, 1, 32, 64, 128, 1000]
        for threshold in valid_abs_thresholds:
            args = RouterArgs(balance_abs_threshold=threshold)
            assert args.balance_abs_threshold == threshold

        # Valid relative thresholds
        valid_rel_thresholds = [1.0, 1.1, 1.5, 2.0, 10.0]
        for threshold in valid_rel_thresholds:
            args = RouterArgs(balance_rel_threshold=threshold)
            assert args.balance_rel_threshold == threshold

    def test_timeout_validation(self):
        """Test timeout parameter validation."""
        # Valid timeouts
        valid_timeouts = [1, 30, 60, 300, 600, 1800, 3600]

        for timeout in valid_timeouts:
            args = RouterArgs(
                worker_startup_timeout_secs=timeout,
                worker_startup_check_interval=timeout,
                request_timeout_secs=timeout,
                queue_timeout_secs=timeout,
            )
            assert args.worker_startup_timeout_secs == timeout
            assert args.worker_startup_check_interval == timeout
            assert args.request_timeout_secs == timeout
            assert args.queue_timeout_secs == timeout

    def test_retry_parameter_validation(self):
        """Test retry parameter validation."""
        # Valid retry parameters
        valid_retry_counts = [0, 1, 3, 5, 10]
        for count in valid_retry_counts:
            args = RouterArgs(retry_max_retries=count)
            assert args.retry_max_retries == count

        # Valid backoff parameters
        valid_backoff_ms = [1, 50, 100, 1000, 30000]
        for backoff in valid_backoff_ms:
            args = RouterArgs(
                retry_initial_backoff_ms=backoff, retry_max_backoff_ms=backoff
            )
            assert args.retry_initial_backoff_ms == backoff
            assert args.retry_max_backoff_ms == backoff

        # Valid multiplier parameters
        valid_multipliers = [1.0, 1.5, 2.0, 3.0]
        for multiplier in valid_multipliers:
            args = RouterArgs(retry_backoff_multiplier=multiplier)
            assert args.retry_backoff_multiplier == multiplier

        # Valid jitter parameters
        valid_jitter = [0.0, 0.1, 0.2, 0.5]
        for jitter in valid_jitter:
            args = RouterArgs(retry_jitter_factor=jitter)
            assert args.retry_jitter_factor == jitter

    def test_circuit_breaker_parameter_validation(self):
        """Test circuit breaker parameter validation."""
        # Valid failure thresholds
        valid_failure_thresholds = [1, 3, 5, 10, 20]
        for threshold in valid_failure_thresholds:
            args = RouterArgs(cb_failure_threshold=threshold)
            assert args.cb_failure_threshold == threshold

        # Valid success thresholds
        valid_success_thresholds = [1, 2, 3, 5]
        for threshold in valid_success_thresholds:
            args = RouterArgs(cb_success_threshold=threshold)
            assert args.cb_success_threshold == threshold

        # Valid timeout durations
        valid_timeouts = [10, 30, 60, 120, 300]
        for timeout in valid_timeouts:
            args = RouterArgs(
                cb_timeout_duration_secs=timeout, cb_window_duration_secs=timeout
            )
            assert args.cb_timeout_duration_secs == timeout
            assert args.cb_window_duration_secs == timeout

    def test_health_check_parameter_validation(self):
        """Test health check parameter validation."""
        # Valid failure thresholds
        valid_failure_thresholds = [1, 2, 3, 5, 10]
        for threshold in valid_failure_thresholds:
            args = RouterArgs(health_failure_threshold=threshold)
            assert args.health_failure_threshold == threshold

        # Valid success thresholds
        valid_success_thresholds = [1, 2, 3, 5]
        for threshold in valid_success_thresholds:
            args = RouterArgs(health_success_threshold=threshold)
            assert args.health_success_threshold == threshold

        # Valid timeouts and intervals
        valid_times = [1, 5, 10, 30, 60, 120]
        for time_val in valid_times:
            args = RouterArgs(
                health_check_timeout_secs=time_val, health_check_interval_secs=time_val
            )
            assert args.health_check_timeout_secs == time_val
            assert args.health_check_interval_secs == time_val

    def test_rate_limiting_parameter_validation(self):
        """Test rate limiting parameter validation."""
        # Valid concurrent request limits
        valid_limits = [1, 10, 64, 256, 512, 1000]
        for limit in valid_limits:
            args = RouterArgs(max_concurrent_requests=limit)
            assert args.max_concurrent_requests == limit

        # Valid queue sizes
        valid_queue_sizes = [0, 10, 50, 100, 500, 1000]
        for size in valid_queue_sizes:
            args = RouterArgs(queue_size=size)
            assert args.queue_size == size

        # Valid token rates
        valid_rates = [1, 10, 50, 100, 500, 1000]
        for rate in valid_rates:
            args = RouterArgs(rate_limit_tokens_per_second=rate)
            assert args.rate_limit_tokens_per_second == rate

    def test_tree_size_validation(self):
        """Test tree size parameter validation."""
        # Valid tree sizes (powers of 2)
        valid_sizes = [2**10, 2**20, 2**24, 2**26, 2**28, 2**30]

        for size in valid_sizes:
            args = RouterArgs(max_tree_size=size)
            assert args.max_tree_size == size

    def test_payload_size_validation(self):
        """Test payload size parameter validation."""
        # Valid payload sizes
        valid_sizes = [
            1024,  # 1KB
            1024 * 1024,  # 1MB
            10 * 1024 * 1024,  # 10MB
            100 * 1024 * 1024,  # 100MB
            512 * 1024 * 1024,  # 512MB
            1024 * 1024 * 1024,  # 1GB
        ]

        for size in valid_sizes:
            args = RouterArgs(max_payload_size=size)
            assert args.max_payload_size == size


class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_pd_mode_validation(self):
        """Test PD mode configuration validation."""
        # Valid PD configuration
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000)],
            decode_urls=["http://decode1:8001"],
        )

        assert args.pd_disaggregation is True
        assert len(args.prefill_urls) > 0
        assert len(args.decode_urls) > 0

    def test_service_discovery_validation(self):
        """Test service discovery configuration validation."""
        # Valid service discovery configuration
        args = RouterArgs(
            service_discovery=True,
            selector={"app": "worker", "env": "prod"},
            service_discovery_port=8080,
            service_discovery_namespace="default",
        )

        assert args.service_discovery is True
        assert args.selector == {"app": "worker", "env": "prod"}
        assert args.service_discovery_port == 8080
        assert args.service_discovery_namespace == "default"

    def test_pd_service_discovery_validation(self):
        """Test PD service discovery configuration validation."""
        # Valid PD service discovery configuration
        args = RouterArgs(
            pd_disaggregation=True,
            service_discovery=True,
            prefill_selector={"app": "prefill"},
            decode_selector={"app": "decode"},
        )

        assert args.pd_disaggregation is True
        assert args.service_discovery is True
        assert args.prefill_selector == {"app": "prefill"}
        assert args.decode_selector == {"app": "decode"}

    def test_policy_validation(self):
        """Test policy configuration validation."""
        # Valid policies
        valid_policies = ["random", "round_robin", "cache_aware", "power_of_two"]

        for policy in valid_policies:
            args = RouterArgs(policy=policy)
            assert args.policy == policy

    def test_pd_policy_validation(self):
        """Test PD policy configuration validation."""
        # Valid PD policies
        valid_policies = ["random", "round_robin", "cache_aware", "power_of_two"]

        for prefill_policy in valid_policies:
            for decode_policy in valid_policies:
                args = RouterArgs(
                    pd_disaggregation=True,
                    prefill_urls=[("http://prefill1:8000", None)],
                    decode_urls=["http://decode1:8001"],
                    prefill_policy=prefill_policy,
                    decode_policy=decode_policy,
                )
                assert args.prefill_policy == prefill_policy
                assert args.decode_policy == decode_policy

    def test_cors_validation(self):
        """Test CORS configuration validation."""
        # Valid CORS origins
        valid_origins = [
            [],
            ["http://localhost:3000"],
            ["https://example.com"],
            ["http://localhost:3000", "https://example.com"],
            ["*"],  # Wildcard (if supported)
        ]

        for origins in valid_origins:
            args = RouterArgs(cors_allowed_origins=origins)
            assert args.cors_allowed_origins == origins

    def test_logging_validation(self):
        """Test logging configuration validation."""
        # Valid log levels
        valid_log_levels = ["debug", "info", "warning", "error", "critical"]

        for level in valid_log_levels:
            args = RouterArgs(log_level=level)
            assert args.log_level == level

    def test_prometheus_validation(self):
        """Test Prometheus configuration validation."""
        # Valid Prometheus configuration
        args = RouterArgs(prometheus_port=29000, prometheus_host="127.0.0.1")

        assert args.prometheus_port == 29000
        assert args.prometheus_host == "127.0.0.1"

    def test_tokenizer_validation(self):
        """Test tokenizer configuration validation."""
        # Note: model_path and tokenizer_path are not available in current RouterArgs
        pytest.skip("Tokenizer configuration not available in current implementation")

    def test_request_id_headers_validation(self):
        """Test request ID headers configuration validation."""
        # Valid request ID headers
        valid_headers = [
            ["x-request-id"],
            ["x-request-id", "x-trace-id"],
            ["x-request-id", "x-trace-id", "x-correlation-id"],
            ["custom-header"],
        ]

        for headers in valid_headers:
            args = RouterArgs(request_id_headers=headers)
            assert args.request_id_headers == headers


class TestLaunchValidation:
    """Test launch-time validation logic."""

    def test_pd_mode_allows_empty_urls(self):
        """Test that PD mode now allows empty URLs (URLs are optional)."""
        # PD mode without URLs is now allowed
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=False,
        )

        # Should not raise validation error - URLs are now optional
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # This should succeed without raising an error
            launch_router(args)
            router_mod.from_args.assert_called_once()

    def test_pd_mode_with_service_discovery_allows_empty_urls(self):
        """Test that PD mode with service discovery allows empty URLs."""
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

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_regular_mode_allows_empty_worker_urls(self):
        """Test that regular mode allows empty worker URLs."""
        args = RouterArgs(worker_urls=[], service_discovery=False)

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_launch_with_valid_config(self):
        """Test launching with valid configuration."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"],
            policy="cache_aware",
        )

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_launch_with_pd_config(self):
        """Test launching with valid PD configuration."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
        )

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_launch_with_service_discovery_config(self):
        """Test launching with valid service discovery configuration."""
        args = RouterArgs(
            service_discovery=True,
            selector={"app": "worker"},
            service_discovery_port=8080,
        )

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()
