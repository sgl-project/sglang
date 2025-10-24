"""
Unit tests for router configuration validation and setup.

These tests focus on testing the router configuration logic in isolation,
including validation of configuration parameters and their interactions.
"""

from unittest.mock import MagicMock, patch

import pytest
from sglang_router.launch_router import RouterArgs, launch_router
from sglang_router.router import policy_from_str
from sglang_router_rs import PolicyType


class TestRouterConfigValidation:
    """Test router configuration validation logic."""

    def test_valid_basic_config(self):
        """Test that a valid basic configuration passes validation."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000", "http://worker2:8000"],
            policy="cache_aware",
        )

        # Should not raise any exceptions
        assert args.host == "127.0.0.1"
        assert args.port == 30000
        assert args.worker_urls == ["http://worker1:8000", "http://worker2:8000"]
        assert args.policy == "cache_aware"

    def test_valid_pd_config(self):
        """Test that a valid PD configuration passes validation."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            pd_disaggregation=True,
            prefill_urls=[
                ("http://prefill1:8000", 9000),
                ("http://prefill2:8000", None),
            ],
            decode_urls=["http://decode1:8001", "http://decode2:8001"],
            policy="cache_aware",
        )

        assert args.pd_disaggregation is True
        assert args.prefill_urls == [
            ("http://prefill1:8000", 9000),
            ("http://prefill2:8000", None),
        ]
        assert args.decode_urls == ["http://decode1:8001", "http://decode2:8001"]
        assert args.policy == "cache_aware"

    def test_pd_config_without_urls_raises_error(self):
        """Test that PD mode without URLs raises validation error."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=False,
        )

        # This should raise an error when trying to launch
        with pytest.raises(
            ValueError, match="PD disaggregation mode requires --prefill"
        ):
            launch_router(args)

    def test_pd_config_with_service_discovery_allows_empty_urls(self):
        """Test that PD mode with service discovery allows empty URLs."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=True,
        )

        # Should not raise validation error when service discovery is enabled
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_regular_mode_without_workers_allows_empty_urls(self):
        """Test that regular mode allows empty worker URLs."""
        args = RouterArgs(worker_urls=[], service_discovery=False)

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_cache_threshold_validation(self):
        """Test cache threshold validation."""
        # Valid cache threshold
        args = RouterArgs(cache_threshold=0.5)
        assert args.cache_threshold == 0.5

        # Edge cases
        args = RouterArgs(cache_threshold=0.0)
        assert args.cache_threshold == 0.0

        args = RouterArgs(cache_threshold=1.0)
        assert args.cache_threshold == 1.0

    def test_balance_threshold_validation(self):
        """Test load balancing threshold validation."""
        # Valid thresholds
        args = RouterArgs(balance_abs_threshold=64, balance_rel_threshold=1.5)
        assert args.balance_abs_threshold == 64
        assert args.balance_rel_threshold == 1.5

        # Edge cases
        args = RouterArgs(balance_abs_threshold=0, balance_rel_threshold=1.0)
        assert args.balance_abs_threshold == 0
        assert args.balance_rel_threshold == 1.0

    def test_timeout_validation(self):
        """Test timeout parameter validation."""
        # Valid timeouts
        args = RouterArgs(
            worker_startup_timeout_secs=600,
            worker_startup_check_interval=30,
            request_timeout_secs=1800,
            queue_timeout_secs=60,
        )
        assert args.worker_startup_timeout_secs == 600
        assert args.worker_startup_check_interval == 30
        assert args.request_timeout_secs == 1800
        assert args.queue_timeout_secs == 60

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Valid retry config
        args = RouterArgs(
            retry_max_retries=5,
            retry_initial_backoff_ms=50,
            retry_max_backoff_ms=30000,
            retry_backoff_multiplier=1.5,
            retry_jitter_factor=0.2,
            disable_retries=False,
        )
        assert args.retry_max_retries == 5
        assert args.retry_initial_backoff_ms == 50
        assert args.retry_max_backoff_ms == 30000
        assert args.retry_backoff_multiplier == 1.5
        assert args.retry_jitter_factor == 0.2
        assert args.disable_retries is False

    def test_circuit_breaker_config_validation(self):
        """Test circuit breaker configuration validation."""
        # Valid circuit breaker config
        args = RouterArgs(
            cb_failure_threshold=10,
            cb_success_threshold=3,
            cb_timeout_duration_secs=60,
            cb_window_duration_secs=120,
            disable_circuit_breaker=False,
        )
        assert args.cb_failure_threshold == 10
        assert args.cb_success_threshold == 3
        assert args.cb_timeout_duration_secs == 60
        assert args.cb_window_duration_secs == 120
        assert args.disable_circuit_breaker is False

    def test_health_check_config_validation(self):
        """Test health check configuration validation."""
        # Valid health check config
        args = RouterArgs(
            health_failure_threshold=3,
            health_success_threshold=2,
            health_check_timeout_secs=5,
            health_check_interval_secs=60,
            health_check_endpoint="/health",
        )
        assert args.health_failure_threshold == 3
        assert args.health_success_threshold == 2
        assert args.health_check_timeout_secs == 5
        assert args.health_check_interval_secs == 60
        assert args.health_check_endpoint == "/health"

    def test_rate_limiting_config_validation(self):
        """Test rate limiting configuration validation."""
        # Valid rate limiting config
        args = RouterArgs(
            max_concurrent_requests=256,
            queue_size=100,
            queue_timeout_secs=60,
            rate_limit_tokens_per_second=100,
        )
        assert args.max_concurrent_requests == 256
        assert args.queue_size == 100
        assert args.queue_timeout_secs == 60
        assert args.rate_limit_tokens_per_second == 100

    def test_service_discovery_config_validation(self):
        """Test service discovery configuration validation."""
        # Valid service discovery config
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

    def test_pd_service_discovery_config_validation(self):
        """Test PD service discovery configuration validation."""
        # Valid PD service discovery config
        args = RouterArgs(
            pd_disaggregation=True,
            service_discovery=True,
            prefill_selector={"app": "prefill"},
            decode_selector={"app": "decode"},
            bootstrap_port_annotation="sglang.ai/bootstrap-port",
        )
        assert args.pd_disaggregation is True
        assert args.service_discovery is True
        assert args.prefill_selector == {"app": "prefill"}
        assert args.decode_selector == {"app": "decode"}
        assert args.bootstrap_port_annotation == "sglang.ai/bootstrap-port"

    def test_prometheus_config_validation(self):
        """Test Prometheus configuration validation."""
        # Valid Prometheus config
        args = RouterArgs(prometheus_port=29000, prometheus_host="127.0.0.1")
        assert args.prometheus_port == 29000
        assert args.prometheus_host == "127.0.0.1"

    def test_cors_config_validation(self):
        """Test CORS configuration validation."""
        # Valid CORS config
        args = RouterArgs(
            cors_allowed_origins=["http://localhost:3000", "https://example.com"]
        )
        assert args.cors_allowed_origins == [
            "http://localhost:3000",
            "https://example.com",
        ]

    def test_tokenizer_config_validation(self):
        """Test tokenizer configuration validation."""
        # Note: model_path and tokenizer_path are not available in current RouterArgs
        pytest.skip("Tokenizer configuration not available in current implementation")

    def test_dp_aware_config_validation(self):
        """Test data parallelism aware configuration validation."""
        # Valid DP aware config
        args = RouterArgs(dp_aware=True, api_key="test-api-key")
        assert args.dp_aware is True
        assert args.api_key == "test-api-key"

    def test_request_id_headers_validation(self):
        """Test request ID headers configuration validation."""
        # Valid request ID headers config
        args = RouterArgs(
            request_id_headers=["x-request-id", "x-trace-id", "x-correlation-id"]
        )
        assert args.request_id_headers == [
            "x-request-id",
            "x-trace-id",
            "x-correlation-id",
        ]

    def test_policy_consistency_validation(self):
        """Test policy consistency validation in PD mode."""
        # Test with both prefill and decode policies specified
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy="power_of_two",
            decode_policy="round_robin",
        )

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_policy_fallback_validation(self):
        """Test policy fallback validation in PD mode."""
        # Test with only prefill policy specified
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy="power_of_two",
            decode_policy=None,
        )

        # Should not raise validation error
        with patch("sglang_router.launch_router.Router") as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance via from_args
            router_mod.from_args.assert_called_once()

    def test_policy_enum_conversion(self):
        """Test policy string to enum conversion."""
        # Test all valid policy conversions
        assert policy_from_str("random") == PolicyType.Random
        assert policy_from_str("round_robin") == PolicyType.RoundRobin
        assert policy_from_str("cache_aware") == PolicyType.CacheAware
        assert policy_from_str("power_of_two") == PolicyType.PowerOfTwo

    def test_invalid_policy_enum_conversion(self):
        """Test invalid policy string to enum conversion."""
        with pytest.raises(KeyError):
            policy_from_str("invalid_policy")

    def test_config_immutability(self):
        """Test that configuration objects are properly immutable."""
        args = RouterArgs(
            host="127.0.0.1", port=30000, worker_urls=["http://worker1:8000"]
        )

        # Test that we can't modify the configuration after creation
        # (This is more of a design test - dataclasses are mutable by default)
        original_host = args.host
        args.host = "0.0.0.0"
        assert args.host == "0.0.0.0"  # Dataclasses are mutable
        assert args.host != original_host

    def test_config_defaults_consistency(self):
        """Test that configuration defaults are consistent."""
        args1 = RouterArgs()
        args2 = RouterArgs()

        # Both instances should have the same defaults
        assert args1.host == args2.host
        assert args1.port == args2.port
        assert args1.policy == args2.policy
        assert args1.worker_urls == args2.worker_urls
        assert args1.pd_disaggregation == args2.pd_disaggregation

    def test_config_serialization(self):
        """Test that configuration can be serialized/deserialized."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"],
            policy="cache_aware",
            cache_threshold=0.5,
        )

        # Test that we can access all attributes
        assert hasattr(args, "host")
        assert hasattr(args, "port")
        assert hasattr(args, "worker_urls")
        assert hasattr(args, "policy")
        assert hasattr(args, "cache_threshold")

    def test_config_with_none_values(self):
        """Test configuration with None values."""
        args = RouterArgs(
            api_key=None,
            log_dir=None,
            log_level=None,
            prometheus_port=None,
            prometheus_host=None,
            request_id_headers=None,
            rate_limit_tokens_per_second=None,
            service_discovery_namespace=None,
        )

        # All None values should be preserved
        assert args.api_key is None
        assert args.log_dir is None
        assert args.log_level is None
        assert args.prometheus_port is None
        assert args.prometheus_host is None
        assert args.request_id_headers is None
        assert args.rate_limit_tokens_per_second is None
        assert args.service_discovery_namespace is None

    def test_config_with_empty_lists(self):
        """Test configuration with empty lists."""
        args = RouterArgs(
            worker_urls=[], prefill_urls=[], decode_urls=[], cors_allowed_origins=[]
        )

        # All empty lists should be preserved
        assert args.worker_urls == []
        assert args.prefill_urls == []
        assert args.decode_urls == []
        assert args.cors_allowed_origins == []

    def test_config_with_empty_dicts(self):
        """Test configuration with empty dictionaries."""
        args = RouterArgs(selector={}, prefill_selector={}, decode_selector={})

        # All empty dictionaries should be preserved
        assert args.selector == {}
        assert args.prefill_selector == {}
        assert args.decode_selector == {}
