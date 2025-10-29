"""
Unit tests for argument parsing functionality in sglang_router.

These tests focus on testing the argument parsing logic in isolation,
without starting actual router instances.
"""

from types import SimpleNamespace

import pytest
from sglang_router.launch_router import RouterArgs, parse_router_args
from sglang_router.router import policy_from_str


class TestRouterArgs:
    """Test RouterArgs dataclass and its methods."""

    def test_default_values(self):
        """Test that RouterArgs has correct default values."""
        args = RouterArgs()

        # Test basic defaults
        assert args.host == "0.0.0.0"
        assert args.port == 30000
        assert args.policy == "cache_aware"
        assert args.worker_urls == []
        assert args.pd_disaggregation is False
        assert args.prefill_urls == []
        assert args.decode_urls == []

        # Test PD-specific defaults
        assert args.prefill_policy is None
        assert args.decode_policy is None

        # Test service discovery defaults
        assert args.service_discovery is False
        assert args.selector == {}
        assert args.service_discovery_port == 80
        assert args.service_discovery_namespace is None

        # Test retry and circuit breaker defaults
        assert args.retry_max_retries == 5
        assert args.cb_failure_threshold == 10
        assert args.disable_retries is False
        assert args.disable_circuit_breaker is False

    def test_parse_selector_valid(self):
        """Test parsing valid selector arguments."""
        # Test single key-value pair
        result = RouterArgs._parse_selector(["app=worker"])
        assert result == {"app": "worker"}

        # Test multiple key-value pairs
        result = RouterArgs._parse_selector(["app=worker", "env=prod", "version=v1"])
        assert result == {"app": "worker", "env": "prod", "version": "v1"}

        # Test empty list
        result = RouterArgs._parse_selector([])
        assert result == {}

        # Test None
        result = RouterArgs._parse_selector(None)
        assert result == {}

    def test_parse_selector_invalid(self):
        """Test parsing invalid selector arguments."""
        # Test malformed selector (no equals sign)
        result = RouterArgs._parse_selector(["app"])
        assert result == {}

        # Test multiple equals signs (should use first one)
        result = RouterArgs._parse_selector(["app=worker=extra"])
        assert result == {"app": "worker=extra"}

    def test_parse_prefill_urls_valid(self):
        """Test parsing valid prefill URL arguments."""
        # Test with bootstrap port
        result = RouterArgs._parse_prefill_urls([["http://prefill1:8000", "9000"]])
        assert result == [("http://prefill1:8000", 9000)]

        # Test with 'none' bootstrap port
        result = RouterArgs._parse_prefill_urls([["http://prefill1:8000", "none"]])
        assert result == [("http://prefill1:8000", None)]

        # Test without bootstrap port
        result = RouterArgs._parse_prefill_urls([["http://prefill1:8000"]])
        assert result == [("http://prefill1:8000", None)]

        # Test multiple prefill URLs
        result = RouterArgs._parse_prefill_urls(
            [
                ["http://prefill1:8000", "9000"],
                ["http://prefill2:8000", "none"],
                ["http://prefill3:8000"],
            ]
        )
        expected = [
            ("http://prefill1:8000", 9000),
            ("http://prefill2:8000", None),
            ("http://prefill3:8000", None),
        ]
        assert result == expected

        # Test empty list
        result = RouterArgs._parse_prefill_urls([])
        assert result == []

        # Test None
        result = RouterArgs._parse_prefill_urls(None)
        assert result == []

    def test_parse_prefill_urls_invalid(self):
        """Test parsing invalid prefill URL arguments."""
        # Test invalid bootstrap port
        with pytest.raises(ValueError, match="Invalid bootstrap port"):
            RouterArgs._parse_prefill_urls([["http://prefill1:8000", "invalid"]])

    def test_parse_decode_urls_valid(self):
        """Test parsing valid decode URL arguments."""
        # Test single decode URL
        result = RouterArgs._parse_decode_urls([["http://decode1:8001"]])
        assert result == ["http://decode1:8001"]

        # Test multiple decode URLs
        result = RouterArgs._parse_decode_urls(
            [["http://decode1:8001"], ["http://decode2:8001"]]
        )
        assert result == ["http://decode1:8001", "http://decode2:8001"]

        # Test empty list
        result = RouterArgs._parse_decode_urls([])
        assert result == []

        # Test None
        result = RouterArgs._parse_decode_urls(None)
        assert result == []

    def test_from_cli_args_basic(self):
        """Test creating RouterArgs from basic CLI arguments."""
        args = SimpleNamespace(
            host="0.0.0.0",
            port=30001,
            worker_urls=["http://worker1:8000", "http://worker2:8000"],
            policy="round_robin",
            prefill=None,
            decode=None,
            router_policy="round_robin",
            router_pd_disaggregation=False,
            router_prefill_policy=None,
            router_decode_policy=None,
            router_worker_startup_timeout_secs=300,
            router_worker_startup_check_interval=15,
            router_cache_threshold=0.7,
            router_balance_abs_threshold=128,
            router_balance_rel_threshold=2.0,
            router_eviction_interval=180,
            router_max_tree_size=2**28,
            router_max_payload_size=1024 * 1024 * 1024,  # 1GB
            router_dp_aware=True,
            router_api_key="test-key",
            router_log_dir="/tmp/logs",
            router_log_level="debug",
            router_service_discovery=True,
            router_selector=["app=worker", "env=test"],
            router_service_discovery_port=8080,
            router_service_discovery_namespace="default",
            router_prefill_selector=["app=prefill"],
            router_decode_selector=["app=decode"],
            router_prometheus_port=29000,
            router_prometheus_host="0.0.0.0",
            router_request_id_headers=["x-request-id", "x-trace-id"],
            router_request_timeout_secs=1200,
            router_max_concurrent_requests=512,
            router_queue_size=200,
            router_queue_timeout_secs=120,
            router_rate_limit_tokens_per_second=100,
            router_cors_allowed_origins=["http://localhost:3000"],
            router_retry_max_retries=3,
            router_retry_initial_backoff_ms=100,
            router_retry_max_backoff_ms=10000,
            router_retry_backoff_multiplier=2.0,
            router_retry_jitter_factor=0.1,
            router_cb_failure_threshold=5,
            router_cb_success_threshold=2,
            router_cb_timeout_duration_secs=30,
            router_cb_window_duration_secs=60,
            router_disable_retries=False,
            router_disable_circuit_breaker=False,
            router_health_failure_threshold=2,
            router_health_success_threshold=1,
            router_health_check_timeout_secs=3,
            router_health_check_interval_secs=30,
            router_health_check_endpoint="/healthz",
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

        # Test basic configuration
        assert router_args.host == "0.0.0.0"
        assert router_args.port == 30001
        assert router_args.worker_urls == ["http://worker1:8000", "http://worker2:8000"]
        assert router_args.policy == "round_robin"

        # Test PD configuration
        assert router_args.pd_disaggregation is False
        assert router_args.prefill_urls == []
        assert router_args.decode_urls == []

        # Test service discovery
        assert router_args.service_discovery is True
        assert router_args.selector == {"app": "worker", "env": "test"}
        assert router_args.service_discovery_port == 8080
        assert router_args.service_discovery_namespace == "default"
        assert router_args.prefill_selector == {"app": "prefill"}
        assert router_args.decode_selector == {"app": "decode"}

        # Test other configurations
        assert router_args.dp_aware is True
        assert router_args.api_key == "test-key"
        assert router_args.log_dir == "/tmp/logs"
        assert router_args.log_level == "debug"
        assert router_args.prometheus_port == 29000
        assert router_args.prometheus_host == "0.0.0.0"
        assert router_args.request_id_headers == ["x-request-id", "x-trace-id"]
        assert router_args.request_timeout_secs == 1200
        assert router_args.max_concurrent_requests == 512
        assert router_args.queue_size == 200
        assert router_args.queue_timeout_secs == 120
        assert router_args.rate_limit_tokens_per_second == 100
        assert router_args.cors_allowed_origins == ["http://localhost:3000"]

        # Test retry configuration
        assert router_args.retry_max_retries == 3
        assert router_args.retry_initial_backoff_ms == 100
        assert router_args.retry_max_backoff_ms == 10000
        assert router_args.retry_backoff_multiplier == 2.0
        assert router_args.retry_jitter_factor == 0.1

        # Test circuit breaker configuration
        assert router_args.cb_failure_threshold == 5
        assert router_args.cb_success_threshold == 2
        assert router_args.cb_timeout_duration_secs == 30
        assert router_args.cb_window_duration_secs == 60
        assert router_args.disable_retries is False
        assert router_args.disable_circuit_breaker is False

        # Test health check configuration
        assert router_args.health_failure_threshold == 2
        assert router_args.health_success_threshold == 1
        assert router_args.health_check_timeout_secs == 3
        assert router_args.health_check_interval_secs == 30
        assert router_args.health_check_endpoint == "/healthz"

        # Note: model_path and tokenizer_path are not available in current RouterArgs

    def test_from_cli_args_pd_mode(self):
        """Test creating RouterArgs from CLI arguments in PD mode."""
        args = SimpleNamespace(
            host="127.0.0.1",
            port=30000,
            worker_urls=[],
            policy="cache_aware",
            prefill=[
                ["http://prefill1:8000", "9000"],
                ["http://prefill2:8000", "none"],
            ],
            decode=[["http://decode1:8001"], ["http://decode2:8001"]],
            router_prefill=[
                ["http://prefill1:8000", "9000"],
                ["http://prefill2:8000", "none"],
            ],
            router_decode=[["http://decode1:8001"], ["http://decode2:8001"]],
            router_policy="cache_aware",
            router_pd_disaggregation=True,
            router_prefill_policy="power_of_two",
            router_decode_policy="round_robin",
            # Include all required fields with defaults
            router_worker_startup_timeout_secs=600,
            router_worker_startup_check_interval=30,
            router_cache_threshold=0.3,
            router_balance_abs_threshold=64,
            router_balance_rel_threshold=1.5,
            router_eviction_interval=120,
            router_max_tree_size=2**26,
            router_max_payload_size=512 * 1024 * 1024,
            router_dp_aware=False,
            router_api_key=None,
            router_log_dir=None,
            router_log_level=None,
            router_service_discovery=False,
            router_selector=None,
            router_service_discovery_port=80,
            router_service_discovery_namespace=None,
            router_prefill_selector=None,
            router_decode_selector=None,
            router_prometheus_port=None,
            router_prometheus_host=None,
            router_request_id_headers=None,
            router_request_timeout_secs=1800,
            router_max_concurrent_requests=256,
            router_queue_size=100,
            router_queue_timeout_secs=60,
            router_rate_limit_tokens_per_second=None,
            router_cors_allowed_origins=[],
            router_retry_max_retries=5,
            router_retry_initial_backoff_ms=50,
            router_retry_max_backoff_ms=30000,
            router_retry_backoff_multiplier=1.5,
            router_retry_jitter_factor=0.2,
            router_cb_failure_threshold=10,
            router_cb_success_threshold=3,
            router_cb_timeout_duration_secs=60,
            router_cb_window_duration_secs=120,
            router_disable_retries=False,
            router_disable_circuit_breaker=False,
            router_health_failure_threshold=3,
            router_health_success_threshold=2,
            router_health_check_timeout_secs=5,
            router_health_check_interval_secs=60,
            router_health_check_endpoint="/health",
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)

        # Test PD configuration
        assert router_args.pd_disaggregation is True
        assert router_args.prefill_urls == [
            ("http://prefill1:8000", 9000),
            ("http://prefill2:8000", None),
        ]
        assert router_args.decode_urls == ["http://decode1:8001", "http://decode2:8001"]
        assert router_args.prefill_policy == "power_of_two"
        assert router_args.decode_policy == "round_robin"
        assert router_args.policy == "cache_aware"  # Main policy still set

    def test_from_cli_args_without_prefix(self):
        """Test creating RouterArgs from CLI arguments without router prefix."""
        args = SimpleNamespace(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"],
            policy="random",
            prefill=None,
            decode=None,
            pd_disaggregation=False,
            prefill_policy=None,
            decode_policy=None,
            worker_startup_timeout_secs=600,
            worker_startup_check_interval=30,
            cache_threshold=0.3,
            balance_abs_threshold=64,
            balance_rel_threshold=1.5,
            eviction_interval=120,
            max_tree_size=2**26,
            max_payload_size=512 * 1024 * 1024,
            dp_aware=False,
            api_key=None,
            log_dir=None,
            log_level=None,
            service_discovery=False,
            selector=None,
            service_discovery_port=80,
            service_discovery_namespace=None,
            prefill_selector=None,
            decode_selector=None,
            prometheus_port=None,
            prometheus_host=None,
            request_id_headers=None,
            request_timeout_secs=1800,
            max_concurrent_requests=256,
            queue_size=100,
            queue_timeout_secs=60,
            rate_limit_tokens_per_second=None,
            cors_allowed_origins=[],
            retry_max_retries=5,
            retry_initial_backoff_ms=50,
            retry_max_backoff_ms=30000,
            retry_backoff_multiplier=1.5,
            retry_jitter_factor=0.2,
            cb_failure_threshold=10,
            cb_success_threshold=3,
            cb_timeout_duration_secs=60,
            cb_window_duration_secs=120,
            disable_retries=False,
            disable_circuit_breaker=False,
            health_failure_threshold=3,
            health_success_threshold=2,
            health_check_timeout_secs=5,
            health_check_interval_secs=60,
            health_check_endpoint="/health",
            model_path=None,
            tokenizer_path=None,
        )

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=False)

        assert router_args.host == "127.0.0.1"
        assert router_args.port == 30000
        assert router_args.worker_urls == ["http://worker1:8000"]
        assert router_args.policy == "random"
        assert router_args.pd_disaggregation is False


class TestPolicyFromStr:
    """Test policy string to enum conversion."""

    def test_valid_policies(self):
        """Test conversion of valid policy strings."""
        from sglang_router_rs import PolicyType

        assert policy_from_str("random") == PolicyType.Random
        assert policy_from_str("round_robin") == PolicyType.RoundRobin
        assert policy_from_str("cache_aware") == PolicyType.CacheAware
        assert policy_from_str("power_of_two") == PolicyType.PowerOfTwo

    def test_invalid_policy(self):
        """Test conversion of invalid policy string."""
        with pytest.raises(KeyError):
            policy_from_str("invalid_policy")


class TestParseRouterArgs:
    """Test the parse_router_args function."""

    def test_parse_basic_args(self):
        """Test parsing basic router arguments."""
        args = [
            "--host",
            "0.0.0.0",
            "--port",
            "30001",
            "--worker-urls",
            "http://worker1:8000",
            "http://worker2:8000",
            "--policy",
            "round_robin",
        ]

        router_args = parse_router_args(args)

        assert router_args.host == "0.0.0.0"
        assert router_args.port == 30001
        assert router_args.worker_urls == ["http://worker1:8000", "http://worker2:8000"]
        assert router_args.policy == "round_robin"

    def test_parse_pd_args(self):
        """Test parsing PD disaggregated mode arguments."""
        args = [
            "--pd-disaggregation",
            "--prefill",
            "http://prefill1:8000",
            "9000",
            "--prefill",
            "http://prefill2:8000",
            "none",
            "--decode",
            "http://decode1:8001",
            "--decode",
            "http://decode2:8001",
            "--prefill-policy",
            "power_of_two",
            "--decode-policy",
            "round_robin",
        ]

        router_args = parse_router_args(args)

        assert router_args.pd_disaggregation is True
        assert router_args.prefill_urls == [
            ("http://prefill1:8000", 9000),
            ("http://prefill2:8000", None),
        ]
        assert router_args.decode_urls == ["http://decode1:8001", "http://decode2:8001"]
        assert router_args.prefill_policy == "power_of_two"
        assert router_args.decode_policy == "round_robin"

    def test_parse_service_discovery_args(self):
        """Test parsing service discovery arguments."""
        args = [
            "--service-discovery",
            "--selector",
            "app=worker",
            "env=prod",
            "--service-discovery-port",
            "8080",
            "--service-discovery-namespace",
            "default",
        ]

        router_args = parse_router_args(args)

        assert router_args.service_discovery is True
        assert router_args.selector == {"app": "worker", "env": "prod"}
        assert router_args.service_discovery_port == 8080
        assert router_args.service_discovery_namespace == "default"

    def test_parse_retry_and_circuit_breaker_args(self):
        """Test parsing retry and circuit breaker arguments."""
        args = [
            "--retry-max-retries",
            "3",
            "--retry-initial-backoff-ms",
            "100",
            "--retry-max-backoff-ms",
            "10000",
            "--retry-backoff-multiplier",
            "2.0",
            "--retry-jitter-factor",
            "0.1",
            "--disable-retries",
            "--cb-failure-threshold",
            "5",
            "--cb-success-threshold",
            "2",
            "--cb-timeout-duration-secs",
            "30",
            "--cb-window-duration-secs",
            "60",
            "--disable-circuit-breaker",
        ]

        router_args = parse_router_args(args)

        # Test retry configuration
        assert router_args.retry_max_retries == 3
        assert router_args.retry_initial_backoff_ms == 100
        assert router_args.retry_max_backoff_ms == 10000
        assert router_args.retry_backoff_multiplier == 2.0
        assert router_args.retry_jitter_factor == 0.1
        assert router_args.disable_retries is True

        # Test circuit breaker configuration
        assert router_args.cb_failure_threshold == 5
        assert router_args.cb_success_threshold == 2
        assert router_args.cb_timeout_duration_secs == 30
        assert router_args.cb_window_duration_secs == 60
        assert router_args.disable_circuit_breaker is True

    def test_parse_rate_limiting_args(self):
        """Test parsing rate limiting arguments."""
        args = [
            "--max-concurrent-requests",
            "512",
            "--queue-size",
            "200",
            "--queue-timeout-secs",
            "120",
            "--rate-limit-tokens-per-second",
            "100",
        ]

        router_args = parse_router_args(args)

        assert router_args.max_concurrent_requests == 512
        assert router_args.queue_size == 200
        assert router_args.queue_timeout_secs == 120
        assert router_args.rate_limit_tokens_per_second == 100

    def test_parse_health_check_args(self):
        """Test parsing health check arguments."""
        args = [
            "--health-failure-threshold",
            "2",
            "--health-success-threshold",
            "1",
            "--health-check-timeout-secs",
            "3",
            "--health-check-interval-secs",
            "30",
            "--health-check-endpoint",
            "/healthz",
        ]

        router_args = parse_router_args(args)

        assert router_args.health_failure_threshold == 2
        assert router_args.health_success_threshold == 1
        assert router_args.health_check_timeout_secs == 3
        assert router_args.health_check_interval_secs == 30
        assert router_args.health_check_endpoint == "/healthz"

    def test_parse_cors_args(self):
        """Test parsing CORS arguments."""
        args = [
            "--cors-allowed-origins",
            "http://localhost:3000",
            "https://example.com",
        ]

        router_args = parse_router_args(args)

        assert router_args.cors_allowed_origins == [
            "http://localhost:3000",
            "https://example.com",
        ]

    def test_parse_tokenizer_args(self):
        """Test parsing tokenizer arguments."""
        # Note: model-path and tokenizer-path arguments are not available in current implementation
        # This test is skipped until those arguments are added
        pytest.skip("Tokenizer arguments not available in current implementation")

    def test_parse_invalid_args(self):
        """Test parsing invalid arguments."""
        # Test invalid policy
        with pytest.raises(SystemExit):
            parse_router_args(["--policy", "invalid_policy"])

        # Test invalid bootstrap port
        with pytest.raises(ValueError, match="Invalid bootstrap port"):
            parse_router_args(
                [
                    "--pd-disaggregation",
                    "--prefill",
                    "http://prefill1:8000",
                    "invalid_port",
                ]
            )

    def test_help_output(self):
        """Test that help output is generated correctly."""
        with pytest.raises(SystemExit) as exc_info:
            parse_router_args(["--help"])

        # SystemExit with code 0 indicates help was displayed
        assert exc_info.value.code == 0
