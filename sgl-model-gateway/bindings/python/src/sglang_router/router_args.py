import argparse
import dataclasses
import logging
import os
from typing import Dict, List, Optional

from sglang_router.sglang_router_rs import get_available_tool_call_parsers

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RouterArgs:
    # Worker configuration
    worker_urls: List[str] = dataclasses.field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = 30000

    # PD-specific configuration
    mini_lb: bool = False
    pd_disaggregation: bool = False  # Enable PD disaggregated mode
    prefill_urls: List[tuple] = dataclasses.field(
        default_factory=list
    )  # List of (url, bootstrap_port)
    decode_urls: List[str] = dataclasses.field(default_factory=list)

    # Routing policy
    policy: str = "cache_aware"
    prefill_policy: Optional[str] = None  # Specific policy for prefill nodes in PD mode
    decode_policy: Optional[str] = None  # Specific policy for decode nodes in PD mode
    worker_startup_timeout_secs: int = 1800
    worker_startup_check_interval: int = 30
    cache_threshold: float = 0.3
    balance_abs_threshold: int = 64
    balance_rel_threshold: float = 1.5
    eviction_interval_secs: int = 60
    max_tree_size: int = 2**26
    max_idle_secs: int = 4 * 3600
    assignment_mode: str = "random"  # Mode for manual policy new routing key assignment
    max_payload_size: int = 512 * 1024 * 1024  # 512MB default for large batches
    bucket_adjust_interval_secs: int = 5
    dp_aware: bool = False
    enable_igw: bool = False  # Enable IGW (Inter-Gateway) mode for multi-model support
    api_key: Optional[str] = None
    log_dir: Optional[str] = None
    log_level: Optional[str] = None
    # Service discovery configuration
    service_discovery: bool = False
    selector: Dict[str, str] = dataclasses.field(default_factory=dict)
    service_discovery_port: int = 80
    service_discovery_namespace: Optional[str] = None
    # PD service discovery configuration
    prefill_selector: Dict[str, str] = dataclasses.field(default_factory=dict)
    decode_selector: Dict[str, str] = dataclasses.field(default_factory=dict)
    bootstrap_port_annotation: str = "sglang.ai/bootstrap-port"
    # Prometheus configuration
    prometheus_port: Optional[int] = None
    prometheus_host: Optional[str] = None
    prometheus_duration_buckets: Optional[List[float]] = None
    # Request ID headers configuration
    request_id_headers: Optional[List[str]] = None
    # Request timeout in seconds
    request_timeout_secs: int = 1800
    # Grace period in seconds to wait for in-flight requests during shutdown
    shutdown_grace_period_secs: int = 180
    # Max concurrent requests for rate limiting (-1 to disable)
    max_concurrent_requests: int = -1
    # Queue size for pending requests when max concurrent limit reached
    queue_size: int = 100
    # Maximum time (in seconds) a request can wait in queue before timing out
    queue_timeout_secs: int = 60
    # Token bucket refill rate (tokens per second). If not set, defaults to max_concurrent_requests
    rate_limit_tokens_per_second: Optional[int] = None
    # CORS allowed origins
    cors_allowed_origins: List[str] = dataclasses.field(default_factory=list)
    # Retry configuration
    retry_max_retries: int = 5
    retry_initial_backoff_ms: int = 50
    retry_max_backoff_ms: int = 30_000
    retry_backoff_multiplier: float = 1.5
    retry_jitter_factor: float = 0.2
    disable_retries: bool = False
    # Health check configuration
    health_failure_threshold: int = 3
    health_success_threshold: int = 2
    health_check_timeout_secs: int = 5
    health_check_interval_secs: int = 60
    health_check_endpoint: str = "/health"
    disable_health_check: bool = False
    # Circuit breaker configuration
    cb_failure_threshold: int = 10
    cb_success_threshold: int = 3
    cb_timeout_duration_secs: int = 60
    cb_window_duration_secs: int = 120
    disable_circuit_breaker: bool = False
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    chat_template: Optional[str] = None
    # Tokenizer cache configuration
    tokenizer_cache_enable_l0: bool = False
    tokenizer_cache_l0_max_entries: int = 10000
    tokenizer_cache_enable_l1: bool = False
    tokenizer_cache_l1_max_memory: int = 50 * 1024 * 1024  # 50MB
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    # MCP server configuration
    mcp_config_path: Optional[str] = None
    # Backend selection
    backend: str = "sglang"
    # History backend configuration
    history_backend: str = "memory"
    oracle_wallet_path: Optional[str] = None
    oracle_tns_alias: Optional[str] = None
    oracle_connect_descriptor: Optional[str] = None
    oracle_username: Optional[str] = None
    oracle_password: Optional[str] = None
    oracle_pool_min: int = 1
    oracle_pool_max: int = 16
    oracle_pool_timeout_secs: int = 30
    postgres_db_url: Optional[str] = None
    postgres_pool_max: int = 16
    redis_url: Optional[str] = None
    redis_pool_max: int = 16
    redis_retention_days: int = 30
    # mTLS configuration for worker communication
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None
    ca_cert_paths: List[str] = dataclasses.field(default_factory=list)
    # Server TLS configuration
    server_cert_path: Optional[str] = None
    server_key_path: Optional[str] = None
    # Trace
    enable_trace: bool = False
    otlp_traces_endpoint: str = "localhost:4317"
    # Control plane authentication
    # API keys for control plane auth (list of tuples: id, name, key, role)
    control_plane_api_keys: List[tuple] = dataclasses.field(default_factory=list)
    control_plane_audit_enabled: bool = False
    # JWT/OIDC configuration for control plane auth
    jwt_issuer: Optional[str] = None
    jwt_audience: Optional[str] = None
    jwt_jwks_uri: Optional[str] = None
    jwt_role_mapping: Dict[str, str] = dataclasses.field(default_factory=dict)

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
        use_router_prefix: bool = False,
        exclude_host_port: bool = False,
    ):
        """
        Add router-specific arguments to an argument parser.

        Args:
            parser: The argument parser to add arguments to
            use_router_prefix: If True, prefix all arguments with 'router-' to avoid conflicts
            exclude_host_port: If True, don't add host and port arguments (used when inheriting from server)
        """
        prefix = "router-" if use_router_prefix else ""

        # Create argument groups for organized --help output
        worker_group = parser.add_argument_group(
            "Worker Configuration", "Settings for worker connections and URLs"
        )
        routing_group = parser.add_argument_group(
            "Routing Policy", "Load balancing and routing configuration"
        )
        pd_group = parser.add_argument_group(
            "PD Disaggregation", "Prefill-Decode disaggregated mode settings"
        )
        k8s_group = parser.add_argument_group(
            "Service Discovery (Kubernetes)", "Kubernetes-based worker discovery"
        )
        logging_group = parser.add_argument_group("Logging", "Log output configuration")
        prometheus_group = parser.add_argument_group(
            "Prometheus Metrics", "Metrics export configuration"
        )
        request_group = parser.add_argument_group(
            "Request Handling", "Request timeout and ID configuration"
        )
        rate_limit_group = parser.add_argument_group(
            "Rate Limiting", "Concurrent request and queue limits"
        )
        retry_group = parser.add_argument_group(
            "Retry Configuration", "Automatic retry behavior for failed requests"
        )
        cb_group = parser.add_argument_group(
            "Circuit Breaker", "Circuit breaker pattern configuration"
        )
        health_group = parser.add_argument_group(
            "Health Checks", "Worker health monitoring settings"
        )
        tokenizer_group = parser.add_argument_group(
            "Tokenizer", "Tokenizer and chat template configuration"
        )
        parser_group = parser.add_argument_group(
            "Parsers", "Reasoning and tool-call parser settings"
        )
        backend_group = parser.add_argument_group(
            "Backend", "Backend runtime and history storage selection"
        )
        oracle_group = parser.add_argument_group(
            "Oracle Database", "Oracle database backend configuration"
        )
        postgres_group = parser.add_argument_group(
            "PostgreSQL Database", "PostgreSQL database backend configuration"
        )
        redis_group = parser.add_argument_group(
            "Redis Database", "Redis database backend configuration"
        )
        tls_group = parser.add_argument_group(
            "TLS/mTLS Security", "TLS certificates for server and worker communication"
        )
        trace_group = parser.add_argument_group(
            "Tracing (OpenTelemetry)", "Distributed tracing configuration"
        )
        auth_group = parser.add_argument_group(
            "Control Plane Authentication", "API key and JWT/OIDC authentication"
        )

        # Worker configuration
        if not exclude_host_port:
            worker_group.add_argument(
                "--host",
                type=str,
                default=RouterArgs.host,
                help="Host address to bind the router server. Supports IPv4, IPv6 (e.g., ::, ::1), or 0.0.0.0 for all interfaces",
            )
            worker_group.add_argument(
                "--port",
                type=int,
                default=RouterArgs.port,
                help="Port number to bind the router server",
            )

        worker_group.add_argument(
            "--worker-urls",
            type=str,
            nargs="*",
            default=[],
            help="List of worker URLs. Supports IPv4 and IPv6 addresses (use brackets for IPv6, e.g., http://[::1]:8000 http://192.168.1.1:8000)",
        )

        # Routing policy configuration
        routing_group.add_argument(
            f"--{prefix}policy",
            type=str,
            default=RouterArgs.policy,
            choices=["random", "round_robin", "cache_aware", "power_of_two", "manual"],
            help="Load balancing policy to use. In PD mode, this is used for both prefill and decode unless overridden",
        )
        routing_group.add_argument(
            f"--{prefix}prefill-policy",
            type=str,
            default=None,
            choices=[
                "random",
                "round_robin",
                "cache_aware",
                "power_of_two",
                "manual",
                "bucket",
            ],
            help="Specific policy for prefill nodes in PD mode. If not specified, uses the main policy",
        )
        routing_group.add_argument(
            f"--{prefix}decode-policy",
            type=str,
            default=None,
            choices=["random", "round_robin", "cache_aware", "power_of_two", "manual"],
            help="Specific policy for decode nodes in PD mode. If not specified, uses the main policy",
        )
        routing_group.add_argument(
            f"--{prefix}cache-threshold",
            type=float,
            default=RouterArgs.cache_threshold,
            help="Cache threshold (0.0-1.0) for cache-aware routing",
        )
        routing_group.add_argument(
            f"--{prefix}balance-abs-threshold",
            type=int,
            default=RouterArgs.balance_abs_threshold,
            help="Absolute threshold for load difference. Balancing is triggered if `(max_load - min_load) > abs_threshold` and the relative threshold is also met.",
        )
        routing_group.add_argument(
            f"--{prefix}balance-rel-threshold",
            type=float,
            default=RouterArgs.balance_rel_threshold,
            help="Relative threshold for load difference. Balancing is triggered if `max_load > min_load * rel_threshold` and the absolute threshold is also met.",
        )
        routing_group.add_argument(
            f"--{prefix}bucket-adjust-interval-secs",
            type=int,
            default=RouterArgs.bucket_adjust_interval_secs,
            help="Interval in seconds between bucket boundary adjustment operations",
        )
        routing_group.add_argument(
            f"--{prefix}eviction-interval-secs",
            type=int,
            default=RouterArgs.eviction_interval_secs,
            help="Interval in seconds between cache eviction operations",
        )
        routing_group.add_argument(
            f"--{prefix}max-tree-size",
            type=int,
            default=RouterArgs.max_tree_size,
            help="Maximum size of the approximation tree for cache-aware routing",
        )
        routing_group.add_argument(
            f"--{prefix}max-idle-secs",
            type=int,
            default=RouterArgs.max_idle_secs,
            help="Maximum idle time in seconds before eviction (for manual policy)",
        )
        routing_group.add_argument(
            f"--{prefix}assignment-mode",
            type=str,
            default=RouterArgs.assignment_mode,
            choices=["random", "min_load", "min_group"],
            help="Mode for assigning new routing keys in manual policy: random (default), min_load (worker with fewest requests), min_group (worker with fewest routing keys)",
        )
        routing_group.add_argument(
            f"--{prefix}max-payload-size",
            type=int,
            default=RouterArgs.max_payload_size,
            help="Maximum payload size in bytes",
        )
        routing_group.add_argument(
            f"--{prefix}dp-aware",
            action="store_true",
            help="Enable data parallelism aware schedule",
        )
        routing_group.add_argument(
            f"--{prefix}enable-igw",
            action="store_true",
            help="Enable IGW (Inference-Gateway) mode for multi-model support",
        )

        # PD-specific arguments
        pd_group.add_argument(
            f"--{prefix}mini-lb",
            action="store_true",
            help="Enable MiniLB",
        )
        pd_group.add_argument(
            f"--{prefix}pd-disaggregation",
            action="store_true",
            help="Enable PD (Prefill-Decode) disaggregated mode",
        )
        pd_group.add_argument(
            f"--{prefix}prefill",
            nargs="+",
            action="append",
            help="Prefill server URL and optional bootstrap port. Can be specified multiple times. "
            "Format: --prefill URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, 'none', or omitted (defaults to none).",
        )
        pd_group.add_argument(
            f"--{prefix}decode",
            nargs=1,
            action="append",
            metavar=("URL",),
            help="Decode server URL. Can be specified multiple times.",
        )
        pd_group.add_argument(
            f"--{prefix}worker-startup-timeout-secs",
            type=int,
            default=RouterArgs.worker_startup_timeout_secs,
            help="Timeout in seconds for worker startup and registration (default: 1800 / 30 minutes). Large models can take significant time to load into GPU memory.",
        )
        pd_group.add_argument(
            f"--{prefix}worker-startup-check-interval",
            type=int,
            default=RouterArgs.worker_startup_check_interval,
            help="Interval in seconds between checks for worker startup",
        )

        # Logging configuration
        logging_group.add_argument(
            f"--{prefix}log-dir",
            type=str,
            default=None,
            help="Directory to store log files. If not specified, logs are only output to console.",
        )
        logging_group.add_argument(
            f"--{prefix}log-level",
            type=str,
            default="info",
            choices=["debug", "info", "warn", "error"],
            help="Set the logging level. If not specified, defaults to INFO.",
        )

        # Service discovery configuration
        k8s_group.add_argument(
            f"--{prefix}service-discovery",
            action="store_true",
            help="Enable Kubernetes service discovery",
        )
        k8s_group.add_argument(
            f"--{prefix}selector",
            type=str,
            nargs="+",
            default={},
            help="Label selector for Kubernetes service discovery (format: key1=value1 key2=value2)",
        )
        k8s_group.add_argument(
            f"--{prefix}service-discovery-port",
            type=int,
            default=RouterArgs.service_discovery_port,
            help="Port to use for discovered worker pods",
        )
        k8s_group.add_argument(
            f"--{prefix}service-discovery-namespace",
            type=str,
            help="Kubernetes namespace to watch for pods. If not provided, watches all namespaces (requires cluster-wide permissions)",
        )
        k8s_group.add_argument(
            f"--{prefix}prefill-selector",
            type=str,
            nargs="+",
            default={},
            help="Label selector for prefill server pods in PD mode (format: key1=value1 key2=value2)",
        )
        k8s_group.add_argument(
            f"--{prefix}decode-selector",
            type=str,
            nargs="+",
            default={},
            help="Label selector for decode server pods in PD mode (format: key1=value1 key2=value2)",
        )
        # Prometheus configuration
        prometheus_group.add_argument(
            f"--{prefix}prometheus-port",
            type=int,
            default=29000,
            help="Port to expose Prometheus metrics (default: 29000).",
        )
        prometheus_group.add_argument(
            f"--{prefix}prometheus-host",
            type=str,
            default="0.0.0.0",
            help="Host address to bind the Prometheus metrics server. Supports IPv4, IPv6 (e.g., ::, ::1), or 0.0.0.0 for all interfaces",
        )
        prometheus_group.add_argument(
            f"--{prefix}prometheus-duration-buckets",
            type=float,
            nargs="+",
            help="Buckets for Prometheus duration metrics",
        )

        # Request handling configuration
        request_group.add_argument(
            f"--{prefix}request-id-headers",
            type=str,
            nargs="*",
            help="Custom HTTP headers to check for request IDs (e.g., x-request-id x-trace-id). If not specified, uses common defaults.",
        )
        request_group.add_argument(
            f"--{prefix}request-timeout-secs",
            type=int,
            default=RouterArgs.request_timeout_secs,
            help="Request timeout in seconds",
        )
        request_group.add_argument(
            f"--{prefix}shutdown-grace-period-secs",
            type=int,
            default=RouterArgs.shutdown_grace_period_secs,
            help="Grace period in seconds to wait for in-flight requests during shutdown",
        )
        request_group.add_argument(
            f"--{prefix}cors-allowed-origins",
            type=str,
            nargs="*",
            default=[],
            help="CORS allowed origins (e.g., http://localhost:3000 https://example.com)",
        )

        # Rate limiting configuration
        rate_limit_group.add_argument(
            f"--{prefix}max-concurrent-requests",
            type=int,
            default=RouterArgs.max_concurrent_requests,
            help="Maximum number of concurrent requests allowed (for rate limiting). Set to -1 to disable rate limiting.",
        )
        rate_limit_group.add_argument(
            f"--{prefix}queue-size",
            type=int,
            default=RouterArgs.queue_size,
            help="Queue size for pending requests when max concurrent limit reached (0 = no queue, return 429 immediately)",
        )
        rate_limit_group.add_argument(
            f"--{prefix}queue-timeout-secs",
            type=int,
            default=RouterArgs.queue_timeout_secs,
            help="Maximum time (in seconds) a request can wait in queue before timing out",
        )
        rate_limit_group.add_argument(
            f"--{prefix}rate-limit-tokens-per-second",
            type=int,
            default=RouterArgs.rate_limit_tokens_per_second,
            help="Token bucket refill rate (tokens per second). If not set, defaults to max_concurrent_requests",
        )

        # Retry configuration
        retry_group.add_argument(
            f"--{prefix}retry-max-retries",
            type=int,
            default=RouterArgs.retry_max_retries,
            help="Maximum number of retry attempts for failed requests",
        )
        retry_group.add_argument(
            f"--{prefix}retry-initial-backoff-ms",
            type=int,
            default=RouterArgs.retry_initial_backoff_ms,
            help="Initial backoff delay in milliseconds before first retry",
        )
        retry_group.add_argument(
            f"--{prefix}retry-max-backoff-ms",
            type=int,
            default=RouterArgs.retry_max_backoff_ms,
            help="Maximum backoff delay in milliseconds between retries",
        )
        retry_group.add_argument(
            f"--{prefix}retry-backoff-multiplier",
            type=float,
            default=RouterArgs.retry_backoff_multiplier,
            help="Multiplier for exponential backoff between retries",
        )
        retry_group.add_argument(
            f"--{prefix}retry-jitter-factor",
            type=float,
            default=RouterArgs.retry_jitter_factor,
            help="Jitter factor (0.0-1.0) to add randomness to retry delays",
        )
        retry_group.add_argument(
            f"--{prefix}disable-retries",
            action="store_true",
            help="Disable retries (equivalent to setting retry_max_retries=1)",
        )

        # Circuit breaker configuration
        cb_group.add_argument(
            f"--{prefix}cb-failure-threshold",
            type=int,
            default=RouterArgs.cb_failure_threshold,
            help="Number of failures before circuit breaker opens",
        )
        cb_group.add_argument(
            f"--{prefix}cb-success-threshold",
            type=int,
            default=RouterArgs.cb_success_threshold,
            help="Number of successes in half-open state before closing circuit",
        )
        cb_group.add_argument(
            f"--{prefix}cb-timeout-duration-secs",
            type=int,
            default=RouterArgs.cb_timeout_duration_secs,
            help="Time in seconds before attempting to close an open circuit",
        )
        cb_group.add_argument(
            f"--{prefix}cb-window-duration-secs",
            type=int,
            default=RouterArgs.cb_window_duration_secs,
            help="Sliding window duration in seconds for tracking failures",
        )
        cb_group.add_argument(
            f"--{prefix}disable-circuit-breaker",
            action="store_true",
            help="Disable circuit breaker (equivalent to setting cb_failure_threshold to u32::MAX)",
        )

        # Health check configuration
        health_group.add_argument(
            f"--{prefix}health-failure-threshold",
            type=int,
            default=RouterArgs.health_failure_threshold,
            help="Number of consecutive health check failures before marking worker unhealthy",
        )
        health_group.add_argument(
            f"--{prefix}health-success-threshold",
            type=int,
            default=RouterArgs.health_success_threshold,
            help="Number of consecutive health check successes before marking worker healthy",
        )
        health_group.add_argument(
            f"--{prefix}health-check-timeout-secs",
            type=int,
            default=RouterArgs.health_check_timeout_secs,
            help="Timeout in seconds for health check requests",
        )
        health_group.add_argument(
            f"--{prefix}health-check-interval-secs",
            type=int,
            default=RouterArgs.health_check_interval_secs,
            help="Interval in seconds between runtime health checks",
        )
        health_group.add_argument(
            f"--{prefix}health-check-endpoint",
            type=str,
            default=RouterArgs.health_check_endpoint,
            help="Health check endpoint path",
        )
        health_group.add_argument(
            f"--{prefix}disable-health-check",
            action="store_true",
            default=RouterArgs.disable_health_check,
            help="Disable all worker health checks at startup",
        )
        # Tokenizer configuration
        tokenizer_group.add_argument(
            f"--{prefix}model-path",
            type=str,
            default=None,
            help="Model path for loading tokenizer (HuggingFace model ID or local path)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-path",
            type=str,
            default=None,
            help="Explicit tokenizer path (overrides model_path tokenizer if provided)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}chat-template",
            type=str,
            default=None,
            help="Chat template path (optional)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-enable-l0",
            action="store_true",
            default=RouterArgs.tokenizer_cache_enable_l0,
            help="Enable L0 (whole-string exact match) tokenizer cache (default: False)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-l0-max-entries",
            type=int,
            default=RouterArgs.tokenizer_cache_l0_max_entries,
            help="Maximum number of entries in L0 tokenizer cache (default: 10000)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-enable-l1",
            action="store_true",
            default=RouterArgs.tokenizer_cache_enable_l1,
            help="Enable L1 (prefix matching) tokenizer cache (default: False)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-l1-max-memory",
            type=int,
            default=RouterArgs.tokenizer_cache_l1_max_memory,
            help="Maximum memory for L1 tokenizer cache in bytes (default: 50MB)",
        )

        # Parser configuration
        parser_group.add_argument(
            f"--{prefix}reasoning-parser",
            type=str,
            default=None,
            help="Specify the parser for reasoning models (e.g., deepseek-r1, qwen3)",
        )
        tool_call_parser_choices = get_available_tool_call_parsers()
        parser_group.add_argument(
            f"--{prefix}tool-call-parser",
            type=str,
            default=None,
            choices=tool_call_parser_choices,
            help=f"Specify the parser for tool-call interactions (e.g., json, qwen)",
        )
        parser_group.add_argument(
            f"--{prefix}mcp-config-path",
            type=str,
            default=None,
            help="Path to MCP (Model Context Protocol) server configuration file",
        )

        # Backend selection
        backend_group.add_argument(
            f"--{prefix}backend",
            type=str,
            default=RouterArgs.backend,
            choices=["sglang", "openai"],
            help="Backend runtime to use (default: sglang)",
        )
        backend_group.add_argument(
            f"--{prefix}history-backend",
            type=str,
            default=RouterArgs.history_backend,
            choices=["memory", "none", "oracle", "postgres", "redis"],
            help="History storage backend for conversations and responses (default: memory)",
        )

        # Oracle configuration
        oracle_group.add_argument(
            f"--{prefix}oracle-wallet-path",
            type=str,
            default=os.getenv("ATP_WALLET_PATH"),
            help="Path to Oracle ATP wallet directory (env: ATP_WALLET_PATH)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-tns-alias",
            type=str,
            default=os.getenv("ATP_TNS_ALIAS"),
            help="Oracle TNS alias from tnsnames.ora (env: ATP_TNS_ALIAS).",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-connect-descriptor",
            type=str,
            default=os.getenv("ATP_DSN"),
            help="Oracle connection descriptor/DSN (full connection string) (env: ATP_DSN)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-username",
            type=str,
            default=os.getenv("ATP_USER"),
            help="Oracle database username (env: ATP_USER)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-password",
            type=str,
            default=os.getenv("ATP_PASSWORD"),
            help="Oracle database password (env: ATP_PASSWORD)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-pool-min",
            type=int,
            default=int(os.getenv("ATP_POOL_MIN", RouterArgs.oracle_pool_min)),
            help="Minimum Oracle connection pool size (default: 1, env: ATP_POOL_MIN)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-pool-max",
            type=int,
            default=int(os.getenv("ATP_POOL_MAX", RouterArgs.oracle_pool_max)),
            help="Maximum Oracle connection pool size (default: 16, env: ATP_POOL_MAX)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-pool-timeout-secs",
            type=int,
            default=int(
                os.getenv("ATP_POOL_TIMEOUT_SECS", RouterArgs.oracle_pool_timeout_secs)
            ),
            help="Oracle connection pool timeout in seconds (default: 30, env: ATP_POOL_TIMEOUT_SECS)",
        )

        # Postgres configuration
        postgres_group.add_argument(
            f"--{prefix}postgres-db-url",
            type=str,
            default=os.getenv("POSTGRES_DB_URL"),
            help="PostgreSQL database connection URL (env: POSTGRES_DB_URL)",
        )
        postgres_group.add_argument(
            f"--{prefix}postgres-pool-max",
            type=int,
            default=int(os.getenv("POSTGRES_POOL_MAX", RouterArgs.postgres_pool_max)),
            help="Maximum PostgreSQL connection pool size (default: 16, env: POSTGRES_POOL_MAX)",
        )

        # Redis configuration
        redis_group.add_argument(
            f"--{prefix}redis-url",
            type=str,
            default=os.getenv("REDIS_URL"),
            help="Redis connection URL (env: REDIS_URL)",
        )
        redis_group.add_argument(
            f"--{prefix}redis-pool-max",
            type=int,
            default=int(os.getenv("REDIS_POOL_MAX", RouterArgs.redis_pool_max)),
            help="Maximum Redis connection pool size (default: 16, env: REDIS_POOL_MAX)",
        )
        redis_group.add_argument(
            f"--{prefix}redis-retention-days",
            type=int,
            default=int(
                os.getenv("REDIS_RETENTION_DAYS", RouterArgs.redis_retention_days)
            ),
            help="Redis data retention in days (-1 for persistent, default: 30, env: REDIS_RETENTION_DAYS)",
        )

        # TLS/mTLS configuration
        tls_group.add_argument(
            f"--{prefix}client-cert-path",
            type=str,
            default=None,
            help="Path to client certificate for mTLS authentication with workers",
        )
        tls_group.add_argument(
            f"--{prefix}client-key-path",
            type=str,
            default=None,
            help="Path to client private key for mTLS authentication with workers",
        )
        tls_group.add_argument(
            f"--{prefix}ca-cert-paths",
            type=str,
            nargs="*",
            default=[],
            help="Path(s) to CA certificate(s) for verifying worker TLS certificates. Can specify multiple CAs.",
        )
        tls_group.add_argument(
            f"--{prefix}tls-cert-path",
            type=str,
            default=None,
            help="Path to server TLS certificate (PEM format)",
        )
        tls_group.add_argument(
            f"--{prefix}tls-key-path",
            type=str,
            default=None,
            help="Path to server TLS private key (PEM format)",
        )

        # Tracing configuration
        trace_group.add_argument(
            f"--{prefix}enable-trace",
            action="store_true",
            help="Enable opentelemetry trace",
        )
        trace_group.add_argument(
            f"--{prefix}otlp-traces-endpoint",
            type=str,
            default="localhost:4317",
            help="Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
        )

        # Control plane authentication
        auth_group.add_argument(
            f"--{prefix}api-key",
            type=str,
            default=None,
            help="The api key used for the authorization with the worker. Useful when the dp aware scheduling strategy is enabled.",
        )
        auth_group.add_argument(
            f"--{prefix}control-plane-api-keys",
            type=str,
            nargs="*",
            default=[],
            help="API keys for control plane authentication. Format: 'id:name:role:key' where role is 'admin' or 'user'. "
            "Example: --control-plane-api-keys 'key1:Service Account:admin:secret123' 'key2:Read Only:user:secret456'",
        )
        auth_group.add_argument(
            f"--{prefix}control-plane-audit-enabled",
            action="store_true",
            default=False,
            help="Enable audit logging for control plane operations",
        )
        auth_group.add_argument(
            f"--{prefix}jwt-issuer",
            type=str,
            default=None,
            help="OIDC issuer URL for JWT authentication (e.g., https://login.microsoftonline.com/{tenant}/v2.0)",
        )
        auth_group.add_argument(
            f"--{prefix}jwt-audience",
            type=str,
            default=None,
            help="Expected audience claim for JWT tokens (usually the client ID or API identifier)",
        )
        auth_group.add_argument(
            f"--{prefix}jwt-jwks-uri",
            type=str,
            default=None,
            help="Explicit JWKS URI. If not provided, discovered from issuer via .well-known/openid-configuration",
        )
        auth_group.add_argument(
            f"--{prefix}jwt-role-mapping",
            type=str,
            nargs="*",
            default=[],
            help="Mapping from IDP role/group names to gateway roles. Format: 'idp_role=gateway_role'. "
            "Example: --jwt-role-mapping 'Gateway.Admin=admin' 'Gateway.User=user'",
        )

    @classmethod
    def from_cli_args(
        cls, args: argparse.Namespace, use_router_prefix: bool = False
    ) -> "RouterArgs":
        """
        Create RouterArgs instance from parsed command line arguments.

        Args:
            args: Parsed command line arguments
            use_router_prefix: If True, look for arguments with 'router-' prefix
        """
        prefix = "router_" if use_router_prefix else ""
        cli_args_dict = vars(args)
        args_dict = {}

        for attr in dataclasses.fields(cls):
            # Auto strip prefix from args
            if f"{prefix}{attr.name}" in cli_args_dict:
                args_dict[attr.name] = cli_args_dict[f"{prefix}{attr.name}"]
            elif attr.name in cli_args_dict:
                args_dict[attr.name] = cli_args_dict[attr.name]

            # Special handling for CLI args with dashes vs dataclass fields with underscores
            # e.g. --tls-cert-path maps to tls_cert_path in args namespace, but we might want server_cert_path in dataclass
            # Wait, dataclass fields are server_cert_path/server_key_path
            # CLI args are tls_cert_path/tls_key_path
            # We need to manually map them if names don't match

        # Map tls args to server cert/key path
        if f"{prefix}tls_cert_path" in cli_args_dict:
            args_dict["server_cert_path"] = cli_args_dict[f"{prefix}tls_cert_path"]
        if f"{prefix}tls_key_path" in cli_args_dict:
            args_dict["server_key_path"] = cli_args_dict[f"{prefix}tls_key_path"]

        # parse special arguments and remove "--prefill" and "--decode" from cli_args_dict
        args_dict["prefill_urls"] = cls._parse_prefill_urls(
            cli_args_dict.get(f"{prefix}prefill", None)
        )
        args_dict["decode_urls"] = cls._parse_decode_urls(
            cli_args_dict.get(f"{prefix}decode", None)
        )
        args_dict["selector"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}selector", None)
        )
        args_dict["prefill_selector"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}prefill_selector", None)
        )
        args_dict["decode_selector"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}decode_selector", None)
        )

        # Mooncake-specific annotation
        args_dict["bootstrap_port_annotation"] = "sglang.ai/bootstrap-port"

        # Parse control plane API keys
        args_dict["control_plane_api_keys"] = cls._parse_control_plane_api_keys(
            cli_args_dict.get(f"{prefix}control_plane_api_keys", [])
        )

        # Parse JWT role mapping
        args_dict["jwt_role_mapping"] = cls._parse_jwt_role_mapping(
            cli_args_dict.get(f"{prefix}jwt_role_mapping", [])
        )

        return cls(**args_dict)

    def _validate_router_args(self):
        # Validate configuration based on mode
        if self.pd_disaggregation:
            # Warn about policy usage in PD mode
            if self.prefill_policy and self.decode_policy and self.policy:
                logger.warning(
                    "Both --prefill-policy and --decode-policy are specified. "
                    "The main --policy flag will be ignored for PD mode."
                )
            elif self.prefill_policy and not self.decode_policy and self.policy:
                logger.info(
                    f"Using --prefill-policy '{self.prefill_policy}' for prefill nodes "
                    f"and --policy '{self.policy}' for decode nodes."
                )
            elif self.decode_policy and not self.prefill_policy and self.policy:
                logger.info(
                    f"Using --policy '{self.policy}' for prefill nodes "
                    f"and --decode-policy '{self.decode_policy}' for decode nodes."
                )

    @staticmethod
    def _parse_selector(selector_list):
        if not selector_list:
            return {}

        # Support `- --selector\n- a=b c=d` case
        if len(selector_list) == 1 and (" " in selector_list[0]):
            selector_list = selector_list[0].split(" ")

        selector = {}
        for item in selector_list:
            if "=" in item:
                key, value = item.split("=", 1)
                selector[key] = value
        return selector

    @staticmethod
    def _parse_prefill_urls(prefill_list):
        """Parse prefill URLs from --prefill arguments.

        Format: --prefill URL [BOOTSTRAP_PORT]
        Example:
            --prefill http://prefill1:8080 9000  # With bootstrap port
            --prefill http://prefill2:8080 none  # Explicitly no bootstrap port
            --prefill http://prefill3:8080       # Defaults to no bootstrap port
        """
        if not prefill_list:
            return []

        prefill_urls = []
        for prefill_args in prefill_list:

            url = prefill_args[0]

            # Handle optional bootstrap port
            if len(prefill_args) >= 2:
                bootstrap_port_str = prefill_args[1]
                # Handle 'none' as None
                if bootstrap_port_str.lower() == "none":
                    bootstrap_port = None
                else:
                    try:
                        bootstrap_port = int(bootstrap_port_str)
                    except ValueError:
                        raise ValueError(
                            f"Invalid bootstrap port: {bootstrap_port_str}. Must be a number or 'none'"
                        )
            else:
                # No bootstrap port specified, default to None
                bootstrap_port = None

            prefill_urls.append((url, bootstrap_port))

        return prefill_urls

    @staticmethod
    def _parse_decode_urls(decode_list):
        """Parse decode URLs from --decode arguments.

        Format: --decode URL
        Example: --decode http://decode1:8081 --decode http://decode2:8081
        """
        if not decode_list:
            return []

        # decode_list is a list of single-element lists due to nargs=1
        return [url[0] for url in decode_list]

    @staticmethod
    def _parse_control_plane_api_keys(api_keys_list):
        """Parse control plane API keys from --control-plane-api-keys arguments.

        Format: id:name:role:key
        Example: --control-plane-api-keys 'key1:Service Account:admin:secret123'
        """
        if not api_keys_list:
            return []

        parsed_keys = []
        for key_str in api_keys_list:
            parts = key_str.split(":", 3)  # Split into at most 4 parts
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid API key format: '{key_str}'. Expected 'id:name:role:key'"
                )
            key_id, name, role, key = parts
            role_lower = role.lower()
            if role_lower not in ("admin", "user"):
                raise ValueError(f"Invalid role: '{role}'. Must be 'admin' or 'user'")
            parsed_keys.append((key_id, name, key, role_lower))
        return parsed_keys

    @staticmethod
    def _parse_jwt_role_mapping(role_mapping_list):
        """Parse JWT role mapping from --jwt-role-mapping arguments.

        Format: idp_role=gateway_role
        Example: --jwt-role-mapping 'Gateway.Admin=admin' 'Gateway.User=user'
        """
        if not role_mapping_list:
            return {}

        mapping = {}
        for mapping_str in role_mapping_list:
            if "=" not in mapping_str:
                raise ValueError(
                    f"Invalid role mapping format: '{mapping_str}'. Expected 'idp_role=gateway_role'"
                )
            idp_role, gateway_role = mapping_str.split("=", 1)
            gateway_role_lower = gateway_role.lower()
            if gateway_role_lower not in ("admin", "user"):
                raise ValueError(
                    f"Invalid gateway role: '{gateway_role}'. Must be 'admin' or 'user'"
                )
            mapping[idp_role] = gateway_role_lower
        return mapping
