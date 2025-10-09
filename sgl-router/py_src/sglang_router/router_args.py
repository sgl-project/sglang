import argparse
import dataclasses
import logging
from typing import Dict, List, Optional

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
    worker_startup_timeout_secs: int = 600
    worker_startup_check_interval: int = 30
    cache_threshold: float = 0.3
    balance_abs_threshold: int = 64
    balance_rel_threshold: float = 1.5
    eviction_interval_secs: int = 120
    max_tree_size: int = 2**26
    max_payload_size: int = 512 * 1024 * 1024  # 512MB default for large batches
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
    # Request ID headers configuration
    request_id_headers: Optional[List[str]] = None
    # Request timeout in seconds
    request_timeout_secs: int = 1800
    # Max concurrent requests for rate limiting
    max_concurrent_requests: int = 256
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
    # Circuit breaker configuration
    cb_failure_threshold: int = 10
    cb_success_threshold: int = 3
    cb_timeout_duration_secs: int = 60
    cb_window_duration_secs: int = 120
    disable_circuit_breaker: bool = False
    # Tokenizer configuration
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    # Parser configuration
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None

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

        # Worker configuration
        if not exclude_host_port:
            parser.add_argument(
                "--host",
                type=str,
                default=RouterArgs.host,
                help="Host address to bind the router server. Supports IPv4, IPv6 (e.g., ::, ::1), or 0.0.0.0 for all interfaces",
            )
            parser.add_argument(
                "--port",
                type=int,
                default=RouterArgs.port,
                help="Port number to bind the router server",
            )

        parser.add_argument(
            "--worker-urls",
            type=str,
            nargs="*",
            default=[],
            help="List of worker URLs. Supports IPv4 and IPv6 addresses (use brackets for IPv6, e.g., http://[::1]:8000 http://192.168.1.1:8000)",
        )

        # Routing policy configuration
        parser.add_argument(
            f"--{prefix}policy",
            type=str,
            default=RouterArgs.policy,
            choices=["random", "round_robin", "cache_aware", "power_of_two"],
            help="Load balancing policy to use. In PD mode, this is used for both prefill and decode unless overridden",
        )
        parser.add_argument(
            f"--{prefix}prefill-policy",
            type=str,
            default=None,
            choices=["random", "round_robin", "cache_aware", "power_of_two"],
            help="Specific policy for prefill nodes in PD mode. If not specified, uses the main policy",
        )
        parser.add_argument(
            f"--{prefix}decode-policy",
            type=str,
            default=None,
            choices=["random", "round_robin", "cache_aware", "power_of_two"],
            help="Specific policy for decode nodes in PD mode. If not specified, uses the main policy",
        )

        # PD-specific arguments
        parser.add_argument(
            f"--{prefix}mini-lb",
            action="store_true",
            help="Enable MiniLB",
        )
        parser.add_argument(
            f"--{prefix}pd-disaggregation",
            action="store_true",
            help="Enable PD (Prefill-Decode) disaggregated mode",
        )
        parser.add_argument(
            f"--{prefix}prefill",
            nargs="+",
            action="append",
            help="Prefill server URL and optional bootstrap port. Can be specified multiple times. "
            "Format: --prefill URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, 'none', or omitted (defaults to none).",
        )
        parser.add_argument(
            f"--{prefix}decode",
            nargs=1,
            action="append",
            metavar=("URL",),
            help="Decode server URL. Can be specified multiple times.",
        )
        parser.add_argument(
            f"--{prefix}worker-startup-timeout-secs",
            type=int,
            default=RouterArgs.worker_startup_timeout_secs,
            help="Timeout in seconds for worker startup",
        )
        parser.add_argument(
            f"--{prefix}worker-startup-check-interval",
            type=int,
            default=RouterArgs.worker_startup_check_interval,
            help="Interval in seconds between checks for worker startup",
        )
        parser.add_argument(
            f"--{prefix}cache-threshold",
            type=float,
            default=RouterArgs.cache_threshold,
            help="Cache threshold (0.0-1.0) for cache-aware routing",
        )
        parser.add_argument(
            f"--{prefix}balance-abs-threshold",
            type=int,
            default=RouterArgs.balance_abs_threshold,
            help="Load balancing is triggered when (max_load - min_load) > abs_threshold AND max_load > min_load * rel_threshold. Otherwise, use cache aware",
        )
        parser.add_argument(
            f"--{prefix}balance-rel-threshold",
            type=float,
            default=RouterArgs.balance_rel_threshold,
            help="Load balancing is triggered when (max_load - min_load) > abs_threshold AND max_load > min_load * rel_threshold. Otherwise, use cache aware",
        )
        parser.add_argument(
            f"--{prefix}eviction-interval-secs",
            type=int,
            default=RouterArgs.eviction_interval_secs,
            help="Interval in seconds between cache eviction operations",
        )
        parser.add_argument(
            f"--{prefix}max-tree-size",
            type=int,
            default=RouterArgs.max_tree_size,
            help="Maximum size of the approximation tree for cache-aware routing",
        )
        parser.add_argument(
            f"--{prefix}max-payload-size",
            type=int,
            default=RouterArgs.max_payload_size,
            help="Maximum payload size in bytes",
        )
        parser.add_argument(
            f"--{prefix}dp-aware",
            action="store_true",
            help="Enable data parallelism aware schedule",
        )
        parser.add_argument(
            f"--{prefix}enable-igw",
            action="store_true",
            help="Enable IGW (Inference-Gateway) mode for multi-model support",
        )
        parser.add_argument(
            f"--{prefix}api-key",
            type=str,
            default=None,
            help="The api key used for the authorization with the worker.  Useful when the dp aware scheduling strategy is enaled.",
        )
        parser.add_argument(
            f"--{prefix}log-dir",
            type=str,
            default=None,
            help="Directory to store log files. If not specified, logs are only output to console.",
        )
        parser.add_argument(
            f"--{prefix}log-level",
            type=str,
            default="info",
            choices=["debug", "info", "warn", "error"],
            help="Set the logging level. If not specified, defaults to INFO.",
        )
        parser.add_argument(
            f"--{prefix}service-discovery",
            action="store_true",
            help="Enable Kubernetes service discovery",
        )
        parser.add_argument(
            f"--{prefix}selector",
            type=str,
            nargs="+",
            default={},
            help="Label selector for Kubernetes service discovery (format: key1=value1 key2=value2)",
        )
        parser.add_argument(
            f"--{prefix}service-discovery-port",
            type=int,
            default=RouterArgs.service_discovery_port,
            help="Port to use for discovered worker pods",
        )
        parser.add_argument(
            f"--{prefix}service-discovery-namespace",
            type=str,
            help="Kubernetes namespace to watch for pods. If not provided, watches all namespaces (requires cluster-wide permissions)",
        )
        parser.add_argument(
            f"--{prefix}prefill-selector",
            type=str,
            nargs="+",
            default={},
            help="Label selector for prefill server pods in PD mode (format: key1=value1 key2=value2)",
        )
        parser.add_argument(
            f"--{prefix}decode-selector",
            type=str,
            nargs="+",
            default={},
            help="Label selector for decode server pods in PD mode (format: key1=value1 key2=value2)",
        )
        # Prometheus configuration
        parser.add_argument(
            f"--{prefix}prometheus-port",
            type=int,
            default=29000,
            help="Port to expose Prometheus metrics. If not specified, Prometheus metrics are disabled",
        )
        parser.add_argument(
            f"--{prefix}prometheus-host",
            type=str,
            default="0.0.0.0",
            help="Host address to bind the Prometheus metrics server. Supports IPv4, IPv6 (e.g., ::, ::1), or 0.0.0.0 for all interfaces",
        )
        parser.add_argument(
            f"--{prefix}request-id-headers",
            type=str,
            nargs="*",
            help="Custom HTTP headers to check for request IDs (e.g., x-request-id x-trace-id). If not specified, uses common defaults.",
        )
        parser.add_argument(
            f"--{prefix}request-timeout-secs",
            type=int,
            default=RouterArgs.request_timeout_secs,
            help="Request timeout in seconds",
        )
        # Retry configuration
        parser.add_argument(
            f"--{prefix}retry-max-retries",
            type=int,
            default=RouterArgs.retry_max_retries,
        )
        parser.add_argument(
            f"--{prefix}retry-initial-backoff-ms",
            type=int,
            default=RouterArgs.retry_initial_backoff_ms,
        )
        parser.add_argument(
            f"--{prefix}retry-max-backoff-ms",
            type=int,
            default=RouterArgs.retry_max_backoff_ms,
        )
        parser.add_argument(
            f"--{prefix}retry-backoff-multiplier",
            type=float,
            default=RouterArgs.retry_backoff_multiplier,
        )
        parser.add_argument(
            f"--{prefix}retry-jitter-factor",
            type=float,
            default=RouterArgs.retry_jitter_factor,
        )
        parser.add_argument(
            f"--{prefix}disable-retries",
            action="store_true",
            help="Disable retries (equivalent to setting retry_max_retries=1)",
        )
        # Circuit breaker configuration
        parser.add_argument(
            f"--{prefix}cb-failure-threshold",
            type=int,
            default=RouterArgs.cb_failure_threshold,
        )
        parser.add_argument(
            f"--{prefix}cb-success-threshold",
            type=int,
            default=RouterArgs.cb_success_threshold,
        )
        parser.add_argument(
            f"--{prefix}cb-timeout-duration-secs",
            type=int,
            default=RouterArgs.cb_timeout_duration_secs,
        )
        parser.add_argument(
            f"--{prefix}cb-window-duration-secs",
            type=int,
            default=RouterArgs.cb_window_duration_secs,
        )
        parser.add_argument(
            f"--{prefix}disable-circuit-breaker",
            action="store_true",
            help="Disable circuit breaker (equivalent to setting cb_failure_threshold to u32::MAX)",
        )
        # Health check configuration
        parser.add_argument(
            f"--{prefix}health-failure-threshold",
            type=int,
            default=RouterArgs.health_failure_threshold,
            help="Number of consecutive health check failures before marking worker unhealthy",
        )
        parser.add_argument(
            f"--{prefix}health-success-threshold",
            type=int,
            default=RouterArgs.health_success_threshold,
            help="Number of consecutive health check successes before marking worker healthy",
        )
        parser.add_argument(
            f"--{prefix}health-check-timeout-secs",
            type=int,
            default=RouterArgs.health_check_timeout_secs,
            help="Timeout in seconds for health check requests",
        )
        parser.add_argument(
            f"--{prefix}health-check-interval-secs",
            type=int,
            default=RouterArgs.health_check_interval_secs,
            help="Interval in seconds between runtime health checks",
        )
        parser.add_argument(
            f"--{prefix}health-check-endpoint",
            type=str,
            default=RouterArgs.health_check_endpoint,
            help="Health check endpoint path",
        )
        parser.add_argument(
            f"--{prefix}max-concurrent-requests",
            type=int,
            default=RouterArgs.max_concurrent_requests,
            help="Maximum number of concurrent requests allowed (for rate limiting)",
        )
        parser.add_argument(
            f"--{prefix}queue-size",
            type=int,
            default=RouterArgs.queue_size,
            help="Queue size for pending requests when max concurrent limit reached (0 = no queue, return 429 immediately)",
        )
        parser.add_argument(
            f"--{prefix}queue-timeout-secs",
            type=int,
            default=RouterArgs.queue_timeout_secs,
            help="Maximum time (in seconds) a request can wait in queue before timing out",
        )
        parser.add_argument(
            f"--{prefix}rate-limit-tokens-per-second",
            type=int,
            default=RouterArgs.rate_limit_tokens_per_second,
            help="Token bucket refill rate (tokens per second). If not set, defaults to max_concurrent_requests",
        )
        parser.add_argument(
            f"--{prefix}cors-allowed-origins",
            type=str,
            nargs="*",
            default=[],
            help="CORS allowed origins (e.g., http://localhost:3000 https://example.com)",
        )
        # Tokenizer configuration
        parser.add_argument(
            f"--{prefix}model-path",
            type=str,
            default=None,
            help="Model path for loading tokenizer (HuggingFace model ID or local path)",
        )
        parser.add_argument(
            f"--{prefix}tokenizer-path",
            type=str,
            default=None,
            help="Explicit tokenizer path (overrides model_path tokenizer if provided)",
        )
        parser.add_argument(
            f"--{prefix}reasoning-parser",
            type=str,
            default=None,
            help="Specify the parser for reasoning models (e.g., deepseek-r1, qwen3)",
        )
        parser.add_argument(
            f"--{prefix}tool-call-parser",
            type=str,
            default=None,
            help="Specify the parser for handling tool-call interactions",
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

        return cls(**args_dict)

    def _validate_router_args(self):
        # Validate configuration based on mode
        if self.pd_disaggregation:
            # Validate PD configuration - skip URL requirements if using service discovery
            if not self.service_discovery:
                if not self.prefill_urls:
                    raise ValueError("PD disaggregation mode requires --prefill")
                if not self.decode_urls:
                    raise ValueError("PD disaggregation mode requires --decode")

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
