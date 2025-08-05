import argparse
import dataclasses
import logging
import sys
from typing import Dict, List, Optional

from sglang_router import Router
from sglang_router_rs import PolicyType


def setup_logger():
    logger = logging.getLogger("router")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[Router (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


@dataclasses.dataclass
class RouterArgs:
    # Worker configuration
    worker_urls: List[str] = dataclasses.field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 30000

    # PD-specific configuration
    pd_disaggregation: bool = False  # Enable PD disaggregated mode
    prefill_urls: List[tuple] = dataclasses.field(
        default_factory=list
    )  # List of (url, bootstrap_port)
    decode_urls: List[str] = dataclasses.field(default_factory=list)

    # Routing policy
    policy: str = "cache_aware"
    prefill_policy: Optional[str] = None  # Specific policy for prefill nodes in PD mode
    decode_policy: Optional[str] = None  # Specific policy for decode nodes in PD mode
    worker_startup_timeout_secs: int = 300
    worker_startup_check_interval: int = 10
    cache_threshold: float = 0.5
    balance_abs_threshold: int = 32
    balance_rel_threshold: float = 1.0001
    eviction_interval: int = 60
    max_tree_size: int = 2**24
    max_payload_size: int = 256 * 1024 * 1024  # 256MB default for large batches
    dp_aware: bool = False
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
    request_timeout_secs: int = 600
    # Max concurrent requests for rate limiting
    max_concurrent_requests: int = 64
    # CORS allowed origins
    cors_allowed_origins: List[str] = dataclasses.field(default_factory=list)

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
                help="Host address to bind the router server",
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
            help="List of worker URLs (e.g., http://worker1:8000 http://worker2:8000)",
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
            f"--{prefix}eviction-interval",
            type=int,
            default=RouterArgs.eviction_interval,
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
            choices=["debug", "info", "warning", "error", "critical"],
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
            help="Label selector for prefill server pods in PD mode (format: key1=value1 key2=value2)",
        )
        parser.add_argument(
            f"--{prefix}decode-selector",
            type=str,
            nargs="+",
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
            default="127.0.0.1",
            help="Host address to bind the Prometheus metrics server",
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
        parser.add_argument(
            f"--{prefix}max-concurrent-requests",
            type=int,
            default=RouterArgs.max_concurrent_requests,
            help="Maximum number of concurrent requests allowed (for rate limiting)",
        )
        parser.add_argument(
            f"--{prefix}cors-allowed-origins",
            type=str,
            nargs="*",
            default=[],
            help="CORS allowed origins (e.g., http://localhost:3000 https://example.com)",
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
        worker_urls = getattr(args, "worker_urls", [])

        # Parse PD URLs
        prefill_urls = cls._parse_prefill_urls(getattr(args, f"{prefix}prefill", None))
        decode_urls = cls._parse_decode_urls(getattr(args, f"{prefix}decode", None))

        return cls(
            worker_urls=worker_urls,
            host=args.host,
            port=args.port,
            pd_disaggregation=getattr(args, f"{prefix}pd_disaggregation", False),
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
            policy=getattr(args, f"{prefix}policy"),
            prefill_policy=getattr(args, f"{prefix}prefill_policy", None),
            decode_policy=getattr(args, f"{prefix}decode_policy", None),
            worker_startup_timeout_secs=getattr(
                args, f"{prefix}worker_startup_timeout_secs"
            ),
            worker_startup_check_interval=getattr(
                args, f"{prefix}worker_startup_check_interval"
            ),
            cache_threshold=getattr(args, f"{prefix}cache_threshold"),
            balance_abs_threshold=getattr(args, f"{prefix}balance_abs_threshold"),
            balance_rel_threshold=getattr(args, f"{prefix}balance_rel_threshold"),
            eviction_interval=getattr(args, f"{prefix}eviction_interval"),
            max_tree_size=getattr(args, f"{prefix}max_tree_size"),
            max_payload_size=getattr(args, f"{prefix}max_payload_size"),
            dp_aware=getattr(args, f"{prefix}dp_aware", False),
            api_key=getattr(args, f"{prefix}api_key", None),
            log_dir=getattr(args, f"{prefix}log_dir", None),
            log_level=getattr(args, f"{prefix}log_level", None),
            service_discovery=getattr(args, f"{prefix}service_discovery", False),
            selector=cls._parse_selector(getattr(args, f"{prefix}selector", None)),
            service_discovery_port=getattr(args, f"{prefix}service_discovery_port"),
            service_discovery_namespace=getattr(
                args, f"{prefix}service_discovery_namespace", None
            ),
            prefill_selector=cls._parse_selector(
                getattr(args, f"{prefix}prefill_selector", None)
            ),
            decode_selector=cls._parse_selector(
                getattr(args, f"{prefix}decode_selector", None)
            ),
            bootstrap_port_annotation="sglang.ai/bootstrap-port",  # Mooncake-specific annotation
            prometheus_port=getattr(args, f"{prefix}prometheus_port", None),
            prometheus_host=getattr(args, f"{prefix}prometheus_host", None),
            request_id_headers=getattr(args, f"{prefix}request_id_headers", None),
            request_timeout_secs=getattr(
                args, f"{prefix}request_timeout_secs", RouterArgs.request_timeout_secs
            ),
            max_concurrent_requests=getattr(
                args,
                f"{prefix}max_concurrent_requests",
                RouterArgs.max_concurrent_requests,
            ),
            cors_allowed_origins=getattr(args, f"{prefix}cors_allowed_origins", []),
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


def policy_from_str(policy_str: str) -> PolicyType:
    """Convert policy string to PolicyType enum."""
    policy_map = {
        "random": PolicyType.Random,
        "round_robin": PolicyType.RoundRobin,
        "cache_aware": PolicyType.CacheAware,
        "power_of_two": PolicyType.PowerOfTwo,
    }
    return policy_map[policy_str]


def launch_router(args: argparse.Namespace) -> Optional[Router]:
    """
    Launch the SGLang router with the configuration from parsed arguments.

    Args:
        args: Namespace object containing router configuration
            Can be either raw argparse.Namespace or converted RouterArgs

    Returns:
        Router instance if successful, None if failed
    """
    logger = logging.getLogger("router")
    try:
        # Convert to RouterArgs if needed
        if not isinstance(args, RouterArgs):
            router_args = RouterArgs.from_cli_args(args)
        else:
            router_args = args

        # Validate configuration based on mode
        if router_args.pd_disaggregation:
            # Validate PD configuration - skip URL requirements if using service discovery
            if not router_args.service_discovery:
                if not router_args.prefill_urls:
                    raise ValueError("PD disaggregation mode requires --prefill")
                if not router_args.decode_urls:
                    raise ValueError("PD disaggregation mode requires --decode")

            # Warn about policy usage in PD mode
            if (
                router_args.prefill_policy
                and router_args.decode_policy
                and router_args.policy
            ):
                logger.warning(
                    "Both --prefill-policy and --decode-policy are specified. "
                    "The main --policy flag will be ignored for PD mode."
                )
            elif (
                router_args.prefill_policy
                and not router_args.decode_policy
                and router_args.policy
            ):
                logger.info(
                    f"Using --prefill-policy '{router_args.prefill_policy}' for prefill nodes "
                    f"and --policy '{router_args.policy}' for decode nodes."
                )
            elif (
                router_args.decode_policy
                and not router_args.prefill_policy
                and router_args.policy
            ):
                logger.info(
                    f"Using --policy '{router_args.policy}' for prefill nodes "
                    f"and --decode-policy '{router_args.decode_policy}' for decode nodes."
                )

        # Create router with unified constructor
        router = Router(
            worker_urls=(
                []
                if router_args.service_discovery or router_args.pd_disaggregation
                else router_args.worker_urls
            ),
            host=router_args.host,
            port=router_args.port,
            policy=policy_from_str(router_args.policy),
            worker_startup_timeout_secs=router_args.worker_startup_timeout_secs,
            worker_startup_check_interval=router_args.worker_startup_check_interval,
            cache_threshold=router_args.cache_threshold,
            balance_abs_threshold=router_args.balance_abs_threshold,
            balance_rel_threshold=router_args.balance_rel_threshold,
            eviction_interval_secs=router_args.eviction_interval,
            max_tree_size=router_args.max_tree_size,
            max_payload_size=router_args.max_payload_size,
            dp_aware=router_args.dp_aware,
            api_key=router_args.api_key,
            log_dir=router_args.log_dir,
            log_level=router_args.log_level,
            service_discovery=router_args.service_discovery,
            selector=router_args.selector,
            service_discovery_port=router_args.service_discovery_port,
            service_discovery_namespace=router_args.service_discovery_namespace,
            prefill_selector=router_args.prefill_selector,
            decode_selector=router_args.decode_selector,
            prometheus_port=router_args.prometheus_port,
            prometheus_host=router_args.prometheus_host,
            request_timeout_secs=router_args.request_timeout_secs,
            pd_disaggregation=router_args.pd_disaggregation,
            prefill_urls=(
                router_args.prefill_urls if router_args.pd_disaggregation else None
            ),
            decode_urls=(
                router_args.decode_urls if router_args.pd_disaggregation else None
            ),
            prefill_policy=(
                policy_from_str(router_args.prefill_policy)
                if router_args.prefill_policy
                else None
            ),
            decode_policy=(
                policy_from_str(router_args.decode_policy)
                if router_args.decode_policy
                else None
            ),
            request_id_headers=router_args.request_id_headers,
            max_concurrent_requests=router_args.max_concurrent_requests,
            cors_allowed_origins=router_args.cors_allowed_origins,
        )

        router.start()
        return router

    except Exception as e:
        logger.error(f"Error starting router: {e}")
        raise e


class CustomHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Custom formatter that preserves both description formatting and shows defaults"""

    pass


def parse_router_args(args: List[str]) -> RouterArgs:
    """Parse command line arguments and return RouterArgs instance."""
    parser = argparse.ArgumentParser(
        description="""SGLang Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode
  python -m sglang_router.launch_router --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode with same policy for both
  python -m sglang_router.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 9000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --policy cache_aware

  # PD mode with optional bootstrap ports
  python -m sglang_router.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 9000 \\    # With bootstrap port
    --prefill http://prefill2:8000 none \\    # Explicitly no bootstrap port
    --prefill http://prefill3:8000 \\         # Defaults to no bootstrap port
    --decode http://decode1:8001 --decode http://decode2:8001

  # PD mode with different policies for prefill and decode
  python -m sglang_router.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --prefill-policy cache_aware --decode-policy power_of_two

    """,
        formatter_class=CustomHelpFormatter,
    )

    RouterArgs.add_cli_args(parser, use_router_prefix=False)
    return RouterArgs.from_cli_args(parser.parse_args(args), use_router_prefix=False)


def main() -> None:
    router_args = parse_router_args(sys.argv[1:])
    launch_router(router_args)


if __name__ == "__main__":
    main()
