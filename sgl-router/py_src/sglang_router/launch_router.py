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

    # Routing policy
    policy: str = "cache_aware"
    worker_startup_timeout_secs: int = 300
    worker_startup_check_interval: int = 10
    cache_threshold: float = 0.5
    balance_abs_threshold: int = 32
    balance_rel_threshold: float = 1.0001
    eviction_interval: int = 60
    max_tree_size: int = 2**24
    max_payload_size: int = 4 * 1024 * 1024  # 4MB
    verbose: bool = False
    log_dir: Optional[str] = None
    # Service discovery configuration
    service_discovery: bool = False
    selector: Dict[str, str] = dataclasses.field(default_factory=dict)
    service_discovery_port: int = 80
    service_discovery_namespace: Optional[str] = None

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
            nargs="+",
            help="List of worker URLs (e.g., http://worker1:8000 http://worker2:8000)",
        )

        # Routing policy configuration
        parser.add_argument(
            f"--{prefix}policy",
            type=str,
            default=RouterArgs.policy,
            choices=["random", "round_robin", "cache_aware"],
            help="Load balancing policy to use",
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
            f"--{prefix}verbose",
            action="store_true",
            help="Enable verbose logging",
        )
        parser.add_argument(
            f"--{prefix}log-dir",
            type=str,
            default=None,
            help="Directory to store log files. If not specified, logs are only output to console.",
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
        worker_urls = args.worker_urls if args.worker_urls is not None else []
        return cls(
            worker_urls=worker_urls,
            host=args.host,
            port=args.port,
            policy=getattr(args, f"{prefix}policy"),
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
            verbose=getattr(args, f"{prefix}verbose", False),
            log_dir=getattr(args, f"{prefix}log_dir", None),
            service_discovery=getattr(args, f"{prefix}service_discovery", False),
            selector=cls._parse_selector(getattr(args, f"{prefix}selector", None)),
            service_discovery_port=getattr(args, f"{prefix}service_discovery_port"),
            service_discovery_namespace=getattr(
                args, f"{prefix}service_discovery_namespace", None
            ),
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


def policy_from_str(policy_str: str) -> PolicyType:
    """Convert policy string to PolicyType enum."""
    policy_map = {
        "random": PolicyType.Random,
        "round_robin": PolicyType.RoundRobin,
        "cache_aware": PolicyType.CacheAware,
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

        router = Router(
            worker_urls=router_args.worker_urls,
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
            verbose=router_args.verbose,
            log_dir=router_args.log_dir,
            service_discovery=router_args.service_discovery,
            selector=router_args.selector,
            service_discovery_port=router_args.service_discovery_port,
            service_discovery_namespace=router_args.service_discovery_namespace,
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
  python -m sglang_router.launch_router --worker-urls http://worker1:8000 http://worker2:8000
  python -m sglang_router.launch_router --worker-urls http://worker1:8000 http://worker2:8000 --cache-threshold 0.7 --balance-abs-threshold 64 --balance-rel-threshold 1.2

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
