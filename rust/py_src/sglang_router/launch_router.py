import argparse
import dataclasses
import logging
import sys
from typing import List, Optional

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
    worker_urls: List[str]
    host: str = "127.0.0.1"
    port: int = 30000

    # Routing policy
    policy: str = "cache_aware"
    cache_threshold: float = 0.5
    balance_abs_threshold: int = 32
    balance_rel_threshold: float = 1.0001
    eviction_interval: int = 60
    max_tree_size: int = 2**24
    verbose: bool = False

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
            f"--{prefix}verbose",
            action="store_true",
            help="Enable verbose logging",
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
        return cls(
            worker_urls=args.worker_urls,
            host=args.host,
            port=args.port,
            policy=getattr(args, f"{prefix}policy"),
            cache_threshold=getattr(args, f"{prefix}cache_threshold"),
            balance_abs_threshold=getattr(args, f"{prefix}balance_abs_threshold"),
            balance_rel_threshold=getattr(args, f"{prefix}balance_rel_threshold"),
            eviction_interval=getattr(args, f"{prefix}eviction_interval"),
            max_tree_size=getattr(args, f"{prefix}max_tree_size"),
            verbose=getattr(args, f"{prefix}verbose", False),
        )


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
            policy=policy_from_str(router_args.policy),
            host=router_args.host,
            port=router_args.port,
            cache_threshold=router_args.cache_threshold,
            balance_abs_threshold=router_args.balance_abs_threshold,
            balance_rel_threshold=router_args.balance_rel_threshold,
            eviction_interval_secs=router_args.eviction_interval,
            max_tree_size=router_args.max_tree_size,
            verbose=router_args.verbose,
        )

        router.start()
        return router

    except Exception as e:
        logger.error(f"Error starting router: {e}", file=sys.stderr)
        return None


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
    logger = setup_logger()
    router_args = parse_router_args(sys.argv[1:])
    router = launch_router(router_args)

    if router is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
