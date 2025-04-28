from typing import List, Optional

from sglang_router_rs import PolicyType
from sglang_router_rs import Router as _Router


class Router:
    """
    A high-performance router for distributing requests across worker nodes.

    Args:
        worker_urls: List of URLs for worker nodes that will handle requests. Each URL should include
            the protocol, host, and port (e.g., ['http://worker1:8000', 'http://worker2:8000'])
        policy: Load balancing policy to use. Options:
            - PolicyType.Random: Randomly select workers
            - PolicyType.RoundRobin: Distribute requests in round-robin fashion
            - PolicyType.CacheAware: Distribute requests based on cache state and load balance
        host: Host address to bind the router server. Default: '127.0.0.1'
        port: Port number to bind the router server. Default: 3001
        worker_startup_timeout_secs: Timeout in seconds for worker startup. Default: 300
        worker_startup_check_interval: Interval in seconds between checks for worker initialization. Default: 10
        cache_threshold: Cache threshold (0.0-1.0) for cache-aware routing. Routes to cached worker
            if the match rate exceeds threshold, otherwise routes to the worker with the smallest
            tree. Default: 0.5
        balance_abs_threshold: Load balancing is triggered when (max_load - min_load) > abs_threshold
            AND max_load > min_load * rel_threshold. Otherwise, use cache aware. Default: 32
        balance_rel_threshold: Load balancing is triggered when (max_load - min_load) > abs_threshold
            AND max_load > min_load * rel_threshold. Otherwise, use cache aware. Default: 1.0001
        eviction_interval_secs: Interval in seconds between cache eviction operations in cache-aware
            routing. Default: 60
        max_payload_size: Maximum payload size in bytes. Default: 4MB
        max_tree_size: Maximum size of the approximation tree for cache-aware routing. Default: 2^24
        verbose: Enable verbose logging. Default: False
        log_dir: Directory to store log files. If None, logs are only output to console. Default: None
    """

    def __init__(
        self,
        worker_urls: List[str],
        policy: PolicyType = PolicyType.RoundRobin,
        host: str = "127.0.0.1",
        port: int = 3001,
        worker_startup_timeout_secs: int = 300,
        worker_startup_check_interval: int = 10,
        cache_threshold: float = 0.50,
        balance_abs_threshold: int = 32,
        balance_rel_threshold: float = 1.0001,
        eviction_interval_secs: int = 60,
        max_tree_size: int = 2**24,
        max_payload_size: int = 4 * 1024 * 1024,  # 4MB
        verbose: bool = False,
        log_dir: Optional[str] = None,
    ):
        self._router = _Router(
            worker_urls=worker_urls,
            policy=policy,
            host=host,
            port=port,
            worker_startup_timeout_secs=worker_startup_timeout_secs,
            worker_startup_check_interval=worker_startup_check_interval,
            cache_threshold=cache_threshold,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
            eviction_interval_secs=eviction_interval_secs,
            max_tree_size=max_tree_size,
            max_payload_size=max_payload_size,
            verbose=verbose,
            log_dir=log_dir,
        )

    def start(self) -> None:
        """Start the router server.

        This method blocks until the server is shut down.
        """
        self._router.start()
