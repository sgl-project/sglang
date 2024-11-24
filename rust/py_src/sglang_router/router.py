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
            - PolicyType.CacheAware: Distribute requests in cache-aware fashion
        host: Host address to bind the router server. Default: '127.0.0.1'
        port: Port number to bind the router server. Default: 3001
        cache_threshold: Cache threshold (0.0-1.0) for cache-aware routing. Routes to cached worker
            if the match rate exceeds threshold, otherwise routes to the worker with the smallest
            tree. Default: 0.5
        cache_routing_prob: Probability of using cache-aware routing (0.0-1.0). Default 1.0 for
            full cache-aware routing, suitable for perfectly divided prefix workloads. For uneven
            workloads, use a lower value to better distribute requests
        eviction_interval_secs: Interval in seconds between cache eviction operations in cache-aware
            routing. Default: 60
        max_tree_size: Maximum size of the approximation tree for cache-aware routing. Default: 2^24
    """

    def __init__(
        self,
        worker_urls: List[str],
        policy: PolicyType = PolicyType.RoundRobin,
        host: str = "127.0.0.1",
        port: int = 3001,
        cache_threshold: float = 0.50,
        cache_routing_prob: float = 1.0,
        eviction_interval_secs: int = 60,
        max_tree_size: int = 2**24,
    ):
        self._router = _Router(
            worker_urls=worker_urls,
            policy=policy,
            host=host,
            port=port,
            cache_threshold=cache_threshold,
            cache_routing_prob=cache_routing_prob,
            eviction_interval_secs=eviction_interval_secs,
            max_tree_size=max_tree_size,
        )

    def start(self) -> None:
        """Start the router server.

        This method blocks until the server is shut down.
        """
        self._router.start()
