from typing import Dict, List, Optional

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
            - PolicyType.PowerOfTwo: Select best of two random workers based on load (PD mode only)
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
        max_payload_size: Maximum payload size in bytes. Default: 256MB
        max_tree_size: Maximum size of the approximation tree for cache-aware routing. Default: 2^24
        verbose: Enable verbose logging. Default: False
        log_dir: Directory to store log files. If None, logs are only output to console. Default: None
        service_discovery: Enable Kubernetes service discovery. When enabled, the router will
            automatically discover worker pods based on the selector. Default: False
        selector: Dictionary mapping of label keys to values for Kubernetes pod selection.
            Example: {"app": "sglang-worker"}. Default: {}
        service_discovery_port: Port to use for service discovery. The router will generate
            worker URLs using this port. Default: 80
        service_discovery_namespace: Kubernetes namespace to watch for pods. If not provided,
            watches pods across all namespaces (requires cluster-wide permissions). Default: None
        prometheus_port: Port to expose Prometheus metrics. Default: None
        prometheus_host: Host address to bind the Prometheus metrics server. Default: None
        pd_disaggregated: Enable PD (Prefill-Decode) disaggregated mode. Default: False
        prefill_urls: List of (url, bootstrap_port) tuples for prefill servers (PD mode only)
        decode_urls: List of URLs for decode servers (PD mode only)
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
        max_payload_size: int = 256 * 1024 * 1024,  # 256MB
        verbose: bool = False,
        log_dir: Optional[str] = None,
        service_discovery: bool = False,
        selector: Dict[str, str] = None,
        service_discovery_port: int = 80,
        service_discovery_namespace: Optional[str] = None,
        prometheus_port: Optional[int] = None,
        prometheus_host: Optional[str] = None,
        pd_disaggregated: bool = False,
        prefill_urls: Optional[List[tuple]] = None,
        decode_urls: Optional[List[str]] = None,
    ):
        if selector is None:
            selector = {}

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
            service_discovery=service_discovery,
            selector=selector,
            service_discovery_port=service_discovery_port,
            service_discovery_namespace=service_discovery_namespace,
            prometheus_port=prometheus_port,
            prometheus_host=prometheus_host,
            pd_disaggregated=pd_disaggregated,
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
        )

    def start(self) -> None:
        """Start the router server.

        This method blocks until the server is shut down.
        """
        self._router.start()
