from typing import Optional

from sglang_router.router_args import RouterArgs
from sglang_router_rs import PolicyType
from sglang_router_rs import Router as _Router


def policy_from_str(policy_str: Optional[str]) -> PolicyType:
    """Convert policy string to PolicyType enum."""
    if policy_str is None:
        return None
    policy_map = {
        "random": PolicyType.Random,
        "round_robin": PolicyType.RoundRobin,
        "cache_aware": PolicyType.CacheAware,
        "power_of_two": PolicyType.PowerOfTwo,
    }
    return policy_map[policy_str]


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
        host: Host address to bind the router server. Supports IPv4, IPv6 (e.g., ::, ::1), or 0.0.0.0 for all interfaces. Default: '0.0.0.0'
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
        dp_aware: Enable data parallelism aware schedule. Default: False
        enable_igw: Enable IGW (Inference-Gateway) mode for multi-model support. When enabled,
            the router can manage multiple models simultaneously with per-model load balancing
            policies. Default: False
        api_key: The api key used for the authorization with the worker.
            Useful when the dp aware scheduling strategy is enabled.
            Default: None
        log_dir: Directory to store log files. If None, logs are only output to console. Default: None
        log_level: Logging level. Options: 'debug', 'info', 'warn', 'error'.
        service_discovery: Enable Kubernetes service discovery. When enabled, the router will
            automatically discover worker pods based on the selector. Default: False
        selector: Dictionary mapping of label keys to values for Kubernetes pod selection.
            Example: {"app": "sglang-worker"}. Default: {}
        service_discovery_port: Port to use for service discovery. The router will generate
            worker URLs using this port. Default: 80
        service_discovery_namespace: Kubernetes namespace to watch for pods. If not provided,
            watches pods across all namespaces (requires cluster-wide permissions). Default: None
        prefill_selector: Dictionary mapping of label keys to values for Kubernetes pod selection
            for prefill servers (PD mode only). Default: {}
        decode_selector: Dictionary mapping of label keys to values for Kubernetes pod selection
            for decode servers (PD mode only). Default: {}
        prometheus_port: Port to expose Prometheus metrics. Default: None
        prometheus_host: Host address to bind the Prometheus metrics server. Default: None
        pd_disaggregation: Enable PD (Prefill-Decode) disaggregated mode. Default: False
        prefill_urls: List of (url, bootstrap_port) tuples for prefill servers (PD mode only)
        decode_urls: List of URLs for decode servers (PD mode only)
        prefill_policy: Specific load balancing policy for prefill nodes (PD mode only).
            If not specified, uses the main policy. Default: None
        decode_policy: Specific load balancing policy for decode nodes (PD mode only).
            If not specified, uses the main policy. Default: None
        request_id_headers: List of HTTP headers to check for request IDs. If not specified,
            uses common defaults: ['x-request-id', 'x-correlation-id', 'x-trace-id', 'request-id'].
            Example: ['x-my-request-id', 'x-custom-trace-id']. Default: None
        bootstrap_port_annotation: Kubernetes annotation name for bootstrap port (PD mode).
            Default: 'sglang.ai/bootstrap-port'
        request_timeout_secs: Request timeout in seconds. Default: 600
        max_concurrent_requests: Maximum number of concurrent requests allowed for rate limiting. Default: 256
        queue_size: Queue size for pending requests when max concurrent limit reached (0 = no queue, return 429 immediately). Default: 100
        queue_timeout_secs: Maximum time (in seconds) a request can wait in queue before timing out. Default: 60
        rate_limit_tokens_per_second: Token bucket refill rate (tokens per second). If not set, defaults to max_concurrent_requests. Default: None
        cors_allowed_origins: List of allowed origins for CORS. Empty list allows all origins. Default: []
        health_failure_threshold: Number of consecutive health check failures before marking worker unhealthy. Default: 3
        health_success_threshold: Number of consecutive health check successes before marking worker healthy. Default: 2
        health_check_timeout_secs: Timeout in seconds for health check requests. Default: 5
        health_check_interval_secs: Interval in seconds between runtime health checks. Default: 60
        health_check_endpoint: Health check endpoint path. Default: '/health'
        model_path: Model path for loading tokenizer (HuggingFace model ID or local path). Default: None
        tokenizer_path: Explicit tokenizer path (overrides model_path tokenizer if provided). Default: None
    """

    def __init__(self, router: _Router):
        self._router = router

    @staticmethod
    def from_args(args: RouterArgs) -> "Router":
        """Create a router from a RouterArgs instance."""

        args_dict = vars(args)
        # Convert RouterArgs to _Router parameters
        args_dict["worker_urls"] = (
            []
            if args_dict["service_discovery"] or args_dict["pd_disaggregation"]
            else args_dict["worker_urls"]
        )
        args_dict["policy"] = policy_from_str(args_dict["policy"])
        args_dict["prefill_urls"] = (
            args_dict["prefill_urls"] if args_dict["pd_disaggregation"] else None
        )
        args_dict["decode_urls"] = (
            args_dict["decode_urls"] if args_dict["pd_disaggregation"] else None
        )
        args_dict["prefill_policy"] = policy_from_str(args_dict["prefill_policy"])
        args_dict["decode_policy"] = policy_from_str(args_dict["decode_policy"])

        # remove mini_lb parameter
        args_dict.pop("mini_lb")

        return Router(_Router(**args_dict))

    def start(self) -> None:
        """Start the router server.

        This method blocks until the server is shut down.
        """
        self._router.start()
