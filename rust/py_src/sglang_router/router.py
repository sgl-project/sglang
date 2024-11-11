from typing import List, Optional

from sglang_router_rs import PolicyType
from sglang_router_rs import Router as _Router


class Router:
    """
    A high-performance router for distributing requests across worker nodes.

    Args:
        worker_urls: List of URLs for worker nodes that will handle requests
        policy: Load balancing policy to use. Options:
            - PolicyType.Random: Randomly select workers
            - PolicyType.RoundRobin: Distribute requests in round-robin fashion
            - PolicyType.ApproxTree: Tree-based routing using tokenizer similarity
        host: Host address to bind the router server
        port: Port number to bind the router server
        tokenizer_path: Path to tokenizer model file (required for ApproxTree policy)
        cache_threshold: Caching threshold value between 0-1

    """

    def __init__(
        self,
        worker_urls: List[str],
        policy: PolicyType = PolicyType.RoundRobin,
        host: str = "127.0.0.1",
        port: int = 3001,
        tokenizer_path: Optional[str] = None,
        cache_threshold: float = 0.50,
    ):

        self._router = _Router(
            worker_urls=worker_urls,
            policy=policy,
            host=host,
            port=port,
            tokenizer_path=tokenizer_path,
            cache_threshold=cache_threshold,
        )

    def start(self) -> None:
        """Start the router server.

        This method blocks until the server is shut down.
        """
        self._router.start()
