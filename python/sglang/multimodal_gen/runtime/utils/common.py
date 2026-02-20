# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import ipaddress
import logging
import os
import platform
import signal
import socket
import sys
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

import psutil
import torch
import zmq

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

# use the native logger to avoid circular import
logger = logging.getLogger(__name__)


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def configure_ipv6(dist_init_addr):
    addr = dist_init_addr
    end = addr.find("]")
    if end == -1:
        raise ValueError("invalid IPv6 address format: missing ']'")

    host = addr[: end + 1]

    # this only validates the address without brackets: we still need the below checks.
    # if it's invalid, immediately raise an error so we know it's not formatting issues.
    if not is_valid_ipv6_address(host[1:end]):
        raise ValueError(f"invalid IPv6 address: {host}")

    port_str = None
    if len(addr) > end + 1:
        if addr[end + 1] == ":":
            port_str = addr[end + 2 :]
        else:
            raise ValueError("received IPv6 address format: expected ':' after ']'")

    if not port_str:
        raise ValueError(
            "a port must be specified in IPv6 address (format: [ipv6]:port)"
        )

    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(f"invalid port in IPv6 address: '{port_str}'")
    return port, host


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False


def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: str,
    bind: bool,
    max_bind_retries: int = 10,
) -> tuple[zmq.Socket, str]:
    """
    Create and configure a ZMQ socket.

    Args:
        context: ZMQ context
        socket_type: Type of ZMQ socket
        endpoint: Endpoint string (e.g., "tcp://localhost:5555")
        bind: Whether to bind (True) or connect (False)
        max_bind_retries: Maximum number of retries if bind fails due to address already in use

    Returns:
        A tuple of (socket, actual_endpoint). The actual_endpoint may differ from the
        requested endpoint if bind retry was needed.
    """
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type in [zmq.DEALER, zmq.REQ, zmq.REP, zmq.ROUTER]:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        # Parse port from endpoint for retry logic
        import re

        port_match = re.search(r":(\d+)$", endpoint)

        if port_match and max_bind_retries > 1:
            original_port = int(port_match.group(1))
            last_exception = None

            for attempt in range(max_bind_retries):
                try:
                    current_endpoint = endpoint
                    if attempt > 0:
                        # Try next port (increment by 42 to match settle_port logic)
                        current_port = original_port + attempt * 42
                        current_endpoint = re.sub(
                            r":(\d+)$", f":{current_port}", endpoint
                        )
                        logger.info(
                            f"ZMQ bind failed for port {original_port + (attempt - 1) * 42}, "
                            f"retrying with port {current_port} (attempt {attempt + 1}/{max_bind_retries})"
                        )

                    socket.bind(current_endpoint)

                    if attempt > 0:
                        logger.warning(
                            f"Successfully bound ZMQ socket to {current_endpoint} after {attempt + 1} attempts. "
                            f"Original port {original_port} was unavailable."
                        )

                    return socket, current_endpoint

                except zmq.ZMQError as e:
                    last_exception = e
                    if e.errno == zmq.EADDRINUSE and attempt < max_bind_retries - 1:
                        # Address already in use, try next port
                        continue
                    elif attempt == max_bind_retries - 1:
                        # Last attempt failed
                        logger.error(
                            f"Failed to bind ZMQ socket after {max_bind_retries} attempts. "
                            f"Original endpoint: {endpoint}, Last tried port: {original_port + attempt * 42}"
                        )
                        raise
                    else:
                        # Different error, raise immediately
                        raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
        else:
            # No retry logic needed (either no port in endpoint or max_bind_retries == 1)
            socket.bind(endpoint)
            return socket, endpoint
    else:
        socket.connect(endpoint)
        return socket, endpoint

    return socket, endpoint


# https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip


@lru_cache(maxsize=1)
def is_host_cpu_x86() -> bool:
    machine = platform.machine().lower()
    return (
        machine in ("x86_64", "amd64", "i386", "i686")
        and hasattr(torch, "cpu")
        and torch.cpu.is_available()
    )


# cuda


def set_cuda_arch():
    capability = torch.cuda.get_device_capability()
    arch = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"


# env var managements

_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = str(value).strip().lower()

    truthy_values = {"1", "true", "yes", "y", "t", "on"}
    falsy_values = {"0", "false", "no", "n", "f", "off", ""}

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) see non-understandable value={value} and treat as false"
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


class DiffusionMetricsCollector:
    """Prometheus metrics for diffusion runtime."""

    def __init__(self):
        from prometheus_client import Counter, Gauge, Histogram

        self.num_queue_reqs = Gauge(
            name="sglang:diffusion_num_queue_reqs",
            documentation="Number of requests in the diffusion scheduler waiting queue.",
            multiprocess_mode="mostrecent",
        )
        self.num_running_reqs = Gauge(
            name="sglang:diffusion_num_running_reqs",
            documentation="Number of currently running diffusion requests.",
            multiprocess_mode="mostrecent",
        )
        self.requests_total = Counter(
            name="sglang:diffusion_requests_total",
            documentation="Total number of diffusion requests by status.",
            labelnames=["status", "is_warmup"],
        )
        self.request_latency_seconds = Histogram(
            name="sglang:diffusion_request_latency_seconds",
            documentation="End-to-end diffusion request latency in seconds.",
            labelnames=["status", "is_warmup"],
            buckets=(
                0.01,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                120,
                300,
            ),
        )
        self.queue_time_seconds = Histogram(
            name="sglang:diffusion_queue_time_seconds",
            documentation="Histogram of queueing time in seconds for diffusion generation requests.",
            buckets=(
                0.0,
                0.1,
                0.2,
                0.5,
                1,
                2,
                3,
                4,
                5,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
            ),
        )
        self.lora_loaded_adapters = Gauge(
            name="sglang:diffusion_lora_loaded_adapters",
            documentation="Number of loaded diffusion LoRA adapters.",
            multiprocess_mode="mostrecent",
        )
        self.lora_active_modules = Gauge(
            name="sglang:diffusion_lora_active_modules",
            documentation="Number of diffusion modules with active LoRA adapters.",
            multiprocess_mode="mostrecent",
        )
        self.lora_active_adapters = Gauge(
            name="sglang:diffusion_lora_active_adapters",
            documentation="Number of unique active diffusion LoRA adapters.",
            multiprocess_mode="mostrecent",
        )
        self.lora_module_active = Gauge(
            name="sglang:diffusion_lora_module_active",
            documentation="Whether LoRA is active for a diffusion module (1 active, 0 inactive).",
            labelnames=["module"],
            multiprocess_mode="mostrecent",
        )
        self._observed_modules: set[str] = set()

    def set_queue_depth(self, queue_depth: int) -> None:
        self.num_queue_reqs.set(max(queue_depth, 0))

    def set_running_reqs(self, running_reqs: int) -> None:
        self.num_running_reqs.set(max(running_reqs, 0))

    def observe_request(self, status: str, is_warmup: bool, latency_s: float) -> None:
        status_label = status if status in ("success", "error") else "unknown"
        warmup_label = "true" if is_warmup else "false"
        labels = {"status": status_label, "is_warmup": warmup_label}
        self.requests_total.labels(**labels).inc()
        self.request_latency_seconds.labels(**labels).observe(max(latency_s, 0.0))

    def observe_queue_time(self, wait_s: float) -> None:
        self.queue_time_seconds.observe(max(wait_s, 0.0))

    def clear_lora_status(self) -> None:
        self.lora_loaded_adapters.set(0)
        self.lora_active_modules.set(0)
        self.lora_active_adapters.set(0)
        for module_name in self._observed_modules:
            self.lora_module_active.labels(module=module_name).set(0)

    def update_lora_status(self, status: dict[str, Any]) -> None:
        loaded_adapters = status.get("loaded_adapters", [])
        loaded_count = len(loaded_adapters) if isinstance(loaded_adapters, list) else 0

        active = status.get("active", {})
        active_map = active if isinstance(active, dict) else {}
        active_modules = set(active_map.keys())

        active_adapters: set[str] = set()
        for module_entries in active_map.values():
            if not isinstance(module_entries, list):
                continue
            for entry in module_entries:
                if not isinstance(entry, dict):
                    continue
                nickname = entry.get("nickname")
                if not nickname:
                    continue
                for adapter in str(nickname).split(","):
                    adapter = adapter.strip()
                    if adapter:
                        active_adapters.add(adapter)

        self.lora_loaded_adapters.set(loaded_count)
        self.lora_active_modules.set(len(active_modules))
        self.lora_active_adapters.set(len(active_adapters))

        all_modules = self._observed_modules | active_modules
        for module_name in all_modules:
            self.lora_module_active.labels(module=module_name).set(
                1 if module_name in active_modules else 0
            )
        self._observed_modules = all_modules


_diffusion_metrics_collector_lock = threading.Lock()
_diffusion_metrics_collector: Optional[DiffusionMetricsCollector] = None


def get_diffusion_metrics_collector(
    server_args: Optional["ServerArgs"] = None,
) -> Optional[DiffusionMetricsCollector]:
    global _diffusion_metrics_collector

    if _diffusion_metrics_collector is not None:
        return _diffusion_metrics_collector

    if server_args is not None and not server_args.enable_metrics:
        return None
    if server_args is None:
        return None

    with _diffusion_metrics_collector_lock:
        if _diffusion_metrics_collector is None:
            _diffusion_metrics_collector = DiffusionMetricsCollector()

    return _diffusion_metrics_collector
