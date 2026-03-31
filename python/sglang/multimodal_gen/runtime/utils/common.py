# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import ipaddress
import logging
import os
import platform
import re
import signal
import socket
import sys
import tempfile
import threading
import time
from functools import lru_cache
from typing import Dict, Tuple

import psutil
import torch
import zmq

# use the native logger to avoid circular import
logger = logging.getLogger(__name__)

# Temporary directory for prometheus multiprocess mode
# Cleaned up automatically when this object is garbage collected
prometheus_multiproc_dir: tempfile.TemporaryDirectory


def set_prometheus_multiproc_dir():
    # Set prometheus multiprocess directory
    # sglang uses prometheus multiprocess mode
    # we need to set this before importing prometheus_client
    # https://prometheus.github.io/client_python/multiprocess/
    global prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        logger.debug("User set PROMETHEUS_MULTIPROC_DIR detected.")
        prometheus_multiproc_dir = tempfile.TemporaryDirectory(
            dir=os.environ["PROMETHEUS_MULTIPROC_DIR"]
        )
    else:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
    logger.debug("PROMETHEUS_MULTIPROC_DIR: %s", os.environ["PROMETHEUS_MULTIPROC_DIR"])


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


def add_prometheus_middleware(app):
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import (
        GC_COLLECTOR,
        PLATFORM_COLLECTOR,
        PROCESS_COLLECTOR,
        CollectorRegistry,
        make_asgi_app,
        multiprocess,
    )
    from starlette.routing import Mount

    registry = CollectorRegistry()
    registry.register(GC_COLLECTOR)
    registry.register(PLATFORM_COLLECTOR)
    registry.register(PROCESS_COLLECTOR)
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


class RefCountedGauge:
    def __init__(self, gauge):
        self._gauge = gauge
        self._refcount: Dict[str, int] = {}

    def inc(self, key: str):
        if key in self._refcount:
            self._refcount[key] += 1
        else:
            self._refcount[key] = 1
            self._gauge.inc()

    def dec(self, key: str):
        if key in self._refcount:
            self._refcount[key] -= 1
            if self._refcount[key] == 0:
                del self._refcount[key]
                self._gauge.dec()


def add_prometheus_track_response_middleware(app):
    from prometheus_client import Counter, Gauge, Histogram, Summary

    http_request_counter = Counter(
        name="sglang:http_requests_total",
        documentation="Total number of HTTP requests by endpoint and method",
        labelnames=["endpoint", "method"],
    )

    http_response_counter = Counter(
        name="sglang:http_responses_total",
        documentation="Total number of HTTP responses by endpoint and status code",
        labelnames=["endpoint", "status_code", "method"],
    )

    http_requests_active = Gauge(
        name="sglang:http_requests_active",
        documentation="Number of currently active HTTP requests",
        labelnames=["endpoint", "method"],
        multiprocess_mode="livesum",
    )

    http_request_size = Summary(
        name="sglang:http_request_size_bytes",
        documentation=(
            "Content length of incoming requests by endpoint. "
            "Only value of header is respected. Otherwise ignored."
        ),
        labelnames=["endpoint"],
    )

    http_response_size = Summary(
        name="sglang:http_response_size_bytes",
        documentation=(
            "Content length of outgoing responses by endpoint. "
            "Only value of header is respected. Otherwise ignored."
        ),
        labelnames=["endpoint"],
    )

    http_request_latency = Histogram(
        name="sglang:http_request_latency_seconds",
        documentation="End-to-end HTTP request latency in seconds",
        labelnames=["endpoint", "status_code", "method"],
        buckets=(
            0.4,
            0.8,
            2.0,
            4.0,
            8.0,
            10.0,
            20.0,
            60.0,
            80.0,
            100.0,
            200.0,
        ),
    )

    routing_keys_active = RefCountedGauge(
        Gauge(
            name="sglang:routing_keys_active",
            documentation="Number of unique routing keys with active requests",
            multiprocess_mode="livesum",
        )
    )

    @app.middleware("http")
    async def track_http_status_code(request, call_next):
        # With recording all requests, we have the risk of high cardinality if requests have arbitrary unhandled paths.
        # But given that SGLang engines with metrics enabled are usually behind routers this looks safe.
        path, _is_handled_path = _get_fastapi_request_path(request)
        method = request.method
        routing_key = request.headers.get("x-smg-routing-key")

        start_time = time.monotonic()
        http_request_counter.labels(endpoint=path, method=method).inc()
        http_requests_active.labels(endpoint=path, method=method).inc()
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                http_request_size.labels(endpoint=path).observe(float(content_length))
            except ValueError:
                pass
        if routing_key:
            routing_keys_active.inc(routing_key)
        status_code = "500"
        response = None
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            http_response_counter.labels(
                endpoint=path,
                method=method,
                status_code=status_code,
            ).inc()

            return response
        finally:
            response_length = (
                response.headers.get("content-length") if response else None
            )
            if response_length is not None:
                try:
                    http_response_size.labels(endpoint=path).observe(
                        float(response_length)
                    )
                except ValueError:
                    pass
            http_request_latency.labels(
                endpoint=path,
                method=method,
                status_code=status_code,
            ).observe(time.monotonic() - start_time)
            http_requests_active.labels(endpoint=path, method=method).dec()
            if routing_key:
                routing_keys_active.dec(routing_key)


def _get_fastapi_request_path(request) -> Tuple[str, bool]:
    from starlette.routing import Match

    for route in request.app.routes:
        match, _child_scope = route.matches(request.scope)
        if match == Match.FULL:
            return route.path, True

    return request.url.path, False


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


# musa


def set_musa_arch():
    capability = torch.cuda.get_device_capability()
    arch = f"{capability[0]}{capability[1]}"
    os.environ["TORCH_MUSA_ARCH_LIST"] = f"{arch}"


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


try:
    import sgl_kernel  # noqa: F401

    is_intel_amx_backend_available = hasattr(
        torch.ops.sgl_kernel, "convert_weight_packed"
    )
except:
    is_intel_amx_backend_available = False

try:
    # move torch._C._cpu._is_amx_tile_supported() from cpu_has_amx_support
    # to support torch compile
    is_amx_tile_supported = torch._C._cpu._is_amx_tile_supported()
except:
    is_amx_tile_supported = False


def cpu_has_amx_support():
    return is_amx_tile_supported and is_intel_amx_backend_available


def use_intel_amx_backend(layer):
    return getattr(layer, "use_intel_amx_backend", False)
