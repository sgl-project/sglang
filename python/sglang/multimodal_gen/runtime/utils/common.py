# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import importlib
import ipaddress
import os
import platform
import signal
import socket
import sys
import threading
from functools import lru_cache

import psutil
import torch
import zmq

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
    context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool
) -> zmq.Socket:
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
    elif socket_type == zmq.DEALER:
        set_send_opt()
        set_recv_opt()
    elif socket_type == zmq.REQ:
        set_send_opt()
        set_recv_opt()
    elif socket_type == zmq.REP:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket


# https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
@lru_cache(maxsize=1)
def is_hip() -> bool:
    return torch.version.hip is not None


@lru_cache(maxsize=1)
def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


@lru_cache(maxsize=1)
def is_cuda_alike():
    return is_cuda() or is_hip()


@lru_cache(maxsize=1)
def is_blackwell():
    if not is_cuda():
        return False
    return torch.cuda.get_device_capability()[0] == 10


@lru_cache(maxsize=1)
def is_hpu() -> bool:
    return hasattr(torch, "hpu") and torch.hpu.is_available()


@lru_cache(maxsize=1)
def is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache(maxsize=1)
def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache(maxsize=1)
def is_host_cpu_x86() -> bool:
    machine = platform.machine().lower()
    return (
        machine in ("x86_64", "amd64", "i386", "i686")
        and hasattr(torch, "cpu")
        and torch.cpu.is_available()
    )


@lru_cache(maxsize=1)
def is_cpu() -> bool:
    return os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1" and is_host_cpu_x86()


# cuda


def set_cuda_arch():
    capability = torch.cuda.get_device_capability()
    arch = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"


def get_bool_env_var(env_var_name: str, default: str | bool = "false") -> bool:
    raw_value = os.getenv(env_var_name, None)
    if raw_value is None:
        raw_value = str(default)

    value_str = str(raw_value).strip().lower()
    truthy = {"1", "true", "yes", "y", "t", "on"}
    falsy = {"0", "false", "no", "n", "f", "off", ""}

    if value_str in truthy:
        return True
    if value_str in falsy:
        return False

    default_bool = str(default).strip().lower() in truthy
    logger.warning(
        "Unrecognized boolean for %s=%r; falling back to default=%r",
        env_var_name,
        raw_value,
        default_bool,
    )
    return default_bool


def is_flashinfer_available():
    """
    Check whether flashinfer is available.
    As of Oct. 6, 2024, it is only available on NVIDIA GPUs.
    """
    # if not get_bool_env_var("SGLANG_IS_FLASHINFER_AVAILABLE", default="true"):
    #     return False
    return importlib.util.find_spec("flashinfer") is not None and is_cuda()


# env var managements

_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    # FIXME: move your environment variable to sglang.srt.environ
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) see non-understandable value={value} and treat as false"
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values
