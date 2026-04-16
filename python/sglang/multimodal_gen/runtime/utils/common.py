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

import psutil
import torch

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
    """Set CUDA architecture for compilation. Only applies to CUDA devices."""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}.{capability[1]}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"
    # For XPU or other platforms, no arch setting needed


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
