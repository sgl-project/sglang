import fcntl
import json
import os
import socket

import sglang

# Default parameters
DEFAULT_MAX_RUNNING_REQUESTS = 200
DEFAULT_MAX_TOTAL_TOKENS = 20480

LOCKFILE = "/tmp/port_lock"
PORT_REGISTRY = "/tmp/port_registry.json"


def init_port_registry():
    # Initialize port registry if it doesn't exist.
    if not os.path.exists(PORT_REGISTRY):
        with open(PORT_REGISTRY, "w") as f:
            json.dump([], f)


def reserve_port(start=30000, end=40000):
    """
    Reserve an available port using a file lock and a registry.
    Returns the allocated port number.
    """
    init_port_registry()
    with open(LOCKFILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(PORT_REGISTRY, "r") as f:
                used = json.load(f)
        except Exception:
            used = []
        for port in range(start, end):
            if port not in used:
                used.append(port)
                with open(PORT_REGISTRY, "w") as f:
                    json.dump(used, f)
                return port
    raise RuntimeError("No free port available")


def release_port(port):
    """
    Release the reserved port by removing it from the registry.
    """
    with open(LOCKFILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(PORT_REGISTRY, "r") as f:
                used = json.load(f)
        except Exception:
            used = []
        if port in used:
            used.remove(port)
        with open(PORT_REGISTRY, "w") as f:
            json.dump(used, f)


import sglang.srt.server_args as server_args_mod

_original_post_init = server_args_mod.ServerArgs.__post_init__


def patched_post_init(self):
    # Call the original __post_init__ then set defaults if needed.
    _original_post_init(self)
    if self.max_running_requests is None:
        self.max_running_requests = DEFAULT_MAX_RUNNING_REQUESTS
    if self.max_total_tokens is None:
        self.max_total_tokens = DEFAULT_MAX_TOTAL_TOKENS
    self.disable_cuda_graph = True


# Replace the __post_init__ method
server_args_mod.ServerArgs.__post_init__ = patched_post_init


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):
    """
    Launch the server with default parameters, reserve a free port, and disable CUDA graph.

    Parameters:
        command (str): Base command to launch the server.
        host (str): Listening address (default "0.0.0.0").
        port (int): Specific port (if None, a free port is reserved).

    Returns:
        tuple: (process_handle, port)
    """
    from sglang.utils import execute_shell_command

    if port is None:
        port = reserve_port()
    extra_flags = (
        f"--max-running-requests {DEFAULT_MAX_RUNNING_REQUESTS} "
        f"--max-total-tokens {DEFAULT_MAX_TOTAL_TOKENS} "
        f"--disable-cuda-graph"
    )
    full_command = f"{command} --port {port} {extra_flags}"
    process = execute_shell_command(full_command)
    return process, port


def terminate_process(process, port=None):
    """
    Terminate the process and release the reserved port if provided.
    """
    from sglang.srt.utils import kill_process_tree

    kill_process_tree(process.pid)
    if port is not None:
        release_port(port)
