import weakref

import nest_asyncio

nest_asyncio.apply()

from sglang.utils import execute_shell_command, reserve_port

DEFAULT_MAX_RUNNING_REQUESTS = 200
DEFAULT_MAX_TOTAL_TOKENS = 20480

import sglang.srt.server_args as server_args_mod

_original_post_init = server_args_mod.ServerArgs.__post_init__


def patched_post_init(self):
    _original_post_init(self)
    if self.max_running_requests is None:
        self.max_running_requests = DEFAULT_MAX_RUNNING_REQUESTS
    if self.max_total_tokens is None:
        self.max_total_tokens = DEFAULT_MAX_TOTAL_TOKENS
    self.disable_cuda_graph = True


server_args_mod.ServerArgs.__post_init__ = patched_post_init

process_socket_map = weakref.WeakKeyDictionary()


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):
    """
    Launch the server using the given command.
    If no port is specified, a free port is reserved.
    """
    if port is None:
        port, lock_socket = reserve_port(host)
    else:
        lock_socket = None

    extra_flags = (
        f"--max-running-requests {DEFAULT_MAX_RUNNING_REQUESTS} "
        f"--max-total-tokens {DEFAULT_MAX_TOTAL_TOKENS} "
        f"--disable-cuda-graph"
    )

    full_command = f"{command} --port {port} {extra_flags}"
    process = execute_shell_command(full_command)

    if lock_socket is not None:
        process_socket_map[process] = lock_socket

    return process, port
