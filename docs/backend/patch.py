import os

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


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):
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
