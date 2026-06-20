"""Connection target for HTTP benchmark scripts.

Owns the launch-vs-connect decision in one place: a benchmark only needs a base
URL, which comes either from a server we launch or one already running.
"""

import dataclasses
import multiprocessing
import os
import time
from typing import Callable, Optional

import requests

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import resolve_base_url

DEFAULT_TIMEOUT = 600

# Field defaults of ServerArgs, used to detect when --host/--port were set
# explicitly (and would be silently ignored in connect mode).
_SERVER_ARGS_DEFAULTS = {f.name: f.default for f in dataclasses.fields(ServerArgs)}


def server_is_up(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
    """Return True if a server answers /v1/models with 200 at base_url."""
    try:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }
        response = requests.get(
            f"{base_url}/v1/models", headers=headers, timeout=timeout
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def _launch_server_target(launch_server_func: Callable, server_args: ServerArgs):
    try:
        launch_server_func(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_or_reuse_server(launch_server_func: Callable, server_args: ServerArgs):
    base_url = resolve_base_url("", server_args.host, server_args.port)

    # Reuse an already-running server instead of forking a second one onto the
    # occupied port, where it would orphan, compete for the GPU, and OOM.
    if server_is_up(base_url, timeout=5):
        print(
            f"WARNING: reusing the server already running at {base_url} "
            f"(--model and server-launch args ignored). Pass --base-url to silence."
        )
        return None, base_url

    proc = multiprocessing.Process(
        target=_launch_server_target,
        args=(
            launch_server_func,
            server_args,
        ),
    )
    proc.start()

    start_time = time.time()
    while time.time() - start_time < DEFAULT_TIMEOUT:
        # Fail fast if the server died during startup (e.g. OOM).
        if not proc.is_alive():
            raise RuntimeError(
                f"Server process exited during startup (exit code "
                f"{proc.exitcode}); see the traceback above for the cause."
            )
        if server_is_up(base_url):
            return proc, base_url
        time.sleep(10)

    # Timed out: kill the half-started server so it does not linger as an orphan.
    kill_process_tree(proc.pid)
    raise TimeoutError("Server failed to start within the timeout period.")


@dataclasses.dataclass
class BenchEndpoint:
    """A base URL plus the lifecycle of any server we launched to back it.
    ``close()`` tears down a launched server; for a connected one it is a no-op.
    """

    base_url: str
    _proc: Optional[multiprocessing.Process] = None

    def close(self) -> None:
        if self._proc is not None:
            kill_process_tree(self._proc.pid)
            self._proc = None


def acquire_endpoint(
    server_args: ServerArgs,
    base_url: str = "",
    launch_server_func: Callable = launch_server,
) -> BenchEndpoint:
    """Resolve the benchmark target -- the single launch-vs-connect decision.

    base_url given: connect to it (server_args is ignored). base_url empty:
    launch a server from server_args. Caller must close() the result.
    """
    if base_url:
        ignored = [
            f"--{name}"
            for name in ("host", "port")
            if getattr(server_args, name) != _SERVER_ARGS_DEFAULTS[name]
        ]
        if ignored:
            print(
                f"WARNING: --base-url is set, so {' / '.join(ignored)} (and other "
                f"launch args) are ignored; benchmarking the server at {base_url}."
            )
        return BenchEndpoint(base_url=base_url)

    proc, url = launch_or_reuse_server(launch_server_func, server_args)
    return BenchEndpoint(base_url=url, _proc=proc)
