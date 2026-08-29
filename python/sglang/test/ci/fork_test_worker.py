"""Execute test files in isolated fork children of a preloaded interpreter."""

import argparse
import json
import os
import runpy
import signal
import sys
import time
import traceback


def _preload_common_modules() -> None:
    """Load the expensive common stack without creating a CUDA context."""
    import numpy  # noqa: F401
    import pytest  # noqa: F401
    import scipy  # noqa: F401
    import torch
    import triton  # noqa: F401

    if torch.cuda.is_initialized():
        raise RuntimeError("fork test worker initialized CUDA before forking")


def _normalize_exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr, flush=True)
    return 1


def _run_file(filename: str) -> int:
    sys.argv = [filename, "-f"]
    try:
        runpy.run_path(filename, run_name="__main__")
        return 0
    except SystemExit as exc:
        return _normalize_exit_code(exc.code)
    except BaseException:
        traceback.print_exc()
        return 1


def _wait_status_to_returncode(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def run_file_in_fork(filename: str) -> tuple[int, float]:
    sys.stdout.flush()
    sys.stderr.flush()
    tic = time.perf_counter()
    child_pid = os.fork()
    if child_pid == 0:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        returncode = _run_file(filename)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(max(0, min(returncode, 255)))

    _, status = os.waitpid(child_pid, 0)
    return _wait_status_to_returncode(status), time.perf_counter() - tic


def main() -> int:
    if not hasattr(os, "fork"):
        raise RuntimeError("fork test worker requires a POSIX platform")

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-fd", type=int, required=True)
    args = parser.parse_args()
    _preload_common_modules()

    with os.fdopen(args.result_fd, "w", buffering=1) as result_stream:
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("command") == "stop":
                break

            filename = command["filename"]
            returncode, elapsed = run_file_in_fork(filename)
            result_stream.write(
                json.dumps(
                    {
                        "filename": filename,
                        "returncode": returncode,
                        "elapsed": elapsed,
                    }
                )
                + "\n"
            )

    return 0


if __name__ == "__main__":
    # Ignore Ctrl-C in the preloader. The active child keeps the default
    # handler, while suite timeouts terminate the complete process tree.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(main())
