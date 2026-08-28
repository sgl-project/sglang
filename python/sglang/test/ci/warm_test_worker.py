"""Run several test files in one interpreter for lower CI startup overhead.

The parent process sends one JSON command per line on stdin. Results are written
to a dedicated file descriptor so test output can keep using stdout/stderr.
"""

import argparse
import gc
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path


def _normalize_exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr, flush=True)
    return 1


def _registered_test_root(filename: str) -> Path | None:
    path = Path(filename).resolve()
    for parent in path.parents:
        if parent.name == "registered" and parent.parent.name == "test":
            return parent
    return None


def _discard_loaded_test_modules(modules_before: set[str], test_root: Path | None):
    """Drop test modules loaded by pytest while retaining expensive libraries."""
    if test_root is None:
        return

    for name in set(sys.modules) - modules_before:
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            Path(module_file).resolve().relative_to(test_root)
        except (OSError, ValueError):
            continue
        sys.modules.pop(name, None)


def run_file(filename: str) -> tuple[int, float]:
    """Execute a test file as ``python filename -f`` would execute it."""
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    old_environ = os.environ.copy()
    old_sys_path = sys.path[:]
    modules_before = set(sys.modules)
    test_root = _registered_test_root(filename)
    tic = time.perf_counter()

    try:
        sys.argv = [filename, "-f"]
        runpy.run_path(filename, run_name="__main__")
        returncode = 0
    except SystemExit as exc:
        returncode = _normalize_exit_code(exc.code)
    except BaseException:
        traceback.print_exc()
        returncode = 1
    finally:
        sys.argv = old_argv
        sys.path[:] = old_sys_path
        os.environ.clear()
        os.environ.update(old_environ)
        os.chdir(old_cwd)
        _discard_loaded_test_modules(modules_before, test_root)
        gc.collect()

    return returncode, time.perf_counter() - tic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-fd", type=int, required=True)
    args = parser.parse_args()

    with os.fdopen(args.result_fd, "w", buffering=1) as result_stream:
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("command") == "stop":
                break

            filename = command["filename"]
            returncode, elapsed = run_file(filename)
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
    raise SystemExit(main())
