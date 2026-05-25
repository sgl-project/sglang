"""Caller-side entry point for ScriptedRuntime tests.

A test invokes :func:`execute_scripted_runtime` with a generator function.
This module:

1. Resolves the generator function to a qualified name so the scheduler
   subprocess can import it back.
2. Allocates a temp file path where the scheduler subprocess writes any
   exception traceback raised by the script.
3. Launches a regular :class:`Engine` with the ScriptedRuntime fields set
   on ``server_args``.
4. Waits for the scheduler subprocess(es) to exit (they exit on their own
   when the script generator finishes).
5. Re-raises any script-side exception on the caller side.

Why qualified-name + importlib instead of cloudpickle:
- ``mp.Process`` already pickles args; functions must be top-level to be
  picklable across the spawn boundary without extra machinery.
- Importable-by-name makes the path inspectable and IDE-friendly.
- Closures / lambdas are rejected with a clear error.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Callable, Optional

from sglang.srt.entrypoints.engine import Engine
from sglang.test.scripted_runtime.runtime import _resolve_fn


def execute_scripted_runtime(
    script_fn: Callable,
    *,
    model_path: str,
    **engine_overrides,
) -> None:
    """Run ``script_fn`` as a ScriptedRuntime generator inside a sglang Engine.

    ``script_fn`` must be a top-level generator function with signature
    ``def script_fn(t: ScriptedRuntime) -> Generator``. It is resolved by
    qualified name in the scheduler subprocess, so it must be importable.

    Blocks until the scheduler subprocess(es) terminate (the script
    generator returning or raising triggers shutdown). On script-side
    exceptions, re-raises an ``AssertionError`` containing the captured
    traceback on the caller side.
    """
    script_fn_path = f"{script_fn.__module__}:{script_fn.__qualname__}"
    resolved = _resolve_fn(script_fn_path)
    if resolved is not script_fn:
        raise ValueError(
            f"script_fn must be a top-level function importable by "
            f"qualified name; resolved {script_fn_path!r} to a different object"
        )

    # Make the script module importable in the scheduler subprocess. Under
    # spawn-mode multiprocessing the subprocess does not inherit the parent's
    # ``sys.path``, so a pytest-collected file under ``test/srt/`` would fail
    # ``importlib.import_module``. We forward the script file's directory.
    sys_path_entry = _module_directory(script_fn.__module__)

    tb_fd, tb_path = tempfile.mkstemp(prefix="scripted_runtime_tb_", suffix=".txt")
    os.close(tb_fd)

    engine: Optional[Engine] = None
    try:
        engine = Engine(
            model_path=model_path,
            scripted_runtime_fn_path=script_fn_path,
            scripted_runtime_traceback_path=tb_path,
            scripted_runtime_sys_path_entry=sys_path_entry,
            **engine_overrides,
        )
        # The subprocess watchdog SIGQUITs the parent if any scheduler subprocess
        # exits non-zero (e.g., script assertion failure). For scripted tests
        # we want to surface the assertion as a normal Python exception, not
        # SIGQUIT the test process. Stop the watchdog before the script can
        # finish.
        _stop_subprocess_watchdog(engine)

        # Scheduler subprocess runs the script in its event loop and exits
        # when the generator finishes. Block here until it does.
        engine._scheduler_init_result.wait_for_completion()

        with open(tb_path) as f:
            tb_text = f.read()
        if tb_text.strip():
            raise AssertionError(f"ScriptedRuntime script failed:\n{tb_text}")
    finally:
        try:
            os.unlink(tb_path)
        except OSError:
            pass
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                pass


def _module_directory(module_name: str) -> Optional[str]:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return os.path.dirname(os.path.abspath(module_file))


def _stop_subprocess_watchdog(engine: Engine) -> None:
    tm = getattr(engine, "tokenizer_manager", None)
    if tm is None:
        return
    watchdog = getattr(tm, "_subprocess_watchdog", None)
    if watchdog is not None:
        watchdog.stop()
