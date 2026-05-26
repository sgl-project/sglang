"""Caller-side entry point for ScriptedRuntime tests.

:func:`execute_scripted_runtime` resolves the script generator to a
qualified name, launches a regular :class:`Engine` with ScriptedRuntime
fields set on ``server_args``, and blocks until the scheduler
subprocess exits.

Why qualified-name + importlib rather than cloudpickle: ``mp.Process``
spawn requires top-level picklable functions anyway; importable-by-name
is inspectable and IDE-friendly, and lambdas / closures are rejected
with a clear error.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Callable, Dict, Literal, Optional

from sglang.srt.entrypoints.engine import Engine
from sglang.test.scripted_runtime.runtime import _resolve_fn


def execute_scripted_runtime(
    script_fn: Callable,
    *,
    model_path: str,
    # === Wishlist kwargs (disagg sidecar mode, see
    # 2026-05-26-round-5-de-skip-and-api-wishlist.md §4.2) ===
    disagg_role: Literal["none", "prefill", "decode"] = "none",
    disagg_sidecar_decode_args: Optional[Dict[str, Any]] = None,
    disagg_sidecar_prefill_args: Optional[Dict[str, Any]] = None,
    disagg_router_args: Optional[Dict[str, Any]] = None,
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

    Disagg sidecar mode (``disagg_role != "none"``) is wishlist — see
    ``2026-05-26-round-5-de-skip-and-api-wishlist.md`` §4.2. In that
    mode this entry point will fork the ScriptedRuntime-controlled
    engine, the opposite-side engine, and the router; today any
    non-default disagg kwarg raises ``NotImplementedError``.

    Consumed by: test_disagg_prefill_per_chunk_kv_send (disagg),
                 test_disagg_overlap_mid_chunk_tmp_end_idx (disagg),
                 test_disagg_retract_resets_send_state (disagg),
                 test_spec_eagle_disagg_chunked (spec).
    """
    _check_disagg_wishlist_kwargs(
        disagg_role=disagg_role,
        disagg_sidecar_decode_args=disagg_sidecar_decode_args,
        disagg_sidecar_prefill_args=disagg_sidecar_prefill_args,
        disagg_router_args=disagg_router_args,
    )
    script_fn_path = f"{script_fn.__module__}:{script_fn.__qualname__}"
    resolved = _resolve_fn(script_fn_path)
    if resolved is not script_fn:
        raise ValueError(
            f"script_fn must be a top-level function importable by "
            f"qualified name; resolved {script_fn_path!r} to a different object"
        )

    # Spawn-mode mp subprocesses don't inherit the parent's ``sys.path``;
    # forward the script's directory so it can be imported by name.
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
        # The watchdog SIGQUITs the parent on any non-zero scheduler exit.
        # Disable it so a script-side assertion surfaces as a Python
        # exception instead of killing the test runner.
        _stop_subprocess_watchdog(engine)

        # Block until the scheduler subprocess exits — it does so on its
        # own when the script generator finishes.
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


def _check_disagg_wishlist_kwargs(
    *,
    disagg_role: Literal["none", "prefill", "decode"],
    disagg_sidecar_decode_args: Optional[Dict[str, Any]],
    disagg_sidecar_prefill_args: Optional[Dict[str, Any]],
    disagg_router_args: Optional[Dict[str, Any]],
) -> None:
    """Raise NotImplementedError if any disagg sidecar kwarg is non-default.

    Centralised so the wishlist mode surfaces with a clear error at the
    call site once any disagg sidecar field is set.
    """
    any_set = (
        disagg_role != "none"
        or disagg_sidecar_decode_args is not None
        or disagg_sidecar_prefill_args is not None
        or disagg_router_args is not None
    )
    if any_set:
        raise NotImplementedError(
            "scripted_runtime: disagg sidecar mode is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )


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
