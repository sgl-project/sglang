"""Subprocess entry point for ScriptedRuntime tests.

:func:`execute_scripted_runtime` runs inside a dedicated ``mp.Process``
(spawned by :class:`ScriptedRuntimeSession`). It resolves the script
generator to a qualified name, builds ``ServerArgs`` with the
ScriptedRuntime fields set, and launches a **real** HTTP server via
:func:`launch_server`. The HTTP server stays up for the whole test class;
``launch_server`` blocks in uvicorn until the scheduler subprocess exits
(which it does once the router script returns on shutdown), at which point
the watchdog tears the server process down.

Running the HTTP server in its own subprocess (rather than a background
thread of the test process) matches a real deployment and sidesteps the
``uvicorn`` "signal only works in main thread" failure — uvicorn installs
its signal handlers in the subprocess's main thread.

Why qualified-name + importlib rather than cloudpickle: ``mp.Process``
spawn requires top-level picklable functions anyway; importable-by-name
is inspectable and IDE-friendly, and lambdas / closures are rejected
with a clear error.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Literal, Optional

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.test.scripted_runtime.runtime import _resolve_fn


def execute_scripted_runtime(
    script_fn: Callable,
    *,
    model_path: str,
    host: str,
    port: int,
    traceback_path: str,
    # === Wishlist kwargs (disagg sidecar mode, see
    # 2026-05-26-round-5-de-skip-and-api-wishlist.md §4.2) ===
    disagg_role: Literal["none", "prefill", "decode"] = "none",
    disagg_sidecar_decode_args: Optional[Dict[str, Any]] = None,
    disagg_sidecar_prefill_args: Optional[Dict[str, Any]] = None,
    disagg_router_args: Optional[Dict[str, Any]] = None,
    **engine_overrides: Any,
) -> None:
    """Run ``script_fn`` as a ScriptedRuntime generator behind a real HTTP server.

    ``script_fn`` must be a top-level generator function with signature
    ``def script_fn(t: ScriptedRuntime) -> Generator``. It is resolved by
    qualified name in the scheduler subprocess, so it must be importable.

    Blocks in ``launch_server`` until the scheduler subprocess(es) terminate
    (the router script returning on shutdown triggers a clean scheduler exit;
    the watchdog then stops the server). On a fatal scheduler-side exception
    the traceback is written to ``traceback_path`` for the session to surface.

    Disagg sidecar mode (``disagg_role != "none"``) is wishlist — see
    ``2026-05-26-round-5-de-skip-and-api-wishlist.md`` §4.2. In that mode this
    entry point will fork the ScriptedRuntime-controlled engine, the
    opposite-side engine, and the router; today any non-default disagg kwarg
    raises ``NotImplementedError``.

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

    server_args = ServerArgs(
        model_path=model_path,
        host=host,
        port=port,
        scripted_runtime_fn_path=script_fn_path,
        scripted_runtime_traceback_path=traceback_path,
        scripted_runtime_sys_path_entry=sys_path_entry,
        **engine_overrides,
    )

    launch_server(server_args)


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
