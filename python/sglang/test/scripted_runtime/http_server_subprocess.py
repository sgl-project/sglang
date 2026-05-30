"""Subprocess entry point for scripted-runtime tests.

:func:`launch_scripted_http_server` runs inside a dedicated ``mp.Process``
(spawned by :class:`ScriptedHttpServer`). It builds ``ServerArgs`` with the
scripted-runtime fields set and launches a **real** HTTP server via
:func:`launch_server`. The HTTP server stays up for the whole test class;
``launch_server`` blocks in uvicorn until the scheduler subprocess exits
(which it does once the dispatch loop returns on shutdown), at which point
the watchdog tears the server process down.

Running the HTTP server in its own subprocess (rather than a background
thread of the test process) matches a real deployment and sidesteps the
``uvicorn`` "signal only works in main thread" failure — uvicorn installs
its signal handlers in the subprocess's main thread.

The scheduler-side dispatch loop (owned by :class:`ScriptedSchedulerHook`)
pulls each caller-requested sub-script over the ZMQ ``PAIR`` socket and
resolves it by qualified name, so user sub-scripts must be importable. The
``scripted_runtime_sys_path_entry`` forwarded here keeps the scripted-runtime
package directory on the spawn-mode subprocess's ``sys.path``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs


def launch_scripted_http_server(
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
    """Run the scripted-runtime dispatch loop behind a real HTTP server.

    Enables the scripted runtime on the ``ServerArgs`` so every scheduler
    subprocess installs a :class:`ScriptedSchedulerHook`; the driver rank's
    hook owns the ZMQ dispatch loop that pulls and runs each caller-requested
    sub-script.

    Blocks in ``launch_server`` until the scheduler subprocess(es) terminate
    (the dispatch loop returning on shutdown triggers a clean scheduler exit;
    the watchdog then stops the server). On a fatal scheduler-side exception
    the traceback is written to ``traceback_path`` for the session to surface.

    Disagg sidecar mode (``disagg_role != "none"``) is wishlist — see
    ``2026-05-26-round-5-de-skip-and-api-wishlist.md`` §4.2. In that mode this
    entry point will fork the scripted-runtime-controlled engine, the
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

    # Spawn-mode mp subprocesses don't inherit the parent's ``sys.path``;
    # forward the scripted-runtime package directory so user sub-scripts can
    # be imported by qualified name in the scheduler subprocess.
    sys_path_entry = os.path.dirname(os.path.abspath(__file__))

    server_args = ServerArgs(
        model_path=model_path,
        host=host,
        port=port,
        enable_scripted_runtime=True,
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
