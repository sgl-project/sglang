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
from typing import Any

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs


def launch_scripted_http_server(
    *,
    model_path: str,
    host: str,
    port: int,
    traceback_path: str,
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
    """
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
