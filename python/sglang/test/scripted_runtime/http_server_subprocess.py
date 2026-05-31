"""Subprocess entry point for scripted-runtime tests.

:func:`launch_scripted_http_server` runs inside a dedicated ``mp.Process``
(spawned by :class:`ScriptedHttpServer`). It sets the
``SGLANG_TEST_SCRIPTED_RUNTIME*`` env vars and launches a **real** HTTP server
via :func:`launch_server`. The HTTP server stays up for the whole test class;
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
``SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY`` set here keeps the
scripted-runtime package directory on the spawn-mode subprocess's ``sys.path``.
"""

from __future__ import annotations

import os
from typing import Any

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.environ import envs
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

    Enables the scripted runtime via the ``SGLANG_TEST_SCRIPTED_RUNTIME*`` env
    vars so every scheduler subprocess installs a :class:`ScriptedSchedulerHook`;
    the driver rank's hook owns the ZMQ dispatch loop that pulls and runs each
    caller-requested sub-script. The env vars are set on this (single-purpose)
    subprocess before :func:`launch_server`, so the scheduler it spawns
    inherits them.

    Blocks in ``launch_server`` until the scheduler subprocess(es) terminate
    (the dispatch loop returning on shutdown triggers a clean scheduler exit;
    the watchdog then stops the server). On a fatal scheduler-side exception
    the traceback is written to ``traceback_path`` for the session to surface.
    """
    # Spawn-mode mp subprocesses don't inherit the parent's ``sys.path``;
    # forward the scripted-runtime package directory so user sub-scripts can
    # be imported by qualified name in the scheduler subprocess.
    sys_path_entry = os.path.dirname(os.path.abspath(__file__))

    envs.SGLANG_TEST_SCRIPTED_RUNTIME.set(True)
    envs.SGLANG_TEST_SCRIPTED_RUNTIME_TRACEBACK_PATH.set(traceback_path)
    envs.SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY.set(sys_path_entry)

    server_args = ServerArgs(
        model_path=model_path,
        host=host,
        port=port,
        **engine_overrides,
    )

    launch_server(server_args)
