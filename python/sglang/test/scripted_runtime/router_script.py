"""Long-lived script that pulls sub-scripts from an IPC socket and runs them.

Installed as the ScriptedRuntime ``script_fn`` by
:class:`ScriptedRuntimeSession`. Owns no test logic of its own — it is a
pure router that ``yield from``s each caller-requested sub-script and
forwards success / failure back over a unix-socket connection.

Crucially, when a sub-script raises (including ``AssertionError``), the
router *captures* the traceback into a socket message and *keeps
running* — it does not re-raise. Re-raising would surface as
``ScriptedRuntimeFinished(ok=False)`` and tear the engine down, voiding
every remaining test in the class.
"""

from __future__ import annotations

import os
import traceback
from multiprocessing.connection import Client
from typing import Generator

from sglang.test.scripted_runtime.runtime import ScriptedRuntime, _resolve_fn


def router_script(t: ScriptedRuntime) -> Generator:
    """Receive ``run`` / ``shutdown`` commands; ``yield from`` each sub-script.

    Runs forever on the scheduler driver rank until the caller sends a
    ``shutdown`` message, at which point the generator returns normally
    and the scheduler tears down.
    """
    addr = os.environ["SGLANG_SCRIPTED_RUNTIME_IPC_ADDR"]
    authkey = bytes.fromhex(os.environ["SGLANG_SCRIPTED_RUNTIME_AUTHKEY"])
    conn = Client(addr, authkey=authkey)
    try:
        while True:
            msg = conn.recv()
            kind = msg["kind"]
            if kind == "shutdown":
                return
            if kind == "run":
                fn = _resolve_fn(msg["fn_path"])
                sub_gen = fn(t)
                try:
                    yield from sub_gen
                except BaseException:
                    conn.send({"kind": "error", "tb": traceback.format_exc()})
                else:
                    conn.send({"kind": "ok"})
                continue
            raise ValueError(f"router: unknown command kind {kind!r}")
    finally:
        conn.close()
