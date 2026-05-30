"""Long-lived script that pulls sub-scripts from an IPC socket and runs them.

Installed as the scripted-runtime ``script_fn`` by
:class:`ScriptedHttpServer`. Owns no test logic of its own — it is a
pure router that ``yield from``s each caller-requested sub-script and
forwards success / failure back over a ZMQ ``PAIR`` socket.

Crucially, when a sub-script raises (including ``AssertionError``), the
router *captures* the traceback into a socket message and *keeps
running* — it does not re-raise. Re-raising would tear the engine down
(the hook ``sys.exit``s the scheduler subprocess), voiding every
remaining test in the class.
"""

from __future__ import annotations

import os
import traceback
from typing import Generator

import zmq

from sglang.srt.utils.network import get_zmq_socket
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
    Shutdown,
)
from sglang.test.scripted_runtime.utils import resolve_fn


def router_script(t: ScriptedContext) -> Generator:
    """Receive :class:`RunScript` / :class:`Shutdown`; ``yield from`` each sub-script.

    Runs forever on the scheduler driver rank until the caller sends a
    :class:`Shutdown`, at which point the generator returns normally and the
    scheduler tears down.
    """
    endpoint = os.environ["SGLANG_SCRIPTED_RUNTIME_IPC_ADDR"]
    ctx = zmq.Context()
    socket = get_zmq_socket(ctx, zmq.PAIR, endpoint, bind=False)
    try:
        # Announce readiness so the server-startup handshake confirms the
        # scheduler subprocess came up.
        socket.send_pyobj(HookReady())
        while True:
            msg = socket.recv_pyobj()
            match msg:
                case Shutdown():
                    return
                case RunScript(fn_path=fn_path, args=args):
                    fn = resolve_fn(fn_path)
                    # Start every sub-script from a clean engine: flush so
                    # radix / pool state from the previous sub-script can't
                    # leak across runs. Visible on the next yield (same as
                    # start_req), hence the explicit yield before the
                    # sub-script observes any state.
                    t.flush_cache()
                    yield
                    sub_gen = fn(t, *args)
                    try:
                        yield from sub_gen
                    except BaseException:
                        socket.send_pyobj(
                            ScriptFailed(traceback=traceback.format_exc())
                        )
                    else:
                        socket.send_pyobj(ScriptSucceeded())
                case _:
                    raise ValueError(f"router: unknown command {msg!r}")
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        ctx.term()
