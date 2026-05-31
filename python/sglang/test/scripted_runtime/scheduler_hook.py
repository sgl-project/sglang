from __future__ import annotations

import logging
import sys
import traceback
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import zmq

from sglang.srt.environ import envs
from sglang.srt.utils.common import broadcast_pyobj
from sglang.srt.utils.network import get_zmq_socket
from sglang.test.scripted_runtime.background_http_poster import BackgroundHttpPoster
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    OutOfBandError,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
    Shutdown,
)
from sglang.test.scripted_runtime.utils import ensure_script_importable, resolve_fn

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


class ScriptedSchedulerHook:

    def __init__(
        self,
        *,
        scheduler: "Scheduler",
        tokenizer_recv_proxy: Optional["ScriptedTokenizerRecvProxy"],
    ) -> None:
        self._scheduler = scheduler
        self._is_driver = (
            scheduler.ps.pp_rank == 0
            and scheduler.ps.tp_rank == 0
            and scheduler.ps.attn_cp_rank == 0
        )

        if self._is_driver:
            ensure_script_importable(
                envs.SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY.get()
            )
            self._http_poster: Optional[BackgroundHttpPoster] = BackgroundHttpPoster()
            self._context: Optional[ScriptedContext] = ScriptedContext(
                scheduler_hook=self,
                tokenizer_recv_proxy=tokenizer_recv_proxy,
                http_poster=self._http_poster,
            )
            self._script_fn_generator: Optional[Generator] = self._run_dispatch_loop()
        else:
            self._http_poster = None
            self._context = None
            self._script_fn_generator = None

    def _run_dispatch_loop(self) -> Generator:
        endpoint = envs.SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR.get()
        ctx_zmq = zmq.Context()
        socket = get_zmq_socket(ctx_zmq, zmq.PAIR, endpoint, bind=False)
        try:
            socket.send_pyobj(HookReady())
            while True:
                msg = socket.recv_pyobj()
                match msg:
                    case Shutdown():
                        return
                    case RunScript(fn_path=fn_path, args=args):
                        fn = resolve_fn(fn_path)
                        ctx = self._context
                        ctx.flush_cache()
                        yield
                        sub_gen = fn(ctx, *args)
                        try:
                            yield from sub_gen
                        except BaseException:
                            socket.send_pyobj(
                                ScriptFailed(traceback=traceback.format_exc())
                            )
                        else:
                            socket.send_pyobj(ScriptSucceeded())
                    case _:
                        raise ValueError(f"dispatch loop: unknown command {msg!r}")
        finally:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            ctx_zmq.term()
            self._http_poster.close()

    def step(self) -> None:
        if self._is_driver:
            payload: List = list(_advance_generator(self._script_fn_generator))
        else:
            payload = []

        payload = broadcast_pyobj(
            data=payload,
            rank=self._scheduler.world_group.rank,
            dist_group=self._scheduler.world_group.cpu_group,
            src=0,
        )
        done, exc_tb = payload[0], payload[1]
        if not done:
            return

        if exc_tb is not None and self._is_driver:
            self._write_out_of_band_error(exc_tb)
        sys.exit(0 if exc_tb is None else 1)

    def _write_out_of_band_error(self, exc_tb: str) -> None:
        path = envs.SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH.get()
        if not path:
            return
        error = OutOfBandError(traceback=exc_tb or "<no traceback>")
        try:
            with open(path, "w") as f:
                f.write(error.to_json())
        except OSError:
            logger.exception(
                "Failed to write scripted_runtime out-of-band error to %s", path
            )


def _advance_generator(generator: Generator) -> Tuple[bool, Optional[str]]:
    try:
        next(generator)
        return (False, None)
    except StopIteration:
        return (True, None)
    except Exception:
        return (True, traceback.format_exc())
