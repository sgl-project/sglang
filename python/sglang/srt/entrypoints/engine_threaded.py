"""ThreadedEngine: scheduler + detokenizer as threads, not processes.

Activated by setting the environment variable:
    SGLANG_THREADED_ENGINE=1

Designed for free-threaded Python (no-GIL).  Eliminates ZMQ serialization
overhead for the scheduler <-> tokenizer/detokenizer path.

Only effective for tp=1.  For tp>1, http_server.py automatically falls
back to the standard multi-process Engine (CPU contention between the
scheduler thread and HTTP/tokenizer threads causes NCCL sync delays
that degrade throughput).

Limitations:
  - tp_size must be 1 (enforced by http_server.py fallback)
  - pp_size must be 1
  - dp_size must be 1
  - No Ray

This file is fully self-contained: it uses monkey-patching at runtime
to adapt behavior of other modules, so ZERO modifications are needed to
existing SGLang source files (except the env check in http_server.py).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import multiprocessing as mp
import os
import signal
import threading
from typing import Callable, Dict, List, Optional, Tuple

import psutil
import torch

from sglang.srt.entrypoints.engine import (
    Engine,
    SchedulerInitResult,
    _compute_parallelism_ranks,
    init_tokenizer_manager,
)
from sglang.srt.managers.channel import ChannelHub
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.scheduler import (
    Scheduler,
    SenderWrapper,
    configure_scheduler_process,
    dispatch_event_loop,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime monkey-patches (applied only when ThreadedEngine is actually used)
# ---------------------------------------------------------------------------

_patches_applied = False


def _apply_runtime_patches(server_args: ServerArgs):
    """Apply minimal runtime patches needed for threaded mode.

    These avoid modifying source files:
    1. fork -> spawn for ProcessPoolExecutor in multimodal processor
    2. Skip SHM tensor transport for tp=1 (objects passed by reference)
    3. Disable piecewise CUDA graph (torch.compile workers can deadlock
       when the scheduler runs as a thread in a free-threaded process)
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # Patch 1: Force "spawn" for all multiprocessing.
    # In threaded mode, fork from a multi-threaded process can deadlock.
    # Also ensures numa_utils assertion (requires start_method == "spawn").
    mp.set_start_method("spawn", force=True)

    _orig_get_context = mp.get_context

    def _safe_get_context(method=None):
        if method == "fork":
            method = "spawn"
        return _orig_get_context(method)

    mp.get_context = _safe_get_context

    # Patch 2: Skip SHM wrapping entirely (zero-copy via queue).
    # In ThreadedEngine, rank-0 scheduler and tokenizer are in the same
    # process, so tensors can be passed by reference. Ranks 1..n get their
    # data from rank 0 via NCCL broadcast, not from the tokenizer.
    import sglang.srt.managers.mm_utils as _mm_utils

    _mm_utils._is_default_tensor_transport = True

    # Patch 3: Disable piecewise CUDA graph. torch.compile spawns
    # inductor compile workers that can deadlock in a multi-threaded,
    # free-threaded process. Regular (non-piecewise) CUDA graphs still work.
    server_args.disable_piecewise_cuda_graph = True


# ---------------------------------------------------------------------------
# Threaded subclasses that skip ZMQ socket creation
# ---------------------------------------------------------------------------


class _ThreadedScheduler(Scheduler):
    """Scheduler that skips ZMQ init — channels wired externally."""

    def init_ipc_channels(self, port_args: PortArgs):
        self.idle_sleeper = None
        self.recv_from_tokenizer = None
        self.recv_from_rpc = None
        self.send_to_tokenizer = SenderWrapper(None)
        self.send_to_detokenizer = SenderWrapper(None)


class _ThreadedDetokenizer(DetokenizerManager):
    """DetokenizerManager that skips ZMQ init."""

    def init_ipc_channels(self, port_args: PortArgs):
        self.recv_from_scheduler = None
        self.send_to_tokenizer = None


class _ThreadedTokenizerManager(TokenizerManager):
    """TokenizerManager that skips ZMQ init."""

    def init_ipc_channels(self, port_args: PortArgs):
        self.recv_from_detokenizer = None
        self.send_to_scheduler = None


# ---------------------------------------------------------------------------
# ThreadedEngine
# ---------------------------------------------------------------------------


class ThreadedEngine(Engine):
    """Engine variant that runs scheduler + detokenizer as threads (tp=1 only).

    All components run in-process with zero IPC overhead.
    """

    def __init__(self, **kwargs):
        load_plugins()

        if "server_args" in kwargs:
            server_args = kwargs["server_args"]
        else:
            if "log_level" not in kwargs:
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        self.server_args = server_args

        assert server_args.tp_size == 1, "ThreadedEngine only supports tp_size=1"
        assert server_args.pp_size == 1, "ThreadedEngine only supports pp_size=1"
        assert server_args.dp_size == 1, "ThreadedEngine only supports dp_size=1"

        _apply_runtime_patches(server_args)

        self.tokenizer_manager = None
        self._shutting_down = False
        atexit.register(self.shutdown)

        port_args = PortArgs.init_new(server_args)
        self.port_args = port_args

        hub = ChannelHub()
        self._channel_hub = hub

        # ---- rank 0 scheduler thread ----
        scheduler_ready = threading.Event()
        scheduler_info_box: Dict = {}
        scheduler_error_box: List = [None]

        gpu_id_0 = server_args.base_gpu_id
        attn_cp_rank_0, moe_dp_rank_0, moe_ep_rank_0 = _compute_parallelism_ranks(
            server_args, 0
        )

        def _scheduler_main():
            try:
                sched_cores = os.environ.get("SGLANG_SCHEDULER_CORES")
                if sched_cores:
                    cores = [int(c) for c in sched_cores.split(",")]
                    os.sched_setaffinity(0, cores)
                    logger.info(f"Scheduler thread pinned to cores {cores}")

                # Disable kill_itself_when_parent_died for thread context —
                # it sets PR_SET_PDEATHSIG which is per-thread on Linux and
                # would kill the entire process when the shell parent exits.
                import sglang.srt.utils.common as _common_utils

                _common_utils.kill_itself_when_parent_died = lambda: None

                configure_scheduler_process(
                    server_args,
                    gpu_id=gpu_id_0,
                    tp_rank=0,
                    attn_cp_rank=attn_cp_rank_0,
                    moe_dp_rank=moe_dp_rank_0,
                    moe_ep_rank=moe_ep_rank_0,
                    pp_rank=0,
                    dp_rank=None,
                )
                sched = _ThreadedScheduler(
                    server_args,
                    port_args,
                    gpu_id=gpu_id_0,
                    tp_rank=0,
                    moe_ep_rank=moe_ep_rank_0,
                    pp_rank=0,
                    attn_cp_rank=attn_cp_rank_0,
                    moe_dp_rank=moe_dp_rank_0,
                    dp_rank=None,
                )
                # Wire queue channels
                sched.recv_from_tokenizer = hub.tokenizer_to_scheduler.receiver
                sched.recv_from_rpc = hub.rpc.receiver
                sched.send_to_tokenizer = SenderWrapper(
                    hub.scheduler_to_tokenizer.sender
                )
                sched.send_to_detokenizer = SenderWrapper(
                    hub.scheduler_to_detokenizer.sender
                )

                scheduler_info_box.update(sched.get_init_info())
                scheduler_ready.set()

                sched.schedule_stream = torch.cuda.Stream(priority=0)
                with torch.cuda.StreamContext(sched.schedule_stream):
                    dispatch_event_loop(sched)
            except Exception:
                if self._shutting_down:
                    return
                import traceback, sys

                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                tb = get_exception_traceback()
                logger.error(f"Scheduler rank-0 thread exception: {tb}")
                scheduler_error_box[0] = tb
                scheduler_ready.set()
                psutil.Process().send_signal(signal.SIGQUIT)

        sched_thread = threading.Thread(
            target=_scheduler_main, name="sglang-scheduler-r0", daemon=True
        )
        sched_thread.start()

        # ---- detokenizer thread ----
        def _detoken_main():
            try:
                detoken_cores = os.environ.get("SGLANG_DETOKENIZER_CORES")
                if detoken_cores:
                    cores = [int(c) for c in detoken_cores.split(",")]
                    os.sched_setaffinity(0, cores)
                    logger.info(f"Detokenizer thread pinned to cores {cores}")

                detoken = _ThreadedDetokenizer(server_args, port_args)
                detoken.recv_from_scheduler = hub.scheduler_to_detokenizer.receiver
                detoken.send_to_tokenizer = hub.detokenizer_to_tokenizer.sender
                detoken.event_loop()
            except Exception:
                tb = get_exception_traceback()
                logger.error(f"Detokenizer thread exception: {tb}")
                psutil.Process().send_signal(signal.SIGQUIT)

        detoken_thread = threading.Thread(
            target=_detoken_main, name="sglang-detokenizer", daemon=True
        )
        detoken_thread.start()

        # ---- wait for scheduler ----
        scheduler_ready.wait(timeout=600)
        if scheduler_error_box[0] is not None:
            raise RuntimeError(
                f"Scheduler rank-0 thread failed:\n{scheduler_error_box[0]}"
            )

        all_infos = [scheduler_info_box]

        # ---- tokenizer manager (main thread) ----
        tokenizer_manager, template_manager = init_tokenizer_manager(
            server_args,
            port_args,
            TokenizerManagerClass=_ThreadedTokenizerManager,
        )
        # Wire queue channels
        tokenizer_manager.send_to_scheduler = hub.tokenizer_to_scheduler.sender
        tokenizer_manager.recv_from_detokenizer = (
            hub.detokenizer_to_tokenizer.async_receiver
        )
        # Fix up FanOutCommunicators that were created with sender=None
        from sglang.srt.managers.tokenizer_control_mixin import _COMMUNICATOR_SPECS

        for spec in _COMMUNICATOR_SPECS:
            comm = getattr(tokenizer_manager, f"{spec[0]}_communicator", None)
            if comm is not None:
                comm._sender = hub.tokenizer_to_scheduler.sender

        tokenizer_manager.max_req_input_len = scheduler_info_box.get(
            "max_req_input_len", 2**31
        )

        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self._sched_thread = sched_thread
        self._detoken_thread = detoken_thread

        # RPC channel
        self.send_to_rpc = hub.rpc.sender

        self._scheduler_init_result = SchedulerInitResult(
            scheduler_infos=all_infos,
            all_child_pids=[],
        )

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def get_all_child_pids(self):
        return []

    def shutdown(self):
        self._shutting_down = True
        if self.tokenizer_manager is not None:
            wdog = getattr(self.tokenizer_manager, "_subprocess_watchdog", None)
            if wdog is not None:
                wdog.stop()
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)


# ---------------------------------------------------------------------------
# launch_threaded_server: full HTTP server backed by ThreadedEngine
# ---------------------------------------------------------------------------


def launch_threaded_server(
    server_args: ServerArgs,
    execute_warmup_func: Optional[Callable] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """Drop-in replacement for launch_server() when SGLANG_THREADED_ENGINE=1."""
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _setup_and_run_http_server,
    )

    if execute_warmup_func is None:
        execute_warmup_func = _execute_server_warmup

    engine = ThreadedEngine(server_args=server_args)

    _setup_and_run_http_server(
        server_args=server_args,
        tokenizer_manager=engine.tokenizer_manager,
        template_manager=engine.template_manager,
        port_args=engine.port_args,
        scheduler_infos=engine._scheduler_init_result.scheduler_infos,
        subprocess_watchdog=None,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
