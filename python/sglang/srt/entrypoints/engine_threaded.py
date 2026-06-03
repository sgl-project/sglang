"""ThreadedEngine: run scheduler + detokenizer as threads, not processes.

Activated via the env var:

    SGLANG_THREADED_ENGINE=1

Designed for free-threaded Python (no-GIL). Eliminates ZMQ serialization
overhead on the scheduler <-> tokenizer / detokenizer path by replacing
the IPC sockets with in-process queues.

Restrictions (enforced at construction):
  - tp_size == 1
  - pp_size == 1
  - dp_size == 1
  - No Ray

Public surface:
  - ``enable_threaded_engine(server_args)`` — apply the global runtime
    patches once. Must be called before constructing a ``ThreadedEngine``.
    ``launch_threaded_server`` does this for you.
  - ``ThreadedEngine`` — Engine subclass that wires queue-backed channels
    into Scheduler / TokenizerManager / DetokenizerManager.
  - ``launch_threaded_server`` — drop-in for ``launch_server`` invoked
    by the ``SGLANG_THREADED_ENGINE=1`` branch in ``http_server.py``.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import traceback
from typing import Callable, Dict, List, Optional

import psutil

from sglang.srt.entrypoints.engine import (
    Engine,
    SchedulerInitResult,
    _compute_parallelism_ranks,
    init_tokenizer_manager,
)
from sglang.srt.environ import envs
from sglang.srt.managers.channel import ChannelHub
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.scheduler import (
    Scheduler,
    configure_scheduler_process,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime patches — applied explicitly via enable_threaded_engine()
# ---------------------------------------------------------------------------

_patches_applied = False
_patches_lock = threading.Lock()

# The hub the next _ThreadedScheduler construction should pick up.  Set by
# ThreadedEngine.__init__ just before instantiating _ThreadedScheduler, read
# by _ThreadedScheduler.init_ipc_channels.
_pending_hub: threading.local = threading.local()


def enable_threaded_engine(server_args: ServerArgs) -> None:
    """Apply runtime adjustments required by ThreadedEngine.

    Idempotent — safe to call multiple times. The first call:
      - forces ``multiprocessing`` start method to ``spawn`` (fork from a
        multi-threaded process can deadlock);
      - flips the multimodal tensor-transport default to in-process refs;
      - forces ``disable_piecewise_cuda_graph=True`` (warns if the user
        had explicitly disabled it).

    sglang's own ``mp.get_context("fork")`` call sites consult
    ``is_threaded_engine_enabled()`` directly and switch to spawn there;
    we do *not* monkey-patch ``mp.get_context`` globally, so third-party
    libraries are unaffected.
    """
    global _patches_applied
    with _patches_lock:
        if _patches_applied:
            return
        _patches_applied = True

        current_method = mp.get_start_method(allow_none=True)
        if current_method not in (None, "spawn"):
            logger.warning(
                "ThreadedEngine is overriding multiprocessing start_method "
                "from %r to 'spawn' (fork from a multi-threaded process can "
                "deadlock).",
                current_method,
            )
        mp.set_start_method("spawn", force=True)

        # In single-process threaded mode the tokenizer and rank-0 scheduler
        # share an address space, so multimodal tensors can be passed by
        # reference rather than wrapped in SHM.
        import sglang.srt.managers.mm_utils as _mm_utils

        _mm_utils._is_default_tensor_transport = True

        # Piecewise CUDA graph spawns inductor compile workers that can
        # deadlock when the scheduler itself is a thread in a free-threaded
        # process. Non-piecewise CUDA graphs still work.
        if not server_args.disable_piecewise_cuda_graph:
            logger.warning(
                "ThreadedEngine forces disable_piecewise_cuda_graph=True; "
                "piecewise CUDA graph compile workers can deadlock when the "
                "scheduler is a thread of the parent process. Non-piecewise "
                "CUDA graphs still work."
            )
            server_args.disable_piecewise_cuda_graph = True


def is_threaded_engine_enabled() -> bool:
    """True iff ``enable_threaded_engine`` has been applied in this process.

    Used by sglang's own subprocess entry points (e.g. multimodal
    processor's ProcessPoolExecutor) so they can opt out of fork without
    a global ``mp.get_context`` monkey-patch.
    """
    return _patches_applied


# ---------------------------------------------------------------------------
# Threaded subclasses
# ---------------------------------------------------------------------------


def _make_threaded_scheduler_ipc(hub: ChannelHub) -> SchedulerIpcChannels:
    """Construct a SchedulerIpcChannels backed by the in-process hub.

    The upstream ``SenderWrapper`` does not care that its ``socket``
    happens to be a ``QueueSender`` — it only calls ``send_pyobj`` on it.
    Similarly, ``recv_from_tokenizer`` only needs ``recv_pyobj(NOBLOCK)``.

    ``recv_from_rpc`` is a queue-backed DEALER stand-in: in single-process
    threaded mode no caller drives RPC, but ``request_receiver`` calls
    ``recv_from_rpc.recv_pyobj(NOBLOCK)`` unconditionally, so it must be
    a real object (not None) that raises ``zmq.Again`` on empty.
    """
    return SchedulerIpcChannels(
        recv_from_tokenizer=hub.tokenizer_to_scheduler.receiver,
        recv_from_rpc=hub.rpc_dealer,
        send_to_tokenizer=SenderWrapper(hub.detokenizer_to_tokenizer.sender),
        send_to_detokenizer=SenderWrapper(hub.scheduler_to_detokenizer.sender),
        send_metrics_from_scheduler=None,
    )


class _ThreadedScheduler(Scheduler):
    """Scheduler that skips ZMQ context creation and idle-sleeper polling.

    ``init_ipc_channels`` is invoked by ``Scheduler.__init__`` and reads
    the hub set on ``_pending_hub`` by ``ThreadedEngine`` just before
    construction.  ``init_idle_sleeper`` is disabled because
    ``zmq.Poller`` can't poll our queue-backed receivers.

    NOTE: this override must mirror any non-socket side-effects that the
    parent's ``init_ipc_channels`` performs — currently
    ``self.load_snapshot_writer = None``. When upstream adds more, they
    have to be replicated here or the scheduler thread will crash with
    ``AttributeError`` deep inside the event loop.
    """

    def init_ipc_channels(self, port_args: PortArgs) -> None:
        hub: Optional[ChannelHub] = getattr(_pending_hub, "hub", None)
        if hub is None:
            raise RuntimeError(
                "_ThreadedScheduler constructed without an active ChannelHub. "
                "Only ThreadedEngine should instantiate this class."
            )
        self.ipc_channels = _make_threaded_scheduler_ipc(hub)
        # Load snapshot writer publishes via ZMQ to peer processes; in
        # single-process threaded mode there are no peers, so disable it.
        self.load_snapshot_writer = None

    def init_idle_sleeper(self) -> None:
        # Queue-backed receivers can't register with zmq.Poller.  Skip the
        # idle sleeper entirely; cost is a tight poll loop, which is exactly
        # what we want anyway in single-process threaded mode.
        self.idle_sleeper = None


class _ThreadedDetokenizer(DetokenizerManager):
    """DetokenizerManager that skips ZMQ context creation.

    Channels are assigned by ``ThreadedEngine`` after construction.
    The two-argument signature matches the upstream parent introduced
    by the multi-detokenizer change.
    """

    def init_ipc_channels(self, port_args: PortArgs, server_args: ServerArgs) -> None:
        self.recv_from_scheduler = None
        self.send_to_tokenizer = None


class _ThreadedTokenizerManager(TokenizerManager):
    """TokenizerManager that skips ZMQ context creation.

    Channels are assigned by ``ThreadedEngine`` after construction.
    """

    def init_ipc_channels(self, port_args: PortArgs) -> None:
        self.recv_from_detokenizer = None
        self.send_to_scheduler = None


# ---------------------------------------------------------------------------
# ThreadedEngine
# ---------------------------------------------------------------------------


def _pin_thread_cores(env_field, role: str) -> None:
    """Honor SGLANG_*_CORES on Linux; warn on platforms without sched_setaffinity."""
    cores_raw = env_field.get()
    if not cores_raw:
        return
    if not hasattr(os, "sched_setaffinity"):
        logger.warning(
            "%s is set but os.sched_setaffinity is unavailable on this "
            "platform; ignoring.",
            env_field.name,
        )
        return
    cores = [int(c) for c in cores_raw.split(",")]
    os.sched_setaffinity(0, cores)
    logger.info("%s thread pinned to cores %s", role, cores)


class ThreadedEngine(Engine):
    """Engine variant that runs scheduler + detokenizer as threads (tp=1 only).

    Call ``enable_threaded_engine(server_args)`` before instantiating.
    ``launch_threaded_server`` does this automatically; direct callers
    must do it themselves.
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

        if not _patches_applied:
            raise RuntimeError(
                "ThreadedEngine requires enable_threaded_engine(server_args) "
                "to be called first (launch_threaded_server does this for you)."
            )

        self.tokenizer_manager: Optional[TokenizerManager] = None
        self._shutting_down = False
        self._sched_thread: Optional[threading.Thread] = None
        self._detoken_thread: Optional[threading.Thread] = None

        port_args = PortArgs.init_new(server_args)
        self.port_args = port_args

        hub = ChannelHub()
        self._channel_hub = hub

        # ---- rank-0 scheduler thread ----
        scheduler_ready = threading.Event()
        scheduler_info_box: Dict = {}
        scheduler_error_box: List[Optional[str]] = [None]

        gpu_id_0 = server_args.base_gpu_id
        attn_cp_rank_0, moe_dp_rank_0, moe_ep_rank_0 = _compute_parallelism_ranks(
            server_args, 0
        )

        def _scheduler_main():
            try:
                _pin_thread_cores(envs.SGLANG_SCHEDULER_CORES, "Scheduler")

                # PR_SET_PDEATHSIG is per-thread on Linux; if the
                # scheduler thread set it, the whole engine would die when
                # the shell parent exits. Skip just that one step here
                # rather than monkey-patching the module-level helper.
                configure_scheduler_process(
                    server_args,
                    gpu_id=gpu_id_0,
                    tp_rank=0,
                    attn_cp_rank=attn_cp_rank_0,
                    moe_dp_rank=moe_dp_rank_0,
                    moe_ep_rank=moe_ep_rank_0,
                    pp_rank=0,
                    dp_rank=None,
                    skip_parent_death_signal=True,
                )
                _pending_hub.hub = hub
                try:
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
                finally:
                    _pending_hub.hub = None

                scheduler_info_box.update(sched.get_init_info())
                scheduler_ready.set()

                # Use the scheduler's own entry point so device-specific
                # setup (schedule_stream, _war_barrier_enabled, MLX dispatch)
                # stays in one place.
                sched.run_event_loop()
            except Exception:
                tb = get_exception_traceback()
                # Always log so the cause isn't masked by an atexit-driven
                # shutdown that may run before threads even start.
                logger.error("Scheduler rank-0 thread exception:\n%s", tb)
                if not self._shutting_down:
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                scheduler_error_box[0] = tb
                scheduler_ready.set()
                if not self._shutting_down:
                    psutil.Process().send_signal(signal.SIGQUIT)

        sched_thread = threading.Thread(
            target=_scheduler_main, name="sglang-scheduler-r0", daemon=True
        )
        sched_thread.start()
        self._sched_thread = sched_thread

        # ---- detokenizer thread ----
        def _detoken_main():
            try:
                _pin_thread_cores(envs.SGLANG_DETOKENIZER_CORES, "Detokenizer")
                detoken = _ThreadedDetokenizer(server_args, port_args)
                detoken.recv_from_scheduler = hub.scheduler_to_detokenizer.receiver
                detoken.send_to_tokenizer = hub.detokenizer_to_tokenizer.sender
                detoken.event_loop()
            except Exception:
                tb = get_exception_traceback()
                logger.error("Detokenizer thread exception:\n%s", tb)
                if not self._shutting_down:
                    psutil.Process().send_signal(signal.SIGQUIT)

        detoken_thread = threading.Thread(
            target=_detoken_main, name="sglang-detokenizer", daemon=True
        )
        detoken_thread.start()
        self._detoken_thread = detoken_thread

        # Register atexit AFTER threads are started so init-time exceptions
        # below propagate cleanly instead of being swallowed by a shutdown
        # that runs before the threads exist.
        atexit.register(self.shutdown)

        # ---- wait for scheduler to come up ----
        if not scheduler_ready.wait(timeout=600):
            raise RuntimeError("Scheduler rank-0 thread did not initialize within 600s")
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
        tokenizer_manager.send_to_scheduler = hub.tokenizer_to_scheduler.sender
        tokenizer_manager.recv_from_detokenizer = hub.detokenizer_to_tokenizer.receiver

        # FanOutCommunicators were constructed with send_to_scheduler=None;
        # patch their private sender field. Assert the field exists so a
        # future renaming of FanOutCommunicator._sender surfaces here rather
        # than silently dropping control-plane messages.
        from sglang.srt.managers.tokenizer_control_mixin import _COMMUNICATOR_SPECS

        for spec in _COMMUNICATOR_SPECS:
            comm = getattr(tokenizer_manager, f"{spec[0]}_communicator", None)
            if comm is None:
                continue
            assert hasattr(comm, "_sender"), (
                f"FanOutCommunicator for '{spec[0]}' no longer has a "
                "_sender attribute — schema changed; update ThreadedEngine."
            )
            comm._sender = hub.tokenizer_to_scheduler.sender

        # Let KeyError propagate: the scheduler always populates this on
        # successful init, so a missing key means init didn't actually
        # finish. Better to fail loudly here than to allow 2-billion-token
        # requests deep into CUDA.
        tokenizer_manager.max_req_input_len = scheduler_info_box["max_req_input_len"]

        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager

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

    def shutdown(self) -> None:
        """Best-effort clean shutdown.

        Threads are daemons so they ultimately die with the process, but
        we still try to join with a timeout so in-flight requests have a
        chance to complete and any background bridge threads inside the
        channel hub exit cleanly.
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        if self.tokenizer_manager is not None:
            wdog = getattr(self.tokenizer_manager, "_subprocess_watchdog", None)
            if wdog is not None:
                wdog.stop()

        # Close hub last so any pending sends from the threads have a
        # chance to land before bridge threads exit.
        for t in (self._sched_thread, self._detoken_thread):
            if t is None or not t.is_alive():
                continue
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning(
                    "ThreadedEngine: %s did not exit within 5s; "
                    "process exit will kill the daemon thread.",
                    t.name,
                )

        try:
            self._channel_hub.close()
        except Exception:
            logger.exception("ThreadedEngine: error while closing channel hub")


# ---------------------------------------------------------------------------
# launch_threaded_server
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

    enable_threaded_engine(server_args)

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
