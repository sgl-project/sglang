"""ThreadedEngine: rank-0 scheduler + detokenizer as threads, not processes.

Designed for free-threaded Python (no-GIL).  Eliminates ZMQ serialization
overhead for the rank-0 ↔ tokenizer/detokenizer path.

For tp>1 the rank-0 scheduler runs as a thread in the main process while
ranks 1..n remain separate processes (required by ``torch.distributed``).
The rank-0 ↔ tokenizer/detokenizer path uses in-process queues instead
of ZMQ, removing all pickle/IPC overhead on the critical path.

Limitations:
  - pp_size must be 1 (no pipeline parallelism)
  - dp_size must be 1 (no data parallelism)
  - No Ray
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import multiprocessing as mp
import os
import signal
import threading
from typing import Callable, List, Tuple

import psutil
import torch

from sglang.srt.entrypoints.engine import (
    Engine,
    SchedulerInitResult,
    _calculate_rank_ranges,
    _compute_parallelism_ranks,
    _wait_for_scheduler_ready,
    init_tokenizer_manager,
)
from sglang.srt.managers.mm_utils import set_skip_shm_transport
from sglang.srt.managers.channel import ChannelHub
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.scheduler import (
    Scheduler,
    SenderWrapper,
    configure_scheduler_process,
    dispatch_event_loop,
    run_scheduler_process,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, maybe_reindex_device_id, numa_utils
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import get_exception_traceback


class _ThreadedScheduler(Scheduler):
    """Scheduler subclass that skips ZMQ socket creation for rank 0."""

    def init_ipc_channels(self, port_args: PortArgs):
        self.idle_sleeper = None
        self.recv_from_tokenizer = None
        self.recv_from_rpc = None
        self.send_to_tokenizer = SenderWrapper(None)
        self.send_to_detokenizer = SenderWrapper(None)


class _ThreadedDetokenizer(DetokenizerManager):
    """DetokenizerManager subclass that skips ZMQ socket creation."""

    def init_ipc_channels(self, port_args: PortArgs):
        self.recv_from_scheduler = None
        self.send_to_tokenizer = None


class _ThreadedTokenizerManager(TokenizerManager):
    """TokenizerManager subclass that skips ZMQ socket creation."""

    def init_ipc_channels(self, port_args: PortArgs):
        self.recv_from_detokenizer = None
        self.send_to_scheduler = None


logger = logging.getLogger(__name__)


class ThreadedEngine(Engine):
    """Engine variant that runs rank-0 scheduler as a thread.

    For tp=1: all components are threads (no ZMQ at all).
    For tp>1: rank 0 is a thread in the main process; ranks 1..n are
              separate processes communicating via NCCL/gloo as usual.
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

        assert server_args.pp_size == 1, "ThreadedEngine only supports pp_size=1"
        assert server_args.dp_size == 1, "ThreadedEngine only supports dp_size=1"

        # tp=1: no broadcast_pyobj, queues pass by reference → skip SHM entirely.
        if server_args.tp_size == 1:
            set_skip_shm_transport(True)

        self.tokenizer_manager = None
        self._shutting_down = False
        atexit.register(self.shutdown)

        port_args = PortArgs.init_new(server_args)
        self.port_args = port_args

        # In-process channels replacing ZMQ for rank 0
        hub = ChannelHub()
        self._channel_hub = hub

        # ---- launch rank 1..n as processes (if tp>1) ----
        scheduler_procs: List[mp.Process] = []
        scheduler_pipe_readers = []
        tp_size = server_args.tp_size
        mp_ctx = mp.get_context("spawn")

        if tp_size > 1:
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=server_args.enable_memory_saver
            )
            for tp_rank in range(1, tp_size):
                reader, writer = mp_ctx.Pipe(duplex=False)
                gpu_id = server_args.base_gpu_id + tp_rank * server_args.gpu_id_step
                attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                    server_args, tp_rank
                )
                with maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp_ctx.Process(
                        target=run_scheduler_process,
                        args=(
                            server_args,
                            port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            0,   # pp_rank
                            None,  # dp_rank
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), \
                         numa_utils.configure_subprocess(server_args, gpu_id):
                        proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        self._scheduler_procs = scheduler_procs

        # ---- rank 0 scheduler thread ----
        scheduler_ready = threading.Event()
        scheduler_info_box: dict = {}
        scheduler_error_box: list = [None]

        gpu_id_0 = server_args.base_gpu_id
        attn_cp_rank_0, moe_dp_rank_0, moe_ep_rank_0 = _compute_parallelism_ranks(
            server_args, 0
        )

        def _scheduler_main():
            try:
                # Pin scheduler thread to dedicated CPU cores to prevent
                # contention with HTTP/tokenizer/IO threads.
                sched_cores = os.environ.get("SGLANG_SCHEDULER_CORES")
                if sched_cores:
                    cores = [int(c) for c in sched_cores.split(",")]
                    os.sched_setaffinity(0, cores)
                    logger.info(f"Scheduler thread pinned to cores {cores}")

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
                    server_args, port_args,
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

        # ---- wait for all schedulers ----
        # Wait for rank-0 thread
        scheduler_ready.wait(timeout=600)
        if scheduler_error_box[0] is not None:
            raise RuntimeError(
                f"Scheduler rank-0 thread failed:\n{scheduler_error_box[0]}"
            )

        # Wait for rank 1..n processes
        if scheduler_pipe_readers:
            other_infos = _wait_for_scheduler_ready(
                scheduler_pipe_readers, scheduler_procs
            )
        else:
            other_infos = []

        all_infos = [scheduler_info_box] + other_infos

        # ---- tokenizer manager (main thread) ----
        tokenizer_manager, template_manager = init_tokenizer_manager(
            server_args, port_args,
            TokenizerManagerClass=_ThreadedTokenizerManager,
        )
        tokenizer_manager.send_to_scheduler = hub.tokenizer_to_scheduler.sender
        tokenizer_manager.recv_from_detokenizer = (
            hub.detokenizer_to_tokenizer.async_receiver
        )
        # Communicators were created with send_to_scheduler=None; fix sender.
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

        # Scheduler init result
        self._scheduler_init_result = SchedulerInitResult(
            scheduler_infos=all_infos,
            all_child_pids=[p.pid for p in scheduler_procs],
        )

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def get_all_child_pids(self):
        return [p.pid for p in self._scheduler_procs]

    def shutdown(self):
        self._shutting_down = True
        if self.tokenizer_manager is not None:
            if hasattr(self.tokenizer_manager, "_subprocess_watchdog"):
                wdog = self.tokenizer_manager._subprocess_watchdog
                if wdog is not None:
                    wdog.stop()
            from sglang.srt.utils import kill_process_tree
            kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)
