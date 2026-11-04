"""A helper process that opens a CUDA context on a PD decode worker's GPU after start-up.

While no other process has opened a CUDA context on the GPU since a decode worker
started, each dependent kernel launch inside the worker's CUDA graphs can wait
longer, which adds up over a speculative decode step. A context that another
process opens after start-up removes that extra latency. An aggregated server's
own server process opens one at its first request and keeps it; a PD decode
worker's never does, so ``--disaggregation-decode-late-cuda-context`` starts a
helper that does, once the first decode batch has been launched.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading

from sglang.srt.runtime_context import get_disagg

logger = logging.getLogger(__name__)

# Opens the context with one CUDA op and reports "ready", then holds it until the
# worker (argv[1]) exits. argv[2] lists the CPUs to run on; empty keeps the
# spawning thread's.
_HELPER = (
    "import os, sys, time\n"
    "parent = int(sys.argv[1])\n"
    "cpus = [int(c) for c in sys.argv[2].split(',') if c]\n"
    "if cpus:\n"
    "    try:\n"
    "        os.sched_setaffinity(0, cpus)\n"
    "    except OSError:\n"
    "        pass\n"
    "import torch\n"
    "torch.ones(1, device='cuda')\n"
    "torch.cuda.synchronize()\n"
    "print('ready', flush=True)\n"
    "while os.getppid() == parent:\n"
    "    time.sleep(5)\n"
)

_started = False


def _physical_device(device_index: int) -> str:
    """This process's logical device as a CUDA_VISIBLE_DEVICES entry for the helper."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return str(device_index)
    return visible.split(",")[device_index].strip()


def _helper_cpus() -> str:
    """The CPUs perfll_host leaves to the process's other threads, or "" when it pinned nothing."""
    from sglang.srt.utils import perfll_host

    # Spawned from the scheduler's main thread, the helper would otherwise inherit
    # that thread's reserved cores and import torch on the decode loop's CPUs.
    return ",".join(str(cpu) for cpu in sorted(perfll_host._other_cores))


def _watch(helper: subprocess.Popen, gpu: str, worker_pid: int) -> None:
    for line in helper.stdout:
        if line.strip() == "ready":
            logger.info(
                "Late CUDA context: helper pid %d holds a context on GPU %s "
                "for decode worker pid %d",
                helper.pid,
                gpu,
                worker_pid,
            )
            break
    # The helper outlives its worker by design, so exiting first means it failed.
    code = helper.wait()
    logger.warning(
        "Late CUDA context: helper pid %d exited with code %s; GPU %s has no late context",
        helper.pid,
        code,
        gpu,
    )


def _start_helper() -> None:
    import torch

    gpu = _physical_device(torch.cuda.current_device())
    helper = subprocess.Popen(
        [sys.executable, "-c", _HELPER, str(os.getpid()), _helper_cpus()],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    threading.Thread(
        target=_watch,
        args=(helper, gpu, os.getpid()),
        name="late-cuda-context-watch",
        daemon=True,
    ).start()


def maybe_open_late_cuda_context() -> None:
    """Start the helper once per process; call after a decode batch has been launched."""
    global _started
    if _started:
        return
    _started = True
    if not get_disagg().disaggregation_decode_late_cuda_context:
        return
    try:
        _start_helper()
    except Exception as err:  # noqa: BLE001
        # The helper only buys speed; it must never stop the decode loop.
        logger.warning("Late CUDA context: helper not started: %r", err)
