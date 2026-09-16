"""Start the worker processes before the launcher imports its HTTP / tokenizer
stack (``SGLANG_PRESPAWN_WORKERS=1``).

A cold ``sglang serve`` spends several seconds importing the server modules
before ``Engine._launch_subprocesses`` can spawn the first worker, and the
workers cannot start their own init (config, tokenizer, weights, CUDA graphs)
until then. ``maybe_prespawn()`` runs the launcher-side preparation that
``_launch_subprocesses`` would run (resolution, environment, validation,
publish, ports) through ``worker_launch``, a light-import copy of that preparation,
spawns the schedulers (or the DP controller) with lazy entry points, and parks
the result here; ``_launch_subprocesses`` adopts it via ``take()`` and skips
the preparation it already did. Combined with
``SGLANG_EARLY_FORKSERVER=1`` the workers are running about a second after the
CLI is entered.
"""

import logging
import os
from typing import Any, List, Optional

import msgspec

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_PRESPAWNED: Optional["Prespawned"] = None


class Prespawned(msgspec.Struct):
    server_args: Any  # the record the workers were spawned from (identity)
    port_args: Any  # PortArgs
    result: Any  # worker_launch.SchedulerInitResult
    procs: Optional[List[Any]]
    context: Any = None  # runtime-context snapshot from before prepare_launch published


def enabled() -> bool:
    return envs.SGLANG_PRESPAWN_WORKERS.get()


def eligible(server_args: ServerArgs) -> bool:
    """Configurations whose launch does extra work before spawning that the
    workers depend on (weight-cache daemons, the engine-info bootstrap server)
    or that do not spawn with multiprocessing (Ray) take the normal path."""
    cfg = resolving_view(server_args)
    if cfg.weight_cache_mode == "daemon":
        return False
    if cfg.remote_instance_weight_loader_start_seed_via_transfer_engine:
        return False
    if cfg.use_ray:
        return False
    return True


def maybe_prespawn(server_args: ServerArgs) -> None:
    """Called from `run_server` before importing the HTTP server module."""
    global _PRESPAWNED
    if not enabled() or _PRESPAWNED is not None:
        return
    import time

    from sglang.srt.entrypoints.worker_launch import (
        allocate_port_args,
        launch_scheduler_processes,
        prepare_launch,
    )
    from sglang.srt.managers import process_entry
    from sglang.srt.runtime_context import restore_context

    if not eligible(server_args):
        logger.info("[prespawn] configuration not eligible; using the normal launch")
        return
    t0 = time.perf_counter()
    context_before_publish = prepare_launch(server_args)
    t1 = time.perf_counter()
    try:
        port_args = allocate_port_args(server_args)
        t2 = time.perf_counter()
        result, procs = launch_scheduler_processes(
            server_args,
            port_args,
            process_entry.run_scheduler_process,
            process_entry.run_data_parallel_controller_process,
        )
    except BaseException:
        restore_context(context_before_publish)
        raise
    logger.info(
        "[prespawn] started %d worker process(es) %.1fs after interpreter start, "
        "before the server modules are imported (prepare %.2fs, ports %.2fs, "
        "spawn %.2fs)",
        len(procs or []),
        _since_process_start(),
        t1 - t0,
        t2 - t1,
        time.perf_counter() - t2,
    )
    _PRESPAWNED = Prespawned(
        server_args=server_args,
        port_args=port_args,
        result=result,
        procs=procs,
        context=context_before_publish,
    )


def take(server_args: ServerArgs) -> Optional[Prespawned]:
    """Hand the pre-spawned workers to `_launch_subprocesses` (once). Only the
    record they were spawned from may adopt them; a launch from any other
    record stops them, so they never outlive the launch that adopts nothing."""
    global _PRESPAWNED
    pre = _PRESPAWNED
    if pre is None:
        return None
    _PRESPAWNED = None
    if pre.server_args is not server_args:
        abandon(pre)
        return None
    return pre


def abandon(pre: Prespawned) -> None:
    """The caller cannot use the pre-spawned workers: stop them and restore the
    runtime context from before they were published, so the normal launch path
    can publish its own record."""
    logger.warning("[prespawn] pre-spawned workers not adopted; terminating them")
    for p in pre.procs or []:
        try:
            p.terminate()
            p.join(timeout=10)
        except Exception:
            pass
    if pre.context is not None:
        from sglang.srt.runtime_context import restore_context

        restore_context(pre.context)


def _since_process_start() -> float:
    """Seconds since this process was created (Linux /proc); -1 elsewhere."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().split(")")[-1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return -1.0
