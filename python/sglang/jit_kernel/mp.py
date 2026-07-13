"""Multi-process / multi-GPU launching utilities (torchrun-based).

Shared `multigpu_launch` helper that both `sglang.jit_kernel.tests.utils` and
`sglang.jit_kernel.benchmark.utils` build their domain-specific entry points on
top of (`multigpu_pytest_main`, `multigpu_bench_main`).

When a script that calls one of those wrappers is run with plain `python`, the
launcher relaunches the same file under `torchrun` once for each `N` in
`num_gpus`. When the inner workers run (identified by an env_key being set),
the same launcher calls `inner()` on every rank, silences stdout on non-zero
ranks, and exits with its return code.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, List, NoReturn, Optional, Sequence

import psutil
import torch

logger = logging.getLogger(__name__)


def register_comm_cleanup(comm: Any) -> None:
    """Register an idempotent shutdown for a custom-AR communicator."""

    def _safe_close() -> None:
        try:
            comm.close()
        except Exception:
            pass
        # Disable both class flavors' early-out paths in __del__/close.
        try:
            comm.disabled = True
        except Exception:
            pass
        # CustomAllReduceV2: drop ``obj`` so close() short-circuits next time.
        try:
            delattr(comm, "obj")
        except Exception:
            pass
        # CustomAllreduce: zero ``_ptr`` so close() short-circuits next time.
        try:
            comm._ptr = 0
        except Exception:
            pass

    atexit.register(_safe_close)


def _kill_pgroup(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _kill_descendants(pid: int) -> None:
    """Snapshot every descendant of `pid` *now* and SIGKILL them all.

    Must be called BEFORE the direct child (torchrun) dies -- once it does,
    its workers get reparented to init and we lose them via the process tree.
    """
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    descendants = root.children(recursive=True)
    for proc in descendants:
        try:
            proc.kill()
        except psutil.Error:
            # NoSuchProcess (already gone) or AccessDenied -- nothing to do.
            pass
    psutil.wait_procs(descendants, timeout=5)


def _extract_num_gpus_override(
    argv: list[str],
) -> tuple[list[int] | None, list[str]]:
    """Pop `--num-gpu(s)` flags out of `argv` and return them separately.

    Accepts `--num-gpu N`, `--num-gpu=N`, `--num-gpus ...`, and comma-separated
    lists like `--num-gpu 2,4,8`. May be repeated.
    """
    override: list[int] = []
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--num-gpu", "--num-gpus"):
            if i + 1 >= len(argv):
                raise ValueError(f"missing value for {a} (expected e.g. `{a} 2,4`)")
            override.extend(int(x) for x in argv[i + 1].split(","))
            i += 2
        elif a.startswith("--num-gpu=") or a.startswith("--num-gpus="):
            _, val = a.split("=", 1)
            override.extend(int(x) for x in val.split(","))
            i += 1
        else:
            remaining.append(a)
            i += 1
    return (override if override else None), remaining


def multigpu_launch(
    name: str,
    file: str,
    num_gpus: Sequence[int],
    env_key: str,
    inner: Callable[[], int],
    kind: str,
    pre_launch_fn: Optional[Callable[[List[int]], None]] = None,
    timeout: Optional[int] = None,
) -> NoReturn | None:
    """Shared torchrun-based launcher.

    See module docstring. `name` is the caller's `__name__`; `file` is its
    `__file__`. `env_key` is a unique string per kind (test/benchmark) used to
    detect the inside-torchrun state. `inner` returns an exit code.

    `pre_launch_fn`, if given, runs once in the outer process *before* any
    torchrun child is spawned. It receives the list of world sizes that will
    actually be launched (already filtered against the host's GPU count and
    any ``--num-gpu`` override). Use it for parallel JIT precompilation so the
    on-disk kernel cache is warm by the time the torchrun children import
    their kernels.

    `timeout`, if given, bounds each per-world-size torchrun invocation (in
    seconds). On expiry the child's whole process group is killed and the
    launcher exits non-zero. `None` (the default) waits indefinitely.
    """
    pid_key = env_key + "_PID"
    if env_key in os.environ:
        assert pid_key in os.environ
        if name != "__main__":
            return
        rank = int(os.environ["LOCAL_RANK"])
        if rank != 0:
            sys.stdout = open(os.devnull, "w")
        torch.cuda.set_device(rank)
        return sys.exit(inner())
    assert pid_key not in os.environ
    if name != "__main__":
        return logger.warning(
            f"{file} can not directly run with `pytest`. "
            "Use `python` to invoke it, which will internally relaunch it "
            "under torchrun for each requested number of GPUs."
        )
    num_devices = torch.cuda.device_count()
    override, forwarded_args = _extract_num_gpus_override(sys.argv[1:])
    if override is not None:
        logger.info(f"--num-gpu override: running only with {override}")
        num_gpus = override
        for N in num_gpus:
            if N <= 1 or N > num_devices:
                raise ValueError(
                    f"Invalid number of GPUs requested: {N} "
                    f"(available: {num_devices})"
                )
    os.environ[env_key] = "1"
    os.environ[pid_key] = str(os.getpid())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")  # single-machine setup
    # Unbuffered child stdout: when a worker is killed on timeout, pytest's
    # block-buffered progress output is otherwise lost or flushed out of
    # order into the CI log, making it impossible to tell which test hung.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Timestamped INFO on stdout for the outer launcher: CI invokes these
    # files with plain `python3`, where the root logger defaults to WARNING,
    # so the launch/timing breadcrumbs below are otherwise invisible exactly
    # where they are needed to attribute a per-file timeout. No-op if the
    # harness already configured logging.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    signal.signal(signal.SIGINT, signal.default_int_handler)
    runnable: List[int] = []
    for N in sorted(num_gpus):
        assert N > 1
        if N > num_devices:
            logger.warning(f"Skipping {kind} with {N} GPUs ({num_devices} available)")
            continue
        runnable.append(N)
    if pre_launch_fn is not None and runnable:
        logger.info(f"Running pre-launch hook for world sizes {runnable}")
        tic = time.monotonic()
        pre_launch_fn(runnable)
        logger.info(f"Pre-launch hook took {time.monotonic() - tic:.1f}s")
    for N in runnable:
        logger.info(f"Running {kind} with {N} GPUs")
        tic = time.monotonic()
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(N),
            "--local-addr",
            "127.0.0.1",
            file,
        ]
        cmd += forwarded_args
        proc = subprocess.Popen(cmd, start_new_session=True)
        pgid = proc.pid
        returncode = -1
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _kill_descendants(os.getpid())
            _kill_pgroup(pgid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        elapsed = time.monotonic() - tic
        if timed_out:
            logger.error(f"{kind} (nproc={N}) timed out after {timeout} seconds")
            sys.exit(1)
        if returncode != 0:
            logger.error(
                f"{kind} failed with {N} GPUs (exit {returncode}) "
                f"after {elapsed:.1f}s"
            )
            sys.exit(returncode)
        logger.info(f"{kind} with {N} GPUs passed in {elapsed:.1f}s")
    logger.info(f"All {kind}s passed")
