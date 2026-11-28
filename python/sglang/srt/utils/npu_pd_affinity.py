"""SGLang-only HOST_RDMA thread affinity; no MF imports or native hooks.

Python threads are identified explicitly. Native names below are tied to the
audited MF/HCOM versions, not an ownership API. Later native threads rely on
creator affinity inheritance and must be checked after real traffic/reconnect.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.utils.npu_affinity import (
    NpuAffinityAssignment,
    NpuAffinityError,
    NpuAffinityThreadResult,
    NpuPdAffinityAssignment,
    format_cpu_list,
)

logger = logging.getLogger(__name__)

# MF 5d867a7d / HCOM c22937e7: only service-specific, source-verified names.
# Do not match generic python/executor/acc_* or infer ownership from CPU masks.
_NATIVE_PD_NAMES = frozenset(
    {
        "adpt_lkdwn",
        "host_re_conn",
        "grp_listen_evt",
        "config_store_hb",
        "acc_store_timer",
        "rank_state_ts",
        "store_chk_sts",
        "NetAsyncGrpMgr",
        "group_send",
        "group_mgr_drv",
    }
)
_NATIVE_PD_INDEXED = re.compile(
    r"(?:RDMAWkr[0-9]+-[0-9]+-[0-9]+"
    r"|(?:RDMAEvent|HcomPerMgr|HCOMHb|OOBTcpConnHdl|OOBUdsConnHdl)[0-9]+"
    r"|(?:OOBTcpSvr|OOBUdsSvr)[0-9]+-[0-9]+)"
)


@dataclass(frozen=True)
class PythonPdThread:
    native_id: int
    start_time: int
    role: str


def read_thread_identity(pid: int, tid: int) -> tuple[str, int]:
    """Read comm and starttime together; comm may contain spaces/parentheses."""
    try:
        stat = Path(f"/proc/{pid}/task/{tid}/stat").read_text(
            encoding="utf-8", errors="replace"
        )
    except FileNotFoundError as exc:
        raise ProcessLookupError(errno.ESRCH, "Thread exited", tid) from exc
    end = stat.rfind(")")
    # Fields following comm begin with field 3; starttime is field 22.
    return stat[stat.index("(") + 1 : end], int(stat[end + 1 :].split()[19])


def get_pd_affinity_budget(server_args, *, is_npu: bool) -> int:
    if not (
        is_npu
        and server_args.disaggregation_mode in ("prefill", "decode")
        and server_args.disaggregation_transfer_backend == "ascend"
        and os.environ.get("ASCEND_MF_TRANSFER_PROTOCOL", "").lower() == "host_rdma"
    ):
        return 0
    budget = envs.SGLANG_NPU_PD_AFFINITY_PCORES_PER_PROC.get()
    if budget < 0:
        raise NpuAffinityError(
            "SGLANG_NPU_PD_AFFINITY_PCORES_PER_PROC must be >= 0",
            stage="plan_pd_affinity",
        )
    if budget and not envs.SGLANG_SET_CPU_AFFINITY.get():
        raise NpuAffinityError(
            "PD CPU affinity requires SGLANG_SET_CPU_AFFINITY=1",
            stage="plan_pd_affinity",
        )
    return budget


def _bind_current(cpu_ids) -> None:
    target = set(cpu_ids)
    try:
        os.sched_setaffinity(0, target)
        actual = os.sched_getaffinity(0)
    except OSError as exc:
        raise NpuAffinityError(str(exc), stage="bind_pd_thread") from exc
    if actual != target:
        raise NpuAffinityError(
            f"PD affinity read-back mismatch: expected={sorted(target)}, "
            f"actual={sorted(actual)}",
            stage="bind_pd_thread",
        )


class PdThreadAffinity:
    def __init__(self, assignment: NpuPdAffinityAssignment):
        self.assignment = assignment
        self.pid = os.getpid()
        self._threads = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def _register_current(self, role: str) -> None:
        _bind_current(self.assignment.logical_cpu_ids)
        tid = threading.get_native_id()
        _, start_time = read_thread_identity(self.pid, tid)
        with self._lock:
            self._threads[threading.current_thread()] = PythonPdThread(
                tid, start_time, role
            )
        if envs.SGLANG_NPU_AFFINITY_DEBUG_THREADS.get():
            logger.info(
                "PD affinity: pid=%s tid=%s role=%s cpu_mask=%s",
                self.pid,
                tid,
                role,
                format_cpu_list(self.assignment.logical_cpu_ids),
            )

    def snapshot(self) -> tuple[PythonPdThread, ...]:
        with self._lock:
            return tuple(
                record for thread, record in self._threads.items() if thread.is_alive()
            )

    def start_thread(self, target, *, role, args=(), name=None, daemon=None):
        ready = threading.Event()
        failure = []

        def run():
            try:
                self._register_current(role)
            except BaseException as exc:
                failure.append(exc)
                ready.set()
                return
            ready.set()
            try:
                target(*args)
            finally:
                with self._lock:
                    self._threads.pop(threading.current_thread(), None)

        thread = threading.Thread(target=run, name=name, daemon=daemon)
        thread.start()
        # Wait only for the binding handshake, never for network/target work.
        if not ready.wait(30):
            raise NpuAffinityError(
                f"Timed out binding PD thread {role}", stage="start_pd_thread"
            )
        if failure:
            raise NpuAffinityError(
                f"Failed to bind PD thread {role}: {failure[0]}",
                stage="start_pd_thread",
            ) from failure[0]
        return thread

    def initialize_pool_thread(self):
        # Weak Thread keys retain the record across tasks, not just initializer
        # execution. No private ThreadPoolExecutor worker hooks are needed.
        self._register_current("transfer_pool")

    @contextmanager
    def bind_initialization_thread(self):
        saved = os.sched_getaffinity(0)
        try:
            _bind_current(self.assignment.logical_cpu_ids)
            yield
        finally:
            # If both initialization and restoration fail, Python's exception
            # chain retains the initialization error as context.
            _bind_current(saved)


_context: PdThreadAffinity | None = None


def install_pd_thread_affinity(assignment: NpuPdAffinityAssignment | None):
    global _context
    _context = PdThreadAffinity(assignment) if assignment is not None else None
    return _context


def get_pd_thread_affinity() -> PdThreadAffinity | None:
    # Never use another scheduler/parent's state inherited through fork.
    return _context if _context is not None and _context.pid == os.getpid() else None


def make_pd_thread_binder(
    compute: NpuAffinityAssignment,
    pd: NpuPdAffinityAssignment,
    python_threads: tuple[PythonPdThread, ...],
):
    records = {thread.native_id: thread for thread in python_threads}

    def classify(pid, tid, name, start_time):
        if tid != pid:
            record = records.get(tid)
            if record is not None and record.start_time == start_time:
                return pd.logical_cpu_ids, record.role, "python_pd"
            if name in _NATIVE_PD_NAMES or _NATIVE_PD_INDEXED.fullmatch(name):
                return pd.logical_cpu_ids, name, "native_pd"
        return compute.logical_cpu_ids, "compute", "default"

    def bind(pid: int, tid: int) -> NpuAffinityThreadResult:
        name, start_time = read_thread_identity(pid, tid)
        expected, role, source = classify(pid, tid, name, start_time)
        # Two bounded passes handle a name published during final. This is not
        # synchronization with native thread creation/exit or a runtime watcher.
        for _ in range(2):
            os.sched_setaffinity(tid, set(expected))
            new_name, new_start = read_thread_identity(pid, tid)
            if new_start != start_time:
                raise ProcessLookupError(errno.ESRCH, "Thread identity changed", tid)
            new_target = classify(pid, tid, new_name, new_start)
            name = new_name
            if new_target == (expected, role, source):
                actual = tuple(sorted(os.sched_getaffinity(tid)))
                return NpuAffinityThreadResult(
                    tid,
                    "bound" if actual == expected else "mismatched",
                    actual,
                    expected_cpu_ids=expected,
                    role=role,
                    source=source,
                    name=name,
                    start_time=start_time,
                )
            expected, role, source = new_target
        return NpuAffinityThreadResult(
            tid,
            "failed",
            error="Thread classification changed during final; binding not verified",
            expected_cpu_ids=expected,
            role=role,
            source=source,
            name=name,
            start_time=start_time,
        )

    return bind
