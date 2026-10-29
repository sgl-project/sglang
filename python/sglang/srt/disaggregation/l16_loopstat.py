"""PD decode loop timing (SGLANG_TRTL16_LOOPSTAT=<dir>, default off).

Per scheduler-loop iteration that runs a batch: exclusive host time by stage,
queue depths, and two forward-stream CUDA events around run_batch (GPU gap
before the batch, span, period). Per-thread CPU time every 10 s. One CSV per
selected TP rank under <dir>; tools/l16_loop.py joins them offline.

SGLANG_TRTL16_AB_WIN / _AB_POLL ("base,treat") / _AB_COARSE and _AB_TREAT_BS
(bucket subsets padded to in baseline / treatment windows; unset = every captured
bucket) / _AB_DEFER=1 (deferred metadata send) alternate knobs per window of
forward_ct, identical on every TP rank, so all ranks switch together; the row's
knob bits mark the treatment.
"""

from __future__ import annotations

import bisect
import os
import socket
import threading
import time
from typing import List, Optional

import torch

_now = time.perf_counter_ns

STAT: Optional["LoopStat"] = None
# Coarse padding list while the coarse window is active (read by _pad_to_bucket).
COARSE_ACTIVE: Optional[List[int]] = None

_HOST_COLS = (
    "pf", "in", "pq", "pq_rel", "pq_res", "pp", "pp_rs", "pp_hs", "pp_alloc", "pp_send",
    "px", "px_poll", "ar", "nb", "rb", "prw", "pr", "smp",
)
_COLS = ("it", "t_ms", "fct", "B", "gbs", "nnew", "npa", "npre", "nxfer", "nwait", "idle", "ar_n", "knob", "mode") + _HOST_COLS


def coarse_pad(raw_size: int, buckets) -> Optional[int]:
    lst = COARSE_ACTIVE
    if lst is None or raw_size > lst[-1] or buckets[-1] != STAT.fine_max:
        return None
    return lst[bisect.bisect_left(lst, raw_size)]


class LoopStat:
    def __init__(self, scheduler, out_dir: str):
        from sglang.srt.runtime_context import get_parallel

        self.sched = scheduler
        self.tp_rank = get_parallel().tp_rank
        self.write = self.tp_rank % 4 == 0
        os.makedirs(out_dir, exist_ok=True)
        base = f"{out_dir}/{socket.gethostname()}-tp{self.tp_rank}-{os.getpid()}"
        self.f = open(base + ".csv", "a", buffering=1 << 20) if self.write else None
        self.fg = open(base + ".gpu.csv", "a", buffering=1 << 20) if self.write else None
        self.ft = open(base + ".thr.csv", "a") if self.write else None
        if self.write:
            self.f.write(",".join(_COLS) + "\n")
            self.fg.write("fct,gap_us,span_us,period_us\n")
            self.ft.write("t_ms,tid,name,cpu_s\n")
        self.t0 = _now()
        self.it = 0
        self.idle = 0
        self.row = dict.fromkeys(_HOST_COLS, 0)
        self.meta = {}
        self.ar_n = 0
        self.last = 0
        self.rows: List[str] = []
        self.grows: List[str] = []
        self.last_flush = self.t0
        self.thr_prev = {}
        # GPU events: ring of (fct, ev_a, ev_b); harvested in order once ev_b completed.
        self.pending = []
        self.prev_b = None
        self.free_ev: List[torch.cuda.Event] = []
        self.graph_bs = sorted(scheduler.server_args.cuda_graph_bs_decode or [])
        self.fine_max = self.graph_bs[-1] if self.graph_bs else 0
        self.win = int(os.environ.get("SGLANG_TRTL16_AB_WIN", "0") or 0)
        poll = os.environ.get("SGLANG_TRTL16_AB_POLL", "")
        self.ab_poll = [int(x) for x in poll.split(",")] if poll else None
        base = os.environ.get("SGLANG_TRTL16_AB_COARSE", "")
        treat = os.environ.get("SGLANG_TRTL16_AB_TREAT_BS", "")
        self.ab_base_bs = sorted(int(x) for x in base.split(",")) if base else None
        self.ab_treat_bs = sorted(int(x) for x in treat.split(",")) if treat else None
        self.ab_graph = self.ab_base_bs is not None or self.ab_treat_bs is not None
        self.ab_defer = os.environ.get("SGLANG_TRTL16_AB_DEFER", "") == "1"
        self.defer_on = False
        self.sync_counts = {}
        self.knob = 0
        if os.environ.get("SGLANG_TRTL16_SYNC_DEBUG", "") == "1" and self.write:
            self._install_sync_debug()

    # ---- sync debug -----------------------------------------------------------
    def _install_sync_debug(self) -> None:
        """Count synchronizing CUDA calls by Python call site (torch sync debug mode), per knob."""
        import warnings

        prev = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if "synchroniz" not in text:
                return prev(message, category, filename, lineno, file, line)
            try:
                key = (self.knob, filename.rsplit("sglang/", 1)[-1], lineno, text[:60])
                self.sync_counts[key] = self.sync_counts.get(key, 0) + 1
            except Exception:
                pass

        warnings.showwarning = show
        warnings.filterwarnings("always", message=".*synchroniz.*")
        torch.cuda.set_sync_debug_mode(1)
        self.fsync = open(self.f.name.replace(".csv", ".sync.csv"), "a")

    # ---- host stage marks -------------------------------------------------
    def begin(self) -> None:
        self.last = _now()
        self.ar_n = 0

    def mark(self, key: str) -> None:
        t = _now()
        self.row[key] += t - self.last
        self.last = t

    def add(self, key: str, t0: int) -> None:
        self.row[key] += _now() - t0

    # ---- knobs --------------------------------------------------------------
    def apply_knobs(self) -> None:
        """Call before process_decode_queue; forward_ct is identical across TP ranks."""
        global COARSE_ACTIVE
        if not self.win:
            return
        w = self.sched.forward_ct // self.win
        # Each enabled knob takes the next bit of the window index; the row's knob keeps
        # fixed positions (bit0 poll, bit1 graph buckets, bit2 deferred metadata send).
        k = pos = 0
        if self.ab_poll is not None:
            b = (w >> pos) & 1
            pos += 1
            k |= b
            self.sched.polling_interval = self.ab_poll[b]
            if not hasattr(self.sched, "polling_count"):
                self.sched.polling_count = 0
        if self.ab_graph:
            b = (w >> pos) & 1
            pos += 1
            k |= b << 1
            # Bucket subset to pad to (None = every captured bucket): treatment when the bit is set.
            COARSE_ACTIVE = self.ab_treat_bs if b else self.ab_base_bs
        if self.ab_defer:
            b = (w >> pos) & 1
            k |= b << 2
            self.defer_on = bool(b)
        self.knob = k

    # ---- GPU events -----------------------------------------------------------
    def _ev(self) -> torch.cuda.Event:
        return self.free_ev.pop() if self.free_ev else torch.cuda.Event(enable_timing=True)

    def gpu_a(self, stream) -> None:
        ev = self._ev()
        ev.record(stream)
        self.pending.append([self.sched.forward_ct + 1, ev, None])

    def gpu_b(self, stream) -> None:
        ev = self._ev()
        ev.record(stream)
        self.pending[-1][2] = ev

    def _harvest(self) -> None:
        while self.pending and self.pending[0][2] is not None and self.pending[0][2].query():
            fct, a, b = self.pending.pop(0)
            if self.prev_b is not None and self.write:
                pb = self.prev_b
                self.grows.append(
                    f"{fct},{pb.elapsed_time(a) * 1e3:.1f},{a.elapsed_time(b) * 1e3:.1f},{pb.elapsed_time(b) * 1e3:.1f}"
                )
            if self.prev_b is not None:
                self.free_ev.append(self.prev_b)
            self.free_ev.append(a)
            self.prev_b = b

    # ---- per-iteration end ----------------------------------------------------
    def end(self, batch) -> None:
        if batch is None:
            self.idle += 1
            for k in self.row:
                self.row[k] = 0
            return
        self.it += 1
        if self.write:
            m = self.meta
            bs = batch.batch_size()
            gbs = 0
            if self.graph_bs and bs <= self.fine_max:
                lst = COARSE_ACTIVE or self.graph_bs
                gbs = lst[bisect.bisect_left(lst, bs)] if bs <= lst[-1] else self.graph_bs[bisect.bisect_left(self.graph_bs, bs)]
            r = self.row
            self.rows.append(
                f"{self.it},{(_now() - self.t0) / 1e6:.3f},{self.sched.forward_ct},{bs},{gbs},"
                f"{m.get('nnew', 0)},{m.get('npa', 0)},{m.get('npre', 0)},{m.get('nxfer', 0)},{m.get('nwait', 0)},{self.idle},{self.ar_n},{self.knob},{m.get('mode', 0)},"
                + ",".join(str(r[k] // 1000) for k in _HOST_COLS)
            )
        for k in self.row:
            self.row[k] = 0
        self.meta = {}
        self.idle = 0
        self._harvest()
        now = _now()
        if now - self.last_flush > 10_000_000_000:
            self.last_flush = now
            self._flush(now)

    def _flush(self, now: int) -> None:
        if not self.write:
            return
        if self.rows:
            self.f.write("\n".join(self.rows) + "\n")
            self.rows = []
        if self.grows:
            self.fg.write("\n".join(self.grows) + "\n")
            self.grows = []
        if self.sync_counts:
            t = (now - self.t0) / 1e6
            self.fsync.write(
                "".join(f"{t:.0f},{k[0]},{k[1]},{k[2]},{v},{k[3].replace(',', ';')}\n" for k, v in self.sync_counts.items())
            )
            self.fsync.flush()
            self.sync_counts = {}
        self.f.flush()
        self.fg.flush()
        names = {t.native_id: t.name for t in threading.enumerate()}
        tck = os.sysconf("SC_CLK_TCK")
        out = []
        for tid in os.listdir("/proc/self/task"):
            try:
                with open(f"/proc/self/task/{tid}/stat") as fh:
                    s = fh.read()
            except OSError:
                continue
            rp = s.rfind(")")
            comm = s[s.find("(") + 1 : rp]
            fields = s[rp + 2 :].split()
            cpu = (int(fields[11]) + int(fields[12])) / tck
            d = cpu - self.thr_prev.get(tid, 0.0)
            self.thr_prev[tid] = cpu
            if d > 0.05:
                name = names.get(int(tid), comm).replace(",", ";")
                out.append(f"{(now - self.t0) / 1e6:.0f},{tid},{name},{d:.2f}")
        if out:
            self.ft.write("\n".join(out) + "\n")
            self.ft.flush()


_NO_SYNC_ENV = os.environ.get("SGLANG_TRTL16_PREALLOC_NO_SYNC", "").lower() in ("1", "true")


def prealloc_no_sync() -> bool:
    """SGLANG_TRTL16_PREALLOC_NO_SYNC, or its A/B window when loopstat alternates it."""
    return _NO_SYNC_ENV or (STAT is not None and STAT.defer_on)


def init(scheduler) -> Optional[LoopStat]:
    global STAT
    out_dir = os.environ.get("SGLANG_TRTL16_LOOPSTAT", "")
    if not out_dir:
        return None
    STAT = LoopStat(scheduler, out_dir)
    return STAT
