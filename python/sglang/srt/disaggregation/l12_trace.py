"""PD request timing trace (SGLANG_TRTL12_TRACE): wall-clock stage times per request,
joined offline on bootstrap_room across the P and D logs."""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

ENABLED = envs.SGLANG_TRTL12_TRACE.get()
# perf_counter -> wall clock; time_stats fields are perf_counter values.
_OFF = time.time() - time.perf_counter()

# Decode: room -> [first CHUNK_READY, last CHUNK_READY, chunk count, Success arrival].
DECODE_ROOMS: Dict[int, List[float]] = {}

_P_FIELDS = (
    ("recv", "scheduler_recv_time"),
    ("bq", "prefill_bootstrap_queue_entry_time"),
    ("bdone", "bootstrap_done_time"),
    ("wq", "wait_queue_entry_time"),
    ("fwd", "forward_entry_time"),
    ("pfin", "prefill_finished_time"),
    ("xq", "prefill_transfer_queue_entry_time"),
    ("xdone", "prefill_kv_transfer_finish_time"),
)
_D_FIELDS = (
    ("recv", "scheduler_recv_time"),
    ("pq", "decode_prealloc_queue_entry_time"),
    ("bdone", "bootstrap_done_time"),
    ("xq", "decode_transfer_queue_entry_time"),
    ("kv", "wait_queue_entry_time"),
    ("fwd", "forward_entry_time"),
    ("pb", "decode_prebuilt_finish_time"),
    ("done", "completion_time"),
)


def _w(t: float) -> float:
    return round(t + _OFF, 4) if t else 0.0


def decode_chunk_arrived(room: int) -> None:
    now = time.time()
    e = DECODE_ROOMS.get(room)
    if e is None:
        DECODE_ROOMS[room] = [now, now, 1, 0.0]
    else:
        e[1] = now
        e[2] += 1


def decode_success_arrived(room: int) -> None:
    e = DECODE_ROOMS.setdefault(room, [0.0, 0.0, 0, 0.0])
    if not e[3]:
        e[3] = time.time()


def log_request(req, mode: str) -> None:
    ts = req.time_stats
    fields = _P_FIELDS if mode == "prefill" else _D_FIELDS
    parts = [f"{k}={_w(getattr(ts, a, 0.0))}" for k, a in fields]
    if mode == "decode":
        e = DECODE_ROOMS.pop(req.bootstrap_room, None)
        if e is not None:
            parts.append(f"c0={round(e[0], 4)} c1={round(e[1], 4)} nc={e[2]} succ={round(e[3], 4)}")
    logger.info(
        "L12T %s room=%s in=%d cached=%d out=%d %s",
        "P" if mode == "prefill" else "D",
        req.bootstrap_room,
        len(req.origin_input_ids),
        req.cached_tokens,
        len(req.output_ids),
        " ".join(parts),
    )


def log_chunk(kv_chunk, t_deq: float, t_send0: float, t_send1: float) -> None:
    logger.info(
        "L12T PX room=%s last=%d pages=%d enq=%.4f deq=%.4f s0=%.4f s1=%.4f defer=%d",
        kv_chunk.room,
        int(kv_chunk.is_last_chunk),
        len(kv_chunk.prefill_kv_indices),
        getattr(kv_chunk, "l12_enq", 0.0),
        t_deq,
        t_send0,
        t_send1,
        getattr(kv_chunk, "l12_defer", 0),
    )


def log_prefill_batch(batch, t0: float, t1: float, t2: float, nwait: int, ninfl: int) -> None:
    """t0 loop start, t1 batch chosen, t2 run_batch returned, now = result processed."""
    logger.info(
        "L12T PB t0=%.4f t1=%.4f t2=%.4f t3=%.4f bs=%d ntok=%s wait=%d infl=%d",
        t0,
        t1,
        t2,
        time.time(),
        batch.batch_size(),
        batch.extend_num_tokens,
        nwait,
        ninfl,
    )


def log_prefill_loop(t0: float, marks: Dict[str, float], bs: int, nwait: int, ninfl: int) -> None:
    """Sub-phases of one prefill loop iteration that ran a batch (P TP0); now = inflight queue processed."""
    logger.info(
        "L12T PL t0=%.4f %s t4=%.4f bs=%d wait=%d infl=%d",
        t0,
        " ".join(f"{k}={v:.4f}" for k, v in marks.items()),
        time.time(),
        bs,
        nwait,
        ninfl,
    )


def log_dispatch(tokenized_obj) -> None:
    """API-server side of one request: created / dispatch start / dispatch end (wall clock), keyed by bootstrap_room."""
    ts = tokenized_obj.time_stats
    logger.info(
        "L12T T room=%s in=%d ct=%s d0=%s d1=%s",
        tokenized_obj.bootstrap_room,
        len(tokenized_obj.input_ids),
        _w(getattr(ts, "created_time", 0.0)),
        _w(getattr(ts, "api_server_dispatch_time", 0.0)),
        _w(getattr(ts, "api_server_dispatch_finish_time", 0.0)),
    )


def install_gc_log(min_ms: int, where: str) -> None:
    """Log every cyclic-GC pause of at least min_ms in this process (t = wall-clock end of the pause)."""
    import gc

    start = [0.0]

    def _cb(phase, info):
        if phase == "start":
            start[0] = time.perf_counter()
            return
        dur = time.perf_counter() - start[0]
        if dur * 1e3 >= min_ms:
            logger.info(
                "L12T GC where=%s gen=%d dur=%.4f t=%.4f collected=%d",
                where,
                info.get("generation", -1),
                dur,
                time.time(),
                info.get("collected", 0),
            )

    gc.callbacks.append(_cb)
