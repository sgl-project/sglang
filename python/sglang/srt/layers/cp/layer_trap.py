"""Per-layer req_to_token write trap (row-overwrite case).

Structural fact this exploits: on an extend forward, every LEGAL write to
req_to_token happens in the scheduler's alloc stage BEFORE the model
forward. During the layer loop the active rows are write-frozen -- any
value going dirty (negative or >= 2^30, the same predicate as [mf-raw])
inside the loop is the racing writer, caught in the act.

SGLANG_MF_LAYER_TRAP=1 snapshots the active rows at every checkpoint
(``layer_trap_mark`` after each decoder layer): a device-side gather of
[bs, W] followed by a non-blocking copy into a host buffer with a stream
event. The host comparison runs at the NEXT checkpoint (the event is
long finished by then; synchronize() returns immediately), so the
compute stream never waits and the racing-writer timing is untouched --
the same discipline as gather_fp.py. First hit reports the checkpoint
window and the emitting rank's current [cp-size] seq, correlating the
catch with the surrounding collective stream.

Funnel: layer-granularity marks give the LAYER in one reproduction; a
rerun with finer marks (e.g. ``L{i}.pre-attn`` / ``L{i}.post-attn`` /
``L{i}.post-moe`` added inside the suspect layer's forward) splits the
layer into segments; the seq anchor names the last collective before
the catch. Converges in O(1) reproductions per level.

Only extend forwards are trapped: decode/mixed steps legally write
req_to_token between forwards and would false-positive. ``seq_lens_cpu``
sizes the window with no device sync; without it the full row width is
copied (a few MB worst case, still cheap next to a layer's kernels).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_on: dict = {}
_pend: dict = {}


def _enabled() -> bool:
    if "v" not in _on:
        _on["v"] = os.getenv("SGLANG_MF_LAYER_TRAP", "0") == "1"
    return _on["v"]


def _active(forward_batch):
    """(r2t, idx_int64, W) for the extend window, or None. No device sync.

    Rows = every ALLOCATED slot (all minus free_slots), not just this
    batch: under the overlap event loop a victim request's rows are
    clobbered during LATER batches' forwards (its own forward ran clean
    -- observed 2026-08-24), so watching only the active rows would be
    blind exactly where the writer acts. W stays the active window's
    max(ext+prefix): the observed victims sit at deep column offsets of
    long-context requests, and in-flight requests share that depth.
    """
    if _pend.get("fired"):
        return None
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_extend() or mode.is_mixed():
        return None
    pool = getattr(forward_batch, "req_to_token_pool", None)
    r2t = getattr(pool, "req_to_token", None)
    idx = getattr(forward_batch, "req_pool_indices", None)
    if not torch.is_tensor(r2t) or not torch.is_tensor(idx) or idx.numel() == 0:
        return None
    free = getattr(pool, "free_slots", None)
    if free is not None:
        try:
            hi = int(r2t.shape[0])
            free_set = {int(s) for s in free if 0 < int(s) < hi}
            rows = [s for s in range(1, hi) if s not in free_set]
        except Exception:
            rows = None
        if rows is None or not rows:
            rows = idx.tolist()
        else:
            rows = torch.tensor(rows, dtype=torch.int64)
    else:
        rows = idx
    if torch.is_tensor(rows):
        row_t = rows.to(torch.int64) if rows.dtype is not torch.int64 else rows
    else:
        row_t = torch.tensor(rows, dtype=torch.int64)
    # CPU-side lens only (extend_seq_lens_cpu + prefix, else full width).
    ext = getattr(forward_batch, "extend_seq_lens_cpu", None)
    pre = getattr(forward_batch, "prefix_lens_cpu", None)
    try:
        if ext is not None and pre is not None:
            w = int(max(int(a) + int(b) for a, b in zip(ext, pre)))
        else:
            w = int(r2t.shape[1])
    except Exception:
        return None
    w = max(1, min(w, int(r2t.shape[1])))
    return r2t, row_t, w


def _snapshot(forward_batch, label):
    act = _active(forward_batch)
    if act is None:
        return
    r2t, idx, w = act
    try:
        buf = torch.empty(
            (idx.numel(), w), dtype=torch.int32, pin_memory=True
        )
    except Exception:
        try:
            buf = torch.empty((idx.numel(), w), dtype=torch.int32)
        except Exception:
            return
    try:
        # idx is 1-D: advanced+basic indexing yields [bs, W] directly
        # (idx.unsqueeze(1) here would produce [bs, 1, W] and fail copy_).
        buf.copy_(r2t[idx, :w], non_blocking=True)
    except Exception:
        return
    try:
        ev = (
            torch.get_device_module(r2t.device)
            .current_stream()
            .record_event()
        )
    except Exception:
        ev = None
    # ev=None means no event could be recorded (e.g. CPU tensors in unit
    # tests): the copy was synchronous, treat as already complete.
    _pend.clear()
    _pend.update(
        buf=buf, ev=ev, label=label, rows=idx.tolist(), fired=False
    )


def _drain():
    """Compare the previous checkpoint's snapshot; report the first dirty."""
    if not _pend or _pend.get("fired"):
        return
    ev = _pend.get("ev")
    if ev is not None:
        try:
            ev.synchronize()
        except Exception:
            return
    vals = _pend["buf"]
    bad = (vals < 0) | (vals >= 1 << 30)
    if bool(bad.any()):
        b, c = bad.nonzero()[0].tolist()
        try:
            from sglang.srt.layers.cp.size_log import current_seq

            seq = current_seq()
        except Exception:
            seq = -1
        logger.error(
            "[mf-trap] caught after=%s row=%d col=%d val=%d seq=%d "
            "window=%dx%d",
            _pend["label"],
            _pend["rows"][b],
            c,
            int(vals[b, c]),
            seq,
            vals.shape[0],
            vals.shape[1],
        )
        _pend["fired"] = True
        # Keep the evidence buffer; no further snapshots this forward.


# ---- send-path split probe (§24.22: writer outside every trapped window) ----
# The layer trap proved the r2t row is clean through every EXTEND forward's
# layer loop, yet [mf-raw] (post-translate) sees garbage. The residual
# windows are (a) decode/mixed forwards + the between-forward gap (overlap
# loop, other batches' KV sends, scheduler) and (b) the send path itself
# (translate / pool-side mapping). This probe splits them: a device-side
# dirty-count of the PRE-translate segment, read out ONLY inside the
# already-synchronized [mf-raw] branch -- the clean path pays one small
# reduction kernel and zero extra D2H, preserving the §19 no-sync discipline.
_send_probe: dict = {}


def send_probe_record(kv_indices, rid, slot, seg_start, seg_end) -> None:
    """Enqueue the pre-translate dirty-count for the current segment."""
    if not _enabled():
        return
    try:
        n = ((kv_indices < 0) | (kv_indices >= 1 << 30)).sum()
        _send_probe.clear()
        _send_probe.update(
            n=n, rid=str(rid), slot=int(slot),
            s0=int(seg_start), s1=int(seg_end),
        )
    except Exception:
        _send_probe.clear()


def send_probe_read(n_dirty_post: int, n_tot: int) -> None:
    """Inside the [mf-raw] branch: confront pre- vs post-translate dirt.

    pre>0  -> the row was ALREADY dirty at send-loop entry: writer acts in
              decode/mixed forwards or the between-forward gap (NOT translate).
    pre=0  -> garbage entered during translate/pool-side mapping: a new
              kernel-side family (confront translate + scatter paths).
    """
    if not _enabled() or not _send_probe:
        return
    try:
        pre = int(_send_probe["n"].item())
    except Exception:
        return
    logger.error(
        "[mf-trap] send pre-translate rid=%s slot=%d seg=[%d,%d) "
        "n_dirty_pre=%d n_dirty_post=%d n_tot=%d",
        _send_probe["rid"], _send_probe["slot"],
        _send_probe["s0"], _send_probe["s1"],
        pre, n_dirty_post, n_tot,
    )
    _send_probe.clear()


def layer_trap_start(forward_batch) -> None:
    """Baseline snapshot before layer 0 (post-alloc: the legal state)."""
    if not _enabled():
        return
    if not _on.get("armed"):
        # Liveness marker: distinguishes "ran and stayed silent" from
        # "never ran" in a log with no catches (deploy/env auditing).
        _on["armed"] = True
        logger.info(
            "[mf-trap] armed rows=allocated window=active-batch-max "
            "(silent-clean mode; catches print [mf-trap] caught)"
        )
    # Drain a pending gap-probe snapshot (post-target/post-draft of the
    # previous iteration) BEFORE re-baselining, or its window is lost. If
    # it caught, keep the evidence buffer and stop (first-hit discipline).
    _drain()
    if _pend.get("fired"):
        return
    _pend.clear()
    _snapshot(forward_batch, "start")


def layer_trap_mark(forward_batch, label: str) -> None:
    """Checkpoint: drain the previous snapshot, take a new one.

    A dirty hit means the writer acted between the previous label and
    this one (half-open window (prev, this]).
    """
    if not _enabled():
        return
    if _pend.get("fired"):
        return  # First-hit only: minimal log, unchanged timing after.
    _drain()
    if _pend.get("fired"):
        return
    _snapshot(forward_batch, label)


def layer_trap_end(forward_batch) -> None:
    """Final drain after the last layer."""
    if not _enabled():
        return
    _drain()


# ---- gap probes (§24.23: writer outside the layer windows) ---------------
# The layer trap + send-probe split proved the r2t row is clean through the
# layer loop but ALREADY dirty at send-loop entry. The untrapped windows on
# the forward thread are exactly: (1) the target model's post-loop path
# (after layer_trap_end: CP all-gather, norms, lm_head, logits capture) and
# (2) the draft prefill forward (a separate ForwardMode the extend-only
# `_active` gate never traps). These probes bracket them: snapshot at the
# boundary, drain at the next checkpoint (the next iteration's
# layer_trap_start drains a still-pending gap snapshot first).


def _snapshot_pool(pool, label: str) -> None:
    """Snapshot every allocated row of `pool` (full width, batch-free)."""
    if not _enabled() or _pend.get("fired"):
        return
    r2t = getattr(pool, "req_to_token", None)
    if not torch.is_tensor(r2t):
        return
    free = getattr(pool, "free_slots", None)
    hi = int(r2t.shape[0])
    if free is not None:
        try:
            free_set = {int(s) for s in free if 0 < int(s) < hi}
            rows = [s for s in range(1, hi) if s not in free_set]
        except Exception:
            rows = None
        if not rows:
            return
        idx = torch.tensor(rows, dtype=torch.int64)
    else:
        idx = torch.arange(1, hi, dtype=torch.int64)
    try:
        buf = torch.empty(
            (idx.numel(), r2t.shape[1]), dtype=torch.int32, pin_memory=True
        )
    except Exception:
        buf = torch.empty((idx.numel(), r2t.shape[1]), dtype=torch.int32)
    try:
        buf.copy_(r2t[idx], non_blocking=True)
    except Exception:
        return
    try:
        ev = (
            torch.get_device_module(r2t.device).current_stream().record_event()
        )
    except Exception:
        ev = None
    _pend.clear()
    _pend.update(
        buf=buf, ev=ev, label=label, rows=idx.tolist(), fired=False
    )


def gap_probe_post_target(worker) -> None:
    """Call on the forward thread right after the TARGET forward returns
    (eagle_worker_v2.forward_batch_generation, extend branch). Drains any
    prior state, then snapshots: a dirty hit at the next drain convicts the
    target post-loop path (CP all-gather / norm / lm_head / logits capture).
    """
    if not _enabled() or _pend.get("fired"):
        return
    pool = getattr(
        getattr(worker, "_target_worker", None), "model_runner", None
    )
    pool = getattr(pool, "req_to_token_pool", None)
    if pool is None:
        return
    _drain()
    if _pend.get("fired"):
        return
    _snapshot_pool(pool, "gap:post-target")


def gap_probe_post_draft(worker) -> None:
    """Call right after the DRAFT prefill forward returns. Drains the
    post-target snapshot (naming the target post-loop window) and leaves a
    post-draft snapshot for the next iteration's layer_trap_start: a hit
    there convicts the draft prefill forward family.
    """
    if not _enabled() or _pend.get("fired"):
        return
    pool = getattr(
        getattr(worker, "_draft_worker", None), "draft_runner", None
    )
    pool = getattr(pool, "req_to_token_pool", None) or getattr(
        getattr(worker, "_target_worker", None),
        "model_runner",
        None,
    )
    pool = getattr(pool, "req_to_token_pool", None)
    if pool is None:
        return
    _drain()
    if _pend.get("fired"):
        return
    _snapshot_pool(pool, "gap:post-draft")


# ---- glue probes (§24.34: bisect the scheduler glue window) ----------------
# Run 19's re-observation analysis convicted the write to the sub-second
# scheduler glue between the victim iteration's gap:post-draft snapshot
# (clean) and its send-loop read (dirty): eagle-worker return ->
# process_batch_result (copy_done.synchronize / result finalize ->
# maybe_cache_unfinished_req radix insert + FreeDeviceKV free device ops ->
# eagle output extraction incl. hidden_states[i].cpu().clone() D2H) ->
# send entry. These row-scoped marks bisect exactly that window; the first
# mark drains any pending gap snapshot, so the existing chain is preserved
# and glue:pre-send is drained by the next iteration's layer_trap_start.


def _snapshot_row(pool, row: int, width: int, label: str) -> None:
    """Snapshot r2t[row, :width] -- per-request, deep-column capable."""
    if not _enabled() or _pend.get("fired"):
        return
    r2t = getattr(pool, "req_to_token", None)
    if not torch.is_tensor(r2t):
        return
    try:
        w = max(1, min(int(width), int(r2t.shape[1])))
    except Exception:
        return
    try:
        buf = torch.empty((1, w), dtype=torch.int32, pin_memory=True)
    except Exception:
        try:
            buf = torch.empty((1, w), dtype=torch.int32)
        except Exception:
            return
    try:
        buf.copy_(r2t[int(row) : int(row) + 1, :w], non_blocking=True)
    except Exception:
        return
    try:
        ev = (
            torch.get_device_module(r2t.device)
            .current_stream()
            .record_event()
        )
    except Exception:
        ev = None
    _pend.clear()
    _pend.update(buf=buf, ev=ev, label=label, rows=[int(row)], fired=False)


def glue_probe(pool, row: int, width: int, label: str) -> None:
    """Drain the pending snapshot, then snapshot this request's row.

    Labels (call sites in disaggregation/prefill.py,
    process_batch_result_disagg_prefill final-chunk branch):
      glue:pre-cache  -- before maybe_cache_unfinished_req; a catch on the
                         DRAINED snapshot (gap:post-draft) convicts the
                         eagle-return tail / result-resolve prologue.
      glue:post-cache -- right after the cache call; a catch here convicts
                         the RADIX CACHE/FREE PATH (insert + FreeDeviceKV
                         device ops; the §22 swa.py:330 crash family).
      glue:pre-send   -- right before send_kv_chunk; a catch here convicts
                         the eagle output extraction (cpu().clone() D2H) +
                         logprob glue. This snapshot itself is drained at
                         the next iteration's layer_trap_start.
    """
    if not _enabled() or _pend.get("fired"):
        return
    _drain()
    if _pend.get("fired"):
        return
    _snapshot_row(pool, row, width, label)


def glue_probe_pool(pool, label: str) -> None:
    """Drain the pending snapshot, then a full-pool snapshot (batch scope).

    glue:entry (process_batch_result_disagg_prefill head): a catch on the
    drained gap:post-draft snapshot convicts the eagle-worker return tail;
    the entry snapshot itself is compared at the next mark, splitting the
    batch-result prologue (copy_done.synchronize / TopkCaptureOutput
    finalize scatters / move_logprobs_to_cpu D2H) into its own window.
    """
    if not _enabled() or _pend.get("fired"):
        return
    _drain()
    if _pend.get("fired"):
        return
    _snapshot_pool(pool, label)
