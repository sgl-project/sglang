"""Gather-site source fingerprints (row-overwrite case, §24.18).

SGLANG_CP_SIZE_LOG=2 enables fingerprinting at CP gather completion points:
16-position x 4-int32-column windows of the gathered tensor are copied
non-blocking into per-tag pinned host rings with a stream event. Windows
are taken at three anchors (head/mid/tail, recorded under tag/tag.mid/
tag.tail); fp32 maps 4 columns, bf16/fp16 8 columns (int32 pairs), int64 2,
int8 16, and 1-D tensors ride column 0 (stride-1 token streams are the
payload's best-shape candidates). No device sync and no tensor pinning on
the hot path (each window copy is 256 bytes).

At [mf-raw] hit time the prefill send path drains the rings (events are
seconds old by then; only the most recent _SCAN entries per tag are
compared -- the victim forward is the newest one) with diagonal
(phase-tolerant) run matching, which also covers the observed
first_dirty_off = 0..4 shifts. A hit names the gather site whose output
produced the payload stream, plus the column. Torch-only: importable
anywhere.

Accepted limitation: windows sample only three anchors per tensor, so a
writer whose copy starts at an arbitrary interior offset (e.g. 800 into
the chunk) can be missed even if the site is the true source; a no-match
verdict excludes "head/mid/tail-aligned copy", not the site itself.
Cross-run comparison of the unconditional [cp-fp] recent streams remains
valid regardless (x1.5 scaling verdict, §24.18).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_POS = 16  # positions (tokens) per fingerprint window
_COLS = 4  # int32 columns per position (fp32 cols 0-3, bf16 pair cols 0-7)
_CELLS = _POS * _COLS
_RING = 1024  # entries per tag; ~61/forward per site-role => several forwards
_SCAN = 256  # newest entries compared per tag at drain time (~4 forwards)
_RUN_MIN = 4  # diagonal run length that counts as a hit
_WINDOWS_MIN_T = 3 * _POS  # tensors below this record head (+tail) only

_on: dict = {}
_rings: dict = {}
_warned: dict = {}


def _enabled() -> bool:
    if "v" not in _on:
        _on["v"] = os.getenv("SGLANG_CP_SIZE_LOG", "") == "2"
    return _on["v"]


def _windows(T: int) -> list:
    """-> [(role, offset)] window anchors for a length-T leading dim."""
    if T >= _WINDOWS_MIN_T:
        return [("head", 0), ("mid", (T - _POS) // 2), ("tail", T - _POS)]
    if T >= 2 * _POS:
        return [("head", 0), ("tail", T - _POS)]
    return [("head", 0)]


def _cells_of(mat: torch.Tensor, off: int) -> torch.Tensor:
    """mat: [T, C] -> [_CELLS] int32 device cells covering rows [off, off+_POS).

    Per position the source contributes ``16 // element_size`` columns so
    each window is exactly 256 bytes (fp32: cols 0-3; bf16/fp16: 8 pair
    cols; int64: 2; int8: 16). Narrow/1-D inputs keep layout [pos][col]
    with zero-padded trailing columns; layout is row-major, so the column
    stream is entry[i*_COLS+c].
    """
    rows = mat[off : off + _POS]
    src_cols = max(1, min(_POS // rows.element_size(), rows.shape[1]))
    win = rows[:, :src_cols].contiguous().view(torch.int32)
    win = win.reshape(_POS, -1)[:, :_COLS]
    cells = torch.zeros(_CELLS, dtype=torch.int32, device=rows.device)
    cells.view(_POS, _COLS)[:, : win.shape[1]] = win
    return cells


def _put(tag: str, cells: torch.Tensor) -> None:
    if tag not in _rings:
        try:
            host = torch.empty((_RING, _CELLS), dtype=torch.int32, pin_memory=True)
        except Exception:
            host = torch.empty((_RING, _CELLS), dtype=torch.int32)
        _rings[tag] = {"slot": 0, "host": host, "events": [None] * _RING}
    r = _rings[tag]
    slot = r["slot"]
    r["host"][slot].copy_(cells, non_blocking=True)
    if cells.device.type == "cpu":
        # CPU copy is synchronous; a sentinel keeps drain() uniform.
        r["events"][slot] = True
    else:
        r["events"][slot] = (
            torch.get_device_module(cells.device).current_stream().record_event()
        )
    r["slot"] = (slot + 1) % _RING


def _warn_once() -> None:
    if not _warned.get("v"):
        _warned["v"] = True
        try:
            logger.warning("gather_fp.record failed once", exc_info=True)
        except Exception:
            pass


def record(tag: str, tensor) -> None:
    """Copy the fingerprint windows of ``tensor`` into its host rings.

    Non-blocking; event-synced only at [mf-raw] drain time. No-op unless
    SGLANG_CP_SIZE_LOG=2.
    """
    try:
        if not _enabled():
            return
        if tensor is None or tensor.ndim not in (1, 2) or tensor.shape[0] < _POS:
            return
        mat = tensor.unsqueeze(1) if tensor.ndim == 1 else tensor
        for role, off in _windows(tensor.shape[0]):
            _put(tag if role == "head" else f"{tag}.{role}", _cells_of(mat, off))
    except Exception:
        _warn_once()


def _wait(ev) -> bool:
    if ev is None:
        return False
    try:
        if ev is not True:
            ev.synchronize()
        return True
    except Exception:
        return False


def _flatten_key(key) -> list:
    """-> flat int list; tolerates nested lists from 2-D int32 views."""
    out = []
    for k in key:
        if isinstance(k, (list, tuple)):
            out.extend(int(x) for x in k)
        else:
            out.append(int(k))
    return out


def _run_of(entry_cells, key, col) -> int:
    """Longest contiguous match between the key and one fingerprint column
    at any relative phase (covers first_dirty_off shifts in either side).

    ``entry_cells`` is the 64-int list of one entry; column c of position
    i is entry_cells[i*_COLS+c].
    """
    best = run = 0
    for d in range(-(_POS - 1), _POS):
        run = 0
        for i in range(_POS):
            j = i + d
            if 0 <= j < _POS and key[i] == entry_cells[j * _COLS + col]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
    return best


def _scan_ids(r: dict) -> list:
    """Newest recorded slot ids (most recent first), capped at _SCAN."""
    slot = r["slot"]
    n = min(_SCAN, _RING)
    return [(slot - 1 - j) % _RING for j in range(n)]


def drain(key) -> list:
    """-> [(tag, col, run)] fingerprint hits against the payload key.

    ``key`` is the 16-int32 payload stream prefix. Each ring entry holds 4
    columns x 16 positions; a column with a diagonal run >= _RUN_MIN names
    the producing gather site. Waited events are downgraded to the True
    sentinel so repeated drains stay cheap.
    """
    out = []
    try:
        key = _flatten_key(key)
        if len(key) < _POS:
            return out
        for tag, r in _rings.items():
            for i in _scan_ids(r):
                ev = r["events"][i]
                if not _wait(ev):
                    continue
                if ev is not True:
                    r["events"][i] = True
                entry_cells = r["host"][i].tolist()
                for col in range(_COLS):
                    run = _run_of(entry_cells, key, col)
                    if run >= _RUN_MIN:
                        out.append((tag, col, run))
    except Exception:
        pass
    return out


def counts() -> dict:
    """-> {tag: n_recorded_entries} instrument liveness for the no-match line."""
    try:
        return {
            tag: sum(1 for ev in r["events"] if ev is not None)
            for tag, r in _rings.items()
        }
    except Exception:
        return {}


def recent() -> list:
    """-> [(tag, [64 ints])] the last 4 entries per tag (offline analysis)."""
    out = []
    try:
        for tag, r in _rings.items():
            slot = r["slot"]
            for j in range(4):
                i = (slot - 1 - j) % _RING
                if not _wait(r["events"][i]):
                    continue
                out.append((tag, [int(x) for x in r["host"][i].tolist()]))
    except Exception:
        pass
    return out
