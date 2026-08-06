# Author: adlashab <adlashab@amd.com>
#
# Contiguous host-slot packing and a bulk/staging reload fast-path for HiCache.
#
# The host KV pool hands out free slot ids first-fit and appends freed ids on
# free, so after a few fill/evict cycles a re-admitted prefix lands on a scrambled
# set of host offsets and the reload runs the per-index transfer kernel over
# scattered indices. `take_contiguous_run` makes the allocator hand out a
# contiguous run whenever one is free (the pool keeps its free list sorted), so a
# prefix occupies a contiguous host region. `bulk_reload` then moves that region
# in one shot: a single copy_ when the device target is also contiguous, or a bulk
# copy into a contiguous device staging buffer followed by an on-device
# index_copy_ when the device slots are scattered. Any non-contiguous host case,
# dtype/shape mismatch, or non-contiguous destination returns False so the caller
# runs the exact stock kernel, so a reload is never wrong.

import json
import os
import threading

import torch

_staging = {}
_staging_lock = threading.Lock()

# Fire counters so a live run can confirm the fast-path took the contiguous path
# rather than falling back. Inert unless SGLANG_HICACHE_BULK_STATS names a file.
_fire = {"bulk": 0, "stage": 0, "fallback": 0, "alloc_contig": 0, "alloc_frag": 0}
_STATS_PATH = os.environ.get("SGLANG_HICACHE_BULK_STATS")


def bulk_stats():
    return dict(_fire)


def _bump(kind):
    _fire[kind] = _fire.get(kind, 0) + 1
    if _STATS_PATH:
        try:
            with open(_STATS_PATH + ".tmp", "w") as f:
                json.dump(_fire, f)
            os.replace(_STATS_PATH + ".tmp", _STATS_PATH)
        except Exception:
            pass


def take_contiguous_run(free_slots, need_size):
    """Pick which free slots to hand out. `free_slots` is a 1-D int64 tensor kept
    sorted ascending by the pool. Returns (selected, remaining): `selected` is a
    length-`need_size` contiguous unit-stride run when one is available (best fit,
    smallest run that holds it), else the first `need_size` slots. `remaining`
    stays sorted. Same success set as the stock first-`need_size` pick."""
    n = int(free_slots.numel())
    if need_size <= 0:
        return free_slots[:0], free_slots
    if n <= need_size:
        return free_slots, free_slots[:0]
    d = free_slots[1:] - free_slots[:-1]
    brk = torch.nonzero(d != 1, as_tuple=False).flatten()
    starts = torch.cat([free_slots.new_zeros(1), brk + 1])
    ends = torch.cat([brk + 1, free_slots.new_full((1,), n)])
    lens = ends - starts
    fit = lens >= need_size
    if bool(fit.any()):
        cand = torch.nonzero(fit, as_tuple=False).flatten()
        j = int(cand[torch.argmin(lens[cand])])
        p = int(starts[j])
        sel = free_slots[p : p + need_size]
        rem = torch.cat([free_slots[:p], free_slots[p + need_size :]])
        _bump("alloc_contig")
        return sel, rem
    _bump("alloc_frag")
    return free_slots[:need_size], free_slots[need_size:]


def _single_run(idx):
    n = int(idx.numel())
    if n == 0:
        return None
    start = int(idx[0].item())
    ar = torch.arange(start, start + n, dtype=idx.dtype, device=idx.device)
    return (start, n) if torch.equal(idx, ar) else None


def _bulk_copy(src, dst):
    if src.dtype != dst.dtype or src.numel() != dst.numel() or not dst.is_contiguous():
        return False
    dst.view(-1).copy_(src.reshape(-1), non_blocking=True)
    return True


def _get_staging(dst, n):
    row = tuple(dst.shape[1:])
    key = (str(dst.device), dst.dtype, row)
    with _staging_lock:
        buf = _staging.get(key)
        if buf is None or buf.shape[0] < n:
            cap = 1 if buf is None else buf.shape[0]
            while cap < n:
                cap *= 2
            try:
                buf = torch.empty((cap,) + row, dtype=dst.dtype, device=dst.device)
            except Exception:
                return None
            _staging[key] = buf
        return buf


def _bulk_scatter(src_contig, dst, device_indices):
    n = int(src_contig.shape[0])
    if n == 0:
        return True
    if src_contig.dtype != dst.dtype or not dst.is_contiguous():
        return False
    di = device_indices
    if str(di.device) != str(dst.device):
        di = di.to(dst.device, non_blocking=True)
    if di.dtype != torch.long:
        di = di.to(torch.long)
    if int(di.numel()) != n:
        return False
    staging = _get_staging(dst, n)
    if staging is None:
        return False
    staging[:n].copy_(src_contig, non_blocking=True)
    dst.index_copy_(0, di, staging[:n])
    return True


def bulk_reload(src_layer_buf, dst_layer_buf, host_indices, device_indices):
    """Bulk reload one layer when the host indices are one contiguous run. Handles
    both the MLA KV anchor (token runs) and the DSA indexer (page runs); the caller
    passes the matching per-layer buffers. Returns True if it moved the data,
    False to fall back to the stock per-index kernel."""
    hr = _single_run(host_indices)
    if hr is None:
        _bump("fallback")
        return False
    h0, n = hr
    src = src_layer_buf[h0 : h0 + n]
    dr = _single_run(device_indices)
    if dr is not None:
        d0, _ = dr
        if _bulk_copy(src, dst_layer_buf[d0 : d0 + n]):
            _bump("bulk")
            return True
        _bump("fallback")
        return False
    if _bulk_scatter(src, dst_layer_buf, device_indices):
        _bump("stage")
        return True
    _bump("fallback")
    return False
