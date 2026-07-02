"""Expert store: the host backing for all E experts + the page-in transport into the K-slot GPU pool.

The K-slot GPU pool *is* the layer's own expert params (sglang's native loader filled slots 0..K-1). An
``ExpertStore`` holds all E experts per paged tensor on the host and copies the chosen ones into their
slots on a miss. It owns only the *backing and the byte movement* — not the residency *decision* (which
expert goes in which slot, when), which is the pager's job (``pager.py``). Splitting the two lets the
transport vary behind one interface:

* ``PinnedExpertStore`` — page-locked host RAM, paged with sglang's existing ``transfer_kv_per_layer_mla``
  block copy (indices read on-device, dynamic count, capture-safe). The fast default.
* ``PageableExpertStore`` — non-pinned host RAM, paged with a plain indexed copy. Correct but slower; for
  hosts that can't page-lock the full store.

Future tiers (disk-mmap, compressed) are additional ``ExpertStore`` subclasses — they implement the same
``page_in`` contract and need no change to the pager or the forward.
"""

from __future__ import annotations

import math
import mmap
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

# packed-quant scaffolding the fused-MoE kernel never reads on the paged path
_NONPAGED_SUFFIXES = ("_g_idx", "_g_idx_sort_indices", "_weight_shape")


def _host_available_bytes() -> int:
    """Available host memory in bytes (Linux ``/proc/meminfo`` ``MemAvailable``), or 0 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except Exception:
        pass
    return 0


def _alloc_disk_mmap(cold_dir: Optional[str], dims: tuple, dtype: torch.dtype):
    """A host tensor backed by a MAP_SHARED file on disk (P4 cold tier) — RAM use is bounded by the OS page
    cache (clean pages evict back to the file under pressure), so a store far larger than RAM still loads.
    The file is unlinked immediately: the inode lives only as long as the mapping (auto-cleaned on free), so
    no stale multi-GB files are left behind. ``cold_dir`` must be on a real disk with room for the cold tier
    (NOT a tmpfs like /tmp, which would defeat the point); falls back to the system temp dir.

    Returns ``(tensor, mm)`` — the ``mmap`` object is returned too so callers can issue ``madvise`` read-ahead
    hints (the gather otherwise faults one page at a time, serially; see ``WindowedExpertStore.prefetch_cold``).
    """
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()
    d = cold_dir or tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=d, suffix=".paged_experts_cold")
    try:
        os.ftruncate(fd, n_bytes)
        os.unlink(
            path
        )  # anonymous-on-disk: the inode persists while mmap'd, freed on munmap
        mm = mmap.mmap(
            fd, n_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
        try:
            # Best-effort: with THP/large-folio support, faults map 2 MB at a time instead of 4 KB —
            # a multi-MB expert row costs a handful of faults instead of hundreds. Ignored by kernels
            # without file-backed-THP; never fatal.
            mm.madvise(mmap.MADV_HUGEPAGE)
        except (OSError, ValueError, AttributeError):
            pass
    finally:
        os.close(fd)  # the mapping keeps the inode alive after the fd is closed
    # torch.frombuffer keeps mm alive inside the tensor storage (munmap fires when the tensor is freed)
    t = torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
    return t, mm


# --- shared staging machinery for cold-tier page-ins -----------------------------------------------
# One pinned buffer per (param-name, row-shape) reused across ALL layers (staging is sequential): the
# cold tier is pageable RAM or a disk mmap, and an H2D from pageable memory is silently synchronous and
# slow — gathering the rows into a pinned buffer first makes the H2D a fast async DMA. The gather stays
# SERIAL: with the MADV_WILLNEED read-ahead the page faults are already serviced concurrently by the
# kernel, so the copies run against warm pages — threading them was measured net-negative (pool overhead
# on warm memcpys; the fault parallelism was the winnable part and prefetch already claims it).
_STAGE_PIN: Dict = {}


def _stage_pin_buf(name: str, k: int, row_shape, dtype) -> torch.Tensor:
    key = (name, tuple(row_shape), dtype)
    buf = _STAGE_PIN.get(key)
    if buf is None or buf.shape[0] < k:
        buf = torch.empty((k, *row_shape), dtype=dtype, device="cpu", pin_memory=True)
        _STAGE_PIN[key] = buf
    return buf


def discover_paged_params(layer, num_slots: int) -> Dict[str, torch.Tensor]:
    """Per-expert params on ``layer``: leading dim == num_slots (the K-slot pool) and non-empty per-slot."""
    out = {}
    for name, p in list(layer.named_parameters(recurse=False)) + list(
        layer.named_buffers(recurse=False)
    ):
        if any(name.endswith(s) for s in _NONPAGED_SUFFIXES):
            continue
        if p.dim() >= 1 and p.shape[0] == num_slots and p[0].numel() > 0:
            out[name] = p
    return out


class ExpertStore(ABC):
    """Host backing for all E experts + the page-in transport into the K-slot GPU pool.

    Subclasses choose the host backing and the byte movement; the pager decides which expert goes in
    which slot and hands the plan to ``page_in`` as index tensors. ``host[name]`` is an ``[E, *slot_shape]``
    CPU buffer per paged tensor (filled once at load time); ``gpu[name]`` is the layer's K-slot param;
    ``item_bytes[name]`` is the per-expert block size in bytes. The class attr ``pinned`` records whether
    the backing is page-locked (and gates the 8-byte alignment that the ``transfer_kv`` gather requires).
    """

    pinned: bool = False

    def __init__(self, layer, num_experts_E: int, num_resident_K: int, device):
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.gpu = discover_paged_params(
            layer, num_resident_K
        )  # the K-slot GPU pool (layer params)
        assert self.gpu, "no per-expert params found on layer"
        self.host: Dict[str, torch.Tensor] = {}
        self.item_bytes: Dict[str, int] = {}
        for name, p in self.gpu.items():
            self.host[name] = torch.empty(
                (self.E, *p.shape[1:]),
                dtype=p.dtype,
                device="cpu",
                pin_memory=self.pinned,
            )
            self.item_bytes[name] = p[0].numel() * p.element_size()
            # transfer_kv_per_layer_mla requires the per-expert block to be 8-byte aligned. Real weight
            # rows (bf16 / marlin qweight+scales+qzeros) satisfy this; a 1-D per-expert scalar scale
            # (e.g. fp8, 4 B) does not -> that needs the deferred scalar-gather path. The pageable copy
            # path has no such requirement.
            if self.pinned and self.item_bytes[name] % 8 != 0:
                raise RuntimeError(
                    f"[paged-experts] paged tensor {name!r} per-expert size {self.item_bytes[name]} B "
                    "is not 8-byte aligned (transfer_kv requirement); unsupported on the reuse gather "
                    "path. Use --paged-experts-store paged (the pageable copy has no such requirement)."
                )

    @abstractmethod
    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        """Copy ``host[src_experts[i]] -> gpu[dst_slots[i]]`` for every paged tensor.

        ``src_experts`` / ``dst_slots`` are device ``int64`` index tensors from the pager's decision; a
        no-op for an empty plan.
        """

    # --- checkpoint-fill accessors (store-layout-agnostic; used by ``pager.setup_pager``) ---
    # A single ``[E, *]`` host buffer here; ``WindowedExpertStore`` overrides both to route an expert into
    # its hot/cold tier, so the fill code never special-cases the store layout.
    def row(self, name: str, e: int) -> torch.Tensor:
        """Writable host slice backing expert ``e`` for paged tensor ``name`` (per-expert fill)."""
        return self.host[name][e]

    def fill_tensor(self, name: str, full: torch.Tensor) -> None:
        """Fill the whole host backing for ``name`` from a contiguous ``[E, *slot_shape]`` CPU tensor."""
        self.host[name].copy_(full)


class PinnedExpertStore(ExpertStore):
    """Pinned (page-locked) host store, paged with sglang's existing ``transfer_kv_per_layer_mla`` block
    copy — pinned-host -> device, indices read on-device, dynamic count, capture-safe. The fast default.
    """

    pinned = True

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        if src_experts.numel() == 0:
            return
        from sgl_kernel import transfer_kv_per_layer_mla

        for name, gpu_param in self.gpu.items():
            transfer_kv_per_layer_mla(
                src=self.host[name],
                dst=gpu_param.data,
                src_indices=src_experts,
                dst_indices=dst_slots,
                item_size=self.item_bytes[name],
            )


class PageableExpertStore(ExpertStore):
    """Non-pinned host store, paged with a plain indexed copy (gather rows on the host, one H2D, scatter
    into the slots). Correct but slower; for hosts that can't page-lock the full store. ``transfer_kv``
    would read stale data from non-page-locked memory, so it is not used here."""

    pinned = False

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        if src_experts.numel() == 0:
            return
        src_cpu = src_experts.to("cpu")
        for name, gpu_param in self.gpu.items():
            rows = self.host[name].index_select(0, src_cpu).to(gpu_param.device)
            gpu_param.data.index_copy_(0, dst_slots, rows)


class WindowedExpertStore(ExpertStore):
    """Pinned hot window + pageable cold tail — the fallback for stores that can't be fully page-locked.

    The ``W`` hot experts live in a page-locked ``host_hot[name]`` block (paged with ``transfer_kv``, and —
    in the captured path, pr3 — gatherable on-device through its UVA device pointer); the remaining ``E-W``
    cold experts live in a pageable ``host_cold[name]`` block (paged with a plain indexed copy, or — under
    capture — staged out-of-graph on a deferred miss). ``host[name]`` is *not* allocated: there is no single
    ``[E, *]`` buffer, so the fill goes through ``row`` / ``fill_tensor``.

    Membership defaults to the static ``[0, W)`` split (``hot_pos`` / ``cold_pos``); a frequency profile may
    later pin the hottest ``W`` (the maps make that a fill-order change, not a layout change). This is the
    >pin-ceiling path: ``W`` = what actually fits page-locked, the rest stays pageable but still served.
    """

    pinned = True  # the hot window is page-locked; the cold tail is pageable by design

    def __init__(
        self,
        layer,
        num_experts_E: int,
        num_resident_K: int,
        device,
        *,
        window_W: int,
        cold_backing: str = "ram",
        cold_dir: Optional[str] = None,
    ):
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.W = max(0, min(int(window_W), num_experts_E))
        self.cold_backing = (
            cold_backing  # "ram" (pageable) | "disk" (mmap'd file, page-cache-bounded)
        )
        self.gpu = discover_paged_params(layer, num_resident_K)
        assert self.gpu, "no per-expert params found on layer"
        self.host_hot: Dict[str, torch.Tensor] = (
            {}
        )  # [W, *shape] PINNED (transfer_kv / UVA gather)
        self.host_cold: Dict[str, torch.Tensor] = (
            {}
        )  # [E-W, *shape] cold tier (RAM pageable | disk mmap)
        self.item_bytes: Dict[str, int] = {}
        self._cold_mm: Dict[str, mmap.mmap] = (
            {}
        )  # disk tier mmap objects, for madvise read-ahead hints
        on_disk = cold_backing == "disk"
        for name, p in self.gpu.items():
            self.host_hot[name] = torch.empty(
                (self.W, *p.shape[1:]), dtype=p.dtype, device="cpu", pin_memory=True
            )
            cold_dims = (self.E - self.W, *p.shape[1:])
            # disk: a >RAM cold tier mmap'd to a file (page-cache-bounded) so the store can exceed RAM;
            # ram: a plain pageable tensor (the cold tier must fit RAM).
            if on_disk:
                self.host_cold[name], self._cold_mm[name] = _alloc_disk_mmap(
                    cold_dir, cold_dims, p.dtype
                )
            else:
                self.host_cold[name] = torch.empty(
                    cold_dims, dtype=p.dtype, device="cpu", pin_memory=False
                )
            self.item_bytes[name] = p[0].numel() * p.element_size()
            # the hot tier feeds transfer_kv -> same 8-byte alignment requirement as the pinned store
            # (see ExpertStore.__init__). The pageable cold tier has none.
            if self.item_bytes[name] % 8 != 0:
                raise RuntimeError(
                    f"[paged-experts] paged tensor {name!r} per-expert size {self.item_bytes[name]} B "
                    "is not 8-byte aligned (transfer_kv requirement on the pinned window); unsupported. "
                    "Use --paged-experts-store paged (the pageable copy has no such requirement)."
                )
        # expert -> (tier, row). v1: static split -> hot experts [0, W), cold experts [W, E). hot_pos[e] is
        # the row of e in host_hot (-1 if cold); cold_pos[e] the row in host_cold (-1 if hot).
        self.hot_pos = torch.full((self.E,), -1, dtype=torch.int64)
        self.cold_pos = torch.full((self.E,), -1, dtype=torch.int64)
        self.hot_pos[: self.W] = torch.arange(self.W, dtype=torch.int64)
        self.cold_pos[self.W :] = torch.arange(self.E - self.W, dtype=torch.int64)

    def prefetch_cold(self, experts) -> None:
        """Issue MADV_WILLNEED for the disk-mmap rows of ``experts`` so the kernel does parallel async
        read-ahead (high queue depth) instead of the serial one-page-fault-at-a-time the gather would do.
        No-op unless the cold tier is disk-backed. madvise needs a page-aligned start, so we round the row
        offset down to a page and extend the length to cover the row."""
        if not self._cold_mm:
            return
        page = mmap.PAGESIZE
        # Hoist row resolution and coalesce adjacent/overlapping page ranges into one madvise each:
        # small rows (fp8 block scales) share pages, and large sorted batches otherwise cost thousands
        # of syscalls per step in the wave regime.
        rows = sorted(r for r in (int(self.cold_pos[e]) for e in experts) if r >= 0)
        if not rows:
            return
        for name, mm in self._cold_mm.items():
            stride = self.item_bytes[name]
            size = len(mm)
            m_start = m_end = None
            for r in rows:
                off = r * stride
                start = (off // page) * page
                end = min(off + stride, size)
                if m_end is not None and start <= m_end:
                    m_end = max(m_end, end)
                    continue
                if m_end is not None:
                    try:
                        mm.madvise(mmap.MADV_WILLNEED, m_start, m_end - m_start)
                    except (OSError, ValueError):
                        pass  # best-effort hint; never fatal
                m_start, m_end = start, end
            if m_end is not None:
                try:
                    mm.madvise(mmap.MADV_WILLNEED, m_start, m_end - m_start)
                except (OSError, ValueError):
                    pass

    def prefetch_cold_all(self) -> None:
        """Issue MADV_WILLNEED over the ENTIRE cold tier (one call per mmap). Used by the wave path to
        read the NEXT layer's cold file ahead while the current layer transfers/computes: at wave-regime
        batch sizes the distinct set saturates toward E, so the next layer's whole cold tier is a
        predictable read — one syscall queues it at full depth. No-op unless disk-backed; best-effort.
        Gated on memory pressure: read-ahead larger than the available page cache would evict the pages
        the CURRENT layer is reading (IO amplification on true >RAM stores).
        """
        if not self._cold_mm:
            return
        total = sum(len(m) for m in self._cold_mm.values())
        avail = _host_available_bytes()
        if avail and total > avail // 2:
            return
        for mm in self._cold_mm.values():
            try:
                mm.madvise(mmap.MADV_WILLNEED, 0, len(mm))
            except (OSError, ValueError):
                pass

    def is_hot(self, e: int) -> bool:
        return bool(self.hot_pos[e] >= 0)

    # --- fill accessors: route per expert into the hot/cold tier (no single [E,*] buffer) ---
    def row(self, name: str, e: int) -> torch.Tensor:
        hp = int(self.hot_pos[e])
        if hp >= 0:
            return self.host_hot[name][hp]
        return self.host_cold[name][int(self.cold_pos[e])]

    def fill_tensor(self, name: str, full: torch.Tensor) -> None:
        # v1 membership is the contiguous [0, W) split, so the tiers are full[:W] / full[W:]. (A frequency
        # profile would gather by hot_pos/cold_pos instead — a fill-order change, deferred to P3.)
        self.host_hot[name].copy_(full[: self.W])
        self.host_cold[name].copy_(full[self.W :])

    def set_window_membership(self, hot_experts) -> None:
        """Re-pin the window to hold ``hot_experts`` (the top-W by routing frequency) instead of the static
        ``[0, W)`` — the P3 freq-ranked window. Runs once, out-of-graph, after a short profiling period;
        the GPU slots keep their (expert-indexed) data unchanged, so only the page-in *source* tier moves.

        Δ-SET: only the experts that actually CHANGE tier move — each promoted expert (cold -> hot) swaps
        rows with a demoted one (hot -> cold); everything else stays in place (``hot_pos``/``cold_pos`` map
        expert -> row, so row order within a tier is free). The previous full-store rewrite read+wrote
        every expert row per tensor — on a disk cold tier that meant re-reading AND dirtying the entire
        cold file (a multi-second-to-minutes stall on one token, plus page-cache eviction); the Δ set is
        typically a small fraction of E.
        """
        hot = [int(e) for e in list(hot_experts)[: self.W]]
        assert len(set(hot)) == len(hot), "hot set has duplicates"
        hot_set = set(hot)
        old_hot = set(e for e in range(self.E) if int(self.hot_pos[e]) >= 0)
        promoted = [e for e in hot if e not in old_hot]  # cold -> hot
        demoted = [e for e in old_hot if e not in hot_set]  # hot -> cold
        assert len(promoted) == len(
            demoted
        ), "window size W is fixed; tier moves must pair up"
        if not promoted:
            return  # membership unchanged
        # Disk cold tier: queue read-ahead for the promoted rows so the swap below faults them in parallel.
        self.prefetch_cold(promoted)
        pairs = [
            (p, d, int(self.hot_pos[d]), int(self.cold_pos[p]))
            for p, d in zip(promoted, demoted)
        ]
        for name in self.gpu:
            hh, hc = self.host_hot[name], self.host_cold[name]
            tmp = torch.empty_like(hh[0])
            for _p, _d, hot_row, cold_row in pairs:
                tmp.copy_(hh[hot_row])  # save the demoted expert's data
                hh[hot_row].copy_(hc[cold_row])  # promoted: cold row -> freed hot row
                hc[cold_row].copy_(
                    tmp
                )  # demoted: -> the promoted expert's old cold row
        for p, d, hot_row, cold_row in pairs:
            self.hot_pos[p] = hot_row
            self.hot_pos[d] = -1
            self.cold_pos[d] = cold_row
            self.cold_pos[p] = -1

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        if src_experts.numel() == 0:
            return
        src_cpu = src_experts.to("cpu")
        hot_mask = (
            self.hot_pos[src_cpu] >= 0
        )  # which planned experts live in the pinned window
        # hot experts -> transfer_kv from the pinned window (fast path), remapped to host_hot rows
        if bool(hot_mask.any()):
            sel = hot_mask.to(dst_slots.device)
            hot_src_rows = self.hot_pos[src_cpu[hot_mask]].to(src_experts.device)
            hot_dst = dst_slots[sel]
            from sgl_kernel import transfer_kv_per_layer_mla

            for name, gpu_param in self.gpu.items():
                transfer_kv_per_layer_mla(
                    src=self.host_hot[name],
                    dst=gpu_param.data,
                    src_indices=hot_src_rows,
                    dst_indices=hot_dst,
                    item_size=self.item_bytes[name],
                )
        # cold experts -> staged copy from the pageable/disk tail, remapped to host_cold rows
        cold_mask = ~hot_mask
        if bool(cold_mask.any()):
            cold_ids = [int(e) for e in src_cpu[cold_mask].tolist()]
            # Disk cold tier: read ahead the cold group in parallel (MADV_WILLNEED) before the copies
            # below fault — unless the wave path already prefetched the WHOLE step's set upfront
            # (avoids re-issuing the same ranges once per wave).
            if not getattr(self, "_step_prefetched", False):
                self.prefetch_cold(cold_ids)
            cold_rows = self.cold_pos[src_cpu[cold_mask]].tolist()
            cold_dst = dst_slots[cold_mask.to(dst_slots.device)].tolist()
            n = len(cold_rows)
            # Gather into PINNED buffers, then direct async H2D per slot: the old
            # index_select -> pageable .to() -> index_copy_ chain crossed the bytes through a pageable
            # temp AND copied device->device again.
            for name, gpu_param in self.gpu.items():
                buf = _stage_pin_buf(
                    name, max(self.K, n), gpu_param.shape[1:], gpu_param.dtype
                )
                for i, r in enumerate(cold_rows):
                    buf[i].copy_(self.host_cold[name][r])
                for i, s in enumerate(cold_dst):
                    gpu_param.data[s].copy_(buf[i], non_blocking=True)
            # the shared pinned bufs must not be reused (next layer / next wave) while H2D is in flight
            torch.cuda.current_stream().synchronize()


def make_expert_store(
    layer,
    num_experts_E: int,
    num_resident_K: int,
    device,
    *,
    pin_host: bool,
    window_W: int = 0,
    cold_backing: str = "ram",
    cold_dir: Optional[str] = None,
) -> ExpertStore:
    """Build the host expert store. ``window_W > 0`` and ``< E`` (with ``pin_host``) selects the windowed
    fallback (pinned hot window + cold tail) for stores that exceed the page-lock ceiling; else pinned (fast
    ``transfer_kv``) or pageable (plain indexed copy). ``cold_backing='disk'`` mmaps the windowed cold tier
    to a file (page-cache-bounded) so the store may exceed RAM (P4)."""
    if pin_host and 0 < window_W < num_experts_E:
        return WindowedExpertStore(
            layer,
            num_experts_E,
            num_resident_K,
            device,
            window_W=window_W,
            cold_backing=cold_backing,
            cold_dir=cold_dir,
        )
    cls = PinnedExpertStore if pin_host else PageableExpertStore
    return cls(layer, num_experts_E, num_resident_K, device)
