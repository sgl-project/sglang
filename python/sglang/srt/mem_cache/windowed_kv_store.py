"""Windowed KV store for KV-streaming: ``WindowedKVStore`` keeps ONE request's live-decode KV as a per-layer
DEVICE W-slot FIFO ring (the recent W tokens) plus a per-layer mapped-UVA HOST tail (all older tokens). The
host tail lives in pinned host memory aliased as CUDA tensors, so the triton decode reads it directly over
PCIe — no per-step re-staging to HBM (proven bit-exact + capture-safe + link-rate).

The attention backend attends the ring (device) + the tail (host-UVA) in two passes and LSE-merges them,
reproducing full-context attention while the device holds only ~W tokens. FIFO by token position — no
SWA/radix/LRU/scheduler entanglement. Write paths: ``append`` (eager) / ``append_device`` (CUDA-graph
capturable) for decode, ``ingest_chunk`` for prefill.

Contrast with HiCache's host pool (``memory_pool_host.py``): that is ``cudaHostRegister``-pinned (copy-only —
a kernel cannot dereference it) and tiers ACROSS requests (prefix reuse). This tiers WITHIN one live sequence
and is UVA-mapped so attention reads it in place.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

# Reuse the fork's proven mapped-pinned UVA machinery (same as paged-experts + the KV probes):
from sglang.jit_kernel.paged_experts_decide import paged_experts_host_devptr
from sglang.srt.layers.moe.paged_experts.store import _pinned_empty


# CUDA-array-interface uint typestr per element width. We alias the pinned buffer as an unsigned int of the
# same byte width (types that lack a CAI typestr — bf16, fp8 — have none of their own), then ``.view(dtype)``
# reinterprets it. Covers every per-element KV dtype: fp8_e4m3/e5m2 (1B), fp16/bf16 (2B), fp32 (4B).
_UINT_TYPESTR = {1: "|u1", 2: "<u2", 4: "<u4"}


def _uva_buffer(shape, dtype: torch.dtype):
    """A CUDA tensor ALIASING mapped-UVA pinned host memory (kernel-readable over PCIe, host-resident).

    Aliases the pinned bytes as a same-width uint (via ``__cuda_array_interface__``) then ``.view(dtype)``, so
    any 1/2/4-byte KV element dtype works (fp8, fp16, bf16, fp32). Returns ``(device_view, host_tensor)``; keep
    the host tensor alive for the lifetime of the view (its pages back the UVA mapping)."""
    typestr = _UINT_TYPESTR.get(dtype.itemsize)
    if typestr is None:
        # Sub-byte-packed KV (e.g. nvfp4, 2 values/byte) can't be addressed as per-element rows here.
        raise NotImplementedError(
            f"windowed KV store supports 1/2/4-byte KV element dtypes (fp8/fp16/bf16/fp32); got {dtype} "
            f"(itemsize {dtype.itemsize}). For sub-byte-packed KV (nvfp4) run without SGLANG_KV_WINDOW."
        )
    host = _pinned_empty(tuple(shape), dtype)
    host.view(torch.uint8).zero_()  # dtype-agnostic zero-fill (fp8 has no reliable .zero_())
    cai = SimpleNamespace()
    cai.__cuda_array_interface__ = {
        "shape": tuple(shape),
        "typestr": typestr,  # element-size-only alias; reinterpreted below
        "data": (int(paged_experts_host_devptr(host)), False),
        "version": 3,
        "strides": None,
    }
    device_view = torch.as_tensor(cai, device="cuda").view(dtype)
    return device_view, host


class WindowedKVStore:
    """Self-contained windowed KV cache (KV-streaming): a per-layer DEVICE W-slot FIFO ring holding the
    recent W tokens + a per-layer mapped-UVA HOST tail holding the older tokens (positional: tail slot == token
    position, no allocator/map). Backend-owned; attention reads the ring (recent W) + the host tail (direct-UVA,
    ALL older tokens) and flash-merges them. FIFO by position — NO SWA/radix/LRU/scheduler entanglement. bs=1;
    decode writes via ``append`` (eager, python-int pos) or ``append_device`` (CUDA-graph capturable).

    append(layer, pos, k, v): write token `pos`'s KV into ring slot pos%W; if the ring is full, first evict the
    token being overwritten (pos-W) into host tail[pos-W]. Attention then reads ring[:min(pos+1,W)] (device) and
    tail[:max(0,pos-W+1)] (host-UVA), order-independent (softmax over keys)."""

    def __init__(self, W, max_ctx, layer_num, head_num, head_dim, dtype=torch.float16, device="cuda"):
        self.W = W
        self.max_ctx = max_ctx
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.rk = [torch.zeros(W, head_num, head_dim, dtype=dtype, device=device) for _ in range(layer_num)]
        self.rv = [torch.zeros(W, head_num, head_dim, dtype=dtype, device=device) for _ in range(layer_num)]
        self.tk, self.tv, self._host = [], [], []
        for _ in range(layer_num):
            kd, kh = _uva_buffer((max_ctx, head_num, head_dim), dtype)
            vd, vh = _uva_buffer((max_ctx, head_num, head_dim), dtype)
            self.tk.append(kd)
            self.tv.append(vd)
            self._host += [kh, vh]
        # Capture support: pre-built constant index buffers (kv_indices for the ring/tail decode passes —
        # the kernel reads the first `num` of them via a device kv_indptr) + a scratch tail slot for
        # evictions while pos < W (never attended). All device tensors so append/attend are graph-safe.
        self.dummy = max_ctx - 1
        self.ring_idx = torch.arange(W, dtype=torch.int32, device=device)
        self.tail_idx = torch.arange(max_ctx, dtype=torch.int32, device=device)
        # Prefill perf: a reused VRAM scratch to stage the host tail into before the extend tail pass.
        # extend_attention_fwd re-reads prefix keys per query block; over UVA those re-reads re-fetch from
        # host (uncached) → ~24× slower than VRAM (where re-reads hit L2). Staging (one contiguous UVA→VRAM
        # copy at link rate) then extending over VRAM collapses that penalty. The per-layer tail fits VRAM
        # (≤~56 MB at 32K), so no tiling. Decode (reads each key once) is already UVA-fast → no staging there.
        self.scratch_k = torch.empty(max_ctx, head_num, head_dim, dtype=dtype, device=device)
        self.scratch_v = torch.empty(max_ctx, head_num, head_dim, dtype=dtype, device=device)

    def append(self, layer_id: int, pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        slot = pos % self.W
        if pos >= self.W:  # ring full: slot currently holds token pos-W -> evict to host tail (device->UVA)
            self.tk[layer_id][pos - self.W] = self.rk[layer_id][slot]
            self.tv[layer_id][pos - self.W] = self.rv[layer_id][slot]
        self.rk[layer_id][slot] = k.reshape(self.head_num, self.head_dim)
        self.rv[layer_id][slot] = v.reshape(self.head_num, self.head_dim)

    def ingest_chunk(self, layer_id: int, chunk_start: int, ck: torch.Tensor, cv: torch.Tensor) -> None:
        """Incremental ingest of a prefill chunk [chunk_start, chunk_start+Ne) into the ring (recent W) +
        host tail (older) — reading ONLY the chunk's own KV (ck,cv: [Ne,H,D]) + the prior device ring,
        NEVER the prior context's pool slots. So it stays correct once the device pool rings those slots
        away (the prefill pool ring). Vectorized; handles Ne<W and Ne>=W. Call AFTER the chunk's attention (which reads
        the prior [0,chunk_start) window+tail)."""
        Ne = ck.shape[0]
        W = self.W
        E = chunk_start + Ne
        dev = self.device
        rk, rv, tk, tv = self.rk[layer_id], self.rv[layer_id], self.tk[layer_id], self.tv[layer_id]
        ck = ck.reshape(Ne, self.head_num, self.head_dim)
        cv = cv.reshape(Ne, self.head_num, self.head_dim)
        # (1) prior-ring positions leaving the window -> tail: [max(0,chunk_start-W), min(chunk_start, E-W))
        lo_e, hi_e = max(0, chunk_start - W), min(chunk_start, E - W)
        if hi_e > lo_e:
            ep = torch.arange(lo_e, hi_e, device=dev)
            tk[ep] = rk[ep % W]
            tv[ep] = rv[ep % W]
        # (2) chunk positions skipping the window straight to tail (Ne>W): [chunk_start, E-W)
        if E - W > chunk_start:
            tp = torch.arange(chunk_start, E - W, device=dev)
            tk[tp] = ck[tp - chunk_start]
            tv[tp] = cv[tp - chunk_start]
        # (3) chunk positions entering the ring: [max(chunk_start, E-W), E)
        rp = torch.arange(max(chunk_start, E - W), E, device=dev)
        rk[rp % W] = ck[rp - chunk_start]
        rv[rp % W] = cv[rp - chunk_start]

    def append_device(self, layer_id: int, pos_dev: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Capturable append+evict: pos_dev is a [1] int64 DEVICE tensor (no python-int branching). Evicts
        ring[pos%W] (which holds token pos-W) to tail[pos-W], or to the scratch `dummy` slot when pos<W
        (torch.where — no host branch), then writes ring[pos%W]=new. All device-index scatters (index_copy_),
        so this replays correctly under a CUDA graph (slot varies with the device pos on each replay)."""
        slot = (pos_dev % self.W).reshape(1)
        evict_idx = torch.where(
            pos_dev >= self.W, pos_dev - self.W, pos_dev.new_full((1,), self.dummy)
        ).reshape(1)
        rk, rv, tk, tv = self.rk[layer_id], self.rv[layer_id], self.tk[layer_id], self.tv[layer_id]
        # evict OLD ring[slot] (device-index gather makes a copy) BEFORE overwriting; dummy slot when pos<W
        tk[evict_idx] = rk[slot]
        tv[evict_idx] = rv[slot]
        rk[slot] = k.reshape(1, self.head_num, self.head_dim)
        rv[slot] = v.reshape(1, self.head_num, self.head_dim)
