# SPDX-License-Identifier: Apache-2.0
"""Streaming prefill for KT MoE: stream ALL routed experts per layer DDR->HBM.

Env-gated bypass of the hybrid (32-resident + CPU) MoE path. During a long prefill
(``KT_PREFILL_STREAM=1`` and chunk token count ``M >= KT_PREFILL_STREAM_THRESHOLD``),
each layer streams its full 256-expert int8 weights from a pinned DDR pool into a single
reused HBM slot, then runs the production NPU operator ``npu_fused_experts`` over all 256
experts (no CPU, no submit/sync). See streaming_prefill_integration_design.md (2c-ii).

Design notes:
- Serial single slot (2b: double-buffer gives <=2% and costs more) -> simplest, least HBM.
- Pool built lazily on first qualifying forward; chunked ND->NZ cast bounds peak HBM.
- Pure bypass: never touches submit/sync/gpu_method -> orthogonal to the hybrid path.
- Any failure falls back to hybrid (returns None); env off -> zero behavior change.
"""
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

_KT_PREFILL_STREAM = os.environ.get("KT_PREFILL_STREAM", "") == "1"
_T = int(os.environ.get("KT_PREFILL_STREAM_THRESHOLD", "512"))
# KT_HOT_TAIL_TOKENS=N (opt-in; default 0 = off = whole-prompt selection): pick the decode resident
# hot-pool from only the LAST N prompt tokens' routing instead of the whole prefill. Rationale:
# decode continues from the prompt tail, so the tail's expert distribution predicts decode routing
# better -> higher decode hit-rate. Validated on natural + synthetic prompts (hit-rate measured in
# eager mode, throughput with graph on): smaller window = better; N=64 optimal -> LONG
# context (8k-32k) decode hit-rate +10-13pp and decode tps +3-8% vs N=0 (tps tracks hit-rate; damped
# because CPU-MoE is partly hidden under the NPU side stream). BUT short prompts (<~2k tok) slightly
# REGRESS (-2-4%: a short prompt is already all "recent", a 64-tok sub-window just adds noise) -> so
# default OFF; enable with =64 for long-context-heavy deployments. Pure perf knob: only changes which
# experts are NPU-resident vs CPU, NOT the computed output (bit-exact).
_HOT_TAIL = int(os.environ.get("KT_HOT_TAIL_TOKENS", "0") or "0")
_CKPT = os.environ.get("KT_PREFILL_STREAM_CKPT", "")
_NZ_CHUNK = int(os.environ.get("KT_PREFILL_STREAM_NZ_CHUNK", "64"))  # experts/chunk for ND->NZ
_ACL_FORMAT_FRACTAL_NZ = 29

# DEPOOL (KT_MXFP4_DEPOOL=1): instead of a 277GB resident W8A8 NZ pool, store the MXFP4 codes+scale
# (~137GB, 4-bit) and convert MXFP4 -> W8A8-NZ on the fly per layer with a fused AscendC kernel
# (~82ms/layer, hidden under H2D). Removes the pin tax (no W8A8 pool to pin). Fully gated: when off,
# the W8A8 path below is byte-identical to before. MXFP4 weights are the original safetensors
# (.weight = codes, .scale = e8m0), NOT the W8A8 checkpoint.
_KT_MXFP4_DEPOOL = os.environ.get("KT_MXFP4_DEPOOL", "") == "1"
# KT_MXFP4_POOL_NO_PIN=1: store the MXFP4 pool in pageable (unpinned) host memory. Pinning the
# ~140GB pool inflates the decode CPU-MoE wall (pin tax, ~2x off_cpu); unpinning removes it at the
# cost of slower streaming-prefill H2D (no async DMA). Default pinned (fast prefill).
_PIN_MXFP4 = os.environ.get("KT_MXFP4_POOL_NO_PIN", "") != "1"
_MXFP4_CKPT = os.environ.get("KT_MXFP4_CKPT", "")


def _require_ckpt_dir(path: str, env_name: str, what: str) -> str:
    """Return the checkpoint dir, or raise with guidance when the env var is unset.

    No environment-specific path is hardcoded: the launch script derives both vars from
    MODEL_PATH, so this only triggers when the feature is used outside that script.
    """
    if not path:
        raise ValueError(
            f"{env_name} is not set, but it is required to load the {what} checkpoint. "
            f"Set {env_name}=/path/to/checkpoint (the launch script derives it from MODEL_PATH)."
        )
    return path


def _ckpt_dir() -> str:
    return _require_ckpt_dir(_CKPT, "KT_PREFILL_STREAM_CKPT", "W8A8 (NPU-side)")


def _mxfp4_ckpt_dir() -> str:
    return _require_ckpt_dir(_MXFP4_CKPT, "KT_MXFP4_CKPT", "native MXFP4 (CPU-side)")
_MXFP4_POOL: dict = {}      # layer_idx -> (c13, s13, c2, s2) pinned host MXFP4 (codes+e8m0)
_MXFP4_POOL_BUILT = False   # set once the pool is fully populated (load-time parallel or lazy serial)
_MXSTAGE: dict = {}         # shape -> reused pinned [K,...] staging buf for the dyn-resident switch
_MXIDX = None               # cached weight_map of the MXFP4 checkpoint index

# GGUF DEDUP (KT_MXFP4_GGUF_DEDUP=1, requires KT_MXFP4_DEPOOL=1): the CPU MoE already mmaps the
# per-layer MXFP4 GGUF (KT_GGUF_TEMPLATE, block_mxfp4 = e8m0 + half-block-packed codes). Instead of
# ALSO keeping a separate ~137GB pinned codes pool (native safetensors, consecutive packing), read
# the layer's codes straight from that same GGUF mmap on the fly: the page cache is shared with
# kt-kernel, so the pinned pool's ~137GB DDR is recovered. Per streaming forward we H2D the raw
# GGUF blocks into a reused pinned staging buffer, de-interleave (scale|codes) on device, and feed
# the fused kernel with packing="halfblock". The kernel + scale->block mapping are packing-agnostic;
# only the convert post-step differs (validated bit-identical to the consecutive pool, E=256).
_KT_GGUF_DEDUP = os.environ.get("KT_MXFP4_GGUF_DEDUP", "") == "1"
_GGUF_TMPL = os.environ.get("KT_GGUF_TEMPLATE", "")
_GGUF_READERS: dict = {}    # layer_idx -> GGUFReader (memmap; page cache shared with CPU MoE)
_GGUF_BLOCKS: dict = {}     # layer_idx -> (gate, up, down) np memmap views [E,N,nb*17] block_mxfp4



def _repo_root():
    """Walk .../python/sglang/srt/layers/moe/<file>.py up 8 levels to the repo root."""
    root = os.path.abspath(__file__)
    for _ in range(8):
        root = os.path.dirname(root)
    return root


def _add_sys_path(d):
    import sys
    if d not in sys.path:
        sys.path.insert(0, d)


def _gguf_py_on_path():
    _add_sys_path(os.path.join(_repo_root(), "third_party", "llama.cpp", "gguf-py"))


def _gguf_layer_blocks(layer: int):
    """Return (gate, up, down) block_mxfp4 memmap views [E,N,nb*17] for one layer from the CPU
    MoE's GGUF (KT_GGUF_TEMPLATE). Lazily opens+caches one GGUFReader per layer; t.data is a
    file-backed memmap whose pages are shared with kt-kernel's mmap (no extra resident DDR)."""
    blk = _GGUF_BLOCKS.get(layer)
    if blk is None:
        _gguf_py_on_path()
        from gguf import GGUFReader
        r = _GGUF_READERS.get(layer)
        if r is None:
            r = GGUFReader(_GGUF_TMPL.format(layer_idx=layer))
            _GGUF_READERS[layer] = r
        byname = {t.name: t for t in r.tensors}
        blk = tuple(byname[f"blk.{layer}.{n}.weight"].data
                    for n in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"))
        _GGUF_BLOCKS[layer] = blk
    return blk


# ----- prefetch (double-buffered) -----
# The per-layer CPU memcpy mmap->pinned (~0.3s) is device-independent, so a worker thread fills
# layer L+1's pinned staging while the main thread is blocked in layer L's convert syncs. Needs
# PING-PONG buffers (2 per key, alternating by layer parity): single-buffer would race the in-flight
# H2D. Correctness needs no device event: the main thread is serial and each convert syncs, so
# layer L-1's H2D (from the buffer the worker reuses for L+1) is finished before L starts.
_KT_PREFETCH = os.environ.get("KT_MXFP4_PREFETCH", "1") == "1"
_MX_PP: dict = {}          # key -> [buf0, buf1] pinned ping-pong staging
_PF = {"ex": None, "futs": {}, "next": None}   # worker, layer->future, expected next layer

def _pp_buf(key, parity, E, OUT, nb17):
    bufs = _MX_PP.get(key)
    if bufs is None:
        bufs = [None, None]
        _MX_PP[key] = bufs
    b = bufs[parity]
    if b is None or tuple(b.shape) != (E, OUT, nb17):
        b = torch.empty((E, OUT, nb17), dtype=torch.uint8, pin_memory=True)
        bufs[parity] = b
    return b


_COPY_POOL = None
_COPY_NTHREADS = int(os.environ.get("KT_MXFP4_COPY_THREADS", "32"))


def _par_copy(dst, src_np):
    """Copy src_np (GGUF memmap, WARM in page cache) -> dst (pinned), parallelised over the expert
    (first) dim. torch.copy_ runs SINGLE-THREADED in the server (the OMP pool is saturated by the
    KT_CPUINFER kt-cpuinfer threads / pinned to 1) -> ~1.6 GB/s on K920's slow single-core BW ->
    ~90s for the 147GB GGUF and the whole long-prefill bottleneck. Explicit threads each release the
    GIL inside copy_ -> ~15 GB/s (the cores are idle during a streaming prefill). KT_MXFP4_COPY_THREADS=0
    -> the old single-threaded copy."""
    global _COPY_POOL
    src = torch.from_numpy(src_np)
    E = src.shape[0]
    n = _COPY_NTHREADS
    if n <= 1 or E < n:
        dst.copy_(src)
        return
    if _COPY_POOL is None:
        import concurrent.futures
        _COPY_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=n)
    futs = [_COPY_POOL.submit(lambda lo, hi: dst[lo:hi].copy_(src[lo:hi]),
                              E * k // n, E * (k + 1) // n) for k in range(n)]
    for f in futs:
        f.result()


def _fill_stage(layer):
    """Worker/main: CPU memcpy this layer's GGUF blocks into its parity's pinned ping-pong buffers
    (w13 = cat(gate,up) along OUT, w2 = down), parallelised (_par_copy). H2D + de-interleave stay
    on the main thread / in the kernel."""
    gate, up, down = _gguf_layer_blocks(layer)
    par = layer % 2
    E = gate.shape[0]
    b13 = _pp_buf("w13", par, E, gate.shape[1] + up.shape[1], gate.shape[2])
    _par_copy(b13[:, :gate.shape[1]], gate)
    _par_copy(b13[:, gate.shape[1]:], up)
    b2 = _pp_buf("w2", par, E, down.shape[1], down.shape[2])
    _par_copy(b2, down)


def _prefetch_ensure(layer, num_layers):
    """Make sure layer's buffers are filled (wait for its prefetch, or fill synchronously on a new
    prefill / out-of-sequence layer), then kick off layer+1's fill on the worker. Returns layer%2."""
    import concurrent.futures
    if _PF["ex"] is None:
        _PF["ex"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    if _PF["next"] == layer and layer in _PF["futs"]:
        _PF["futs"].pop(layer).result()        # warm: worker already filling/filled it
    else:
        _PF["futs"].clear()                     # new prefill / resync -> fill this layer now
        _fill_stage(layer)
    nxt = layer + 1
    if nxt < num_layers:
        _PF["futs"][nxt] = _PF["ex"].submit(_fill_stage, nxt)
    _PF["next"] = nxt
    return layer % 2


def _stage_pin_h2d(src, idx_cpu, dev):
    """Gather src[idx_cpu] (K hot experts) into a REUSED pinned buffer, then DMA to dev.
    Plain `src[idx]` (advanced indexing) returns an UNpinned tensor -> the following H2D loses
    DMA (~246ms/layer measured). index_select into a pinned out buffer keeps the copy on the DMA
    path (~20ms/layer). Works whether or not the pool itself is pinned."""
    K = int(idx_cpu.numel())
    shp = (K,) + tuple(src.shape[1:])
    buf = _MXSTAGE.get(shp)
    if buf is None:
        buf = torch.empty(shp, dtype=src.dtype, pin_memory=True)
        _MXSTAGE[shp] = buf
    torch.index_select(src, 0, idx_cpu, out=buf)
    return buf.to(dev, non_blocking=True)


def _mxfp4_op_dir_on_path():
    d = os.environ.get("KT_MXFP4_OP_DIR") or os.path.join(_repo_root(), "tools", "ascendc_mxfp4")
    _add_sys_path(d)


def _mxfp4_convert_fn():
    """Lazily import the fused-kernel wrapper (tools/ascendc_mxfp4/mxfp4_fused_op.py)."""
    _mxfp4_op_dir_on_path()
    from mxfp4_fused_op import mxfp4_layer_to_nz_slots
    return mxfp4_layer_to_nz_slots


def _mxfp4_convert_blk_fn():
    """Wrapper that converts straight from RAW GGUF blocks (in-kernel de-interleave; no slow
    host/device 16-of-17 strided de-interleave). KT_MXFP4_BLK_KERNEL=1 (default)."""
    _mxfp4_op_dir_on_path()
    from mxfp4_fused_op import mxfp4_layer_to_nz_slots_blk
    return mxfp4_layer_to_nz_slots_blk


_KT_BLK_KERNEL = os.environ.get("KT_MXFP4_BLK_KERNEL", "1") == "1"

# Module-level singletons (shared across all layers / wrapper instances).
_POOL: dict = {}            # layer_idx -> (w13_host_nz, w2_host_nz, s13_bf16_npu, s2_bf16_npu)
_SLOT: dict = {}           # 'w13'/'w2' -> reused NZ HBM slot
_POOL_BUILT = False
_SLOT_RESERVED = False

# Goal 2 (sub-goal 2): dynamic decode-resident expert pool. During a streaming prefill we
# count per-layer expert activations (device-side bincount, cheap); at the last layer the
# per-layer top-K (K = current resident slots, 32) replaces the static-prefix resident set:
# weights are gathered from the DDR NZ pool into pinned staging and copied whole-tensor into
# layer.w13_weight/w2_weight (per-slot NZ slice copy is byte-WRONG — validated; staging path
# is bitwise == fresh cast), scales via device indexing, and all three routing structures are
# updated IN PLACE (same storage, so decode graph replay and the C++ side see them):
#   1. KTEP wrapper.gpu_experts_mask / logical_to_gpu_index (device tensors)
#   2. kt_kernel wrapper.gpu_experts_mask (pinned CPU bool, shared with C++ by pointer)
_KT_DYN_RESIDENT = os.environ.get("KT_DYNAMIC_RESIDENT", "") == "1"
_REQ_HIST: dict = {}    # layer_idx -> int64 device tensor [E] (current prefill pass)
_REGISTRY: dict = {}    # layer_idx -> (layer_module, ktep_wrapper)

# 2c-ii-c3 incremental build: capture MATERIALIZES each expert's int8 weight straight into
# that layer's FINAL pinned ND buffer as the load loop reads it (no 277GB ref stage — refs
# were lazy mmap views whose real read only happened at assemble time, under cold cache).
# When a layer's 1536 tensors (256 experts x 3 weights x 2) are all in, the layer is NZ-cast
# IN PLACE (chunked: pinned ND -> HBM -> transpose+format_cast -> bytes back into the SAME
# pinned region; ND[E,2I,H] and NZ[E,H,2I] have identical byte counts; chunked==full cast
# was proven bitwise-equal in tools/longseq_dbg/test_stream_module.py). The whole build is
# spread inside the model-load loop; peak extra DDR = 0 (the pinned pool IS the product).
_CFG: dict = {}     # E,H,I,num_layers from checkpoint config.json (lazy)
_LBUF: dict = {}    # layer -> {flat13,flat2(pinned int8),s13,s2(cpu fp32),count}


def _get_cfg():
    if not _CFG:
        cfg = json.load(open(os.path.join(_ckpt_dir(), "config.json")))
        _CFG["E"] = int(cfg["n_routed_experts"])
        _CFG["H"] = int(cfg["hidden_size"])
        _CFG["I"] = int(cfg["moe_intermediate_size"])
        _CFG["num_layers"] = int(cfg["num_hidden_layers"])
    return _CFG["E"], _CFG["H"], _CFG["I"], _CFG["num_layers"]


def _layer_buf(L: int, E: int, H: int, I: int) -> dict:
    b = _LBUF.get(L)
    if b is None:
        b = {
            "flat13": torch.empty(E * 2 * I * H, dtype=torch.int8, pin_memory=True),
            "flat2": torch.empty(E * H * I, dtype=torch.int8, pin_memory=True),
            "s13": torch.empty(E, 2 * I, 1, dtype=torch.float32),
            "s2": torch.empty(E, H, 1, dtype=torch.float32),
            "count": 0,
        }
        _LBUF[L] = b
    return b


# ----- parallel O_DIRECT pool reader (2c-ii-c5; roofline §2c-ii-c4) -----
# The build bottleneck is reading 277GB of expert int8 from the W8A8 checkpoint. The loader's
# buffered single-thread reads cap at ~0.7 GB/s; parallel O_DIRECT (bypassing page cache) +
# per-expert rearrange reaches ~1.2-1.5 GB/s across threads. Raw O_DIRECT is 3.5 GB/s (NVMe
# cap) but the Python per-expert rearrange (experts are expert-major but scattered on disk)
# is the hard cap -> NVMe speed needs a C++ reader.
_NVME_ALIGN = 4096
_IDX = None


def _index():
    global _IDX
    if _IDX is None:
        _IDX = json.load(open(os.path.join(_ckpt_dir(), "model.safetensors.index.json")))["weight_map"]
    return _IDX


def _shard_header(path):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _read_layer_odirect(L, E, H, I, scratch) -> None:
    """Fill _LBUF[L] (pinned flat13/flat2 + scales) for layer L via O_DIRECT large reads +
    per-expert rearrange. scratch: a page-aligned mmap >= one shard's expert region (reused)."""
    from safetensors import safe_open

    idx = _index()
    b = _layer_buf(L, E, H, I)
    f13 = b["flat13"].view(E, 2 * I, H)
    f2 = b["flat2"].view(E, H, I)
    byfile = {}
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            byfile.setdefault(idx[f"layers.{L}.ffn.experts.{e}.{w}.weight"], []).append((e, w))
    for fn, items in byfile.items():
        path = os.path.join(_ckpt_dir(), fn)
        hdr, base = _shard_header(path)
        offs = {(e, w): hdr[f"layers.{L}.ffn.experts.{e}.{w}.weight"]["data_offsets"]
                for e, w in items}
        lo = min(o[0] for o in offs.values())
        hi = max(o[1] for o in offs.values())
        a_lo = ((base + lo) // _NVME_ALIGN) * _NVME_ALIGN
        skip = (base + lo) - a_lo
        need = (base + hi) - a_lo
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            dv = memoryview(scratch)
            got = 0
            while got < need:
                n = os.preadv(fd, [dv[got:]], a_lo + got)
                if n <= 0:
                    break
                got += n
        finally:
            os.close(fd)
        region = memoryview(scratch)[skip : skip + (hi - lo)]
        for e, w in items:
            o0, o1 = offs[(e, w)]
            blk = torch.frombuffer(region[o0 - lo : o1 - lo], dtype=torch.int8)
            if w == "w1":
                f13[e, 0:I].copy_(blk.view(I, H))
            elif w == "w3":
                f13[e, I : 2 * I].copy_(blk.view(I, H))
            else:
                f2[e].copy_(blk.view(H, I))
    # scales (tiny) via normal get_tensor
    sfiles = {}
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            sfiles.setdefault(idx[f"layers.{L}.ffn.experts.{e}.{w}.weight_scale"], []).append((e, w))
    for fn, items in sfiles.items():
        with safe_open(os.path.join(_ckpt_dir(), fn), framework="pt") as f:
            for e, w in items:
                t = f.get_tensor(f"layers.{L}.ffn.experts.{e}.{w}.weight_scale")
                if w == "w1":
                    b["s13"][e, 0:I] = t.reshape(I, 1)
                elif w == "w3":
                    b["s13"][e, I : 2 * I] = t.reshape(I, 1)
                else:
                    b["s2"][e] = t.reshape(H, 1)


_BG = {"ex": None, "done_q": None, "t_start": 0.0, "started": False}


def _start_bg_reads(E, H, I, num_layers, nworkers=8) -> None:
    """Start O_DIRECT read workers in the BACKGROUND (host-only, no HBM) so they overlap the rest
    of model load (GGUF loads, construction). Reads contend with the load on NVMe but use its
    NPU/CPU phases freely. NZ-cast happens later in _finish_bg_build (needs HBM scratch)."""
    import time, threading, queue, mmap
    from concurrent.futures import ThreadPoolExecutor

    _BG["done_q"] = queue.Queue()
    _BG["t_start"] = time.perf_counter()
    _BG["ex"] = ThreadPoolExecutor(max_workers=nworkers)
    _tls = threading.local()

    def rd(L):
        if not hasattr(_tls, "scratch"):
            _tls.scratch = mmap.mmap(-1, 8 * 1024 ** 3)  # page-aligned, per-thread, reused
        _read_layer_odirect(L, E, H, I, _tls.scratch)
        _BG["done_q"].put(L)

    for L in range(num_layers):
        _BG["ex"].submit(rd, L)
    logger.info("[KT_STREAM] background reads started (%d layers, %d workers) — overlapping load",
                num_layers, nworkers)


def _finish_bg_build(num_layers, dev) -> None:
    """Drain the background reads, NZ-casting each layer as its read completes. By now most reads
    have finished (overlapped with load), so this is mostly the NZ pass (~80s)."""
    import time

    _free_slot()  # NZ-cast HBM scratch
    t0 = time.perf_counter()
    for _ in range(num_layers):
        L = _BG["done_q"].get()
        _finalize_layer(L, dev)
    _BG["ex"].shutdown()
    print(f"[KT_STREAM] bg build done: total {time.perf_counter()-_BG['t_start']:.0f}s "
          f"(NZ-drain {time.perf_counter()-t0:.0f}s, {num_layers} layers)", flush=True)


def _build_pool_parread(E, H, I, num_layers, dev, nworkers=8) -> None:
    """Non-overlapped path (lazy fallback): start reads then immediately drain+NZ."""
    _start_bg_reads(E, H, I, num_layers, nworkers)
    _finish_bg_build(num_layers, dev)


def _inplace_nz(flat: torch.Tensor, E: int, A: int, B: int, dev) -> torch.Tensor:
    """Chunked in-place ND[E,A,B] -> FRACTAL_NZ[E,B,A] over the same pinned bytes."""
    import torch_npu

    nd = flat.view(E, A, B)
    nz_host = flat.view(E, B, A)
    for c in range(0, E, _NZ_CHUNK):
        sub = nd[c : c + _NZ_CHUNK].to(dev).transpose(1, 2).contiguous()
        nz = torch_npu.npu_format_cast(sub, _ACL_FORMAT_FRACTAL_NZ)
        nz_host[c : c + _NZ_CHUNK].copy_(nz)
        del sub, nz
    torch.npu.empty_cache()
    return nz_host


def _finalize_layer(L: int, dev) -> None:
    """NZ-cast a completed layer in place and publish it to the pool."""
    global _POOL_BUILT
    import time

    E, H, I, num_layers = _get_cfg()
    b = _LBUF[L]
    t0 = time.perf_counter()
    h13 = _inplace_nz(b["flat13"], E, 2 * I, H, dev)   # -> [E, H, 2I] NZ view
    h2 = _inplace_nz(b["flat2"], E, H, I, dev)         # -> [E, I, H] NZ view
    s13b = b["s13"].squeeze(-1).to(torch.bfloat16).to(dev)
    s2b = b["s2"].squeeze(-1).to(torch.bfloat16).to(dev)
    _POOL[L] = (h13, h2, s13b, s2b)
    b["s13"] = b["s2"] = None
    if len(_POOL) == num_layers:
        _POOL_BUILT = True
    print(f"[KT_STREAM] layer {L} NZ-finalized in-loop ({time.perf_counter()-t0:.1f}s, "
          f"{len(_POOL)}/{num_layers} done)", flush=True)


def _is_prefill() -> bool:
    try:
        return not torch.npu.is_current_stream_capturing()
    except Exception:
        return True


# ---------------------------------------------------------------------------
#  DEPOOL: load MXFP4 codes+scale (instead of W8A8) and build a small pinned pool
# ---------------------------------------------------------------------------
def _as_u8(t):
    return (t if t.dtype == torch.uint8 else t.view(torch.uint8)).contiguous()


def _load_layer_mxfp4(layer: int, E: int, H: int, I: int):
    """Read one layer's E experts of native MXFP4 (codes + e8m0 scale) from the MXFP4 checkpoint and build
    combined w13 = cat(w1,w3) along OUT. Returns pinned host tensors:
      c13 [E,2I,H/2] u8, s13 [E,2I,H/32] u8, c2 [E,H,I/2] u8, s2 [E,H,I/32] u8."""
    from safetensors import safe_open

    idx = json.load(open(os.path.join(_mxfp4_ckpt_dir(), "model.safetensors.index.json")))["weight_map"]
    cache: dict = {}

    def _open(k):
        sh = idx[k]
        if sh not in cache:
            cache[sh] = safe_open(os.path.join(_mxfp4_ckpt_dir(), sh), framework="pt")
        return cache[sh]

    def stack(proj):
        cs, ss = [], []
        for e in range(E):
            wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
            sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
            h = _open(wk)
            cs.append(_as_u8(h.get_tensor(wk)))
            ss.append(_as_u8(h.get_tensor(sk)))
        return torch.stack(cs), torch.stack(ss)

    _pin = (lambda t: t.pin_memory()) if _PIN_MXFP4 else (lambda t: t)
    c1, s1 = stack("w1")
    c3, s3 = stack("w3")
    c13 = _pin(torch.cat([c1, c3], dim=1))
    s13 = _pin(torch.cat([s1, s3], dim=1))
    c2, s2 = stack("w2")
    return c13, s13, _pin(c2), _pin(s2)


def _build_mxfp4_pool(E: int, H: int, I: int, num_layers: int) -> None:
    """Serial fallback: fill _MXFP4_POOL with pinned MXFP4 codes+scale per layer (~140GB, 4-bit).
    The fast path is the load-time parallel O_DIRECT build (_start_bg_reads_mxfp4); this serial
    safe_open reader (~8s/layer) only runs if that failed or was never started (env/timing)."""
    global _MXFP4_POOL_BUILT
    import time
    if _MXFP4_POOL_BUILT:
        return
    _MXFP4_POOL.clear()  # drop any partial buffers from a failed parallel build
    t0 = time.perf_counter()
    print(f"[KT_STREAM][depool] building MXFP4 pool (serial): {num_layers} layers from {_mxfp4_ckpt_dir()}",
          flush=True)
    for L in range(num_layers):
        _MXFP4_POOL[L] = _load_layer_mxfp4(L, E, H, I)
        print(f"[KT_STREAM][depool] mxfp4 layer {L + 1}/{num_layers} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    _MXFP4_POOL_BUILT = True
    print(f"[KT_STREAM][depool] MXFP4 pool built in {time.perf_counter() - t0:.0f}s", flush=True)


# ----- load-time parallel O_DIRECT MXFP4 pool reader (mirrors the W8A8 _start_bg_reads pattern) -----
# The depool pool is just pinned host MXFP4 codes+scale (no NZ-cast — the bytes ARE the product), so
# building it is purely a read problem. The serial safe_open reader above is ~8s/layer x 43 = ~347s,
# all charged to the first long request. Instead read all layers in parallel with O_DIRECT large reads
# (bypassing the page cache, ~5x the buffered single-thread rate) started at model-load time so they
# overlap the rest of the load. MXFP4 is 4-bit (~3.4GB/layer vs W8A8's ~12GB), so reads are cheap.
_BG_MX = {"ex": None, "done_q": None, "t_start": 0.0, "started": False}


def _mxfp4_index():
    global _MXIDX
    if _MXIDX is None:
        _MXIDX = json.load(
            open(os.path.join(_mxfp4_ckpt_dir(), "model.safetensors.index.json")))["weight_map"]
    return _MXIDX


def _odirect_region(path, base, lo, hi, scratch):
    """O_DIRECT read file bytes [base+lo, base+hi) into the page-aligned scratch mmap; return a
    memoryview of exactly the [lo,hi) payload. Same read loop as the W8A8 _read_layer_odirect."""
    a_lo = ((base + lo) // _NVME_ALIGN) * _NVME_ALIGN
    skip = (base + lo) - a_lo
    need = (base + hi) - a_lo
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        dv = memoryview(scratch)
        got = 0
        while got < need:
            n = os.preadv(fd, [dv[got:]], a_lo + got)
            if n <= 0:
                break
            got += n
    finally:
        os.close(fd)
    return memoryview(scratch)[skip : skip + (hi - lo)]


def _mxfp4_layer_buf(L, E, H, I):
    """Get (or allocate) layer L's pinned destination tensors. The pinned buffer IS the pool
    product (no NZ round-trip), so we fill it in place: c13/s13 = cat(w1,w3) along OUT, c2/s2 = w2."""
    b = _MXFP4_POOL.get(L)
    if b is None:
        pin = _PIN_MXFP4
        b = (
            torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, pin_memory=pin),   # c13 codes
            torch.empty(E, 2 * I, H // 32, dtype=torch.uint8, pin_memory=pin),  # s13 e8m0
            torch.empty(E, H, I // 2, dtype=torch.uint8, pin_memory=pin),       # c2  codes
            torch.empty(E, H, I // 32, dtype=torch.uint8, pin_memory=pin),      # s2  e8m0
        )
        _MXFP4_POOL[L] = b
    return b


def _read_layer_mxfp4_odirect(L, E, H, I, scratch) -> None:
    """Fill _MXFP4_POOL[L] (pinned c13/s13/c2/s2) for layer L via O_DIRECT reads + per-expert
    rearrange. Codes (.weight) and scales (.scale) sit in two separate contiguous on-disk blocks,
    so each is one tight region read per shard file. byte-equivalent to _load_layer_mxfp4."""
    idx = _mxfp4_index()
    c13, s13, c2, s2 = _mxfp4_layer_buf(L, E, H, I)
    for suf, (dst13, dst2, w13, w2_n) in (
        ("weight", (c13, c2, H // 2, I // 2)),
        ("scale", (s13, s2, H // 32, I // 32)),
    ):
        byfile = {}
        for e in range(E):
            for proj in ("w1", "w2", "w3"):
                k = f"layers.{L}.ffn.experts.{e}.{proj}.{suf}"
                byfile.setdefault(idx[k], []).append((e, proj))
        for fn, items in byfile.items():
            path = os.path.join(_mxfp4_ckpt_dir(), fn)
            hdr, base = _shard_header(path)
            offs = {(e, proj): hdr[f"layers.{L}.ffn.experts.{e}.{proj}.{suf}"]["data_offsets"]
                    for e, proj in items}
            lo = min(o[0] for o in offs.values())
            hi = max(o[1] for o in offs.values())
            region = _odirect_region(path, base, lo, hi, scratch)
            for e, proj in items:
                o0, o1 = offs[(e, proj)]
                blk = torch.frombuffer(region[o0 - lo : o1 - lo], dtype=torch.uint8)
                if proj == "w1":
                    dst13[e, 0:I].copy_(blk.view(I, w13))
                elif proj == "w3":
                    dst13[e, I : 2 * I].copy_(blk.view(I, w13))
                else:
                    dst2[e].copy_(blk.view(H, w2_n))


def _start_bg_reads_mxfp4(E, H, I, num_layers, nworkers=8) -> None:
    """Submit O_DIRECT MXFP4 read workers in the BACKGROUND (host-only, no HBM) so they overlap the
    rest of model load. Each worker fills a layer's pinned pool buffers in place; the build is done
    once all are drained (no NZ pass for depool)."""
    import time, threading, queue, mmap
    from concurrent.futures import ThreadPoolExecutor

    _BG_MX["done_q"] = queue.Queue()
    _BG_MX["t_start"] = time.perf_counter()
    _BG_MX["ex"] = ThreadPoolExecutor(max_workers=nworkers)
    _tls = threading.local()

    def rd(L):
        try:
            if not hasattr(_tls, "scratch"):
                _tls.scratch = mmap.mmap(-1, 4 * 1024 ** 3)  # >= one layer's ~3.4GB weight region
            _read_layer_mxfp4_odirect(L, E, H, I, _tls.scratch)
            _BG_MX["done_q"].put(L)
        except Exception as e:
            _BG_MX["done_q"].put(("ERR", L, repr(e)[:200]))

    for L in range(num_layers):
        _BG_MX["ex"].submit(rd, L)
    print(f"[KT_STREAM][depool] MXFP4 background reads started ({num_layers} layers, {nworkers} "
          f"workers) — overlapping load", flush=True)


def _finish_bg_build_mxfp4(num_layers) -> None:
    """Drain the background MXFP4 reads (no NZ pass). Raises if any layer read failed so the caller
    can fall back to the serial builder."""
    global _MXFP4_POOL_BUILT
    import time

    errs = []
    for _ in range(num_layers):
        item = _BG_MX["done_q"].get()
        if isinstance(item, tuple):
            errs.append(item)
    _BG_MX["ex"].shutdown()
    if errs:
        raise RuntimeError(f"MXFP4 parallel read failed on {len(errs)} layer(s): {errs[:3]}")
    _MXFP4_POOL_BUILT = True
    print(f"[KT_STREAM][depool] MXFP4 pool built in {time.perf_counter() - _BG_MX['t_start']:.0f}s "
          f"(parallel O_DIRECT, {num_layers} layers)", flush=True)


def reserve_slot(E: int, H: int, I: int, dev) -> None:
    """Allocate the reused NZ HBM streaming slot EARLY (during model load, before KV-pool
    sizing). _profile_available_bytes measures free HBM post-load, so the slot reserved here
    is automatically excluded from the KV pool -> no lazy-alloc contention/OOM. Idempotent."""
    global _SLOT_RESERVED
    if _SLOT_RESERVED:
        return
    import torch_npu

    s13 = torch_npu.npu_format_cast(
        torch.empty(E, H, 2 * I, dtype=torch.int8, device=dev), _ACL_FORMAT_FRACTAL_NZ)
    s2 = torch_npu.npu_format_cast(
        torch.empty(E, I, H, dtype=torch.int8, device=dev), _ACL_FORMAT_FRACTAL_NZ)
    _SLOT["w13"], _SLOT["w2"] = s13, s2
    _SLOT_RESERVED = True
    print(f"[KT_STREAM] reserved streaming slot {tuple(s13.shape)}+{tuple(s2.shape)} "
          f"({(s13.numel() + s2.numel()) / 1e9:.2f}GB) at model-load time", flush=True)


def reserve_slot_depool(E: int, H: int, I: int, dev) -> None:
    """Reserve the depool streaming convert-output slot as PLAIN ND torch.empty (NOT format_cast).
    The depool convert fills it via out_nz[c:ce].copy_(nz_chunk): an ND-tagged dest takes a raw
    byte copy of the NZ bytes (what the W8A8 op then consumes), whereas a slice-copy into an
    NZ-FORMATTED dest triggers a full-tensor de-format round-trip (a fresh ~4.3GB alloc -> OOM
    on the ~8GB serving headroom). Reserved at load so the KV pool is sized around it. Idempotent.
    Shapes match (E,)+nz.shape[1:]: w13 [E,H,2I], w2 [E,I,H]."""
    global _SLOT_RESERVED
    if _SLOT_RESERVED:
        return
    _SLOT["w13"] = torch.empty(E, H, 2 * I, dtype=torch.int8, device=dev)
    _SLOT["w2"] = torch.empty(E, I, H, dtype=torch.int8, device=dev)
    _SLOT_RESERVED = True
    print(f"[KT_STREAM][depool] reserved ND streaming slot {tuple(_SLOT['w13'].shape)}+"
          f"{tuple(_SLOT['w2'].shape)} "
          f"({(_SLOT['w13'].numel() + _SLOT['w2'].numel()) / 1e9:.2f}GB) at model-load time",
          flush=True)


def _ensure_slot(w13_shape, w2_shape, dev):
    import torch_npu

    if "w13" not in _SLOT:
        s13 = torch.empty(w13_shape, dtype=torch.int8, device=dev)
        s2 = torch.empty(w2_shape, dtype=torch.int8, device=dev)
        _SLOT["w13"] = torch_npu.npu_format_cast(s13, _ACL_FORMAT_FRACTAL_NZ)
        _SLOT["w2"] = torch_npu.npu_format_cast(s2, _ACL_FORMAT_FRACTAL_NZ)
    return _SLOT["w13"], _SLOT["w2"]


def _wrapper_dims(wrapper):
    E = int(getattr(wrapper, "global_num_experts", 0) or 0)
    H = int(getattr(wrapper, "hidden_size", 0) or 0)
    I = int(getattr(wrapper, "intermediate_size_per_partition", 0) or 0)
    num_layers = int(getattr(wrapper.kt_config, "num_layers", 0) or 0)
    return E, H, I, num_layers


def _free_slot() -> None:
    """Release the reserved slot so its HBM can serve as build scratch (the NZ cast must
    round-trip through HBM). Build runs before streaming, so the slot is idle then; it is
    re-allocated by _ensure_slot after the build. Net HBM the feature needs stays at 1 slot."""
    global _SLOT_RESERVED
    _SLOT.clear()
    _SLOT_RESERVED = False
    torch.npu.empty_cache()


def maybe_reserve_slot(wrapper, dev, layer=None) -> None:
    """Called from process_weights_after_loading (model-load time) when streaming is enabled.
    Reserves the 6.4GB streaming slot BEFORE KV-pool sizing so the KV pool auto-accounts for it
    (the pool itself is built lazily on the first long prefill — that mid-forward window has the
    HBM headroom; model-load time does NOT, peak load memory leaves ~nothing). The reserved slot
    doubles as the build's HBM scratch via _free_slot (see maybe_streaming_forward).
    Also registers (layer, wrapper) for the dynamic decode-resident pool update."""
    if not _KT_PREFILL_STREAM or getattr(wrapper, "tp_rank", 0) != 0:
        return
    try:
        if layer is not None:
            _REGISTRY[wrapper.kt_config.layer_idx] = (layer, wrapper)
            if _KT_DYN_RESIDENT:
                # REMAP the resident weight params off the model's weight-memory region (loaded
                # weights are flush/coherence-optimized as read-only) into caching-allocator memory.
                # Writing the weight region at runtime triggers a device coherence/flush that stalls
                # the NSA per-layer .item() syncs (~+100s). dummy-test proved writing an identical NZ
                # caching-allocator tensor is free; remapping makes the resident params such tensors.
                # Done at model-load time = BEFORE npu-graph capture, so the graph captures the new
                # storage (else decode would read stale weights from the old address).
                for _nm in ("w13_weight", "w2_weight",
                            "w13_weight_scale_bf16", "w2_weight_scale_bf16"):
                    _p = getattr(layer, _nm, None)
                    if _p is not None:
                        _p.data = _p.data.clone()
                # Same remap for the resident MASK buffers: _set_resident_masks rewrites
                # gpu_experts_mask / logical_to_gpu_index every prefill, and on the weight region that
                # write triggers the SAME NSA decode stall as the weight write (root-caused: skipping
                # the resident weight write still leaves decode slow -> it's the mask rewrite, not the
                # weight write; writing identical values is also slow -> the act of writing
                # the weight-region tensor, not its content). Clone -> caching-allocator so the
                # per-prefill mask rewrite is free (decode recovers ~12 -> ~18 tok/s).
                for _mnm in ("gpu_experts_mask", "logical_to_gpu_index"):
                    _mp = getattr(wrapper, _mnm, None)
                    if _mp is not None and getattr(_mp, "device", None) is not None \
                            and _mp.device.type == "npu":
                        setattr(wrapper, _mnm, _mp.clone())
        if _KT_MXFP4_DEPOOL and _KT_GGUF_DEDUP:
            # GGUF dedup: no codes pool to build (each layer is read from the CPU MoE's GGUF mmap on
            # the fly). Mark built so the lazy serial builder in maybe_streaming_forward is skipped.
            # Still reserve the streaming convert-output slot (the full 256-expert NZ produced per
            # layer): _profile_available_bytes measures free HBM POST-reserve, so the slot is excluded
            # from the KV pool -> the per-layer convert writes into it instead of competing for HBM
            # mid-forward (else the KV pool, sized to fill HBM, leaves no room -> OOM at full context).
            global _MXFP4_POOL_BUILT
            _MXFP4_POOL_BUILT = True
            if not _GGUF_TMPL:
                logger.warning("[KT_STREAM][dedup] KT_MXFP4_GGUF_DEDUP=1 but KT_GGUF_TEMPLATE empty")
            E, H, I, _nl = _wrapper_dims(wrapper)
            if E and H and I:
                reserve_slot_depool(E, H, I, dev)
            return
        if _KT_MXFP4_DEPOOL:
            # Depool builds no W8A8 pool. Reserve the streaming convert-output slot (same reason as
            # the dedup branch above) so the KV pool is sized around it. Build the small MXFP4 pool
            # via parallel O_DIRECT reads started on the first process_weights call (host-only,
            # overlapping load), drained on the last layer.
            E, H, I, num_layers = _wrapper_dims(wrapper)
            if E and H and I:
                reserve_slot_depool(E, H, I, dev)
            if E and H and I and num_layers and not _MXFP4_POOL_BUILT:
                if not _BG_MX["started"]:
                    _BG_MX["started"] = True
                    _start_bg_reads_mxfp4(E, H, I, num_layers)
                if wrapper.kt_config.layer_idx == num_layers - 1:
                    _finish_bg_build_mxfp4(num_layers)
            return
        E, H, I, num_layers = _wrapper_dims(wrapper)
        if E and H and I:
            reserve_slot(E, H, I, dev)
        # Overlap the pool build's O_DIRECT reads with the rest of model load: start the
        # background reads on the FIRST process_weights call (they run host-only while the GGUF
        # loads + construction continue), then drain+NZ on the LAST call. Hides ~100s of read
        # inside the ~150s load; only the NZ pass (~80s) remains after.
        if E and H and I and num_layers and not _POOL_BUILT:
            if not _BG["started"]:
                _BG["started"] = True
                _start_bg_reads(E, H, I, num_layers)
            if wrapper.kt_config.layer_idx == num_layers - 1:
                _finish_bg_build(num_layers, dev)
                reserve_slot(E, H, I, dev)
    except Exception as e:
        logger.warning("[KT_STREAM] reserve/build at load failed (%s); lazy fallback", repr(e)[:160])


def npu_fused_experts(
    hidden_states,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    top_k,
    **kwargs,
):
    # Vendored: the int8 fused-expert kernel this streaming path has always used. It previously
    # lived in hardware_backend/npu/quantization/fused_moe_method_npu.py, a module this tree does
    # not ship; the function is self-contained over torch.ops.npu and carried here verbatim.
    import torch

    w13_offset = kwargs.get("w13_offset", None)
    w2_offset = kwargs.get("w2_offset", None)
    use_wna16 = kwargs.get("use_wna16", False)

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    if not use_wna16:
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        scale_args13 = {
            "scale": [w13_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args13 = {
            "antiquant_scale": [w13_scale],
            "antiquant_offset": [w13_offset],
        }

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    if not use_wna16:
        hidden_states, pertoken_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            activate_left=True,
            quant_mode=1,
        )

        scale_args2 = {
            "scale": [w2_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        scale_args2 = {"antiquant_scale": [w2_scale], "antiquant_offset": [w2_offset]}
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def _streaming_forward(layer_idx, x, topk_output, top_k):
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    if _KT_MXFP4_DEPOOL:
        E, H, I, num_layers = _get_cfg()
        dev = x.device
        # Reserved streaming slot (maybe_reserve_slot): convert writes the 256-expert NZ straight into
        # it, reused across all layers, so no per-layer ~6.4GB output alloc competes with the KV pool.
        # None (reserve skipped/failed) -> convert allocates fresh (back-compat).
        slot13, slot2 = _SLOT.get("w13"), _SLOT.get("w2")
        if _KT_GGUF_DEDUP:
            # No pinned codes pool: read this layer's MXFP4 straight from the CPU MoE's GGUF mmap
            # (block_mxfp4). The prefetch worker memcpys the next layer's raw blocks into a pinned
            # ping-pong buffer while this layer converts (else fill inline). We H2D the RAW blocks and
            # let the AscendC kernel de-interleave (scale|codes) in UB via Gather (KT_MXFP4_BLK_KERNEL,
            # default) -- the software 16-of-17 strided de-interleave was the prefill bottleneck
            # (~65s; in-kernel gather is ~0.6s/prefill). Fallback (=0): device de-interleave + base
            # kernel.
            if _KT_PREFETCH:
                par = _prefetch_ensure(layer_idx, num_layers)
            else:
                _fill_stage(layer_idx)
                par = layer_idx % 2
            blk13 = _MX_PP["w13"][par].to(dev, non_blocking=True)
            blk2 = _MX_PP["w2"][par].to(dev, non_blocking=True)
            if _KT_BLK_KERNEL:
                w13, s13b, w2, s2b = _mxfp4_convert_blk_fn()(blk13, blk2, H, I,
                                                             out_w13=slot13, out_w2=slot2)
            else:
                def _di(d):
                    E_, OUT_, n17 = d.shape
                    nbq = n17 // 17
                    b = d.view(E_, OUT_, nbq, 17)
                    return (b[..., 1:17].reshape(E_, OUT_, nbq * 16).contiguous(),
                            b[..., 0].contiguous())
                c13d, s13d = _di(blk13)
                c2d, s2d = _di(blk2)
                w13, s13b, w2, s2b = _mxfp4_convert_fn()(c13d, s13d, c2d, s2d, H, I,
                                                         packing="halfblock", out_w13=slot13,
                                                         out_w2=slot2)
        else:
            # H2D this layer's MXFP4 (4-bit, ~half the W8A8 bytes) then convert -> W8A8-NZ on chip.
            c13, s13, c2, s2 = _MXFP4_POOL[layer_idx]
            c13d = c13.to(dev, non_blocking=True)
            s13d = s13.to(dev, non_blocking=True)
            c2d = c2.to(dev, non_blocking=True)
            s2d = s2.to(dev, non_blocking=True)
            w13, s13b, w2, s2b = _mxfp4_convert_fn()(c13d, s13d, c2d, s2d, H, I,
                                                     out_w13=slot13, out_w2=slot2)
    else:
        h13, h2, s13b, s2b = _POOL[layer_idx]
        slot13, slot2 = _ensure_slot(h13.shape, h2.shape, x.device)
        slot13.copy_(h13)  # H2D this layer's 256 experts (default stream, serial single slot)
        slot2.copy_(h2)
        w13, w2 = slot13, slot2
    out = npu_fused_experts(
        hidden_states=x,
        w13=w13,
        w13_scale=s13b,
        w2=w2,
        w2_scale=s2b,
        topk_weights=topk_output.topk_weights.to(x.dtype),
        topk_ids=topk_output.topk_ids.to(torch.int32),
        top_k=top_k,
    )

    # dynamic-resident: W8A8 path gathers from the resident W8A8 _POOL; depool path converts the
    # hot-K experts' MXFP4 -> resident W8A8-NZ on the fly (see _apply_dynamic_residency).
    if _KT_DYN_RESIDENT:
        E, _, _, num_layers = _get_cfg()
        if _KT_MXFP4_DEPOOL:
            # Reuse the just-converted 256 NZ (w13/w2/s13b/s2b) — gather hot-K into resident slots.
            # ~free per layer (no re-convert/format_cast), folded into prefill. Needs the resident
            # params remapped to caching-allocator memory at load (maybe_reserve_slot) — else the
            # write stalls NSA .item() syncs ~100s.
            try:
                _apply_resident_layer_depool(layer_idx, topk_output, w13, s13b, w2, s2b)
            except Exception as e:
                logger.warning("[KT_STREAM] inline resident L%d failed (%s); static kept",
                               layer_idx, repr(e)[:140])
        else:
            # W8A8 path: gather from the resident _POOL at end of prefill (post-pass, unchanged).
            if layer_idx == 0:
                _REQ_HIST.clear()
            _REQ_HIST[layer_idx] = torch.bincount(
                _hist_ids(topk_output.topk_ids).to(torch.int64), minlength=E
            )[:E]
            if layer_idx == num_layers - 1 and len(_REQ_HIST) == num_layers:
                try:
                    _apply_dynamic_residency()
                except Exception as e:
                    logger.warning("[KT_STREAM] dynamic residency failed (%s); static set kept",
                                   repr(e)[:160])

    return StandardCombineInput(hidden_states=out)


def _hist_ids(topk_ids):
    """Flattened routed expert ids used to build the resident-set histogram. When
    KT_HOT_TAIL_TOKENS>0, restrict to the last N prompt tokens (recency); else whole prefill.
    topk_ids is [M_tokens, top_k]; slicing rows keeps the last N tokens' routing."""
    if _HOT_TAIL > 0 and topk_ids.dim() == 2 and topk_ids.shape[0] > _HOT_TAIL:
        topk_ids = topk_ids[-_HOT_TAIL:]
    return topk_ids.reshape(-1)


def _pick_resident_top(counts, K):
    """Pick the K resident experts for a layer: top-K by activation, returned ascending int64.
    Shared by the inline (depool) and post-pass (W8A8) paths so both agree on selection."""
    return counts.topk(K).indices.sort().values


def _set_resident_masks(wrap, top_cpu, K, E):
    """Rewrite the routing structures (gpu_experts_mask, logical_to_gpu_index, C++ live mask) so
    the resident set is exactly top_cpu. In place, decode-graph and C++-side safe."""
    new_mask = torch.zeros(E, dtype=torch.bool)
    new_mask[top_cpu] = True
    l2g = torch.full((E,), -1, dtype=torch.int64)
    l2g[top_cpu] = torch.arange(K, dtype=torch.int64)
    wrap.gpu_experts_mask.copy_(new_mask.to(wrap.gpu_experts_mask.device))
    wrap.logical_to_gpu_index.copy_(
        l2g.to(device=wrap.logical_to_gpu_index.device, dtype=wrap.logical_to_gpu_index.dtype))
    if wrap.wrapper is not None:  # pinned CPU mask, C++ reads it live
        wrap.wrapper.gpu_experts_mask.copy_(new_mask)


_RES_PEND = {}  # L -> (wrap, top_device, counts_device): deferred mask updates


def _apply_resident_layer_depool(L, topk_output, w13, s13b, w2, s2b):
    """Depool: per layer, populate the decode hot-expert resident slots by gathering hot-K from the
    streaming forward's already-converted 256-expert NZ (w13/w2/s13b/s2b) DIRECTLY into the resident
    params via index_select (zero-alloc, format-safe NZ first-dim gather). Masks deferred to the
    last layer (no per-layer host sync). Folds the decode hot-expert resident update into the
    streaming prefill at ~0 cost — no separate switch pass.

    Prereq (see maybe_reserve_slot): the resident params must be REMAPPED to caching-allocator memory
    at model load. Writing them on the model's loaded-weight memory region triggers a device-level
    coherence/flush (weights are read-only-optimized) that stalls the NSA per-layer .item() syncs by
    ~100s/prefill; in caching-allocator memory the write is free. (Root-caused via a clone-vs-param
    A/B: writing an identical NZ clone was free, writing the registered param was +100s.)"""
    layer, wrap = _REGISTRY[L]
    K = int(wrap.num_gpu_experts)
    if K <= 0 or wrap.gpu_experts_mask is None or wrap.logical_to_gpu_index is None:
        return
    E, H, I, num_layers = _get_cfg()
    if L == 0:
        _RES_PEND.clear()
    counts = torch.bincount(
        _hist_ids(topk_output.topk_ids).to(torch.int64), minlength=E)[:E]
    top = _pick_resident_top(counts, K)
    torch.index_select(w13, 0, top, out=layer.w13_weight.data)
    torch.index_select(w2, 0, top, out=layer.w2_weight.data)
    torch.index_select(s13b, 0, top, out=layer.w13_weight_scale_bf16.data)
    torch.index_select(s2b, 0, top, out=layer.w2_weight_scale_bf16.data)
    _RES_PEND[L] = (wrap, top, counts)
    if L == num_layers - 1:
        share_sum = 0.0
        for LL, (wr, tp, cnt) in sorted(_RES_PEND.items()):
            _set_resident_masks(wr, tp.cpu(), K, E)
            share_sum += float(cnt[tp].sum().item()) / max(float(cnt.sum().item()), 1.0)
        n = len(_RES_PEND)
        _RES_PEND.clear()
        print(f"[KT_STREAM] inline resident: top-{K} x {n} layers folded into prefill, "
              f"share={share_sum / max(n, 1):.3f}", flush=True)


def _apply_dynamic_residency() -> None:
    """Replace the static-prefix resident expert set with this prefill's per-layer top-K.
    Updates resident weights+scales and all routing structures in place (decode-graph and
    C++-side safe). Called at the end of a streaming prefill pass."""
    import time

    E, H, I, num_layers = _get_cfg()
    for L in range(num_layers):
        _pool_ok = (L in _MXFP4_POOL) if _KT_MXFP4_DEPOOL else (L in _POOL)
        if L not in _REQ_HIST or L not in _REGISTRY or not _pool_ok:
            logger.warning("[KT_STREAM] dyn-resident: layer %d incomplete; abort", L)
            return
    t0 = time.perf_counter()
    share_sum = 0.0
    K = 0
    for L in range(num_layers):
        layer, wrap = _REGISTRY[L]
        K = int(wrap.num_gpu_experts)
        if K <= 0 or wrap.gpu_experts_mask is None or wrap.logical_to_gpu_index is None:
            logger.warning("[KT_STREAM] dyn-resident: no resident slots/masks; abort")
            return
        counts = _REQ_HIST[L]
        top = _pick_resident_top(counts, K)                  # device, ascending logical ids
        top_cpu = top.cpu()
        if _KT_MXFP4_DEPOOL:
            # Convert ONLY the hot-K experts' MXFP4 -> resident W8A8-NZ directly. MXFP4 codes
            # are plain packed bytes (NOT NZ), so a first-dim [top] slice is format-safe; the
            # fused kernel emits resident-shaped [K,IN,OUT] NZ + [K,OUT] bf16 scale straight into
            # the resident slots (no whole-pool H2D / NZ round-trip gather).
            c13, s13, c2, s2 = _MXFP4_POOL[L]
            dev = layer.w13_weight.device
            c13d = _stage_pin_h2d(c13, top_cpu, dev)   # pinned staging -> DMA H2D
            s13d = _stage_pin_h2d(s13, top_cpu, dev)
            c2d = _stage_pin_h2d(c2, top_cpu, dev)
            s2d = _stage_pin_h2d(s2, top_cpu, dev)
            w13_top, s13b_top, w2_top, s2b_top = _mxfp4_convert_fn()(c13d, s13d, c2d, s2d, H, I)
            layer.w13_weight.data.copy_(w13_top)
            layer.w2_weight.data.copy_(w2_top)
            layer.w13_weight_scale_bf16.data.copy_(s13b_top)
            layer.w2_weight_scale_bf16.data.copy_(s2b_top)
        else:
            # Gather resident experts ON THE DEVICE: host pool slices (h13[e]) are format-UNAWARE
            # (NZ bytes sliced as ND -> garbage); NPU slices are format-aware. So H2D the whole
            # pool to the NZ slot, then slice there.
            h13, h2, s13b, s2b = _POOL[L]
            dev = layer.w13_weight.device
            slot13, slot2 = _ensure_slot(h13.shape, h2.shape, dev)   # [E,...] NPU NZ
            slot13.copy_(h13)                                        # whole-tensor H2D (correct NZ)
            slot2.copy_(h2)
            # Gather via ND round-trip: per-slot NZ device copy is bandwidth-pathological
            # (~0.3 GB/s); format_cast NZ->ND runs at full HBM BW, ND fancy-index is cheap,
            # ND->NZ restores format. ~12x faster, bitwise-equivalent (nz_batched_gather_test.py).
            import torch_npu as _tn
            _topd = top.to(dev)
            _nd13 = _tn.npu_format_cast(slot13, 2)                  # whole pool NZ->ND
            _g13 = _tn.npu_format_cast(_nd13[_topd].contiguous(), _ACL_FORMAT_FRACTAL_NZ)
            del _nd13
            layer.w13_weight.data.copy_(_g13)
            del _g13
            _nd2 = _tn.npu_format_cast(slot2, 2)
            _g2 = _tn.npu_format_cast(_nd2[_topd].contiguous(), _ACL_FORMAT_FRACTAL_NZ)
            del _nd2
            layer.w2_weight.data.copy_(_g2)
            del _g2
            layer.w13_weight_scale_bf16.data.copy_(s13b[top])
            layer.w2_weight_scale_bf16.data.copy_(s2b[top])
        _set_resident_masks(wrap, top_cpu, K, E)
        share_sum += float(counts[top].sum().item()) / max(float(counts.sum().item()), 1.0)
    torch.npu.synchronize()
    logger.info("[KT_STREAM] dynamic resident applied: top-%d x %d layers in %.1fs, "
                "prefill top-K activation share=%.3f",
                K, num_layers, time.perf_counter() - t0, share_sum / num_layers)


# Option B: run the first N real prefills through the HYBRID path (not streamed) to prime the CPU
# MoE (kt_kernel). Streamed prefills never invoke kt_kernel, so a stream-everything server (low
# THRESHOLD) keeps the CPU MoE cold and decode runs ~11 tps until enough hybrid traffic warms it.
# OS page cache is NOT the issue (the GGUF is already cached); the warming is process-local
# (kt_kernel threadpool/buffers + first-touch PTEs). A few hybrid prefills fix it.
_KT_STREAM_WARMUP = int(os.environ.get("KT_STREAM_WARMUP", "0") or "0")
_STREAM_WARMUP_STATE: dict = {}


def maybe_streaming_forward(wrapper, layer_idx, x, topk_output):
    """Entry from KTEPWrapperMethod.apply. Returns a CombineInput if streaming handled this
    layer, else None (caller falls through to the hybrid path). Never raises."""
    if not _KT_PREFILL_STREAM or getattr(wrapper, "tp_rank", 0) != 0:
        return None
    if not _is_prefill():
        return None
    # Option B startup warmup: force the first N real (multi-token) prefills to HYBRID to prime
    # kt_kernel. MUST be BEFORE the threshold gate -- else at a high THRESHOLD every sub-threshold
    # prefill (incl. the small sglang startup-warmup prefill) returns at the gate WITHOUT counting,
    # so the counter instead lands on the first LONG user prefill and wrongly forces it hybrid (the
    # slow ~73s path streaming was meant to avoid). The sglang startup warmup is the first
    # multi-token prefill (before any user request) -> it consumes the budget; real user prefills
    # then follow the normal threshold gate below and stream as expected.
    if _KT_STREAM_WARMUP > 0 and x.shape[0] > 1:
        st = _STREAM_WARMUP_STATE
        if layer_idx == 0:
            st["seen"] = st.get("seen", 0) + 1
        if st.get("seen", 0) <= _KT_STREAM_WARMUP:
            if layer_idx == 0:
                print(f"[KT_STREAM] warmup prefill {st['seen']}/{_KT_STREAM_WARMUP} -> hybrid "
                      f"(prime CPU MoE)", flush=True)
            return None
    if x.shape[0] < _T:
        return None
    try:
        E = int(getattr(wrapper, "global_num_experts", 0) or 0)
        H = int(getattr(wrapper, "hidden_size", x.shape[-1]))
        I = int(getattr(wrapper, "intermediate_size_per_partition", 0) or 0)
        num_layers = int(getattr(wrapper.kt_config, "num_layers", 0) or 0)
        if not (E and I and num_layers):
            logger.warning("[KT_STREAM] missing dims (E=%s I=%s L=%s) -> hybrid", E, I, num_layers)
            return None
        if _KT_MXFP4_DEPOOL:
            if _KT_GGUF_DEDUP:
                pass  # no pool: each layer is read from the CPU MoE's GGUF mmap on the fly
            elif not _MXFP4_POOL_BUILT:
                # Normally built at model-load (maybe_reserve_slot, parallel O_DIRECT); this serial
                # builder only runs if that path failed or never started.
                _build_mxfp4_pool(E, H, I, num_layers)
        elif not _POOL_BUILT:
            # Lazy fallback (pool not built at load): parallel O_DIRECT build. _build_pool_parread
            # frees the slot internally for NZ scratch; _streaming_forward re-allocates it after.
            _build_pool_parread(E, H, I, num_layers, x.device)
        top_k = topk_output.topk_ids.shape[1]
        return _streaming_forward(layer_idx, x, topk_output, top_k)
    except Exception as e:  # any failure -> fall back to hybrid, never crash the forward
        logger.warning("[KT_STREAM] streaming failed (%s) -> hybrid fallback", repr(e)[:160])
        return None
