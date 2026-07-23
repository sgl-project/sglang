"""Deterministic DSA top-k selector — CuTe DSL SIMT (SM90 / SM100).

Implements the DSA top-k backend contract (`DSATopKBackend.topk_func`):

    cute_topk_func(score (B, L) fp32, lengths (B) i32, topk,
                   row_starts (B) i32 | None) -> (B, topk) int32

Candidates for row r: columns c in [row_starts[r], row_starts[r] + lengths[r])
with score[r, c] != -inf (exact -inf excluded — matches the torch backend's
masked-fill convention; finite scores are never excluded). Returned indices
are LOCAL (column - row_starts[r]), front-packed, -1 padded — same as
sgl_kernel.fast_topk_v2 / the torch fallback — and emitted in ASCENDING
INDEX order (the slot order is part of this backend's determinism
contract; no sglang consumer depends on any particular slot order, the
default kernel's is arrival-nondeterministic).

Why this kernel (vs the default sgl-kernel CUDA radix top-k):

1. DETERMINISM: bitwise-identical output for identical input, always, in
   canonical order (INDEX ASCENDING; the selected set is an exact top-k
   with smallest-index tie-breaking, see 3.). The default kernel emits
   winners in smem-atomic ARRIVAL order and resolves the k-th-boundary tie
   set by arrival order, so equal logits can produce different index
   sets/orders run to run. No atomic in this kernel can affect any output
   bit: histogram atomics only ADD (commutative counts); staging atomics
   only permute buffers that feed order-independent histograms or the
   canonical index sort; the scan-emit output stage has no atomics at all.
   Both output stages (below) emit the same canonical output bit-for-bit.
2. EXACTNESS under adversarial distributions: the default kernel stages
   threshold-bin candidates into a fixed 4K-entry smem window and silently
   skips overflow (see sgl-project/sglang#17747), leaving uninitialized
   output slots; this kernel falls back to full-row rescans instead —
   identical results, just slower.
3. Tie policy is well-defined: at the k-th boundary the selected tie subset
   is the SMALLEST indices among bitwise-equal scores.

Algorithm (one CTA per row, 512 threads):
  ukey  = monotonic uint32 of the fp32 bits (b < 0 ? ~b : b | 0x80000000)
  * pass 1: 256-bucket histogram of ukey byte3 over the window (full-row
    scan #1), suffix-scan -> threshold bucket b*, count above s_above,
    bucket population c1, n_valid.
  * scan #2: elements in bucket b* are staged to a survivor buffer
    (composite int64) when c1 <= _SURV_CAP; on the index-sort output stage
    (below) elements in buckets > b* are definitively selected and go
    straight to the sort buffer. Refinement never touches gmem again on
    the staged path.
  * MSB-first 8-bit radix refinement over the survivors (<=3 passes with
    early exit) -> exact threshold key K_th and t_rem, the number of slots
    left for ukey == K_th elements.
  * If the tie set is larger than t_rem: 8-bit radix select over
    nidx24 = (~idx) & 0xFFFFFF among ukey == K_th (<=3 passes) -> N_th.
    Selection is then a pure per-element predicate:
        sel = ukey > K_th || (ukey == K_th && nidx >= N_th)
    which selects EXACTLY min(topk, n_valid) elements, smallest-index
    preferred at the boundary.
  * OUTPUT STAGE — two shape-selected variants, SAME canonical output
    (ascending index, front-packed, -1 padded), both bitwise deterministic:
    - scan-emit (grids that saturate the GPU, or short rows): one more
      ordered full-window pass; per element the pure predicate above; warp
      ballot+popc gives the lane prefix, a 16-warp smem scan plus a running
      cross-chunk base gives the global slot. The slot of each winner is
      its rank in index order among winners — a pure function of the
      input. No sort buffer, no output-affecting atomic.
    - index-sort collect (long rows at small B, small k): winners are
      staged to a sort buffer in (arbitrary) arrival order as int64 keys
      ~idx, then a bitonic sort canonicalizes to index-ascending order.
      Unique keys => ONE canonical permutation regardless of staging
      arrival order. Costs the k-slot sort but skips the third full-row
      scan — cheaper exactly where rows are long and the grid is small.
    The variant is chosen per shape at launch (compile-time constant, see
    _pick_scanemit); SGLANG_DSA_TOPK_CUTEDSL_STAGE={auto,scanemit,sort}
    overrides for perf work (output is identical either way).
  * c1 > _SURV_CAP (near-degenerate rows, e.g. all-equal) falls back to
    full-row scans for refinement — identical results, just slower.

smem: 512-entry i32 histogram + SURV_CAP int64 survivor buffer + vars
(+ NSORT = next_pow2(topk) int64 sort buffer on the index-sort stage
only). SURV_CAP is occupancy-aware: 4096 (~34KB scan-emit / ~50KB
index-sort at topk=2048) for grids that fill the GPU, 20480 (~162KB,
1 CTA/SM — free when B <= #SMs, e.g. decode) so wide/tie-heavy rows stay
on the staged path instead of the full-row-rescan fallback. No
KV-dependent smem: any row width up to 2^24-1 (nidx24 domain) with
B*L < 2^31 (flat int32 indexing) is supported; topk <= 4096 (production
DSA topk = 2048).

SGLANG_DSA_TOPK_CUTEDSL_SURVCAP overrides the survivor-stage capacity
(perf tuning only; results are identical for any capacity that fits in
shared memory). SGLANG_DSA_TOPK_CUTEDSL_STAGE={auto,scanemit,sort}
overrides the output-stage choice (perf tuning only; output is
identical).

Input contract notes: row_starts/lengths are trusted metadata (same as the
default kernel) — windows are clamped to [0, L) so out-of-range values can
never fault, but invalid metadata (e.g. negative row_starts) yields
window-clamped results rather than an error. NaN scores are not excluded:
they rank in the same bitwise-monotone total order (positive NaN above
+inf, negative NaN below all finite values), which is deterministic but
differs from torch.topk's NaN-maximal convention; DSA indexer logits are
expected NaN-free.
"""

import os
import threading

import torch

HAVE_CUTE_TOPK = False
_IMPORT_ERROR = None
try:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import make_ptr

    HAVE_CUTE_TOPK = torch.cuda.is_available()
except Exception as e:  # pragma: no cover - import guard
    _IMPORT_ERROR = e


_NT = 512  # threads per CTA (also histogram zero-stride; keep 512)
_NW = _NT // 32  # warps per CTA (scan-emit ballot+scan output stage)
_RADIX = 256
_MAX_TOPK = 4096  # sort smem cap (8B * 4096 = 32KB; production DSA = 2048)
_SURV_CAP_SMALL = 4096  # big grids: 32KB stage
_SURV_CAP_LARGE = 20480  # small grids (B <= #SMs): 160KB stage, 1 CTA/SM is
# free when the grid can't fill the GPU anyway;
# keeps wide rows staged (worst observed coarse-bin
# population ~18.7K on randn 64k rows) instead of
# the full-row-rescan fallback.
_I64_MIN = -(1 << 63)

_CAP_OK = {}  # device index -> compute capability >= 9.0


def _sm90(device: torch.device) -> bool:
    """Per-device capability gate (the tensor's device, not the current one)."""
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    ok = _CAP_OK.get(idx)
    if ok is None:
        ok = torch.cuda.get_device_capability(idx)[0] >= 9
        _CAP_OK[idx] = ok
    return ok


def supports(score: torch.Tensor, topk: int) -> bool:
    """CuTe path constraints; call sites fall back to the default otherwise."""
    if not HAVE_CUTE_TOPK:
        return False
    if not (score.is_cuda and score.dtype == torch.float32 and score.dim() == 2):
        return False
    if not _sm90(score.device):
        return False
    b, kv = score.shape
    if b <= 0 or kv <= 0:
        return False
    return (
        0 < topk <= _MAX_TOPK
        and kv < (1 << 24)  # nidx24 tie-select domain
        and b * kv < (1 << 31)  # flat int32 indexing (input)
        and b * topk < (1 << 31)
    )  # flat int32 indexing (output)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pick_scanemit(b: int, kv: int, topk: int) -> bool:
    """Output-stage heuristic (B200-measured crossover; see PR perf table).

    scan-emit wins when rows are short enough that one extra ordered row
    scan is cheaper than the k-slot bitonic sort — decisively so when the
    grid saturates the GPU (prefill-scale B, where the sort is the
    bottleneck at full occupancy: 2.6-3.8x) and for k >> 512 at any B.
    index-sort wins on long rows (the third full-row scan and its
    per-chunk CTA barriers dominate) and at small k where the sort is
    cheap. Output is bit-identical either way; this is perf-only.
    """
    return kv <= 16384 and (b >= 512 or topk > 512)


if HAVE_CUTE_TOPK:

    _EXT = (1 << 31) - 1  # flat extent placeholder (no bounds checks)

    def _warp_scan_pick(sHist, sVar, lane, k_rem):
        """Single-warp suffix-scan + threshold pick over the 256-bucket
        histogram (branchless; module-level so loops stay python). Lane l
        owns buckets [8l, 8l+8). Computes, for suffix-inclusive sums
        S[b] = sum_{u >= b} hist[u], the unique b* with
        S[b*] >= k_rem > S[b*+1], and writes (b*, S[b*+1], hist[b*],
        n_valid=S[0]) to sVar[0..2] and sVar[5] — every lane writes the same
        value (benign duplicate stores). Histogram is left untouched."""
        vals = [sHist[lane * 8 + b] for b in range(8)]
        suf = [None] * 8
        acc = vals[7]
        suf[7] = acc
        for b in (6, 5, 4, 3, 2, 1, 0):
            acc = acc + vals[b]
            suf[b] = acc
        tot = acc
        # incl_from_me = sum of lane totals for lanes >= me (shfl.down scan,
        # branchless OOB guard: shfl.down with clamp returns own value)
        x = tot
        for off in (1, 2, 4, 8, 16):
            y = cute.arch.shuffle_sync_down(x, off)
            x = x + y * cutlass.Int32(lane + off < 32)
        above = x - tot  # totals of lanes strictly above
        n_valid = cute.arch.warp_reduction_sum(tot, threads_in_group=32)
        bacc = cutlass.Int32(0)
        sacc = cutlass.Int32(0)
        cacc = cutlass.Int32(0)
        for b in range(8):
            s_here = suf[b] + above
            s_next = (suf[b + 1] + above) if b < 7 else above
            ind = cutlass.Int32(s_here >= k_rem) * cutlass.Int32(s_next < k_rem)
            bacc = bacc + (lane * 8 + b) * ind
            sacc = sacc + s_next * ind
            cacc = cacc + vals[b] * ind
        bacc = cute.arch.warp_reduction_sum(bacc, threads_in_group=32)
        sacc = cute.arch.warp_reduction_sum(sacc, threads_in_group=32)
        cacc = cute.arch.warp_reduction_sum(cacc, threads_in_group=32)
        sVar[0] = bacc
        sVar[1] = sacc
        sVar[2] = cacc
        sVar[5] = n_valid

    # ----------------------------------------------------------------------
    # Sort emission helpers (index-sort output stage). These are PLAIN
    # module-level python functions: the DSL AST rewrite only applies to
    # @cute.kernel/@cute.jit bodies (and their nested defs), so loops here
    # stay python-unrolled and may index python lists of register values.
    # Consequence: NO dynamic `if` allowed here — every compare-exchange is
    # branchless select arithmetic. The dynamic short-row guards live in
    # the kernel body.
    # ----------------------------------------------------------------------

    def _cmpx_reg_local(regs, tidx, k, j, E):
        # in-register pairs (e, e|j), j < E
        for e in range(E):
            if e & j == 0:
                a = regs[e]
                b = regs[e | j]
                d = b - a
                s = cutlass.Int64(b > a)
                m = a + d * s  # max(a, b)
                n = b - d * s  # min(a, b)
                desc = ((tidx * E + e) & k) == 0
                dI = cutlass.Int64(desc)
                lo_v = n + (m - n) * dI  # lo slot: desc -> max
                regs[e] = lo_v
                regs[e | j] = (m + n) - lo_v

    def _cmpx_reg_shfl(regs, tidx, k, j, E):
        # cross-lane pairs, E <= j < 32E: partner lane = t ^ (j//E)
        xm = j // E
        desc = ((tidx * E) & k) == 0  # k >= 2E: e-invariant
        i_lo = (tidx & xm) == 0
        km = cutlass.Int64(cutlass.Boolean(i_lo == desc))
        for e in range(E):
            a = regs[e]
            hi = cutlass.Int32(a >> 32)
            lo = cutlass.Int32((a << 32) >> 32)
            ph = cute.arch.shuffle_sync_bfly(hi, xm)
            pl = cute.arch.shuffle_sync_bfly(lo, xm)
            b = (cutlass.Int64(ph) << 32) | (
                cutlass.Int64(pl) & cutlass.Int64(0xFFFFFFFF)
            )
            d = b - a
            s = cutlass.Int64(b > a)
            m = a + d * s
            n = b - d * s
            regs[e] = n + (m - n) * km

    # NOTE: tail smem stages (npairs < NT, i.e. nsort = 512) need a dynamic
    # thread predicate — a wrap-alias (p &= npairs-1) would let two threads
    # in different warps rewrite the same pair unsynchronized. Dynamic `if`
    # requires the DSL AST rewrite, so the stage function is @cute.jit with
    # Constexpr shape params (python loops in the PLAIN helpers around it
    # stay python-unrolled).

    def _cmpx_smem_pair(sBuf, p, k, j):
        # one compare-exchange (branchless select arithmetic; plain helper)
        lo = ((p & ~(j - 1)) << 1) | (p & (j - 1))
        hi_i = lo | j
        a = sBuf[lo]
        b = sBuf[hi_i]
        d = b - a
        s = cutlass.Int64(b > a)
        m = a + d * s
        n = b - d * s
        dI = cutlass.Int64((lo & k) == 0)  # descending segment
        lo_v = n + (m - n) * dI
        sBuf[lo] = lo_v
        sBuf[hi_i] = (m + n) - lo_v

    @cute.jit
    def _cmpx_smem_stage(
        sBuf: cute.Tensor,
        tidx: cutlass.Int32,
        k: cutlass.Constexpr,
        j: cutlass.Constexpr,
        npairs: cutlass.Constexpr,
    ):
        # smem pairs, j >= 32E; surplus threads predicated off on tail
        # stages, npairs >= NT stages keep the unpredicated codegen.
        # Caller adds the barrier.
        for i in range(max(npairs // _NT, 1)):
            p = tidx + i * _NT
            if cutlass.const_expr(npairs < _NT):
                if p < npairs:
                    _cmpx_smem_pair(sBuf, p, k, j)
            else:
                _cmpx_smem_pair(sBuf, p, k, j)

    def _sort_phase_b_group(sBuf, tidx, nsort, kk):
        # one k-group with k = kk > 32E: high-j stages in smem, tail in regs
        E = nsort // _NT
        wspan = 32 * E
        npairs = nsort // 2
        j = kk // 2
        while j >= wspan:
            _cmpx_smem_stage(sBuf, tidx, kk, j, npairs)
            cute.arch.sync_threads()
            j //= 2
        regs = [sBuf[tidx * E + e] for e in range(E)]
        while j >= 1:
            if j >= E:
                _cmpx_reg_shfl(regs, tidx, kk, j, E)
            else:
                _cmpx_reg_local(regs, tidx, kk, j, E)
            j //= 2
        for e in range(E):
            sBuf[tidx * E + e] = regs[e]

    def _sort_phase_a(sBuf, tidx, nsort):
        # k = 2 .. 32E in registers/shuffles only; zero barriers
        E = nsort // _NT
        wspan = 32 * E
        regs = [sBuf[tidx * E + e] for e in range(E)]
        kk = 2
        while kk <= wspan:
            j = kk // 2
            while j >= 1:
                if j >= E:
                    _cmpx_reg_shfl(regs, tidx, kk, j, E)
                else:
                    _cmpx_reg_local(regs, tidx, kk, j, E)
                j //= 2
            kk *= 2
        for e in range(E):
            sBuf[tidx * E + e] = regs[e]

    @cute.kernel
    def _topk_kernel(
        mIn: cute.Tensor,  # flat [b * kv] fp32
        mKs: cute.Tensor,  # [b] int32 (row_starts)
        mLen: cute.Tensor,  # [b] int32 (lengths)
        mOut: cute.Tensor,  # flat [b * topk] int32
        kv: cutlass.Int32,
        topk_c: cutlass.Constexpr,  # int
        nsort_c: cutlass.Constexpr,  # int, pow2 >= topk_c (index-sort stage)
        surv_cap_c: cutlass.Constexpr,  # survivor-stage capacity (smem)
        scanemit_c: cutlass.Constexpr,  # bool: scan-emit output stage
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        hist_p = smem.allocate_array(cutlass.Int32, 2 * _RADIX, byte_alignment=4)
        var_p = smem.allocate_array(cutlass.Int32, 8, byte_alignment=4)
        sBuf = None
        if cutlass.const_expr(not scanemit_c):
            buf_p = smem.allocate_array(cutlass.Int64, nsort_c, byte_alignment=8)
            sBuf = cute.make_tensor(buf_p, cute.make_layout(nsort_c))
        surv_p = smem.allocate_array(cutlass.Int64, surv_cap_c, byte_alignment=8)
        sHist = cute.make_tensor(hist_p, cute.make_layout(2 * _RADIX))
        sVar = cute.make_tensor(var_p, cute.make_layout(8))
        sSurv = cute.make_tensor(surv_p, cute.make_layout(surv_cap_c))

        ks = mKs[row]
        ln = mLen[row]
        if ks < 0:
            ks = cutlass.Int32(0)
        if ln < 0:
            ln = cutlass.Int32(0)
        ke = ks + ln
        if ke > kv:
            ke = kv
        base = row * kv
        span = ke - ks
        iters = cutlass.Int32(0)
        if span > 0:
            iters = (span + _NT - 1) // _NT

        # ------------------------------------------------------------------
        # traced helpers (nested defs; all traced values passed explicitly —
        # the DSL forbids closure capture inside dynamic control flow)
        # ------------------------------------------------------------------
        def hist_zero(sHist, tidx):
            # 512 threads x 512 entries: exactly one store per thread
            sHist[tidx] = 0

        def elem(it, tidx, ks, ke, base, mIn):
            """Load element it*NT+tidx of the window; returns
            (valid, idx, ukey, nidx24)."""
            idx = ks + it * _NT + tidx
            inw = idx < ke
            v = cutlass.Float32(0.0)
            if inw:
                v = mIn[base + idx]
            b = v.bitcast(cutlass.Int32)
            bu = b.bitcast(cutlass.Uint32)
            ukey = bu | cutlass.Uint32(0x80000000)
            if b < 0:
                ukey = ~bu
            # -inf (bits 0xFF800000) excluded (torch-backend parity)
            valid = inw and (b != cutlass.Int32(-8388608))
            nidx = (~idx).bitcast(cutlass.Uint32) & cutlass.Uint32(0xFFFFFF)
            return valid, idx, ukey, nidx

        def comp_of(idx, ukey):
            """Unique canonical-order composite: (skey << 32) | (~idx)."""
            skey = (ukey ^ cutlass.Uint32(0x80000000)).bitcast(cutlass.Int32)
            return (cutlass.Int64(skey) << 32) | (
                cutlass.Int64(~idx) & cutlass.Int64(0xFFFFFFFF)
            )

        def surv_decode(comp):
            """Recover (ukey, nidx24) from a staged composite."""
            hi = cutlass.Int32(comp >> 32)
            ukey = hi.bitcast(cutlass.Uint32) ^ cutlass.Uint32(0x80000000)
            nidx = cutlass.Int32(comp).bitcast(cutlass.Uint32) & cutlass.Uint32(
                0xFFFFFF
            )
            return ukey, nidx

        def suffix_scan_and_pick(sHist, sVar, tidx, k_rem):
            """Warp-0 suffix-scan + pick over hist[0:256] (see
            _warp_scan_pick). Returns (bstar, s_above, c_bucket),
            CTA-uniform; row candidate total lands in sVar[5]."""
            if tidx < 32:
                _warp_scan_pick(sHist, sVar, tidx, k_rem)
            cute.arch.sync_threads()
            return sVar[0], sVar[1], sVar[2]

        # ------------------------------------------------------------------
        # init: counters, sort-buffer sentinel padding (index-sort stage),
        # pass-1 histogram (ukey byte3 over all valid candidates; full-row
        # scan #1)
        # ------------------------------------------------------------------
        hist_zero(sHist, tidx)
        if tidx < 8:
            sVar[tidx] = 0
        if cutlass.const_expr(not scanemit_c):
            for i in range(nsort_c // _NT):
                sBuf[tidx + i * _NT] = cutlass.Int64(_I64_MIN)
        cute.arch.sync_threads()
        for it in cutlass.range(iters):
            valid, idx, ukey, nidx = elem(it, tidx, ks, ke, base, mIn)
            if valid:
                byte = (ukey >> 24).bitcast(cutlass.Int32)
                cute.arch.atomic_add(
                    sHist.iterator + byte, cutlass.Int32(1), scope="cta"
                )
        cute.arch.sync_threads()

        # pass-0 pick; row candidate total (S[0]) comes back in sVar[5]
        bstar, s_above, c_bkt = suffix_scan_and_pick(
            sHist, sVar, tidx, cutlass.Int32(topk_c)
        )
        n_valid = sVar[5]
        cute.arch.sync_threads()

        kth = cutlass.Uint32(0)
        nth = cutlass.Uint32(0)
        out_v = n_valid
        if n_valid <= topk_c:
            # short row: every valid candidate is selected (kth=nth=0).
            # Index-sort staging only — scan-emit handles short rows
            # entirely in the ordered emission pass below. Sort keys are
            # int64 ~idx: unique, and descending-sorting them yields
            # ascending index order.
            if cutlass.const_expr(not scanemit_c):
                for it in cutlass.range(iters):
                    valid, idx, ukey, nidx = elem(it, tidx, ks, ke, base, mIn)
                    if valid:
                        pos = cute.arch.atomic_add(
                            sVar.iterator + 3, cutlass.Int32(1), scope="cta"
                        )
                        if pos < nsort_c:
                            sBuf[pos] = cutlass.Int64(~idx)
        else:
            out_v = cutlass.Int32(topk_c)
            # pass-1 pick is valid (S[0] = n_valid > topk)
            bstar_u = bstar.bitcast(cutlass.Uint32)
            prefix = cutlass.Uint32(bstar_u << 24)
            k_rem = cutlass.Int32(topk_c) - s_above
            c1 = c_bkt
            done = c_bkt == k_rem
            staged = c1 <= surv_cap_c
            # scan #2 (full-row): winners above bucket b* go straight to
            # the sort buffer (index-sort stage only — scan-emit re-finds
            # them in the ordered emission pass); bucket-b* members are
            # staged to the smem survivor buffer for refinement.
            for it in cutlass.range(iters):
                valid, idx, ukey, nidx = elem(it, tidx, ks, ke, base, mIn)
                if valid:
                    byte3 = ukey >> 24
                    if cutlass.const_expr(not scanemit_c):
                        if byte3 > bstar_u:
                            pos = cute.arch.atomic_add(
                                sVar.iterator + 3, cutlass.Int32(1), scope="cta"
                            )
                            if pos < nsort_c:
                                sBuf[pos] = cutlass.Int64(~idx)
                        elif staged and byte3 == bstar_u:
                            sp = cute.arch.atomic_add(
                                sVar.iterator + 4, cutlass.Int32(1), scope="cta"
                            )
                            if sp < surv_cap_c:
                                sSurv[sp] = comp_of(idx, ukey)
                    else:
                        if staged and byte3 == bstar_u:
                            sp = cute.arch.atomic_add(
                                sVar.iterator + 4, cutlass.Int32(1), scope="cta"
                            )
                            if sp < surv_cap_c:
                                sSurv[sp] = comp_of(idx, ukey)
            cute.arch.sync_threads()
            iters_s = (c1 + _NT - 1) // _NT

            # MSB-first radix refinement to the exact threshold key
            for shift in (16, 8, 0):
                if not done:
                    hist_zero(sHist, tidx)
                    cute.arch.sync_threads()
                    if staged:
                        for it in cutlass.range(iters_s):
                            si = it * _NT + tidx
                            if si < c1:
                                ukey, nidx = surv_decode(sSurv[si])
                                if (ukey >> (shift + 8)) == (prefix >> (shift + 8)):
                                    byte = (
                                        (ukey >> shift) & cutlass.Uint32(0xFF)
                                    ).bitcast(cutlass.Int32)
                                    cute.arch.atomic_add(
                                        sHist.iterator + byte,
                                        cutlass.Int32(1),
                                        scope="cta",
                                    )
                    else:
                        for it in cutlass.range(iters):
                            valid, idx, ukey, nidx = elem(it, tidx, ks, ke, base, mIn)
                            if valid and (ukey >> (shift + 8)) == (
                                prefix >> (shift + 8)
                            ):
                                byte = ((ukey >> shift) & cutlass.Uint32(0xFF)).bitcast(
                                    cutlass.Int32
                                )
                                cute.arch.atomic_add(
                                    sHist.iterator + byte, cutlass.Int32(1), scope="cta"
                                )
                    cute.arch.sync_threads()
                    bstar, s_above, c_bkt = suffix_scan_and_pick(
                        sHist, sVar, tidx, k_rem
                    )
                    prefix = prefix | (bstar.bitcast(cutlass.Uint32) << shift)
                    k_rem = k_rem - s_above
                    if shift != 0:
                        done = c_bkt == k_rem
            kth = prefix
            # tie boundary: need k_rem smallest indices among ukey == kth,
            # i.e. the k_rem LARGEST nidx24 = (~idx) & 0xFFFFFF
            if (not done) and c_bkt > k_rem:
                nprefix = cutlass.Uint32(0)
                done2 = cutlass.Boolean(False)
                for shift in (16, 8, 0):
                    if not done2:
                        hist_zero(sHist, tidx)
                        cute.arch.sync_threads()
                        if staged:
                            for it in cutlass.range(iters_s):
                                si = it * _NT + tidx
                                if si < c1:
                                    ukey, nidx = surv_decode(sSurv[si])
                                    if ukey == kth and (nidx >> (shift + 8)) == (
                                        nprefix >> (shift + 8)
                                    ):
                                        byte = (
                                            (nidx >> shift) & cutlass.Uint32(0xFF)
                                        ).bitcast(cutlass.Int32)
                                        cute.arch.atomic_add(
                                            sHist.iterator + byte,
                                            cutlass.Int32(1),
                                            scope="cta",
                                        )
                        else:
                            for it in cutlass.range(iters):
                                valid, idx, ukey, nidx = elem(
                                    it, tidx, ks, ke, base, mIn
                                )
                                if (
                                    valid
                                    and ukey == kth
                                    and (nidx >> (shift + 8))
                                    == (nprefix >> (shift + 8))
                                ):
                                    byte = (
                                        (nidx >> shift) & cutlass.Uint32(0xFF)
                                    ).bitcast(cutlass.Int32)
                                    cute.arch.atomic_add(
                                        sHist.iterator + byte,
                                        cutlass.Int32(1),
                                        scope="cta",
                                    )
                        cute.arch.sync_threads()
                        bstar, s_above, c_bkt = suffix_scan_and_pick(
                            sHist, sVar, tidx, k_rem
                        )
                        nprefix = nprefix | (bstar.bitcast(cutlass.Uint32) << shift)
                        k_rem = k_rem - s_above
                        if shift != 0:
                            done2 = c_bkt == k_rem
                nth = nprefix

            # final selection within bucket b* (index-sort stage only):
            # pure predicate; arrival order only permutes the buffer the
            # index sort canonicalizes. Scan-emit applies the same
            # predicate inside the ordered emission pass instead.
            if cutlass.const_expr(not scanemit_c):
                if staged:
                    for it in cutlass.range(iters_s):
                        si = it * _NT + tidx
                        if si < c1:
                            comp = sSurv[si]
                            ukey, nidx = surv_decode(comp)
                            sel = ukey > kth
                            if ukey == kth and nidx >= nth:
                                sel = cutlass.Boolean(True)
                            if sel:
                                pos = cute.arch.atomic_add(
                                    sVar.iterator + 3, cutlass.Int32(1), scope="cta"
                                )
                                if pos < nsort_c:
                                    # low 32 bits of the composite are ~idx
                                    sBuf[pos] = cutlass.Int64(cutlass.Int32(comp))
                else:
                    for it in cutlass.range(iters):
                        valid, idx, ukey, nidx = elem(it, tidx, ks, ke, base, mIn)
                        if valid and (ukey >> 24) == bstar_u:
                            sel = ukey > kth
                            if ukey == kth and nidx >= nth:
                                sel = cutlass.Boolean(True)
                            if sel:
                                pos = cute.arch.atomic_add(
                                    sVar.iterator + 3, cutlass.Int32(1), scope="cta"
                                )
                                if pos < nsort_c:
                                    sBuf[pos] = cutlass.Int64(~idx)
        cute.arch.sync_threads()

        if cutlass.const_expr(scanemit_c):
            # --------------------------------------------------------------
            # scan-emit output stage (scan #3): deterministic ORDERED
            # emission. Selection is the pure predicate on (kth, nth); the
            # output slot of each winner is the exclusive count of winners
            # before it in index order (warp ballot + 16-warp scan +
            # running cross-chunk base). No sort, no arrival-order atomic
            # anywhere near the output => bitwise deterministic, canonical
            # (index ascending).
            # --------------------------------------------------------------
            if tidx == 0:
                sVar[3] = 0
            cute.arch.sync_threads()
            obase = row * topk_c
            lane = tidx % 32
            wid = tidx // 32
            for it in cutlass.range(iters):
                valid, idx, ukey, nidx = elem(it, tidx, ks, ke, base, mIn)
                sel = cutlass.Boolean(False)
                if valid:
                    if n_valid <= topk_c:
                        sel = cutlass.Boolean(True)
                    elif ukey > kth:
                        sel = cutlass.Boolean(True)
                    elif ukey == kth and nidx >= nth:
                        sel = cutlass.Boolean(True)
                mask = cute.arch.vote_ballot_sync(sel)
                lpre = cute.arch.popc(mask & cute.arch.lanemask_lt())
                if lane == 0:
                    sHist[_RADIX + wid] = cute.arch.popc(mask)
                cute.arch.sync_threads()
                if tidx < 32:
                    t_w = cutlass.Int32(0)
                    if tidx < _NW:
                        t_w = sHist[_RADIX + tidx]
                    run0 = sVar[3]
                    x = t_w
                    for off in (1, 2, 4, 8):
                        # DSL gotcha: shuffle_sync_up's default
                        # mask_and_clamp (31) is the DOWN clamp; up MUST
                        # pass 0 or every lane silently reads itself back.
                        y = cute.arch.shuffle_sync_up(x, off, mask_and_clamp=0)
                        x = x + y * cutlass.Int32(tidx >= off)
                    if tidx < _NW:
                        sHist[_RADIX + 32 + tidx] = run0 + x - t_w
                        if tidx == _NW - 1:
                            sVar[6] = run0 + x
                cute.arch.sync_threads()
                if sel:
                    pos = sHist[_RADIX + 32 + wid] + lpre
                    if pos < topk_c:
                        mOut[obase + pos] = idx - ks  # LOCAL index
                if tidx == 0:
                    sVar[3] = sVar[6]
                cute.arch.sync_threads()

            # -1 fill beyond out_v (selected count is analytic:
            # min(topk, n_valid))
            for i in range(max((topk_c + _NT - 1) // _NT, 1)):
                o = tidx + i * _NT
                if o < topk_c:
                    if o >= out_v:
                        mOut[obase + o] = cutlass.Int32(-1)
        else:
            # --------------------------------------------------------------
            # index-sort output stage: descending bitonic sort of nsort_c
            # int64 keys ~idx (unique; descending ~idx == ascending idx —
            # the SAME canonical order the scan-emit stage produces) via
            # the module-level emission helpers above. Short rows skip
            # k-groups under CTA-uniform guards: with out_v real elements
            # compacted into [0, out_v) and INT64_MIN sentinels elsewhere,
            # the k <= next_pow2(out_v) prefix of the network fully sorts
            # block 0 (larger-k stages only permute sentinel blocks).
            # Phase B is always exactly 4 groups: kk = NSORT/8, /4, /2,
            # NSORT (32E = NSORT/16).
            # --------------------------------------------------------------
            if out_v > 1:
                _sort_phase_a(sBuf, tidx, nsort_c)
            cute.arch.sync_threads()
            if out_v * 2 > nsort_c // 8:
                _sort_phase_b_group(sBuf, tidx, nsort_c, nsort_c // 8)
            cute.arch.sync_threads()
            if out_v * 2 > nsort_c // 4:
                _sort_phase_b_group(sBuf, tidx, nsort_c, nsort_c // 4)
            cute.arch.sync_threads()
            if out_v * 2 > nsort_c // 2:
                _sort_phase_b_group(sBuf, tidx, nsort_c, nsort_c // 2)
            cute.arch.sync_threads()
            if out_v * 2 > nsort_c:
                _sort_phase_b_group(sBuf, tidx, nsort_c, nsort_c)
            cute.arch.sync_threads()

            # emit: slot i -> LOCAL index (~key - row_start) if i < out_v
            # else -1
            obase = row * topk_c
            for i in range(max((topk_c + _NT - 1) // _NT, 1)):
                o = tidx + i * _NT
                if o < topk_c:
                    res = cutlass.Int32(-1)
                    if o < out_v:
                        low = cutlass.Int32((~sBuf[o]) & cutlass.Int64(0xFFFFFFFF))
                        res = low - ks
                    mOut[obase + o] = res

    @cute.jit
    def _topk_launch(
        in_ptr: cute.Pointer,
        ks_ptr: cute.Pointer,
        len_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        b_rows: cutlass.Int32,
        kv: cutlass.Int32,
        stream: cuda_driver.CUstream,
        topk_c: cutlass.Constexpr,
        nsort_c: cutlass.Constexpr,
        surv_cap_c: cutlass.Constexpr,
        scanemit_c: cutlass.Constexpr,
    ):
        mIn = cute.make_tensor(in_ptr, cute.make_layout(_EXT))
        mKs = cute.make_tensor(ks_ptr, cute.make_layout(_EXT))
        mLen = cute.make_tensor(len_ptr, cute.make_layout(_EXT))
        mOut = cute.make_tensor(out_ptr, cute.make_layout(_EXT))
        _topk_kernel(
            mIn, mKs, mLen, mOut, kv, topk_c, nsort_c, surv_cap_c, scanemit_c
        ).launch(grid=(b_rows, 1, 1), block=(_NT, 1, 1), stream=stream)

    _COMPILED = {}
    _COMPILE_LOCK = threading.Lock()
    _ZEROS = {}
    _SM_COUNTS = {}

    def _sm_count(device):
        idx = device.index
        n = _SM_COUNTS.get(idx)
        if n is None:
            n = torch.cuda.get_device_properties(device).multi_processor_count
            _SM_COUNTS[idx] = n
        return n

    def _f32_ptr(t):
        return make_ptr(
            cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )

    def _i32_ptr(t):
        return make_ptr(
            cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )

    def _zeros_i32(n, device):
        key = (device.index,)
        z = _ZEROS.get(key)
        if z is None or z.numel() < n:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "cute_topk_func: the default row_starts buffer would be "
                    "allocated under CUDA graph capture; run one eager "
                    "warmup step first."
                )
            z = torch.zeros(max(n, 8192), dtype=torch.int32, device=device)
            # one-time: make the zero-fill visible to any stream that later
            # reads this cached buffer
            torch.cuda.current_stream(device).synchronize()
            _ZEROS[key] = z
        return z


def cute_topk_func(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor = None,
) -> torch.Tensor:
    """sglang DSATopKBackend.topk_func-compatible entry (see module doc).

    score (B, L) fp32 CUDA; lengths (B) int32; optional row_starts (B) int32.
    Returns (B, topk) int32 LOCAL indices, -1 padded, in ascending index
    order (canonical; see module docstring). Bitwise deterministic.
    """
    if not HAVE_CUTE_TOPK:
        raise RuntimeError(
            "cute_topk_func requires a CUDA SM90+ device and nvidia-cutlass-dsl"
            + (
                f" (import failed: {_IMPORT_ERROR!r})"
                if _IMPORT_ERROR is not None
                else ""
            )
        )
    if not (score.is_cuda and score.dtype == torch.float32 and score.dim() == 2):
        raise ValueError(
            "cute_topk_func: score must be a CUDA (B, L) float32 tensor, got "
            f"{score.device} {tuple(score.shape)} {score.dtype}"
        )
    if not _sm90(score.device):
        raise RuntimeError(
            f"cute_topk_func: device {score.device} has compute capability "
            f"{torch.cuda.get_device_capability(score.device)}, needs SM90+"
        )
    score = score.contiguous()
    b, kv = score.shape
    # validate the full documented envelope BEFORE the empty-shape early
    # returns, so out-of-envelope arguments fail loudly even on B=0/L=0
    if not (
        0 <= topk <= _MAX_TOPK
        and kv < (1 << 24)
        and b * kv < (1 << 31)
        and b * topk < (1 << 31)
    ):
        raise ValueError(
            f"cute_topk_func: unsupported shape B={b}, L={kv}, topk={topk} "
            f"(requires topk <= {_MAX_TOPK}, L < 2**24, B*L < 2**31, "
            "B*topk < 2**31; see supports())"
        )
    if lengths.numel() < b:
        raise ValueError(
            f"cute_topk_func: lengths has {lengths.numel()} elements, needs {b}"
        )
    if row_starts is not None and row_starts.numel() < b:
        raise ValueError(
            f"cute_topk_func: row_starts has {row_starts.numel()} elements, "
            f"needs {b}"
        )
    out = torch.empty((b, topk), dtype=torch.int32, device=score.device)
    if b == 0 or topk == 0:
        return out
    if kv == 0:
        return out.fill_(-1)
    lengths = lengths.to(device=score.device, dtype=torch.int32).contiguous()
    if row_starts is None:
        row_starts = _zeros_i32(b, score.device)
    else:
        row_starts = row_starts.to(device=score.device, dtype=torch.int32).contiguous()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(score.device).cuda_stream)
    args = (
        _f32_ptr(score),
        _i32_ptr(row_starts),
        _i32_ptr(lengths),
        _i32_ptr(out),
        cutlass.Int32(b),
        cutlass.Int32(kv),
        stream,
    )
    cap_env = os.environ.get("SGLANG_DSA_TOPK_CUTEDSL_SURVCAP", "")
    if cap_env:
        try:
            surv_cap = int(cap_env)
        except ValueError:
            raise ValueError(
                f"SGLANG_DSA_TOPK_CUTEDSL_SURVCAP={cap_env!r} is not an integer"
            ) from None
        if surv_cap <= 0:
            raise ValueError(
                f"SGLANG_DSA_TOPK_CUTEDSL_SURVCAP must be > 0, got {surv_cap}"
            )
    else:
        # occupancy-aware: grids that cannot fill the GPU (1 CTA/SM anyway)
        # take the large stage for free; full grids keep the small stage.
        surv_cap = _SURV_CAP_LARGE if b <= _sm_count(score.device) else _SURV_CAP_SMALL
    stage_env = os.environ.get("SGLANG_DSA_TOPK_CUTEDSL_STAGE", "auto")
    if stage_env == "scanemit":
        scanemit = True
    elif stage_env == "sort":
        scanemit = False
    elif stage_env in ("auto", ""):
        scanemit = _pick_scanemit(b, kv, topk)
    else:
        raise ValueError(
            "SGLANG_DSA_TOPK_CUTEDSL_STAGE must be one of 'auto', 'scanemit', "
            f"'sort', got {stage_env!r}"
        )
    nsort = max(_next_pow2(topk), _NT)
    key = (topk, nsort, surv_cap, scanemit, score.get_device())
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cute_topk_func: first call for config "
                f"{key} happened under CUDA graph capture; run one eager "
                "warmup step first."
            )
        with torch.cuda.nvtx.range("cute_topk_compile"):
            with torch.cuda.device(score.device), _COMPILE_LOCK:
                fn = _COMPILED.get(key)
                if fn is None:
                    fn = cute.compile(
                        _topk_launch, *args, topk, nsort, surv_cap, scanemit
                    )
                    _COMPILED[key] = fn
    fn(*args)
    return out
