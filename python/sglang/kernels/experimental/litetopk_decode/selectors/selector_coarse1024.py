# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generated BATCHED isolated derivative (grid (R, B)), FUSED PHYSICAL MAPPING. Original NVIDIA Apache-2.0 license retained.
# Original device SHA256 d375890c30be2257d1773ae11e99a062a17ebe002e075e61344042200670d4a5. No upstream file is changed.
import _gvr_hist_prologue_acqrel_hsplitq_original_20260918 as _original
globals().update({name: val for name, val in vars(_original).items() if not name.startswith("__")})

# Batched derivative: byte offset of the per-row (strict | slab) cursor pairs, appended
# after the stock GVR2 workspace. MAX_ROWS pairs = 256 B on a 20 MiB workspace.
WS_TAIL_OFF__batched = _original.workspace_bytes()
MAX_ROWS__batched = 32

@cute.jit
def half_value(raw):
    raw = cutlass.Uint32(raw)
    exponent = (raw >> cutlass.Uint32(10)) & cutlass.Uint32(31)
    mantissa = raw & cutlass.Uint32(1023)
    sign = raw & cutlass.Uint32(0x8000)
    result = cutlass.Float32(0.0)
    if exponent == cutlass.Uint32(0):
        result = cutlass.Float32(mantissa) * cutlass.Float32(5.960464477539063e-8)
        if sign != cutlass.Uint32(0):
            result = -result
    else:
        packed = (sign << cutlass.Uint32(16)) | ((exponent + cutlass.Uint32(112)) << cutlass.Uint32(23)) | (mantissa << cutlass.Uint32(13))
        result = C.f32_of_u32(packed)
    return result


@cute.jit
def coarse_floor(tau):
    # Lower FP32 boundary of one grouped ordered-FP16 bin.
    result = cutlass.Float32(float("inf"))
    if tau >= cutlass.Int32(1008):
        result = cutlass.Float32(float("-inf"))
    elif tau >= cutlass.Int32(15):
        maximum = cutlass.Uint32((tau << cutlass.Int32(6)) | cutlass.Int32(63))
        accepted = maximum
        if maximum < cutlass.Uint32(0x8000):
            accepted = cutlass.Uint32(0x7FFF) - maximum
        if accepted == cutlass.Uint32(0):
            result = cutlass.Float32(0.0)
        elif accepted == cutlass.Uint32(0x7C00):
            result = cutlass.Float32(65520.0)
        elif accepted == cutlass.Uint32(0xFBFF):
            result = C.f32_of_u32(C.u32_of_f32(cutlass.Float32(-65520.0)) - cutlass.Uint32(1))
        else:
            rejected = accepted - cutlass.Uint32(1)
            if (accepted & cutlass.Uint32(0x8000)) != cutlass.Uint32(0):
                rejected = accepted + cutlass.Uint32(1)
            result = (half_value(accepted) + half_value(rejected)) * cutlass.Float32(0.5)
            if (accepted & cutlass.Uint32(1)) != cutlass.Uint32(0):
                raw = C.u32_of_f32(result)
                if (raw & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
                    raw = raw - cutlass.Uint32(1)
                else:
                    raw = raw + cutlass.Uint32(1)
                result = C.f32_of_u32(raw)
    return result


class SplitQAcqRelHistogramPrologueGvrKernel(GvrMainKernel):
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        scap_dead: cutlass.Int32,
        cmp_dead: cutlass.Int32,
        R: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        kv_lens: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
        amin: cutlass.Int32,
        sd_en: cutlass.Int32,
        tsh_en: cutlass.Int32,
        histogram: cute.Tensor,
        pdiag: cute.Tensor,
        hrestore: cutlass.Int32,
        diagnostics: cute.Tensor,
        ptable: cute.Tensor,
        pcols: cutlass.Int32,
        npages: cutlass.Int32,
    ):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        KPT = self.kpt
        SCPB = self.scpb
        CMPB = self.cmpb
        PFD = self.pfd
        NATT = self.natt
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()  # 2-D grid (part, row)
        row = by
        part = cutlass.Int32(0)
        if cutlass.const_expr(self.split):
            part = bx
        lane = tidx & cutlass.Int32(31)

        # ================= per-row varlen prologue (varlen mode only) =========
        # Production contract: row r serves request r // next_n with
        # kv_len = kv_lens[r // next_n], n = (kv_len - next_n + r % next_n + 1)
        # >> cr_shift.  The sampling-ladder scalars are then re-derived from
        # this row's n by the EXACT route_dynamic() host formulas (the scalar
        # launch args are dead in this mode, except `n`, which carries the
        # envelope the per-row length is clamped to).  n <= k rows have no runtime
        # `return` in CuTe DSL: they run the body as a zero-work pass
        # (n = 0, TGT = INT_MAX so no rung ever accepts) and the identity/pad
        # emission happens in the epilogue at the end of the kernel.  Every
        # value below is a pure function of `row`, so all R split CTAs of a
        # row (and all threads) compute identical scalars — grid-uniform per
        # row by construction.
        short = cutlass.Int32(0)
        n_row = cutlass.Int32(0)
        # Function-scope default so the tail epilogue can read it unconditionally:
        # the election below assigns `alive` inside `if cutlass.const_expr(self.split)`,
        # which is compile-time true for every shipped build but must not be relied on
        # implicitly by code outside that block.
        alive = cutlass.Int32(0)
        tsh_run = cutlass.Int32(1)
        if cutlass.const_expr(self.varlen):
            req = row // cutlass.Int32(self.next_n)
            rr = row % cutlass.Int32(self.next_n)
            kvl = kv_lens[req]
            nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                self.cr_shift
            )
            if nv < cutlass.Int32(0):
                nv = cutlass.Int32(0)
            # Clamp to the logical envelope carried in the (otherwise dead)
            # `n` launch slot, NOT to the row stride `npad`: on a wider-stride
            # arena view the columns in [n, npad) belong to the arena, never
            # to this row, so an oversized kv_len must not classify them
            # (same clamp the reg / reg_clus / clus varlen prologues apply).
            if nv > n:
                nv = n
            if cutlass.const_expr(self.r_const == 37):
                if nv <= cutlass.Int32(262144):
                    R = cutlass.Int32(16)
            if cutlass.const_expr(self.r_const == 18):
                if nv <= cutlass.Int32(262144):
                    R = cutlass.Int32(8)
            if cutlass.const_expr(self.r_const == 9):
                if nv <= cutlass.Int32(65536):
                    R = cutlass.Int32(4)
            n_row = nv
            if nv <= k:
                short = cutlass.Int32(1)
                n = cutlass.Int32(0)
                SMP = cutlass.Int32(0)
                SS2 = cutlass.Int32(1)
                # "never accepts" sentinels; 2^30-1 so the TGT*2 scan target
                # stays positive (0x7FFFFFFF would overflow to -2 and flip
                # every tot0 >= TGT*2 gate on the all-zero histogram)
                TGT = cutlass.Int32(0x3FFFFFFF)
                TGT2 = cutlass.Int32(0x3FFFFFFF)
                Q = cutlass.Int32(0)
            if short == cutlass.Int32(0):
                n = nv
                n4v = n >> cutlass.Int32(2)
                # Ladder-scalar baselines only: the real SMP/SS2/TGT/TGT2 are
                # derived by warp0 alone in the block below (bit-identical
                # formulas) and published through s_lad — every thread's local
                # copies here are overwritten by the post-barrier smem read.
                SMP = cutlass.Int32(0)
                SS2 = cutlass.Int32(1)
                TGT = cutlass.Int32(0)
                TGT2 = cutlass.Int32(0)
                if cutlass.const_expr(self.split):
                    Q = (n4v + cutlass.Int32(self.r_const - 1)) // cutlass.Int32(self.r_const)
                    if cutlass.const_expr(self.r_const == 37):
                        if R == cutlass.Int32(16):
                            Q = (n4v + cutlass.Int32(15)) // cutlass.Int32(16)
                    if cutlass.const_expr(self.r_const == 18):
                        if R == cutlass.Int32(8):
                            Q = (n4v + cutlass.Int32(7)) // cutlass.Int32(8)
                    if cutlass.const_expr(self.r_const == 9):
                        if R == cutlass.Int32(4):
                            Q = (n4v + cutlass.Int32(3)) // cutlass.Int32(4)
                else:
                    Q = cutlass.Int32(0)
            # per-row TSH-floor runtime gate (CUDA parity: b>15 && k<=1024 in
            # tsh_en, n4 <= 32768 per row)
            tsh_run = cutlass.Int32(0)
            if tsh_en != cutlass.Int32(0):
                if (n >> cutlass.Int32(2)) <= cutlass.Int32(32768):
                    tsh_run = cutlass.Int32(1)

        # ---- shared memory (one blob, compile-time offsets) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)), byte_alignment=128
        )
        s_ws = smem.allocate_tensor(  # unused; byte parity  # noqa: F841
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        # crossing-scan result slots (RES_B/M/ABOVE/TOT/B2/B3)
        s_res = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2 [3]=s_base
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        s_pk = smem.allocate_tensor(
            cutlass.Int64, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=8
        )
        s_tsh = smem.allocate_tensor(
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
        )
        # STATIC smem word for the walk's byte stride — kept out of the blob
        # so dyn_bytes stays equal to the CUDA dispatch's smem. blk==512 VSTG
        # only.
        if cutlass.const_expr(self.vstg and self.blk == 512):
            s_x4 = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
            )
        s_kmm = smem.allocate_tensor(  # [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        if cutlass.const_expr(self.varlen):
            # ladder broadcast slots: [0]=SMP [1]=SS2 [2]=TGT [3]=TGT2
            # (static like s_x4, so dyn_bytes keeps CUDA dispatch parity)
            s_lad = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
            )
        # split broadcast: [0]=split flag [1]=need (0 => every survivor strict) [2]=strict total S
        s_cert = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        blob = smem.allocate_tensor(  # dynamic-equivalent region
            cutlass.Int8, cute.make_ordered_layout((self.dyn_bytes,), order=(0,)), byte_alignment=16
        )
        sbase = blob.iterator.toint()
        s_cbuf = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((SCPB + 4,)),
        )
        s_cbuf2 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((SCPB + 4,)),
        )
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((CMPB + 1,)),
        )

        if part < R:
            # emission smem bases pinned ONCE, outside the attempt/tile loops
            # (asm identity mov) — LLVM otherwise refolds the shared-window
            # materialisation into every _emitc site inside the divergent
            # bit-walk. VSTG-only: the VSTG=False tuples keep their original
            # spellings untouched (64-register wall).
            hb_pin = cutlass.Int32(0)
            cb2_pin = cutlass.Int32(0)
            if cutlass.const_expr(self.vstg):
                hb_pin = _smem_addr_reg(s_hist.iterator.toint())
                cb2_pin = _smem_addr_reg(s_cbuf2.iterator.toint())
            # park the stride 4 in the dedicated smem word and load it back —
            # the LDS result is opaque to ptxas, so the walk's stride register
            # cannot be re-materialized in-loop (asm-mov and shfl forms are
            # folded by ptxas value-tracking). blk==512 family ONLY: the other
            # arms sit at the 64-register wall. Threads are converged here
            # (kernel prologue), so the one extra barrier is safe.
            x4_pin = cutlass.Int32(4)
            if cutlass.const_expr(self.vstg and self.blk == 512):
                if tidx == cutlass.Int32(0):
                    s_x4[0] = cutlass.Int32(4)
                cute.arch.barrier()
                x4_pin = s_x4[0]

            # ---- row bases ----
            row64 = cutlass.Int64(row)
            # _pin_i64: keep the row base a REGISTER across the attempt/tile scf
            # regions (NVVM otherwise re-derives ld.param+%ctaid.y+mul per region)
            x_addr = _pin_i64(logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(4))
            # varlen: pre_idx is REQUEST-level [num_rows/next_n, k] — a request's
            # next_n rows share one hint row (production contract); legacy mode
            # keeps the per-row mapping (next_n == 1 makes them identical).
            prow64 = row64
            if cutlass.const_expr(self.varlen):
                prow64 = cutlass.Int64(row // cutlass.Int32(self.next_n))
            p_addr = pre_idx.iterator.toint() + prow64 * cutlass.Int64(k) * cutlass.Int64(4)
            out_row = out[row, None]
            # The producer has already validated every active physical page.
            tbase = row * pcols
            ws_addr = ws.iterator.toint()
            gdon_addr = ws_addr  # slab views
            goff_addr = ws_addr + cutlass.Int64(C.GVR_WS_OFF_OFF)
            gbuf_addr = ws_addr + cutlass.Int64(C.GVR_WS_BUF_OFF)
            # SPLIT only: row-slab base pinned like x_addr above; the
            # publish/gather/P5/degen consumers spell gbuf_row + i*8 instead of
            # re-deriving gbuf_addr + (row64*GCAP__main + i)*8 per candidate
            # (value-identical by i64 distributivity).
            gbuf_row = cutlass.Int64(0)
            if cutlass.const_expr(self.split):
                gbuf_row = _pin_i64(gbuf_addr + row64 * cutlass.Int64(GCAP__main) * cutlass.Int64(8))
            # ---- batched derivative row bases ----
            # hbase: this row's slice of the producer's int32[B,2048] global histogram.
            # pair_addr: this row's 8-byte (strict | slab) split cursor, in a region
            # appended AFTER the stock GVR2 workspace so [ws, ws+WS_BYTES) stays
            # byte-compatible with upstream (the canonical u32 g_off is stride 4 and a
            # u64 there would alias rows 2r / 2r+1).
            hbase = row * cutlass.Int32(1024)
            pair_addr = ws_addr + cutlass.Int64(WS_TAIL_OFF__batched) + row64 * cutlass.Int64(8)

            n4 = n >> cutlass.Int32(2)
            c0 = cutlass.Int32(0)
            c1 = n4
            if cutlass.const_expr(self.split):
                c0 = part * Q
                c1 = c0 + Q
                if c1 > n4:
                    c1 = n4
            tail0 = n4 << cutlass.Int32(2)
            tailn = cutlass.Int32(0)
            if part == cutlass.Int32(0):
                tailn = n - tail0

            if tidx == cutlass.Int32(0):
                s_scal[0] = cutlass.Int32(0)  # s_bufn
                s_scal[1] = cutlass.Int32(0)
                s_scal[2] = cutlass.Int32(0)
                s_res[C.RES_B2] = cutlass.Int32(-1)
                s_res[C.RES_B3] = cutlass.Int32(-1)
            if tidx < cutlass.Int32(self.hb):  # HB<=BLK always
                s_hist[tidx] = cutlass.Int32(0)

            # Exact certificate from the producer's global coarse histogram. The
            # candidate blob is free during the prologue and serves as scratch:
            # [0..NW) warp sums; [NW+0] crossing bin; [NW+1] strict; [NW+2] count;
            # [NW+3] total; [NW+4] producer-bad flag.
            if tidx == cutlass.Int32(0):
                s_cbuf[NW + 0] = cutlass.Int32(0x7FFFFFFF)
                s_cbuf[NW + 1] = cutlass.Int32(0)
                s_cbuf[NW + 2] = cutlass.Int32(0)
                s_cbuf[NW + 3] = cutlass.Int32(0)
                s_cbuf[NW + 4] = cutlass.Int32(0)
            h0 = cutlass.Int32(0)
            h1 = cutlass.Int32(0)
            if short == cutlass.Int32(0):
                h0 = histogram[hbase + tidx]
            hs = h0
            hincl = C.warp_incl_scan_add(hs, lane)
            cute.arch.barrier()  # scratch init published
            if lane == cutlass.Int32(31):
                s_cbuf[tidx >> cutlass.Int32(5)] = hincl
            if tidx < cutlass.Int32(148):
                if pdiag[tidx] != cutlass.Int32(0):
                    C.atomic_or_cta(s_cbuf.iterator + (NW + 4), cutlass.Int32(1))
            cute.arch.barrier()  # warp sums and diagnostic flag published
            # The preceding barrier publishes all NW=32 warp totals. Replicate
            # their scan within each warp; shuffle its exclusive prefix to all lanes.
            # Integer additions are exact: the whole row contains at most 1M scores.
            hwarp = tidx >> cutlass.Int32(5)
            warp_total = s_cbuf[lane]
            warp_exclusive = C.warp_incl_scan_add(warp_total, lane) - warp_total
            hbefore = cute.arch.shuffle_sync(warp_exclusive, hwarp)
            # Exact inclusive prefix; the crossing is the first bin whose inclusive
            # prefix exceeds k (strict prefix <= k): the frozen helper's semantics.
            run0 = hbefore + hincl - hs
            run1 = run0 + h0
            run2 = run1 + h1
            if short == cutlass.Int32(0):
                if run0 <= k and k < run1:
                    s_cbuf[NW + 0] = tidx
                    s_cbuf[NW + 1] = run0
                    s_cbuf[NW + 2] = h0
                elif run1 <= k and k < run2:
                    s_cbuf[NW + 0] = tidx * cutlass.Int32(2) + cutlass.Int32(1)
                    s_cbuf[NW + 1] = run1
                    s_cbuf[NW + 2] = h1
                if tidx == cutlass.Int32(BLK - 1):
                    s_cbuf[NW + 3] = run2
            cute.arch.barrier()  # crossing, total and flag published
            # Reuse s_lad only before original sampling, or exclusively on the fast path.
            if tidx == cutlass.Int32(0):
                cert_bin = s_cbuf[NW + 0]
                cert_strict = s_cbuf[NW + 1]
                cert_count = s_cbuf[NW + 2]
                cert_total = s_cbuf[NW + 3]
                cert_bad = s_cbuf[NW + 4]
                eligible = cutlass.Int32(0)
                split_flag = cutlass.Int32(0)
                split_need = cutlass.Int32(0)
                cert_lower = cutlass.Float32(0.0)
                cert_upper = cutlass.Float32(0.0)
                cert_expected = cutlass.Int32(0)
                if cert_bad == 0 and short == 0 and cert_total == n:
                    if cert_bin >= 0 and cert_bin <= 1023 and cert_strict >= 0 and cert_strict <= k:
                        cert_quota = k - cert_strict
                        if cert_count > 0 and cert_count <= n and cert_strict + cert_count > k and cert_strict + cert_count <= n:
                            cert_effective = cert_bin
                            cert_expected = cert_strict + cert_count
                            if cert_quota == 0:
                                cert_effective = cert_bin - cutlass.Int32(1)
                                cert_expected = cert_strict
                            cert_lower = coarse_floor(cert_effective)
                            cert_upper = coarse_floor(cert_effective - cutlass.Int32(1))
                            lower_abs = C.u32_of_f32(cert_lower) & cutlass.Uint32(0x7FFFFFFF)
                            upper_abs = C.u32_of_f32(cert_upper) & cutlass.Uint32(0x7FFFFFFF)
                            # `upper_abs > 0` refuses the +0 boundary. The coarse key splits the two
                            # zeros across bins (to_coarse_key: +0.0 -> 1023, -0.0 -> 1024), and
                            # coarse_floor(1023) is exactly +0.0, so a crossing bin of 1024 yields
                            # HIC == +0.0 while cert_lower stays nonzero and passes its own guard.
                            # The P3 filter classifies with a bare `>=`, and -0.0 >= +0.0 is TRUE, so
                            # every -0.0 would be published as STRICT although the certificate places
                            # bin 1024 AT the crossing. That inflates the measured strict cursor, drives
                            # ksc = k - soff to <= 0, makes scan_cross0 record no crossing (its test is
                            # `after < target`), leaves RES_B/ABOVE/M at stale shared values, and the
                            # accept test `RES_TOT >= ksc` is then vacuously true -- so P5 writes
                            # out_row[soff + p] past column k. Refusing the +0 boundary sends the frame
                            # to the frozen sampling ladder instead: correct, and never reached in any
                            # measured frame (cutoff bins observed 438-556 / 1664-1733 over 64 frames).
                            # Same signed-zero hazard as any FP32 boundary predicate:
                            # the +0 boundary admits +0 and rejects -0.
                            if lower_abs > 0 and upper_abs > 0 and lower_abs < cutlass.Uint32(0x7F800000) and upper_abs < cutlass.Uint32(0x7F800000):
                                if cert_upper > cert_lower and cert_expected <= cutlass.Int32(GCAP__main):
                                    eligible = cutlass.Int32(1)
                                    if cert_expected <= cutlass.Int32(SCPB) and cert_quota > 0:
                                        split_flag = cutlass.Int32(1)
                                        split_need = cert_quota
                s_lad[0] = eligible
                s_lad[1] = C.i32_of_f32(cert_lower)
                s_lad[2] = C.i32_of_f32(cert_upper)
                s_lad[3] = cert_expected
                s_cert[0] = split_flag
                s_cert[1] = split_need
                s_cert[2] = cutlass.Int32(0)
            cute.arch.barrier()  # also publishes original local counter/histogram init
            cert_fast = s_lad[0]
            injected_T = C.f32_of_i32(s_lad[1])
            injected_HIC = C.f32_of_i32(s_lad[2])
            expected_pass = s_lad[3]
            split_mode = s_cert[0]
            need_eff = s_cert[1]
            HICs = injected_HIC
            if need_eff == cutlass.Int32(0):
                HICs = injected_T
            soff = cutlass.Int32(0)
            ksc = k
            # Every warp must finish reading before fallback warp0 reuses s_lad.
            cute.arch.barrier()
            if cert_fast != cutlass.Int32(0):
                # Sampling is skipped. This sentinel preserves the existing PRIME-LATE
                # L2 prefetch gate for long slices, not a real sample population.
                SMP = cutlass.Int32(160)

            # ===== varlen: warp0-only ladder mirror + register-free L2 hints =====
            # The sampling-ladder scalars are a pure function of the row; issuing
            # the mirror chain (runtime divides + isqrt fixups) per thread costs
            # more instructions than the rest of the kernel on 1-row launches.
            # warp0 alone walks the chain and publishes the four derived scalars
            # through s_lad; the other warps spend the wait issuing L2 prefetch
            # hints for this CTA's own P3 slice (register-free, so zero pressure
            # on the 64-register arms — the PRIME-LATE register loads below are
            # untouched and simply hit L2).  Values are bit-identical to the
            # per-thread derivation this replaces.
            if cutlass.const_expr(self.varlen):
                if tidx < cutlass.Int32(32) and cert_fast == cutlass.Int32(0):
                    if short == cutlass.Int32(0):
                        # ---- aim ladder (cheap mirror) ----
                        # The ladder scalars steer the sampling rung only —
                        # exactness is schedule-invariant (retry/degen close every
                        # miss), so +-1 drift vs the host double form is allowed.
                        # Serial latency dominates (this chain sits in front of a
                        # barrier): runtime divides become MUFU.RCP multiplies and
                        # the isqrt fixup loops collapse to single steps (the f32
                        # sqrt of an exactly-representable int (6n <= 2^23) is
                        # within 1 of isqrt, so one correction per side suffices).
                        # Q (chunk ownership) stays exact — compile-time divisor.
                        x6 = cutlass.Int32(6) * n
                        ri = cutlass.Int32(cmath.sqrt(cutlass.Float32(x6)))
                        if ri * ri > x6:
                            ri = ri - cutlass.Int32(1)
                        if (ri + cutlass.Int32(1)) * (ri + cutlass.Int32(1)) <= x6:
                            ri = ri + cutlass.Int32(1)
                        r6 = ri
                        if x6 - ri * ri > ri:
                            r6 = ri + cutlass.Int32(1)
                        aim = aim_base
                        if r6 > aim:
                            aim = r6
                        if cutlass.const_expr(self.r_const > 1):
                            if aim < amin:
                                aim = amin
                        scap_c = cutlass.Int32(SCPB)  # SCAP == SCPB for gvr_main (proven identity)
                        if aim > (scap_c >> cutlass.Int32(1)):
                            aim = scap_c >> cutlass.Int32(1)
                        if aim < k:
                            aim = k
                        n4w = n >> cutlass.Int32(2)
                        # pair-sample gate: (n > SCAP or small_dense) and n4 >= 4;
                        # small_dense = k > 1024 and not big and n <= SCAP and n > 2k
                        # (k/big folded into the launch-constant sd_en flag).
                        gate = cutlass.Int32(0)
                        if n > scap_c:
                            gate = cutlass.Int32(1)
                        if sd_en != cutlass.Int32(0):
                            if n <= scap_c:
                                if n > (k << cutlass.Int32(1)):
                                    gate = cutlass.Int32(1)
                        if n4w < cutlass.Int32(4):
                            gate = cutlass.Int32(0)
                        if gate != cutlass.Int32(0):
                            # sel = sfac*n // aim via rcp (sfac*n <= 2^24: f32-exact
                            # to the last unit; quotient error < 1 => +-1 drift)
                            sel = cutlass.Int32(
                                cutlass.Float32(sfac * n) * cute.arch.rcp_approx(cutlass.Float32(aim))
                            )
                            if sel < cutlass.Int32(256):
                                sel = cutlass.Int32(256)
                            nh = n >> cutlass.Int32(1)
                            if sel > nh:
                                sel = nh
                            pairs = sel >> cutlass.Int32(3)
                            if pairs < cutlass.Int32(1):
                                pairs = cutlass.Int32(1)
                            half = n4w >> cutlass.Int32(1)
                            if half < cutlass.Int32(1):
                                half = cutlass.Int32(1)
                            if pairs > half:
                                pairs = half
                            SS2 = cutlass.Int32(
                                cutlass.Float32(half) * cute.arch.rcp_approx(cutlass.Float32(pairs))
                            )
                            if SS2 < cutlass.Int32(1):
                                SS2 = cutlass.Int32(1)
                            SMP = cutlass.Int32(
                                cutlass.Float32(half) * cute.arch.rcp_approx(cutlass.Float32(SS2))
                            )
                            # sample-window guard: the P1 gather indexes up to
                            # ~SMP*SS2*2 f32x4 lines; keep SMP*SS2 <= half so the
                            # window never walks past the row (approx error is
                            # bounded by +1, one decrement closes it)
                            if SMP * SS2 > half:
                                SMP = SMP - cutlass.Int32(1)
                            if SMP < cutlass.Int32(1):
                                SMP = cutlass.Int32(1)
                            # TGT/TGT2: i64 products // n -> f32 mul + one rcp(n).
                            # aim/SMP/k/n are all f32-exact here (<= 2^20); the
                            # quotients are <= 8*aim ~ 2^16, so the approx error
                            # stays far below 1 unit — +-1 at worst on the floor.
                            rn_ = cute.arch.rcp_approx(cutlass.Float32(n))
                            smp8f = cutlass.Float32(SMP) * cutlass.Float32(8.0)
                            TGT = cutlass.Int32(cutlass.Float32(aim) * smp8f * rn_)
                            if TGT < cutlass.Int32(1):
                                TGT = cutlass.Int32(1)
                            TGT2 = cutlass.Int32(cutlass.Float32(k) * smp8f * rn_)
                            if TGT2 < cutlass.Int32(1):
                                TGT2 = cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_lad[0] = SMP
                        s_lad[1] = SS2
                        s_lad[2] = TGT
                        s_lad[3] = TGT2
                # Register-free L2 hints for the first U-batch of this CTA's own
                # P3 slice (clamped in-row): the data P3 touches first starts
                # flowing while warp0 walks the chain. Short rows clamp every
                # hint to the row's last line — harmless.
                plim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
                for uu in cutlass.range_constexpr(U):
                    # NOTE: names must not collide with the PRIME-LATE block's
                    # i_/ic — the DSL kills inner-scope names at region exit and
                    # a later same-name assignment inside a dynamic `if` trips
                    # "is None prior to this if".
                    pic = c0 + tidx + cutlass.Int32(uu * BLK)
                    if pic >= c1:
                        pic = plim4
                    C._prefetch_l2(x_addr + cutlass.Int64(pic) * cutlass.Int64(16))
                cute.arch.barrier()  # publish s_lad (also covers the smem inits)
                if cert_fast == cutlass.Int32(0):
                    SMP = s_lad[0]
                    SS2 = s_lad[1]
                    TGT = s_lad[2]
                    TGT2 = s_lad[3]

            # ============ P1: sample prefetch (hint gather LAZY) =================
            atom128 = C.g2r_atom_f32(128, invariant=True)
            fsa = cute.make_rmem_tensor((4,), cutlass.Float32)
            fsb = cute.make_rmem_tensor((4,), cutlass.Float32)
            fma_ = cute.make_rmem_tensor((4,), cutlass.Float32)  # strided-tail pair bufs
            fmb_ = cute.make_rmem_tensor((4,), cutlass.Float32)
            shas = cutlass.Int32(0)
            if cert_fast == cutlass.Int32(0):
                if tidx < SMP:
                    shas = cutlass.Int32(1)
                if shas != cutlass.Int32(0):
                    p4 = tidx * SS2 * cutlass.Int32(2)
                    C.ld_g_f32x4(atom128, x_addr, p4, fsa)
                    C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)

                # ============ P2: quantile rung from the sample ======================
                smn = cutlass.Float32(float("inf"))
                smx = cutlass.Float32(float("-inf"))
                if shas != cutlass.Int32(0):
                    for t in cutlass.range_constexpr(4):
                        smn = C.fmin_f32(smn, fsa[t])
                        smx = C.fmax_f32(smx, fsa[t])
                    for t in cutlass.range_constexpr(4):
                        smn = C.fmin_f32(smn, fsb[t])
                        smx = C.fmax_f32(smx, fsb[t])
                j = tidx + cutlass.Int32(BLK)  # strided tail
                while j < SMP:
                    p4 = j * SS2 * cutlass.Int32(2)
                    C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                    C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                    for t in cutlass.range_constexpr(4):
                        smn = C.fmin_f32(smn, fma_[t])
                        smx = C.fmax_f32(smx, fma_[t])
                    for t in cutlass.range_constexpr(4):
                        smn = C.fmin_f32(smn, fmb_[t])
                        smx = C.fmax_f32(smx, fmb_[t])
                    j = j + cutlass.Int32(BLK)
                a0 = C.warp_min_u32(C.fkey(smn))
                c0m = C.warp_max_u32(C.fkey(smx))
                if lane == cutlass.Int32(0):
                    s_wmn[tidx >> cutlass.Int32(5)] = a0
                    s_wmx[tidx >> cutlass.Int32(5)] = c0m
                cute.arch.barrier()  # ---- barrier (sample redux publish) ----

            # PRIME-LATE prefetch block: strictly after the barrier.
            lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
            pf = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(PFD, 1))]
            if cutlass.const_expr(self.pf):
                fullsl = cutlass.Int32(0)
                if (c1 - c0) >= cutlass.Int32(BLK * U):
                    fullsl = cutlass.Int32(1)
                if fullsl != cutlass.Int32(0):  # prime, full slice
                    for uu in cutlass.range_constexpr(PFD):
                        C.ld_g_f32x4(atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu])
                else:  # clamped prime
                    for uu in cutlass.range_constexpr(PFD):
                        i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4
                        C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                # asm prefetch site #1: gate (c1-c0)>=2*BLK*U && SMP>=160
                g1 = cutlass.Int32(0)
                if (c1 - c0) >= cutlass.Int32(2 * BLK * U):
                    if SMP >= cutlass.Int32(160):
                        g1 = cutlass.Int32(1)
                if g1 != cutlass.Int32(0):
                    for uu in cutlass.range_constexpr(PFD, U):
                        C._prefetch_l2(
                            x_addr
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
                        )
            if cutlass.const_expr((not self.pf) and (not self.split)):
                fullsl = cutlass.Int32(0)
                if (c1 - c0) >= cutlass.Int32(BLK * U):
                    fullsl = cutlass.Int32(1)
                if fullsl != cutlass.Int32(0):  # prefetch site #2
                    for uu in cutlass.range_constexpr(U):
                        C._prefetch_l2(
                            x_addr
                            + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
                        )
                else:
                    if SMP > cutlass.Int32(0):  # prefetch site #3
                        for uu in cutlass.range_constexpr(U):
                            i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4
                            C._prefetch_l2(x_addr + cutlass.Int64(ic) * cutlass.Int64(16))

            T = injected_T
            HIC = injected_HIC
            GMIN = cutlass.Float32(C.SENT_LO)
            GMAX = cutlass.Float32(C.SENT_HI)
            if cert_fast == cutlass.Int32(0):
                # cross-warp sample reduce
                av = cutlass.Uint32(0xFFFFFFFF)
                cv = cutlass.Uint32(0)
                if lane < cutlass.Int32(NW):
                    av = s_wmn[lane]
                    cv = s_wmx[lane]
                SMIN = C.invkey(C.warp_min_u32(av))
                SMAX = C.invkey(C.warp_max_u32(cv))

                GMIN = cutlass.Float32(C.SENT_LO)  # sentinels
                GMAX = cutlass.Float32(C.SENT_HI)
                T = cutlass.Float32(_NEG_INF)
                HIC = cutlass.Float32(_NEG_INF)
                w = cutlass.Float32(0.0)
                sok = cutlass.Int32(0)
                if SMP > cutlass.Int32(0):
                    if SMAX > SMIN:
                        sok = cutlass.Int32(1)
                if sok != cutlass.Int32(0):  # sample histogram
                    w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
                    # rcp.approx.ftz.f32 = the CUDA arm's --use_fast_math 1.0f/w
                    # (bare MUFU.RCP, no Newton refinement) — bitwise-aligned scale
                    sc_s = cute.arch.rcp_approx(w)
                    if shas != cutlass.Int32(0):
                        for t in cutlass.range_constexpr(4):
                            bq = C.f2s_rz((fsa[t] - SMIN) * sc_s)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                        for t in cutlass.range_constexpr(4):
                            bq = C.f2s_rz((fsb[t] - SMIN) * sc_s)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    j = tidx + cutlass.Int32(BLK)  # tail re-loads
                    while j < SMP:
                        p4 = j * SS2 * cutlass.Int32(2)
                        C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                        C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                        for t in cutlass.range_constexpr(4):
                            bq = C.f2s_rz((fma_[t] - SMIN) * sc_s)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                        for t in cutlass.range_constexpr(4):
                            bq = C.f2s_rz((fmb_[t] - SMIN) * sc_s)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                        j = j + cutlass.Int32(BLK)
                cute.arch.barrier()  # ---- barrier (sample histogram) ----
                # triple-target ZERO scan: TGT / TGT2 / 2*TGT
                # (THREE = SHD || gated-SPLIT)
                C.scan_cross0(
                    s_hist,
                    TGT,
                    tidx,
                    s_res,
                    TGT2,
                    TGT * cutlass.Int32(2),
                    s_hist,
                    nb=NBS,
                    zero=True,
                    two=True,
                    three=(self.shd or self.tshg),
                )
                cute.arch.barrier()  # ---- barrier (scan publish) ----

                tot0 = s_res[C.RES_TOT]
                b1v = s_res[C.RES_B]
                if sok != cutlass.Int32(0):
                    if tot0 >= TGT:
                        T = _fmaf(cutlass.Float32(b1v), w, SMIN)
                Trung = T  # snapshot
                needg = cutlass.Int32(1)  # degenerate sample
                if T > cutlass.Float32(_NEG_INF):
                    needg = cutlass.Int32(0)
                if needg != cutlass.Int32(0):
                    GMIN, GMAX = C.gather_hint(
                        x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=KPT
                    )  # 2 barriers inside
                    T = GMIN
                if sok != cutlass.Int32(0):  # HIC tighten
                    if tot0 >= TGT:
                        b2v = s_res[C.RES_B2]
                        if b2v >= cutlass.Int32(0):
                            Tk = _fmaf(cutlass.Float32(b2v), w, SMIN)
                            anch = T
                            if cutlass.const_expr(not self.split):
                                anch = C.fmin_f32(T, Trung)
                            d_ = C.fmax_f32(Tk - anch, cutlass.Float32(0.0))
                            HIC = C.fmax_f32(
                                _fmaf(cutlass.Float32(4.0), d_, T), _fmaf(cutlass.Float32(8.0), w, T)
                            )
                if cutlass.const_expr(self.shd or self.tshg):  # TSH floor (+gated SPLIT)
                    if tidx == cutlass.Int32(0):
                        t5 = cutlass.Float32(_NEG_INF)
                        if sok != cutlass.Int32(0):
                            if tot0 >= TGT * cutlass.Int32(2):
                                b3v = s_res[C.RES_B3]
                                if b3v >= cutlass.Int32(0):
                                    if T > GMIN:
                                        T3 = _fmaf(cutlass.Float32(b3v), w, SMIN)
                                        if T3 < T:
                                            t5 = T3
                        s_tsh[0] = t5

                if cutlass.const_expr(self.tshg):
                    # TSH-FLOOR STAGING: SPLIT has no retry ladder, so a rung
                    # overshoot (count(>=T) < k) used to hand the LAST CTA a
                    # single-CTA whole-row narrowing.  Stage at the sample's
                    # rank-(2*TGT) floor instead: staged population ~aim -> ~2*aim,
                    # and the merged histogram contains the k-crossing whenever
                    # count(>=TSH) >= k.  TSH miss falls to GMIN/degen unchanged.
                    cute.arch.barrier()
                    t5s = s_tsh[0]
                    # varlen: per-row runtime gate (tsh_run == 1 always in legacy
                    # mode, so legacy codegen semantics are unchanged)
                    if tsh_run != cutlass.Int32(0):
                        if t5s > cutlass.Float32(_NEG_INF):
                            if t5s < T:
                                T = t5s

            if tidx == cutlass.Int32(0):
                drow = row * cutlass.Int32(self.r_const) + part
                diagnostics[drow, 0] = cert_fast
                diagnostics[drow, 1] = n_row
                diagnostics[drow, 2] = expected_pass if cert_fast != 0 else cutlass.Int32(-1)
                diagnostics[drow, 3] = C.i32_of_f32(T)
                diagnostics[drow, 4] = C.i32_of_f32(HIC)
                diagnostics[drow, 5] = split_mode

            # ============ attempt loop — MUST NOT unroll ============
            listN = cutlass.Int32(0)
            above = cutlass.Int32(0)
            m = cutlass.Int32(0)
            need = cutlass.Int32(0)
            B = cutlass.Int32(0)
            SC = cutlass.Float32(1.0)
            TF = T
            complete = cutlass.Int32(0)
            valid = cutlass.Int32(0)
            fromg = cutlass.Int32(0)
            alive = cutlass.Int32(1)

            fr = [
                cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(U - PFD, 1))
            ]  # explicit batch
            att = cutlass.Int32(0)
            running = cutlass.Int32(1)
            while running != cutlass.Int32(0):
                if cutlass.const_expr(not self.split):  # SPLIT never retries (NATT=1)
                    if att > cutlass.Int32(0):  # retry reset
                        if cutlass.const_expr(self.pf):
                            # exactness: re-prime pf[] (holds stale roll data)
                            fullsl = cutlass.Int32(0)
                            if (c1 - c0) >= cutlass.Int32(BLK * U):
                                fullsl = cutlass.Int32(1)
                            if fullsl != cutlass.Int32(0):
                                for uu in cutlass.range_constexpr(PFD):
                                    C.ld_g_f32x4(
                                        atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                                    ic = i_
                                    if ic >= c1:
                                        ic = lim4
                                    C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                        if tidx < cutlass.Int32(NBS):
                            s_hist[tidx] = cutlass.Int32(0)
                        if tidx == cutlass.Int32(0):
                            s_scal[0] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (retry reset) ----

                TF = T  # window
                hi = C.fmax_f32(GMAX, T)
                if HIC > T:
                    if HIC < hi:
                        hi = HIC
                WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
                wdok = cutlass.Int32(0)
                if WD > cutlass.Float32(0.0):
                    wdok = cutlass.Int32(1)
                if wdok == cutlass.Int32(0):
                    WD = cutlass.Float32(1e-30)
                # CUDA compiles its own `1.0f / WD` here to a bare MUFU.RCP
                # (approximate); div.rn's dependent rcp+Newton+CALL chain
                # serializes the attempt prologue. blk==512 ONLY: the
                # (256,8,4,·) family keeps the original div.rn spelling below.
                if cutlass.const_expr(self.blk == 512):
                    SC = cute.arch.rcp_approx(WD)
                else:
                    SC = cutlass.Float32(1.0) / WD

                # ---- P3 row pass ----
                span = c1 - c0
                step = cutlass.Int32(BLK * U)
                nFull = cutlass.Int32(0)
                rem = cutlass.Int32(0)
                if span > cutlass.Int32(0):  # peel
                    nFull = span // step
                    rem = span - nFull * step
                # _pin_i32: the isfull peel predicate reads nFull every tile iter;
                # unpinned, NVVM re-derives the whole ld.param+shr/sel div chain
                # at the loop head
                nFull = _pin_i32(nFull)
                nIt = nFull
                if rem > cutlass.Int32(0):
                    nIt = nIt + cutlass.Int32(1)
                # _pin_i32: stop NVVM re-deriving the ceil-div bound (ld.param n +
                # shr/sel chain) inside the tile-loop condition region per iter
                nIt = _pin_i32(nIt)

                it = cutlass.Int32(0)
                while it < nIt:
                    i0 = c0 + it * step + tidx
                    M = cutlass.Int32(0)
                    isfull = cutlass.Int32(0)
                    if it < nFull:
                        isfull = cutlass.Int32(1)
                    if isfull != cutlass.Int32(0):  # full body
                        for uu in cutlass.range_constexpr(PFD, U):
                            C.ld_g_f32x4(atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD])
                        for uu in cutlass.range_constexpr(U):
                            if cutlass.const_expr(uu < PFD):
                                vv = pf[uu]
                            else:
                                vv = fr[uu - PFD]
                            for q in cutlass.range_constexpr(4):
                                M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                    else:  # partial body
                        for uu in cutlass.range_constexpr(PFD, U):
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= c1:
                                ic = lim4  # clamped address
                            C.ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                        for uu in cutlass.range_constexpr(U):
                            if cutlass.const_expr(uu < PFD):
                                vv = pf[uu]
                            else:
                                vv = fr[uu - PFD]
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            okq = cutlass.Int32(0)
                            if i_ < c1:
                                okq = cutlass.Int32(1)
                            if okq != cutlass.Int32(0):  # ok-gated (+inf-pad escape)
                                for q in cutlass.range_constexpr(4):
                                    M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                    # prefetch roll-forward BEFORE reservation/walk
                    if cutlass.const_expr(self.pf):
                        hasnext = cutlass.Int32(0)
                        if it + cutlass.Int32(1) < nIt:
                            hasnext = cutlass.Int32(1)
                        if hasnext != cutlass.Int32(0):
                            j0 = i0 + step
                            infull = cutlass.Int32(0)  # warp-uniform peel
                            if it + cutlass.Int32(1) < nFull:
                                infull = cutlass.Int32(1)
                            if infull != cutlass.Int32(0):
                                for uu in cutlass.range_constexpr(PFD):
                                    C.ld_g_f32x4(atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu])
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    j_ = j0 + cutlass.Int32(uu * BLK)
                                    jc = j_
                                    if jc >= c1:
                                        jc = lim4
                                    C.ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                    # warp-aggregated reservation
                    cnt = cutlass.Int32(C.popc(M))
                    inc = C.warp_incl_scan_add(cnt, lane)
                    bpos = cutlass.Int32(0)
                    if lane == cutlass.Int32(31):
                        if inc != cutlass.Int32(0):
                            bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
                    pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                    # survivor bit-walk, software-pipelined ONE deep;
                    # reload X[idx] — do NOT hold the U float4s (spills)
                    if M != cutlass.Int32(0):
                        bp = C.ffs_m1(M)
                        M = M & (M - cutlass.Int32(1))
                        idx = (
                            (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK)) << cutlass.Int32(2)
                        ) + (bp & cutlass.Int32(3))
                        if cutlass.const_expr(self.vstg and self.blk == 512):
                            xv = _ldg_f32_rs(x_addr, idx, x4_pin)
                        else:
                            xv = C.ldg_f32(x_addr, idx)
                        while M != cutlass.Int32(0):
                            bp2 = C.ffs_m1(M)
                            M = M & (M - cutlass.Int32(1))
                            idx2 = (
                                (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                << cutlass.Int32(2)
                            ) + (bp2 & cutlass.Int32(3))
                            if cutlass.const_expr(self.vstg and self.blk == 512):
                                xv2 = _ldg_f32_rs(x_addr, idx2, x4_pin)
                            else:
                                xv2 = C.ldg_f32(x_addr, idx2)
                            pos = self._emitc(
                                xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                            )
                            idx = idx2
                            xv = xv2
                        pos = self._emitc(
                            xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                        )
                    it = it + cutlass.Int32(1)
                # scalar tail, part 0 only
                i = tidx
                while i < tailn:
                    x = C.ldg_f32(x_addr, tail0 + i)
                    if x >= TF:
                        post = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                        post = self._emitc(
                            x, tail0 + i, post, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                        )
                    i = i + cutlass.Int32(BLK)
                cute.arch.barrier()  # ---- barrier (row pass) ----
                myn = s_scal[0]

                if cutlass.const_expr(self.split):
                    pubn = myn
                    if split_mode != cutlass.Int32(0):
                        # ---- split publication: one u64 RMW per warp reserves strict (high) and slab (low) ----
                        pgo64 = cute.make_ptr(cutlass.Int64, pair_addr, cute.AddressSpace.gmem, assumed_align=8)
                        it2 = (myn + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            w64 = cutlass.Uint64(0)
                            idv = cutlass.Int32(0)
                            if i < myn:
                                w64 = cutlass.Uint64(s_cbuf2[i])
                                xb = cutlass.Int32(cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF)))
                                idv = cutlass.Int32(cutlass.Uint32(w64 >> cutlass.Uint64(32)))
                                if C.f32_of_i32(xb) >= HICs:
                                    p1 = cutlass.Int32(1)
                                else:
                                    p2 = cutlass.Int32(1)
                            n1 = C.ballot(p1 != cutlass.Int32(0))
                            n2 = C.ballot(p2 != cutlass.Int32(0))
                            bhi = cutlass.Int32(0)
                            blo = cutlass.Int32(0)
                            if lane == cutlass.Int32(0):
                                if (n1 | n2) != cutlass.Int32(0):
                                    oldv = cutlass.Int64(cute.arch.atomic_add(
                                        pgo64,
                                        (cutlass.Int64(C.popc(n1)) << cutlass.Int64(32)) + cutlass.Int64(C.popc(n2)),
                                    ))
                                    bhi = cutlass.Int32(oldv >> cutlass.Int64(32))
                                    blo = cutlass.Int32(oldv & cutlass.Int64(0xFFFFFFFF))
                                    if n2 != cutlass.Int32(0):
                                        C.atomic_add_cta(s_scal.iterator + 2, cutlass.Int32(C.popc(n2)))
                            bhi = cute.arch.shuffle_sync(bhi, cutlass.Int32(0))
                            blo = cute.arch.shuffle_sync(blo, cutlass.Int32(0))
                            lm = cutlass.Int32(cute.arch.lanemask_lt())
                            if p1 != cutlass.Int32(0):
                                p = bhi + cutlass.Int32(C.popc(n1 & lm))
                                if p < k:
                                    out_row[p] = ptable[tbase + (idv >> cutlass.Int32(6))] * cutlass.Int32(64) + (idv & cutlass.Int32(63))
                            if p2 != cutlass.Int32(0):
                                p = blo + cutlass.Int32(C.popc(n2 & lm))
                                if p < cutlass.Int32(GCAP__main):
                                    _st_g_u64(gbuf_row + cutlass.Int64(p) * cutlass.Int64(8), w64)
                            it = it + cutlass.Int32(1)
                    else:
                        # ---- SLAB HAND-OFF; exactly ONE attempt ----
                        if tidx == cutlass.Int32(0):
                            pgo = cute.make_ptr(
                                cutlass.Int32,
                                goff_addr + row64 * cutlass.Int64(4),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            s_scal[3] = cutlass.Int32(cute.arch.atomic_add(pgo, myn))
                        cute.arch.barrier()  # ---- barrier (slab offset) ----
                        base = s_scal[3]
                        if myn <= cutlass.Int32(SCPB):  # coalesced publish
                            i = tidx
                            while i < myn:
                                p = base + i
                                if p < cutlass.Int32(GCAP__main):
                                    _st_g_u64(gbuf_row + cutlass.Int64(p) * cutlass.Int64(8), s_cbuf2[i])
                                i = i + cutlass.Int32(BLK)
                        else:  # overflow re-sweep
                            if tidx == cutlass.Int32(0):
                                s_scal[0] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (overflow reset) ----
                            lo2 = c0 << cutlass.Int32(2)
                            hi2 = c1 << cutlass.Int32(2)
                            i = lo2 + tidx
                            while i < hi2:
                                x = C.ldg_f32(x_addr, i)
                                if x >= TF:
                                    pq = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                                    p = base + pq
                                    if p < cutlass.Int32(GCAP__main):
                                        _st_g_u64(
                                            gbuf_row + cutlass.Int64(p) * cutlass.Int64(8),
                                            (cutlass.Uint64(cutlass.Uint32(i)) << cutlass.Uint64(32))
                                            | cutlass.Uint64(C.u32_of_f32(x)),
                                        )
                                i = i + cutlass.Int32(BLK)
                            i = tidx  # true tail
                            while i < tailn:
                                x = C.ldg_f32(x_addr, tail0 + i)
                                if x >= TF:
                                    pq = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                                    p = base + pq
                                    if p < cutlass.Int32(GCAP__main):
                                        _st_g_u64(
                                            gbuf_row + cutlass.Int64(p) * cutlass.Int64(8),
                                            (
                                                cutlass.Uint64(cutlass.Uint32(tail0 + i))
                                                << cutlass.Uint64(32)
                                            )
                                            | cutlass.Uint64(C.u32_of_f32(x)),
                                        )
                                i = i + cutlass.Int32(BLK)
                    cute.arch.barrier()  # ---- barrier (slab publish) ----
                    if split_mode != cutlass.Int32(0):
                        pubn = s_scal[2]
                    if tidx == cutlass.Int32(0):  # acq_rel RMW, cumulative through the CTA barrier
                        pdon = cute.make_ptr(
                            cutlass.Int64,
                            gdon_addr + row64 * cutlass.Int64(8),
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        s_pk[0] = cutlass.Int64(cute.arch.atomic_add(
                            pdon, cutlass.Int64(1 << 32) + cutlass.Int64(pubn), sem="acq_rel", scope="gpu"
                        ))
                    cute.arch.barrier()  # ---- barrier (arrival word) ----
                    pk = s_pk[0]
                    alive = cutlass.Int32(0)  # last-CTA test
                    if cutlass.Int32(pk >> cutlass.Int64(32)) == R - cutlass.Int32(1):
                        alive = cutlass.Int32(1)
                    if alive != cutlass.Int32(0):
                        if tidx == cutlass.Int32(0):  # ZERO-RESTORE, per row
                            # Strict count S is the high half of this ROW's tail pair.
                            s_cert[2] = C.ld_g_i32(pair_addr, cutlass.Int32(1))
                            _st_g_u64(pair_addr, cutlass.Uint64(0))
                            # The canonical u32 g_off is what the non-split fallback
                            # branch increments; a mixed batch can have one row on each
                            # path, so restore this row's word unconditionally.
                            _st_g_u32(goff_addr + row64 * cutlass.Int64(4), cutlass.Int32(0))
                            _st_g_u64(gdon_addr + row64 * cutlass.Int64(8), cutlass.Uint64(0))
                        if hrestore != cutlass.Int32(0):
                            # Every CTA of this row read its 2048 bins in the prologue
                            # before arriving; the last arrival restores only this row.
                            hz = tidx
                            while hz < cutlass.Int32(1024):
                                histogram[hbase + hz] = cutlass.Int32(0)
                                hz = hz + cutlass.Int32(BLK)
                        total = cutlass.Int32(pk & cutlass.Int64(0xFFFFFFFF)) + pubn
                        if total <= cutlass.Int32(GCAP__main):  # one-pass consume
                            listN = total
                            if total > cutlass.Int32(SCPB):
                                fromg = cutlass.Int32(1)
                            i = tidx
                            while i < listN:
                                gvx, gvy = C._ldcg_v2_i32(
                                    gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                )
                                if fromg == cutlass.Int32(0):
                                    s_cbuf2[i] = (
                                        cutlass.Uint64(cutlass.Uint32(gvy)) << cutlass.Uint64(32)
                                    ) | cutlass.Uint64(cutlass.Uint32(gvx))
                                bq = C.f2s_rz((C.f32_of_i32(gvx) - TF) * SC)
                                if bq > cutlass.Int32(NBS - 1):
                                    bq = cutlass.Int32(NBS - 1)
                                # resultless red off the pinned hist base
                                _red_shared_add1(hb_pin + (bq << cutlass.Int32(2)))
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (slab histogram) ----
                            if split_mode != cutlass.Int32(0):
                                soff = s_cert[2]
                                ksc = k - soff
                            C.scan_cross0(
                                s_hist,
                                ksc,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=False,
                            )
                            cute.arch.barrier()  # ---- barrier (scan publish) ----
                            if s_res[C.RES_TOT] >= ksc:
                                valid = cutlass.Int32(1)
                                complete = cutlass.Int32(1)
                                above = s_res[C.RES_ABOVE]
                                m = s_res[C.RES_M]
                                need = ksc - above
                                B = s_res[C.RES_B]
                    running = cutlass.Int32(0)  # break (NATT==1)
                else:
                    # ---- non-split verify + rung ladder ----
                    C.scan_cross0(
                        s_hist,
                        k,
                        tidx,
                        s_res,
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        s_hist,
                        nb=NBS,
                        zero=False,
                    )
                    cute.arch.barrier()  # ---- barrier (verify scan) ----
                    tot = s_res[C.RES_TOT]
                    acc = cutlass.Int32(0)
                    if tot >= k:
                        acc = cutlass.Int32(1)
                    if acc != cutlass.Int32(0):  # accept
                        valid = cutlass.Int32(1)
                        complete = cutlass.Int32(0)
                        if myn <= cutlass.Int32(SCPB):
                            complete = cutlass.Int32(1)
                        listN = myn
                        above = s_res[C.RES_ABOVE]
                        m = s_res[C.RES_M]
                        need = k - above
                        B = s_res[C.RES_B]
                        running = cutlass.Int32(0)
                    else:
                        if att == cutlass.Int32(NATT - 1):  # ladder exhausted
                            running = cutlass.Int32(0)
                        else:
                            tshtaken = cutlass.Int32(0)  # TSH retry
                            if cutlass.const_expr(self.shd):
                                if att == cutlass.Int32(0):
                                    T5 = s_tsh[0]
                                    if T5 > cutlass.Float32(_NEG_INF):
                                        if T5 < TF:
                                            T = T5
                                            tshtaken = cutlass.Int32(1)
                            if tshtaken != cutlass.Int32(0):
                                cute.arch.barrier()  # ---- barrier (TSH retry) ----
                            else:
                                # LAZY GATHER (sentinel equality flag)
                                if GMIN == cutlass.Float32(C.SENT_LO):
                                    GMIN, GMAX = C.gather_hint(
                                        x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=KPT
                                    )
                                floorhit = cutlass.Int32(1)
                                if T > GMIN:
                                    floorhit = cutlass.Int32(0)
                                if floorhit != cutlass.Int32(0):
                                    running = cutlass.Int32(0)
                                else:
                                    T = GMIN
                                    cute.arch.barrier()  # ---- barrier (floor retry) ----
                att = att + cutlass.Int32(1)

            # ============ classification ============
            if alive != cutlass.Int32(0):
                whole = cutlass.Int32(0)
                if valid != cutlass.Int32(0):
                    if need >= m:
                        whole = cutlass.Int32(1)
                lim1 = above
                if whole != cutlass.Int32(0):
                    lim1 = above + m
                degen = cutlass.Int32(0)
                if valid == cutlass.Int32(0):
                    degen = cutlass.Int32(1)
                if m > cutlass.Int32(CMPB):
                    degen = cutlass.Int32(1)
                mc = cutlass.Int32(0)
                if degen == cutlass.Int32(0):
                    mc = m

                if degen == cutlass.Int32(0):
                    # ---- P5 cursor emit ----
                    if complete != cutlass.Int32(0):
                        i = tidx
                        while i < listN:
                            idv = cutlass.Int32(0)
                            bq = cutlass.Int32(0)
                            xv = cutlass.Float32(0.0)
                            if cutlass.const_expr(self.vstg):
                                vx = cutlass.Int32(0)
                                vy = cutlass.Int32(0)
                                if cutlass.const_expr(self.split):
                                    if fromg != cutlass.Int32(0):
                                        vx, vy = C._ldcg_v2_i32(
                                            gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                        )
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(
                                            cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                        )
                                        vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                else:
                                    pk64 = s_cbuf2[i]
                                    vx = cutlass.Int32(
                                        cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                    )
                                    vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                xv = C.f32_of_i32(vx)
                                idv = vy
                                bq = C.f2s_rz((xv - TF) * SC)
                                if bq > cutlass.Int32(NBS - 1):
                                    bq = cutlass.Int32(NBS - 1)
                            else:
                                wpk = cutlass.Uint32(s_cbuf[i])
                                idv = cutlass.Int32(wpk & cutlass.Uint32(IDXM__main))
                                bq = cutlass.Int32(wpk >> cutlass.Uint32(IDXB__main))
                            if bq >= B:
                                p = C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                                if p < lim1:
                                    out_row[soff + p] = ptable[tbase + (idv >> cutlass.Int32(6))] * cutlass.Int32(64) + (idv & cutlass.Int32(63))
                                else:
                                    if whole == cutlass.Int32(0):
                                        q2 = p - above
                                        if q2 < cutlass.Int32(CMPB):
                                            if cutlass.const_expr(self.vstg):
                                                kk = C.fkey(xv)
                                            else:
                                                kk = C.fkey(C.ldg_f32(x_addr, idv))
                                            s_ck64[q2] = (
                                                cutlass.Uint64(kk) << cutlass.Uint64(32)
                                            ) | cutlass.Uint64(cutlass.Uint32(idv))
                            i = i + cutlass.Int32(BLK)
                    else:
                        # collect overflow: scalar re-sweep, exact tail remap —
                        # zero extra live registers by design
                        lo2 = c0 << cutlass.Int32(2)
                        hi2 = c1 << cutlass.Int32(2)
                        i0_ = lo2 + tidx
                        while i0_ < hi2 + tailn:
                            i_ = i0_
                            if i0_ >= hi2:
                                i_ = tail0 + (i0_ - hi2)
                            x = C.ldg_f32(x_addr, i_)
                            if x >= TF:
                                bq = C.f2s_rz((x - TF) * SC)
                                if bq > cutlass.Int32(NBS - 1):
                                    bq = cutlass.Int32(NBS - 1)
                                if bq >= B:
                                    p = C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                                    if p < lim1:
                                        out_row[p] = ptable[tbase + (i_ >> cutlass.Int32(6))] * cutlass.Int32(64) + (i_ & cutlass.Int32(63))
                                    else:
                                        if whole == cutlass.Int32(0):
                                            q2 = p - above
                                            if q2 < cutlass.Int32(CMPB):
                                                s_ck64[q2] = (
                                                    cutlass.Uint64(C.fkey(x)) << cutlass.Uint64(32)
                                                ) | cutlass.Uint64(cutlass.Uint32(i_))
                            i0_ = i0_ + cutlass.Int32(BLK)

                    # ---- P6 refine ----
                    if whole == cutlass.Int32(0):
                        cute.arch.barrier()  # ---- barrier (emit done) ----
                        if mc <= cutlass.Int32(QUADC_CLUS__main):  # O(mc^2) rank
                            mc2 = mc & cutlass.Int32(~1)
                            i = tidx
                            while i < mc:
                                # NOTE: values crossing a dynamic-while region are
                                # re-wrapped SIGNED by the DSL — every u64 compare
                                # must re-assert Uint64 at the USE site.
                                u64v = s_ck64[i]
                                r_ = cutlass.Int32(0)
                                jq = cutlass.Int32(0)
                                while jq < mc2:  # ulonglong2 16B reads
                                    vlo, vhi = C._lds_v2_u64(ck_addr + jq * cutlass.Int32(8))
                                    r_ = (
                                        r_
                                        + cutlass.Int32(vlo > cutlass.Uint64(u64v))
                                        + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                    )
                                    jq = jq + cutlass.Int32(2)
                                if mc2 < mc:  # odd tail
                                    r_ = r_ + cutlass.Int32(
                                        cutlass.Uint64(s_ck64[mc2]) > cutlass.Uint64(u64v)
                                    )
                                if r_ < need:
                                    tie_id = cutlass.Int32(
                                        cutlass.Uint32(
                                            cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                    )
                                    out_row[soff + above + r_] = ptable[tbase + (tie_id >> cutlass.Int32(6))] * cutlass.Int32(64) + (tie_id & cutlass.Int32(63))
                                i = i + cutlass.Int32(BLK)
                        else:
                            # key-space narrowing over ck64
                            if tidx == cutlass.Int32(0):
                                s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                                s_kmm[1] = cutlass.Uint32(0)
                            if tidx < cutlass.Int32(NBS):  # cleared ONCE
                                s_hist[tidx] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (narrowing init) ----
                            i = tidx
                            while i < mc:
                                kk = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                C.atomic_min_cta(s_kmm.iterator + 0, kk)
                                C.atomic_max_cta(s_kmm.iterator + 1, kk)
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (key range) ----
                            rlo = s_kmm[0]
                            rhi = s_kmm[1]
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            aboveC = cutlass.Int32(0)
                            needC = need
                            mm = mc
                            brk = cutlass.Int32(0)
                            lev = cutlass.Int32(0)
                            while brk == cutlass.Int32(0):  # <=6 levels
                                if needC == mm:
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                    aboveC = aboveC + mm
                                    needC = cutlass.Int32(0)
                                    brk = cutlass.Int32(1)
                                elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    brk = cutlass.Int32(1)
                                elif lev >= cutlass.Int32(6):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    brk = cutlass.Int32(1)
                                else:
                                    d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                    b2_ = cutlass.Int32(32) - C.clz_i32(
                                        cutlass.Int32(d2 | cutlass.Uint32(1))
                                    )
                                    sh2 = b2_ - cutlass.Int32(self.lb)
                                    if sh2 < cutlass.Int32(0):
                                        sh2 = cutlass.Int32(0)
                                    sh2u = cutlass.Uint32(sh2)
                                    i = tidx
                                    while i < mc:  # re-bin
                                        uq = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                        if uq >= cutlass.Uint32(rlo):
                                            if uq <= cutlass.Uint32(rhi):
                                                du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                                if du > cutlass.Uint32(NBS - 1):
                                                    du = cutlass.Uint32(NBS - 1)
                                                C.atomic_add_cta(
                                                    s_hist.iterator + cutlass.Int32(du),
                                                    cutlass.Int32(1),
                                                )
                                        i = i + cutlass.Int32(BLK)
                                    cute.arch.barrier()  # ---- barrier (level hist) ----
                                    C.scan_cross0(
                                        s_hist,
                                        needC,
                                        tidx,
                                        s_res,
                                        cutlass.Int32(0),
                                        cutlass.Int32(0),
                                        s_hist,
                                        nb=NBS,
                                        zero=True,
                                    )
                                    cute.arch.barrier()  # ---- barrier (level scan) ----
                                    aboveC = aboveC + s_res[C.RES_ABOVE]
                                    needC = needC - s_res[C.RES_ABOVE]
                                    mm = s_res[C.RES_M]
                                    sB = s_res[C.RES_B]
                                    nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                    if sB != cutlass.Int32(NBS - 1):
                                        rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                                    rlo = nlo
                                    lev = lev + cutlass.Int32(1)
                            if tidx == cutlass.Int32(0):
                                s_scal[1] = cutlass.Int32(0)
                                s_scal[2] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (emit counters) ----
                            it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                            it = cutlass.Int32(0)
                            while it < it2:  # ballot emit
                                i = it * cutlass.Int32(BLK) + tidx
                                p1 = cutlass.Int32(0)
                                p2 = cutlass.Int32(0)
                                idv = cutlass.Int32(0)
                                if i < mc:
                                    w64 = s_ck64[i]
                                    iu = cutlass.Int64(cutlass.Uint32(w64 >> cutlass.Uint64(32)))
                                    idv = cutlass.Int32(
                                        cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF))
                                    )
                                    if iu > ethr:
                                        p1 = cutlass.Int32(1)
                                    if iu == ethr:
                                        p2 = cutlass.Int32(1)
                                self._ballot_pair_emit(
                                    p1,
                                    p2,
                                    idv,
                                    soff + above,
                                    aboveC,
                                    soff + above + aboveC,
                                    needC,
                                    out_row,
                                    s_scal,
                                    lane,
                                )
                                it = it + cutlass.Int32(1)
                else:
                    dga = cutlass.Int32(0)  # gate: valid && complete
                    if valid != cutlass.Int32(0):
                        if complete != cutlass.Int32(0):
                            dga = cutlass.Int32(1)
                    if dga != cutlass.Int32(0):
                        # ---- degen A: narrowing over STAGED candidates ----
                        rlo = cutlass.Uint32(0)
                        rhi = cutlass.Uint32(0xFFFFFFFF)
                        above2 = cutlass.Int32(0)
                        need2 = k
                        m2 = listN
                        ethr = cutlass.Int64(0)
                        tie_m = cutlass.Int32(1)
                        if tidx < cutlass.Int32(NBS):
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (degen A init) ----
                        brk = cutlass.Int32(0)
                        lev = cutlass.Int32(0)
                        while brk == cutlass.Int32(0):  # <=8 levels
                            if need2 == m2:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                above2 = above2 + m2
                                need2 = cutlass.Int32(0)
                                tie_m = cutlass.Int32(0)
                                brk = cutlass.Int32(1)
                            elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            elif lev >= cutlass.Int32(8):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            else:
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2_ = cutlass.Int32(32) - C.clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = b2_ - cutlass.Int32(self.lb)
                                if sh2 < cutlass.Int32(0):
                                    sh2 = cutlass.Int32(0)
                                sh2u = cutlass.Uint32(sh2)
                                i = tidx
                                while i < listN:
                                    uq = cutlass.Uint32(0)
                                    if cutlass.const_expr(self.vstg):
                                        vx = cutlass.Int32(0)
                                        vy = cutlass.Int32(0)
                                        if cutlass.const_expr(self.split):
                                            if fromg != cutlass.Int32(0):
                                                vx, vy = C._ldcg_v2_i32(
                                                    gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                                )
                                            else:
                                                pk64 = s_cbuf2[i]
                                                vx = cutlass.Int32(
                                                    cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                                )
                                        else:
                                            pk64 = s_cbuf2[i]
                                            vx = cutlass.Int32(
                                                cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                            )
                                        uq = C.fkey_bits(cutlass.Uint32(vx))
                                    else:
                                        id0 = cutlass.Int32(
                                            cutlass.Uint32(s_cbuf[i]) & cutlass.Uint32(IDXM__main)
                                        )
                                        uq = C.fkey(C.ldg_f32(x_addr, id0))
                                    if uq >= cutlass.Uint32(rlo):
                                        if uq <= cutlass.Uint32(rhi):
                                            du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                            if du > cutlass.Uint32(NBS - 1):
                                                du = cutlass.Uint32(NBS - 1)
                                            C.atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # ---- barrier (level hist) ----
                                C.scan_cross0(
                                    s_hist,
                                    need2,
                                    tidx,
                                    s_res,
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                    s_hist,
                                    nb=NBS,
                                    zero=True,
                                )
                                cute.arch.barrier()  # ---- barrier (level scan) ----
                                above2 = above2 + s_res[C.RES_ABOVE]
                                need2 = need2 - s_res[C.RES_ABOVE]
                                m2 = s_res[C.RES_M]
                                sB = s_res[C.RES_B]
                                nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                if sB != cutlass.Int32(NBS - 1):
                                    rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):
                            s_scal[1] = cutlass.Int32(0)
                            s_scal[2] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (emit counters) ----
                        nA = k
                        nT = cutlass.Int32(0)
                        if tie_m != cutlass.Int32(0):
                            nA = above2
                            nT = need2
                        it2 = (listN + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            idv = cutlass.Int32(0)
                            if i < listN:
                                uq = cutlass.Uint32(0)
                                if cutlass.const_expr(self.vstg):
                                    vx = cutlass.Int32(0)
                                    vy = cutlass.Int32(0)
                                    if cutlass.const_expr(self.split):
                                        if fromg != cutlass.Int32(0):
                                            vx, vy = C._ldcg_v2_i32(
                                                gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                            )
                                        else:
                                            pk64 = s_cbuf2[i]
                                            vx = cutlass.Int32(
                                                cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                            )
                                            vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(
                                            cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                        )
                                        vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                    uq = C.fkey_bits(cutlass.Uint32(vx))
                                    idv = vy
                                else:
                                    idv = cutlass.Int32(
                                        cutlass.Uint32(s_cbuf[i]) & cutlass.Uint32(IDXM__main)
                                    )
                                    uq = C.fkey(C.ldg_f32(x_addr, idv))
                                iu = cutlass.Int64(uq)
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if tie_m != cutlass.Int32(0):
                                    if iu == ethr:
                                        p2 = cutlass.Int32(1)
                            self._ballot_pair_emit(
                                p1, p2, idv, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                            )
                            it = it + cutlass.Int32(1)
                    else:
                        # ---- degen B: whole-row narrowing ----
                        rlo = cutlass.Uint32(0)
                        rhi = cutlass.Uint32(0xFFFFFFFF)
                        above2 = cutlass.Int32(0)
                        need2 = k
                        m2 = n
                        ethr = cutlass.Int64(0)
                        tie_m = cutlass.Int32(1)
                        if tidx < cutlass.Int32(NBS):
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (degen B init) ----
                        brk = cutlass.Int32(0)
                        lev = cutlass.Int32(0)
                        while brk == cutlass.Int32(0):  # <=8 levels
                            if need2 == m2:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                above2 = above2 + m2
                                need2 = cutlass.Int32(0)
                                tie_m = cutlass.Int32(0)
                                brk = cutlass.Int32(1)
                            elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            elif lev >= cutlass.Int32(8):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            else:
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2_ = cutlass.Int32(32) - C.clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = b2_ - cutlass.Int32(self.lb)
                                if sh2 < cutlass.Int32(0):
                                    sh2 = cutlass.Int32(0)
                                sh2u = cutlass.Uint32(sh2)
                                i = tidx
                                while i < n:  # whole row
                                    uq = C.fkey(C.ldg_f32(x_addr, i))
                                    if uq >= cutlass.Uint32(rlo):
                                        if uq <= cutlass.Uint32(rhi):
                                            du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                            if du > cutlass.Uint32(NBS - 1):
                                                du = cutlass.Uint32(NBS - 1)
                                            C.atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # ---- barrier (level hist) ----
                                C.scan_cross0(
                                    s_hist,
                                    need2,
                                    tidx,
                                    s_res,
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                    s_hist,
                                    nb=NBS,
                                    zero=True,
                                )
                                cute.arch.barrier()  # ---- barrier (level scan) ----
                                above2 = above2 + s_res[C.RES_ABOVE]
                                need2 = need2 - s_res[C.RES_ABOVE]
                                m2 = s_res[C.RES_M]
                                sB = s_res[C.RES_B]
                                nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                if sB != cutlass.Int32(NBS - 1):
                                    rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):
                            s_scal[1] = cutlass.Int32(0)
                            s_scal[2] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (emit counters) ----
                        nA = k
                        nT = cutlass.Int32(0)
                        if tie_m != cutlass.Int32(0):
                            nA = above2
                            nT = need2
                        it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            if i < n:
                                uq = C.fkey(C.ldg_f32(x_addr, i))
                                iu = cutlass.Int64(uq)
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if tie_m != cutlass.Int32(0):
                                    if iu == ethr:
                                        p2 = cutlass.Int32(1)
                            self._ballot_pair_emit(
                                p1, p2, i, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                            )
                            it = it + cutlass.Int32(1)

            # ---- varlen short-row epilogue (production heuristicTopKDecode
            # convention): every valid position is in the top-K — emit identity
            # indices and pad the tail with -1.  The body above ran as a
            # zero-work pass for these rows (n = 0, TGT = INT_MAX) so nothing
            # was written; only part 0 of a SPLIT row emits.
            if cutlass.const_expr(self.varlen):
                if short != cutlass.Int32(0):
                    if part == cutlass.Int32(0):
                        i = tidx
                        while i < n_row:
                            out_row[i] = ptable[tbase + (i >> cutlass.Int32(6))] * cutlass.Int32(64) + (i & cutlass.Int32(63))
                            i = i + cutlass.Int32(BLK)
                        j = n_row + tidx
                        while j < k:
                            out_row[j] = cutlass.Int32(-1)
                            j = j + cutlass.Int32(BLK)


        else:
            if tidx < cutlass.Int32(6):
                drow = row * cutlass.Int32(self.r_const) + part
                diagnostics[drow, tidx] = cutlass.Int32(-2)

    # ------------------------------------------------------------------
    # host launcher (grid dim3(R, b); MINB wall via min_blocks_per_mp)
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        scap_dead: cutlass.Int32,
        cmp_dead: cutlass.Int32,
        R: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        kv_lens: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
        amin: cutlass.Int32,
        sd_en: cutlass.Int32,
        tsh_en: cutlass.Int32,
        histogram: cute.Tensor,
        pdiag: cute.Tensor,
        hrestore: cutlass.Int32,
        diagnostics: cute.Tensor,
        ptable: cute.Tensor,
        pcols: cutlass.Int32,
        npages: cutlass.Int32,
        stream,
    ):
        b = logits.shape[0]
        self.kern(
            logits,
            pre_idx,
            out,
            ws,
            n,
            npad,
            k,
            scap_dead,
            cmp_dead,
            R,
            SMP,
            TGT,
            Q,
            SS2,
            TGT2,
            kv_lens,
            aim_base,
            sfac,
            amin,
            sd_en,
            tsh_en,
            histogram,
            pdiag,
            hrestore,
            diagnostics,
            ptable,
            pcols,
            npages,
        ).launch(grid=(R, b, 1), block=(self.blk, 1, 1), stream=stream, min_blocks_per_mp=self.minb)
