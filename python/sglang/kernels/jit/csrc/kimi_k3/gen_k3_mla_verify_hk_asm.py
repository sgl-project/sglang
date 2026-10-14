"""Generate k3_mla_verify_hk_asm.inc: the hand-scheduled inner loop of the
K3 MLA verify kernel (compute waves).

Register map (compute waves; the compiler is limited to v0..v47 via
amdgpu_num_vgpr(48), everything above is owned by these asm blocks):
  a[0:255]    O^T accumulator, 16 blocks of 16 (block cb = V cols 32cb..32cb+31)
  v[48:119]   Q fp8, B operand of S^T = K Q^T, k-step ks at v[48+8ks : 55+8ks]
  v[120:151]  S buffer A (tb0 v120.., tb1 v136..)
  v[152:183]  S buffer B
  v[184:191]  P fp8 (B operand of O^T += V^T P^T)
  v[192:223]  K fragment ring (4 x 8)
  v[224:255]  V fragment ring (4 x 8); softmax temps inside the QK block
"""

import sys

NV_COMPILER = 48
Q0 = 48
SBUF = (120, 152)
P0 = 184
KR = 192
T = 240  # temps (QK block exp sums: v240..255; K ring uses v192..239)
KSLOTS = 6  # K ring slots (v192.. during QK, V ring unused)
PDK = 4  # K reads issued this many MFMAs ahead
VSLOTS = 6  # V ring v[208:255] (v192..207: decision temps)
VRING = 256 - 8 * VSLOTS
PDV = 3  # V reads issued this many MFMAs ahead
DELTA = 3.0  # rescale headroom (log2 units) added to a new running max

out = []
w = out.append


def clob_v(lo, hi):
    return ", ".join(f'"v{i}"' for i in range(lo, hi + 1))


ALL_A = ", ".join(f'"a{i}"' for i in range(256))
ALL_V = clob_v(NV_COMPILER, 255)


def asm_block(lines, outs="", ins="", clob=""):
    body = "\n".join(f'      "{l}\\n"' for l in lines)
    return f"  asm volatile(\n{body}\n      : {outs}\n      : {ins}\n      : {clob});"


# ------------------------------------------------------------------ O helpers
w("__device__ __forceinline__ void o_zero() {")
w(asm_block([f"v_accvgpr_write_b32 a{i}, 0" for i in range(256)], clob=ALL_A))
w("}")

w("__device__ __forceinline__ void o_scale(float alpha) {")
for cb in range(16):
    l = ["s_nop 7", "s_nop 7", "s_nop 4"] if cb == 0 else []
    l += [f"v_accvgpr_read_b32 v{T + r}, a{16 * cb + r}" for r in range(16)]
    l += [f"v_mul_f32_e32 v{T + r}, %[al], v{T + r}" for r in range(16)]
    l += [f"v_accvgpr_write_b32 a{16 * cb + r}, v{T + r}" for r in range(16)]
    w(asm_block(l, ins='[al] "v"(alpha)', clob=clob_v(T, T + 15) + ", " + ALL_A))
w("}")

w("template <int CB> __device__ __forceinline__ v16f o_read();")
for cb in range(16):
    l = ["s_nop 7", "s_nop 7", "s_nop 7"] if cb == 0 else []
    l += [f"v_accvgpr_read_b32 %[o{r}], a{16 * cb + r}" for r in range(16)]
    outs = ", ".join(f'[o{r}] "=v"(x[{r}])' for r in range(16))
    w(f"template <> __device__ __forceinline__ v16f o_read<{cb}>() {{")
    w("  float x[16];")
    w(asm_block(l, outs=outs))
    w("  return v16f{" + ", ".join(f"x[{r}]" for r in range(16)) + "};")
    w("}")

# ------------------------------------------------------------------ Q set
w("template <int KS> __device__ __forceinline__ void q_set(v8i q);")
for ks in range(9):
    l = [f"v_mov_b32 v{Q0 + 8 * ks + k}, %[q{k}]" for k in range(8)]
    ins = ", ".join(f'[q{k}] "v"(q[{k}])' for k in range(8))
    w(f"template <> __device__ __forceinline__ void q_set<{ks}>(v8i q) {{")
    w(asm_block(l, ins=ins, clob=clob_v(Q0 + 8 * ks, Q0 + 8 * ks + 7)))
    w("}")


# ------------------------------------------------------------------ QK pieces
def kreads(k):
    """ds reads of the K fragment for MFMA k = 2 ks + tb into ring slot k % 4."""
    ks, tb = divmod(k, 2)
    r = KR + 8 * (k % KSLOTS)
    off = (64 * ks) + 18432 * tb
    return [
        f"ds_read_b128 v[{r}:{r + 3}], %[ke] offset:{off}",
        f"ds_read_b128 v[{r + 4}:{r + 7}], %[ko] offset:{off}",
    ]


def kmfma(k, sbase):
    ks, tb = divmod(k, 2)
    r = KR + 8 * (k % KSLOTS)
    d = sbase + 16 * tb
    q = Q0 + 8 * ks
    c = "0" if ks == 0 else f"v[{d}:{d + 15}]"
    return f"v_mfma_f32_32x32x64_f8f6f4 v[{d}:{d + 15}], v[{r}:{r + 7}], v[{q}:{q + 7}], {c}"


def qk_schedule(sbase, valu):
    """18 QK MFMAs into sbase with K reads PDK MFMAs ahead; valu chunks interleaved."""
    n = 18
    lines = []
    for k in range(min(PDK, n)):
        lines += kreads(k)
    chunks = [[] for _ in range(n)]
    if valu:
        per = -(-len(valu) // n)
        for i, ins in enumerate(valu):
            chunks[min(i // per, n - 1)].append(ins)
    for k in range(n):
        if k + PDK < n:
            lines += kreads(k + PDK)
        ahead = min(PDK, n - 1 - k)  # MFMAs whose reads were issued after k's
        lines.append(f"s_waitcnt lgkmcnt({2 * ahead})")
        lines.append(kmfma(k, sbase))
        lines += chunks[k]
    return lines


def decide_valu(cur, tmp):
    """Row max of S[cur] -> lazy rescale decision. Outputs m (m_used), nb (6 - m_used),
    al (2^(m_old - m_used), 1 if no rescale), rs (wave mask of rows wanting a rescale)."""
    s = [cur + k for k in range(32)]
    t = [tmp + i for i in range(16)]
    v = []
    for i in range(10):
        v.append(f"v_max3_f32 v{t[i]}, v{s[3 * i]}, v{s[3 * i + 1]}, v{s[3 * i + 2]}")
    v.append(f"v_max3_f32 v{t[10]}, v{t[0]}, v{t[1]}, v{s[30]}")
    v.append(f"v_max3_f32 v{t[11]}, v{t[2]}, v{t[3]}, v{s[31]}")
    v.append(f"v_max3_f32 v{t[12]}, v{t[4]}, v{t[5]}, v{t[6]}")
    v.append(f"v_max3_f32 v{t[13]}, v{t[7]}, v{t[8]}, v{t[9]}")
    v.append(f"v_max3_f32 v{t[0]}, v{t[10]}, v{t[11]}, v{t[12]}")
    v.append(f"v_max_f32_e32 v{t[0]}, v{t[0]}, v{t[13]}")
    v.append(f"v_mov_b32_e32 v{t[1]}, v{t[0]}")
    v.append("s_nop 1")
    v.append(f"v_permlane32_swap_b32_e32 v{t[0]}, v{t[1]}")
    v.append(f"v_max_f32_e32 v{t[0]}, v{t[0]}, v{t[1]}")
    v.append(f"v_mul_f32_e32 v{t[0]}, %[sc], v{t[0]}")  # mt (scaled)
    v.append(f"v_max_f32_e32 v{t[2]}, %[m], v{t[0]}")  # m_new
    v.append(f"v_sub_f32_e32 v{t[3]}, v{t[2]}, %[m]")
    v.append(f"v_cmp_lt_f32_e64 %[rs], 2.0, v{t[3]}")  # rows with growth > tau
    v.append(
        f"v_add_f32_e32 v{t[2]}, {DELTA}, v{t[2]}"
    )  # headroom: m_used = m_new + delta
    v.append("s_nop 4")
    v.append("s_cmp_lg_u64 %[rs], 0")
    v.append("s_cselect_b64 %[mk], -1, 0")
    v.append("s_nop 2")
    v.append(f"v_cndmask_b32_e64 v{t[4]}, %[m], v{t[2]}, %[mk]")  # m_used
    v.append(f"v_sub_f32_e32 v{t[5]}, %[m], v{t[4]}")
    v.append(f"v_sub_f32_e32 %[nb], 6.0, v{t[4]}")  # 6 - m_used
    v.append(f"v_exp_f32_e32 %[al], v{t[5]}")
    v.append(f"v_mov_b32_e32 %[m], v{t[4]}")
    return v


def exp_valu(cur):
    """p = 2^(s*sc + nb) (x kPScale folded into nb) -> fp8 P; l = l*al + sum(p).
    Software-pipelined order so no instruction consumes a result produced < 4 slots before."""
    s = [cur + k for k in range(32)]
    t = [T + i for i in range(16)]
    fma = [f"v_fma_f32 v{s[k]}, v{s[k]}, %[sc], %[nb]" for k in range(32)]
    ex = [f"v_exp_f32_e32 v{s[k]}, v{s[k]}" for k in range(32)]
    grp = []
    for g8 in range(8):  # group of 4 tokens kb..kb+3 -> P dword
        kb = 4 * g8
        tb, a = kb // 16, (kb % 16) // 4
        pd = P0 + 4 * tb + a
        gg = [
            f"v_cvt_pk_fp8_f32 v{pd}, v{s[kb]}, v{s[kb + 1]}",
            f"v_add_f32_e32 v{t[g8]}, v{s[kb]}, v{s[kb + 1]}",
            f"v_cvt_pk_fp8_f32 v{pd}, v{s[kb + 2]}, v{s[kb + 3]} op_sel:[0,0,1]",
            f"v_add_f32_e32 v{t[8 + g8]}, v{s[kb + 2]}, v{s[kb + 3]}",
        ]
        gg = [gg[0], gg[2]]
        grp.append(gg)
    v = []
    # stage: fma k at step k, exp k at step k+4, group g after exp of its last element + 4
    for step in range(32 + 4 + 8 * 1 + 8):
        if step < 32:
            v.append(fma[step])
        if 0 <= step - 4 < 32:
            v.append(ex[step - 4])
        # group g's last exp at step 4g+3+4 -> pack at >= 4g+3+4+3
        for g8 in range(8):
            if step == 4 * g8 + 3 + 4 + 3:
                v += grp[g8]
    return v


# QK(next) into S[1-D] || exp/pack of S[D]
SIG_QE = "(uint32_t ke, uint32_t ko, float sc, float nb, float al, float& l)"
w("template <int D> __device__ __forceinline__ void qk_exp" + SIG_QE + ";")
for D in (0, 1):
    cur, nxt = SBUF[D], SBUF[1 - D]
    valu = exp_valu(cur)
    lines = qk_schedule(nxt, valu)
    w(f"template <> __device__ __forceinline__ void qk_exp<{D}>{SIG_QE} {{")
    w(
        asm_block(
            lines,
            outs='[l] "+v"(l)',
            ins='[ke] "v"(ke), [ko] "v"(ko), [sc] "v"(sc), [nb] "v"(nb), [al] "v"(al)',
            clob=ALL_V,
        )
    )
    w("}")

SIG_DEC = "(float sc, float& m, float& nb, float& al, uint64_t& rs)"
OUTS_DEC = '[m] "+v"(m), [nb] "=&v"(nb), [al] "=&v"(al), [rs] "=&s"(rs), [mk] "=&s"(mk)'
# standalone decision on S[D] (prologue)
w("template <int D> __device__ __forceinline__ void decide" + SIG_DEC + ";")
for D in (0, 1):
    lines = ["s_nop 7"] * 10 + decide_valu(SBUF[D], KR)
    w(f"template <> __device__ __forceinline__ void decide<{D}>{SIG_DEC} {{")
    w("  uint64_t mk;")
    w(
        asm_block(
            lines, outs=OUTS_DEC, ins='[sc] "v"(sc)', clob=ALL_V + ', "vcc", "scc"'
        )
    )
    w("}")

w("__device__ __forceinline__ void qk_first(uint32_t ke, uint32_t ko) {")
w(asm_block(qk_schedule(SBUF[0], []), ins='[ke] "v"(ke), [ko] "v"(ko)', clob=ALL_V))
w("}")

# ------------------------------------------------------------------ causal / tail mask on S buffer D
w("template <int D> __device__ __forceinline__ void mask_s(int thr);")
for D in (0, 1):
    cur = SBUF[D]
    l = ["s_nop 7"] * 10 + [f"v_mov_b32_e32 v{T}, 0xff800000"]
    for tb in range(2):
        for r in range(16):
            c = 32 * tb + 8 * (r >> 2) + (r & 3)
            reg = cur + 16 * tb + r
            l.append(f"v_cmp_gt_i32_e32 vcc, {c}, %[thr]")
            l.append(f"v_cndmask_b32_e32 v{reg}, v{reg}, v{T}, vcc")
    w(f"template <> __device__ __forceinline__ void mask_s<{D}>(int thr) {{")
    w(asm_block(l, ins='[thr] "v"(thr)', clob=ALL_V + ', "vcc"'))
    w("}")


# ------------------------------------------------------------------ PV
def tm(m):
    return 32 * (m >> 1) + 16 * (m & 1)


def vbuf(cb):
    return VRING + 8 * (cb % VSLOTS)


def vreads(cb):
    buf = vbuf(cb)
    reg = "%[addr_o]" if cb & 1 else "%[addr]"
    return [
        f"ds_read_b64_tr_b8 v[{buf + 2 * m}:{buf + 2 * m + 1}], {reg} offset:{64 * (cb >> 1) + 576 * tm(m)}"
        for m in range(4)
    ]


def pv_schedule(valu, cur):
    body = ["s_waitcnt lgkmcnt(0)", "s_nop 4"]
    for cb in range(min(PDV, 16)):
        body += vreads(cb)
    # row sums of the fp8 P: A = all-ones (fp8 1.0), result in the dead S buffer
    body.append(
        f"v_mfma_f32_32x32x64_f8f6f4 v[{cur}:{cur + 15}], %[ones], v[{P0}:{P0 + 7}], 0"
    )
    valu = [f"v_fma_f32 %[l], %[l], %[al], v{cur}"] + valu
    chunks = [[] for _ in range(16)]
    if valu:
        per = -(-len(valu) // 12)
        for i, ins in enumerate(valu):
            chunks[min(3 + i // per, 15)].append(
                ins
            )  # after the 3rd PV MFMA (QK / rowsum results done)
    for cb in range(16):
        if cb + PDV < 16:
            body += vreads(cb + PDV)
        ahead = min(PDV, 15 - cb)
        body.append(
            f"s_waitcnt lgkmcnt({4 * ahead})"
            if 4 * ahead <= 15
            else "s_waitcnt lgkmcnt(15)"
        )
        buf = vbuf(cb)
        body.append(
            f"v_mfma_f32_32x32x64_f8f6f4 a[{16 * cb}:{16 * cb + 15}], v[{buf}:{buf + 7}], v[{P0}:{P0 + 7}], "
            f"a[{16 * cb}:{16 * cb + 15}]"
        )
        if cb == 2 and valu:
            body.append("s_nop 2")
        body += chunks[cb]
    return body


w(
    "template <int D> __device__ __forceinline__ void pv_tile_asm(uint32_t addr, uint32_t addr_o, v8i ones, float al, float& l);"
)
for D in (0, 1):
    w(
        f"template <> __device__ __forceinline__ void pv_tile_asm<{D}>(uint32_t addr, uint32_t addr_o, v8i ones, float al, float& l) {{"
    )
    w(
        asm_block(
            pv_schedule([], SBUF[D]),
            outs='[l] "+v"(l)',
            ins='[addr] "v"(addr), [addr_o] "v"(addr_o), [ones] "v"(ones), [al] "v"(al)',
            clob=ALL_V + ", " + ALL_A,
        )
    )
    w("}")
SIG_PD = "(uint32_t addr, uint32_t addr_o, float sc, float& m, float& nb, float& al, uint64_t& rs, v8i ones, float& l)"
OUTS_PD = '[m] "+v"(m), [nb] "=&v"(nb), [al] "+v"(al), [rs] "=&s"(rs), [mk] "=&s"(mk), [l] "+v"(l)'
w("template <int D> __device__ __forceinline__ void pv_decide" + SIG_PD + ";")
for D in (0, 1):
    w(f"template <> __device__ __forceinline__ void pv_decide<{D}>{SIG_PD} {{")
    w("  uint64_t mk;")
    w(
        asm_block(
            pv_schedule(decide_valu(SBUF[D], KR), SBUF[1 - D]),
            outs=OUTS_PD,
            ins='[addr] "v"(addr), [addr_o] "v"(addr_o), [sc] "v"(sc), [ones] "v"(ones)',
            clob=ALL_V + ", " + ALL_A + ', "vcc", "scc"',
        )
    )
    w("}")

# ------------------------------------------------------------------ loader: one tile of LDS DMA
# (loader wave only; its own register use: v48..v55 temps, v56..v59 kv-index ring)
w(
    "template <int R> __device__ __forceinline__ void issue_tile_asm(v4u_ rs, uint32_t slot, uint32_t st, uint32_t ba, uint32_t co);"
)
w("template <int R> __device__ __forceinline__ void load_idx_asm(v4u_ rs, int off);")
for R in range(8):
    l = []
    for tg in range(4):
        off = f" offset:{64 * tg}" if tg else ""
        l.append(f"ds_bpermute_b32 v{48 + tg}, %[ba], v{56 + R}{off}")
    l.append("s_waitcnt lgkmcnt(0)")
    for tg in range(4):
        l.append(f"v_mad_u32_u24 v{48 + tg}, v{48 + tg}, %[st], %[co]")
    for k in range(36):
        tg, cg = divmod(k, 9)
        tmp = 52 + (k % 4)
        if cg:
            l.append(f"v_add_u32_e32 v{tmp}, {64 * cg}, v{48 + tg}")
            va = tmp
        else:
            va = 48 + tg
        l.append(f"s_add_u32 m0, %[slot], {1024 * k}")
        l.append("s_nop 0")
        l.append(f"buffer_load_dwordx4 v{va}, %[rs], 0 offen lds")
    w(
        f"template <> __device__ __forceinline__ void issue_tile_asm<{R}>(v4u_ rs, uint32_t slot, uint32_t st, uint32_t ba, uint32_t co) {{"
    )
    w(
        asm_block(
            l,
            ins='[rs] "s"(rs), [slot] "s"(slot), [st] "s"(st), [ba] "v"(ba), [co] "v"(co)',
            clob=clob_v(48, 55) + ', "m0", "scc", "memory"',
        )
    )
    w("}")
    w(
        f"template <> __device__ __forceinline__ void load_idx_asm<{R}>(v4u_ rs, int off) {{"
    )
    w(
        asm_block(
            [f"buffer_load_dword v{56 + R}, %[o], %[rs], 0 offen"],
            ins='[o] "v"(off), [rs] "s"(rs)',
            clob=f'"v{56 + R}", "memory"',
        )
    )
    w("}")


# ------------------------------------------------------------------ Q prologue
# Raw bf16 Q (32 dims x 9 k-steps per lane = 144 dwords) -> v[48:191] with all loads in
# flight at once; per-row amax (int max of |bf16| bits, then across lane halves);
# quantize in place to fp8 v[48:119] (fp8 dword k of k-step ks from raw dwords 2k, 2k+1).
def q_prologue(lds=False, part=0):
    """part 0: whole prologue; 1: loads only (raw bf16 Q in v[48:191]); 2: amax + quantize only."""
    l = []
    if part == 2:
        lds = None
    if lds:
        # coalesced LDS-DMA of this wave's 32 contiguous rows (36 KB) into its slot, then
        # per-lane ds_read_b128 in the MFMA layout (row l32, dims 64 ks + 32 h + 16 jj)
        for k in range(36):
            l.append(f"v_add_u32_e32 v192, {1024 * k}, %[v16]")
            l += [
                f"s_add_u32 m0, %[sl], {1024 * k}",
                "s_nop 0",
                "buffer_load_dwordx4 v192, %[rs], 0 offen lds",
            ]
        if part == 1:
            # zero O (AGPRs) while the Q loads are in flight (off the critical path)
            l += [f"v_accvgpr_write_b32 a{i}, 0" for i in range(256)]
        l.append("s_waitcnt vmcnt(0)")
        for ks in range(9):
            for jj in range(4):
                r = 48 + 16 * ks + 4 * jj
                l.append(
                    f"ds_read_b128 v[{r}:{r + 3}], %[la] offset:{128 * ks + 16 * jj}"
                )
        l.append("s_waitcnt lgkmcnt(0)")
    elif lds is None:
        pass
    else:
        for ks in range(9):
            for jj in range(4):
                r = 48 + 16 * ks + 4 * jj
                l.append(
                    f"global_load_dwordx4 v[{r}:{r + 3}], %[qa], off offset:{128 * ks + 16 * jj}"
                )
        l.append("s_waitcnt vmcnt(0)")
    if part == 1:
        return l
    # |x| max via packed f16 maximum3 on the raw bf16 bits (same ordering for finite
    # values; bf16 |x| >= 2^121 would read as f16 inf/nan -- never in Q): chains of max(x)
    # (v192..195) and max(-x) (v196..199), two dwords per instruction
    A, Bn = [192, 193, 194, 195], [196, 197, 198, 199]
    for p0 in range(0, 72, 4):
        for j in range(4):
            x, y = 48 + 2 * (p0 + j), 49 + 2 * (p0 + j)
            src = f"v{x}, v{y}, v{x}" if p0 == 0 else f"v{A[j]}, v{x}, v{y}"
            l.append(f"v_pk_maximum3_f16 v{A[j]}, {src}")
        for j in range(4):
            x, y = 48 + 2 * (p0 + j), 49 + 2 * (p0 + j)
            if p0 == 0:
                l.append(
                    f"v_pk_maximum3_f16 v{Bn[j]}, v{x}, v{y}, v{x} neg_lo:[1,1,1] neg_hi:[1,1,1]"
                )
            else:
                l.append(
                    f"v_pk_maximum3_f16 v{Bn[j]}, v{Bn[j]}, v{x}, v{y} neg_lo:[0,1,1] neg_hi:[0,1,1]"
                )
    for j in range(4):
        l.append(f"v_pk_maximum3_f16 v{A[j]}, v{A[j]}, v{Bn[j]}, 0")
    l.append("v_pk_max_u16 v192, v192, v193")
    l.append("v_pk_max_u16 v194, v194, v195")
    l.append("v_pk_max_u16 v192, v192, v194")
    l.append("v_lshrrev_b32_e32 v193, 16, v192")
    l.append("v_and_b32_e32 v192, 0xffff, v192")
    l.append("v_max_u32_e32 v192, v192, v193")
    l.append("v_lshlrev_b32_e32 v192, 16, v192")
    l.append("v_mov_b32_e32 v193, v192")
    l.append("s_nop 1")
    l.append("v_permlane32_swap_b32_e32 v192, v193")
    l.append("v_max_f32_e32 v192, v192, v193")
    l.append("v_max_f32_e32 v192, 0x1e3ce508, v192")  # amax >= 1e-20
    # c = 448 / amax truncated to bf16 (so max |q c| <= 448); q * c exactly via v_dot2_f32_bf16
    # with (c, 0) / (0, c): one instruction per element, no bf16 -> f32 unpacking.
    # qscale = 1 / c (consistent with the rounded c).
    l.append("v_rcp_f32_e32 v194, v192")
    l.append("s_nop 1")
    l.append("v_mul_f32_e32 v194, 0x43e00000, v194")  # 448 / amax
    l.append("v_and_b32_e32 v195, 0xffff0000, v194")  # (0, c): c in the high half
    l.append("v_rcp_f32_e32 v193, v195")
    l.append("v_lshrrev_b32_e32 v194, 16, v195")  # (c, 0)
    l.append("s_nop 1")
    l.append("v_mov_b32_e32 %[am], v193")
    for g0 in range(0, 72, 4):
        for idx in range(g0, g0 + 4):
            r0, t = 48 + 2 * idx, 196 + 4 * (idx % 4)
            l += [
                f"v_dot2_f32_bf16 v{t}, v{r0}, v194, 0",
                f"v_dot2_f32_bf16 v{t + 1}, v{r0}, v195, 0",
            ]
        for idx in range(g0, g0 + 4):
            r0, t = 48 + 2 * idx, 196 + 4 * (idx % 4)
            l += [
                f"v_dot2_f32_bf16 v{t + 2}, v{r0 + 1}, v194, 0",
                f"v_dot2_f32_bf16 v{t + 3}, v{r0 + 1}, v195, 0",
            ]
        for idx in range(g0, g0 + 4):
            t = 196 + 4 * (idx % 4)
            l.append(f"v_cvt_pk_fp8_f32 v{48 + idx}, v{t}, v{t + 1}")
        for idx in range(g0, g0 + 4):
            t = 196 + 4 * (idx % 4)
            l.append(f"v_cvt_pk_fp8_f32 v{48 + idx}, v{t + 2}, v{t + 3} op_sel:[0,0,1]")
    return l


w("__device__ __forceinline__ float q_prologue_asm(const uint16_t* qa) {")
w("  float am;")
w(
    asm_block(
        q_prologue(),
        outs='[am] "=v"(am)',
        ins='[qa] "v"(qa)',
        clob=ALL_V + ', "memory"',
    )
)
w("  return am;")
w("}")
w(
    "__device__ __forceinline__ float q_prologue_lds_asm(v4u_ rs, uint32_t sl, uint32_t v16, uint32_t la, uint32_t rot) {"
)
w("  float am;")
w("  uint32_t t0, t1;")
w(
    asm_block(
        q_prologue(True),
        outs='[am] "=v"(am), [t0] "=&s"(t0), [t1] "=&s"(t1)',
        ins='[rs] "s"(rs), [sl] "s"(sl), [v16] "v"(v16), [la] "v"(la), [rot] "s"(rot)',
        clob=ALL_V + ', "m0", "scc", "memory"',
    )
)
w("  return am;")
w("}")
w(
    "__device__ __forceinline__ void q_load_lds_asm(v4u_ rs, uint32_t sl, uint32_t v16, uint32_t la, uint32_t rot) {"
)
w("  uint32_t t0, t1;")
w(
    asm_block(
        q_prologue(True, 1),
        outs='[t0] "=&s"(t0), [t1] "=&s"(t1)',
        ins='[rs] "s"(rs), [sl] "s"(sl), [v16] "v"(v16), [la] "v"(la), [rot] "s"(rot)',
        clob=ALL_V + (", " + ALL_A) + ', "m0", "scc", "memory"',
    )
)
w("}")
w("__device__ __forceinline__ float q_quant_asm() {")
w("  float am;")
w(asm_block(q_prologue(None, 2), outs='[am] "=v"(am)', clob=ALL_V))
w("  return am;")
w("}")

# ------------------------------------------------------------------ shared DMA (compute waves issue their token group)
# mailbox: mb[t % 4][tg][lane] u32 voffsets written by the loader.
w(
    "template <int R> __device__ __forceinline__ void mbox_write_asm(uint32_t mb, uint32_t ba, uint32_t st, uint32_t co);"
)
w(
    "template <int R> __device__ __forceinline__ void issue_tg3_asm(v4u_ rs, uint32_t slot, uint32_t st, uint32_t ba, uint32_t co);"
)
for R in range(8):
    l = []
    for tg in range(3):
        off = f" offset:{64 * tg}" if tg else ""
        l.append(f"ds_bpermute_b32 v{48 + tg}, %[ba], v{56 + R}{off}")
    l.append("s_waitcnt lgkmcnt(0)")
    for tg in range(3):
        l.append(f"v_mad_u32_u24 v{48 + tg}, v{48 + tg}, %[st], %[co]")
    for tg in range(3):
        off = f" offset:{256 * tg}" if tg else ""
        l.append(f"ds_write_b32 %[mb], v{48 + tg}{off}")
    w(
        f"template <> __device__ __forceinline__ void mbox_write_asm<{R}>(uint32_t mb, uint32_t ba, uint32_t st, uint32_t co) {{"
    )
    w(
        asm_block(
            l,
            ins='[mb] "v"(mb), [ba] "v"(ba), [st] "s"(st), [co] "v"(co)',
            clob=clob_v(48, 55) + ', "memory"',
        )
    )
    w("}")
    l = [
        f"ds_bpermute_b32 v51, %[ba], v{56 + R} offset:192",
        "s_waitcnt lgkmcnt(0)",
        "v_mad_u32_u24 v51, v51, %[st], %[co]",
    ]
    for cg in range(9):
        k = 27 + cg
        if cg:
            l.append(f"v_add_u32_e32 v{52 + cg % 4}, {64 * cg}, v51")
            va = 52 + cg % 4
        else:
            va = 51
        l += [
            f"s_add_u32 m0, %[slot], {1024 * k}",
            "s_nop 0",
            f"buffer_load_dwordx4 v{va}, %[rs], 0 offen lds",
        ]
    w(
        f"template <> __device__ __forceinline__ void issue_tg3_asm<{R}>(v4u_ rs, uint32_t slot, uint32_t st, uint32_t ba, uint32_t co) {{"
    )
    w(
        asm_block(
            l,
            ins='[rs] "s"(rs), [slot] "s"(slot), [st] "s"(st), [ba] "v"(ba), [co] "v"(co)',
            clob=clob_v(48, 55) + ', "m0", "scc", "memory"',
        )
    )
    w("}")

# compute wave w: read its voffset from the mailbox, issue 9 LDS-DMA loads (token group w)
w(
    "template <int W> __device__ __forceinline__ void cdma_asm(uint32_t mb, v4u_ rs, uint32_t slot);"
)
for W in range(3):
    l = ["ds_read_b32 v240, %[mb]", "s_waitcnt lgkmcnt(0)"]
    for cg in range(9):
        k = 9 * W + cg
        if cg:
            l.append(f"v_add_u32_e32 v{241 + cg % 4}, {64 * cg}, v240")
            va = 241 + cg % 4
        else:
            va = 240
        l += [
            f"s_add_u32 m0, %[slot], {1024 * k}",
            "s_nop 0",
            f"buffer_load_dwordx4 v{va}, %[rs], 0 offen lds",
        ]
    w(
        f"template <> __device__ __forceinline__ void cdma_asm<{W}>(uint32_t mb, v4u_ rs, uint32_t slot) {{"
    )
    w(
        asm_block(
            l,
            ins='[mb] "v"(mb), [rs] "s"(rs), [slot] "s"(slot)',
            clob=clob_v(240, 244) + ', "m0", "scc", "memory"',
        )
    )
    w("}")

# ------------------------------------------------------------------ row-linear loader
# Instruction k / lane i fills LDS bytes 16 n, n = 64 k + i = 36 t + p: token t, position p,
# holding global chunk c = p ^ ((t >> 2) & 3). Per-lane constants (bpermute address 4 t and
# 16 c) precomputed once into v[80:115] / v[116:151]; temps v[152:160].
w("__device__ __forceinline__ void rl_precompute(uint32_t lane) {")
l = []
for k in range(36):
    l += [
        f"v_add_u32_e32 v152, {64 * k}, %[ln]",  # n
        "v_mul_u32_u24_e32 v153, 0x71d, v152",  # n * 1821
        "v_lshrrev_b32_e32 v153, 16, v153",  # t = n / 36
        "v_mul_u32_u24_e32 v154, 36, v153",
        "v_sub_u32_e32 v154, v152, v154",  # p
        "v_bfe_u32 v155, v153, 2, 2",  # (t >> 2) & 3
        "v_xor_b32_e32 v154, v154, v155",  # c
        f"v_lshlrev_b32_e32 v{80 + k}, 2, v153",
        f"v_lshlrev_b32_e32 v{116 + k}, 4, v154",
    ]
w(asm_block(l, ins='[ln] "v"(lane)', clob=clob_v(80, 160)))
w("}")
w(
    "template <int R> __device__ __forceinline__ void issue_tile_rl_asm(v4u_ rs, uint32_t slot, uint32_t st);"
)
for R in range(8):
    l = []
    for b9 in range(4):
        ks = range(9 * b9, 9 * b9 + 9)
        for j, k in enumerate(ks):
            l.append(f"ds_bpermute_b32 v{152 + j}, v{80 + k}, v{56 + R}")
        l.append("s_waitcnt lgkmcnt(0)")
        for j, k in enumerate(ks):
            l.append(f"v_mad_u32_u24 v{152 + j}, v{152 + j}, %[st], v{116 + k}")
        for j, k in enumerate(ks):
            l += [
                f"s_add_u32 m0, %[slot], {1024 * k}",
                "s_nop 0",
                f"buffer_load_dwordx4 v{152 + j}, %[rs], 0 offen lds",
            ]
    w(
        f"template <> __device__ __forceinline__ void issue_tile_rl_asm<{R}>(v4u_ rs, uint32_t slot, uint32_t st) {{"
    )
    w(
        asm_block(
            l,
            ins='[rs] "s"(rs), [slot] "s"(slot), [st] "s"(st)',
            clob=clob_v(152, 160) + ', "m0", "scc", "memory"',
        )
    )
    w("}")

open(sys.argv[1], "w").write(
    "// generated by gen/gen_asm.py -- do not edit\n" + "\n".join(out) + "\n"
)
