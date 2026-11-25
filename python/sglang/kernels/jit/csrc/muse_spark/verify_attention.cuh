// SM90 DFlash verify attention (paged FP8 KV; fused QK-norm, RoPE, KV write, norm gate) for
// kernels/ops/attention/muse_spark_verify_sm90.py, which defines PS (1: push rows to the SP o_proj owners).
#include "ptx.cuh"
#define HD 64
#define STG 16384
#define WG_VTB 2304
#define XR 272
#define VTBUF (2 * WG_VTB)
DEVI void cp16(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVI void mbar_cp_arrive(u32 a) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
}
// not .nc: ptxas hoists ld.global.nc above griddepcontrol.wait
DEVI uint2 ldq_u2(const void* a) {
  uint2 v;
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(a) : "memory");
  return v;
}
DEVI i64 ldq_s64(const i64* a) {
  i64 v;
  asm volatile("ld.global.s64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
  return v;
}
DEVI int ldg_s32p(const int* a, bool pr) {
  int v = 0;
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p ld.global.nc.s32 %0, [%1];\n}"
               : "+r"(v)
               : "l"(a), "r"((int)pr)
               : "memory");
  return v;
}
DEVI void lds4(u32 (&a)[4], u32 addr) {
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr)
               : "memory");
}
DEVI void ldsm4t(u32 addr, u32 (&x)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "r"(addr)
               : "memory");
}
// stores ldsm4t's x with byte pairs regrouped
DEVI void stsm4t(u32 addr, const u32 (&x)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr),
               "r"(prmt(x[0], x[1], 0x6420)),
               "r"(prmt(x[0], x[1], 0x7531)),
               "r"(prmt(x[2], x[3], 0x6420)),
               "r"(prmt(x[2], x[3], 0x7531))
               : "memory");
}
// bits [a, b) of a 64-column tile
DEVI u64 cmask(int a, int b) {
  a = max(a, 0);
  b = min(b, 64);
  if (b <= a) return 0;
  const u64 hi = b >= 64 ? ~0ull : ((1ull << b) - 1);
  return hi & ~((1ull << a) - 1);
}
// 8 new keys at tile column nk
DEVI u64 nmask(u32 anc, int nk) {
  if (nk >= 64 || nk <= -8) return 0;
  return nk >= 0 ? (u64)anc << nk : (u64)(anc >> -nk);
}
// S columns whose vm[rh] bit is clear -> -inf
DEVI void apply_cmask(float (&s)[32], const u64 (&vm)[2], int q) {
  for (int rh = 0; rh < 2; ++rh) {
    const u64 sh = vm[rh] >> (2 * q);
    const u32 w0 = (u32)sh, w1 = (u32)(sh >> 32);
    for (int j = 0; j < 32; ++j) {
      if (((j >> 1) & 1) != rh) continue;
      const int c = j >> 2, bit = 8 * (c & 3) + (j & 1);
      if (!(((c < 4 ? w0 : w1) >> bit) & 1u)) s[j] = NEG_INF;
    }
  }
}
// online softmax: rescale to the new row max mn
DEVI void sm_step(float& m, float& l, float& a, float& mb, float mn, float sc) {
  const float mcur = mn == NEG_INF ? 0.f : mn;
  a = ex2((m - mcur) * sc);
  m = mn;
  mb = mcur * sc - 8.f;
  l *= a;
}
struct WGS {
  float o[32] = {}, m[2] = {NEG_INF, NEG_INF}, l[2] = {};
};
struct WGC {
  u64 kd, vd;
  u32 ldsm, stsm;
};
DEVI void wg_setup(WGC& c, u32 rb, u32 vb, int h, int wgw, int lane) {
  const int k = lane >> 3, r = lane & 7;
  c.kd = wg_desc(rb + h * 64, 16, 1024, 1);
  c.vd = wg_desc(vb, 128, 256, 0);
  c.ldsm = rb + 8192 + (8 * k + r) * 128 + (((h * 4 + wgw) ^ r) << 4);
  c.stsm = vb + (2 * wgw + (k & 1)) * 256 + (k >> 1) * 128 + r * 16;
}
DEVI void tile(
    WGS& w,
    const WGC& c,
    u32 qs,
    u32 st,
    u32 vt,
    int q,
    int barid,
    bool need_mask,
    const int (&lo)[2],
    const int (&hi)[2],
    float sc,
    const u32 (&an)[2],
    bool resc) {
  u32 qa[2][4];
  for (int i = 0; i < 2; ++i)
    lds4(qa[i], qs + 16 * i);
  float s[32];
  wg_fence();
  const u64 dk = dadd(c.kd, st >> 4);
  wg_qk<0>(s, qa[0], dk);
  wg_qk<1>(s, qa[1], dadd(dk, 2));
  wg_commit();
  for (int nb = 0; nb < 2; ++nb) {
    u32 x[4];
    ldsm4t(c.ldsm + st + nb * 32 * 128, x);
    stsm4t(c.stsm + vt + nb * WG_VTB, x);
  }
  wg_wait<0>();
  acc_pin(s);
  if (need_mask) {
    u64 vm[2];
    for (int rh = 0; rh < 2; ++rh)
      vm[rh] = cmask(lo[rh], hi[rh]) | nmask(an[rh], hi[rh]);
    apply_cmask(s, vm, q);
  }
  float alpha[2], mb[2], mx[2] = {NEG_INF, NEG_INF};
  for (int j = 0; j < 32; ++j)
    mx[(j >> 1) & 1] = fmaxf(mx[(j >> 1) & 1], s[j]);
  for (int rh = 0; rh < 2; ++rh)
    sm_step(w.m[rh], w.l[rh], alpha[rh], mb[rh], fmaxf(w.m[rh], quad_max(mx[rh])), sc);
  u32 pa[2][4];
  float pf[32];
  for (int j = 0; j < 32; ++j) {
    pf[j] = ex2(fmaf(s[j], sc, -mb[(j >> 1) & 1]));
    w.l[(j >> 1) & 1] += pf[j];
  }
  for (int nb = 0; nb < 2; ++nb)
    p_frag(pa[nb], pf + 16 * nb);
  if (resc)
    for (int i = 0; i < 32; ++i)
      w.o[i] *= alpha[(i >> 1) & 1];
  fence_async();
  bar_sync(barid, 128);
  wg_fence();
  for (int k = 0; k < 2; ++k)
    wg_pv(w.o, pa[k], dadd(c.vd, (vt + k * WG_VTB) >> 4));
  wg_commit();
  wg_wait<0>();
  acc_pin(w.o);
}
// = FusedQKNorm + .to(e4m3)
DEVI float rms_scale(float ss) {
  return rsqrtf(ss * (1.0f / HD) + 1e-5f);
}
// o = bf16(o il); returns the RMSNorm scale over W lanes
template <int W, int N>
DEVI float bf_rms(float (&o)[N], float il) {
  float ss = 0.f;
  for (int e = 0; e < N; ++e) {
    o[e] = bfr(o[e] * il);
    ss += o[e] * o[e];
  }
  for (int m = 1; m < W; m <<= 1)
    ss += shfl_xor(ss, m);
  return rms_scale(ss);
}
DEVI void rope2(float& ev, float& od, float2 t) {
  const float e = bfr(ev), o = bfr(od);
  ev = bfr(e * t.x - o * t.y);
  od = bfr(e * t.y + o * t.x);
}
DEVI void q_rms16(float (&f)[16], const uint2 (&qx)[4]) {
  for (int cg = 0; cg < 4; ++cg) {
    const uint2 x = qx[cg];
    f[4 * cg] = bf_lo(x.x);
    f[4 * cg + 1] = bf_hi(x.x);
    f[4 * cg + 2] = bf_lo(x.y);
    f[4 * cg + 3] = bf_hi(x.y);
  }
  float ss = 0.f;
  for (int e = 0; e < 16; ++e)
    ss += f[e] * f[e];
  const float rs = rms_scale(quad_sum(ss));
  for (int e = 0; e < 16; ++e)
    f[e] *= rs;
}
DEVI void q_frag(u32 (&qa)[2][4], const uint2 (&qx)[2][4], const float2* rope, int r0, int r1, int tig) {
  for (int rh = 0; rh < 2; ++rh) {
    float f[16];
    q_rms16(f, qx[rh]);
    if (rope) {
      for (int e = 0; e < 8; ++e)
        rope2(f[2 * e], f[2 * e + 1], rope[(rh ? r1 : r0) * 32 + 8 * (e >> 1) + 2 * tig + (e & 1)]);
    } else {
      for (int e = 0; e < 16; ++e)
        f[e] = bfr(f[e]);
    }
    u32 wv[4];
    for (int e = 0; e < 4; ++e)
      wv[e] = e4m3x4(f[4 * e], f[4 * e + 1], f[4 * e + 2], f[4 * e + 3]);
    qa[0][rh] = wv[0];
    qa[0][2 + rh] = wv[1];
    qa[1][rh] = wv[2];
    qa[1][2 + rh] = wv[3];
  }
}
DEVI void new_kv(uint2 kraw, uint2 vraw, const float2* t, u32& kq, u32& vq) {
  float k0 = bf_lo(kraw.x), k1 = bf_hi(kraw.x), k2 = bf_lo(kraw.y), k3 = bf_hi(kraw.y);
  float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
  for (int o = 1; o < 16; o <<= 1)
    ss += shfl_xor(ss, o);
  const float rs = rms_scale(ss);
  k0 *= rs;
  k1 *= rs;
  k2 *= rs;
  k3 *= rs;
  if (t) {
    rope2(k0, k1, t[0]);
    rope2(k2, k3, t[1]);
  } else {
    k0 = bfr(k0);
    k1 = bfr(k1);
    k2 = bfr(k2);
    k3 = bfr(k3);
  }
  kq = e4m3x4(k0, k1, k2, k3);
  vq = e4m3x4(bf_lo(vraw.x), bf_hi(vraw.x), bf_lo(vraw.y), bf_hi(vraw.y));
}
DEVI float4 as_f4(const float* a) {
  return {a[0], a[1], a[2], a[3]};
}
DEVI float2 rope_cs(int pos, float ifq) {
  float sn, cs;
  sincosf(__fmul_rn((float)pos, ifq), &sn, &cs);
  return {cs, sn};
}
// packed by _args()
struct AP {
  const u16* qkv;
  const int* cache_seqlens;
  const i64* cache_loc;
  u8* k_cache;
  u8* v_cache;
  const int* page_table;
  u16* out;
  i64 qkv_stride;
  int max_pages;
  int window;
  float sc;
  u8* ws;
  int B;
  int G;
  float ifq[32];
  const u8* anc;
  const u8* dep;
  const i64* rdst;
  int rrows;
  i64 rld;
  int rcol, rbase;
};
DEVI u16* prow(const AP& p, i64 tok) {
  const int g = p.rbase + (int)tok, o = g / p.rrows;
  return (u16*)p.rdst[o] + (g - o * p.rrows) * p.rld + p.rcol;
}
#define ODST(tok) (PS ? prow(p, tok) : p.out + (tok) * 16 * HD)
// request b's 8 anc / depth bytes
struct TR {
  uint2 a, d;
};
DEVI TR tree_rows(const AP& p, int b) {
  return {
      p.anc ? ldq_u2(p.anc + 8 * b) : make_uint2(0x0f070301u, 0xff7f3f1fu),
      p.dep ? ldq_u2(p.dep + 8 * b) : make_uint2(0x03020100u, 0x07060504u)};
}
// cache_seqlens - 8, clamped to the page-table width: history reads stay in the row
DEVI int past(const AP& p, int b) {
  return max(min(ldg_s32p(p.cache_seqlens + b, true) - 8, p.max_pages), 0);
}
DEVI int tr_byte(uint2 w, int i) {
  return (((i & 4) ? w.y : w.x) >> (8 * (i & 3))) & 0xffu;
}
DEVI void q_load(uint2 (&qx)[2][4], const AP& p, i64 tok0, int h, int mt, int g, int tig) {
  for (int rh = 0; rh < 2; ++rh)
    for (int cg = 0; cg < 4; ++cg)
      qx[rh][cg] = ldq_u2(p.qkv + (tok0 + 2 * mt + rh) * p.qkv_stride + (8 * h + g) * HD + 16 * cg + 4 * tig);
}
DEVI void new_raw(const AP& p, i64 tok, int hh, int q4, uint2& kraw, uint2& vraw, i64& crow) {
  const u16* src = p.qkv + tok * p.qkv_stride;
  kraw = ldq_u2(src + (16 + hh) * HD + q4 * 4);
  vraw = ldq_u2(src + (18 + hh) * HD + q4 * 4);
  crow = ldq_s64(p.cache_loc + tok);
}
DEVI void kv_put(const AP& p, i64 crow, int hh, int q4, u32 kq, u32 vq) {
  const i64 o = crow * 128 + hh * HD + q4 * 4;
  *(u32*)(p.k_cache + o) = kq;
  *(u32*)(p.v_cache + o) = vq;
}
// VS (sliding): C-CTA cluster per request; CTA `part` = window keys [u0, u1) (+ 8 new on the last part) in 64-key
// tiles, newest first, a stage each, filled once: 256 loader cp.async arrivals (+ 256 builder ones on the last part).
// Part q owns rows [32 q / C, 32 (q + 1) / C): peers bulk-push them on q's mbr, armed before the cluster barrier.
template <int C, int T>
DEVI void vs(const AP& p) {
  constexpr int L = 8;
  extern __shared__ __align__(1024) u8 smem[];
  const u32 sb0 = smem_u32(smem), rb = (sb0 + 1023u) & ~1023u;
  u8* stg = smem + (rb - sb0);
  u8* vtb = stg + T * STG;
  float2* rtab = (float2*)(vtb + 8 * VTBUF);
  u32* qxs = (u32*)(rtab + 256);
  // C == 3 receives into the stages, after a cluster barrier
  constexpr bool RCV = C != 3;
  const u32 mb0 = smem_u32(qxs + 2 * 128 * 8), mbr = mb0 + 8 * T;
  u8* rcv = smem + ((mbr + 24u - sb0 + 127u) & ~127u);
  TR* trs = (TR*)(smem + (mbr + 8u - sb0));

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int wgi = shfl(warp >> 2, 0), h = wgi & 1, s = wgi >> 1, mt = warp & 3, tl = tid & 127, b = blockIdx.x / C;
  const int part = cl_rank();
  const bool lastp = part == C - 1, ldr = tid < 256;
  const int seq = past(p, b);
  if (tid == 0) {
    for (int k = 0; k < T; ++k)
      mbar_init(mb0 + 8 * k, 256 + (lastp ? 256 : 0));
    mbar_init(mbr, 1);
    if (RCV) mbar_expect(mbr, 4 * (C - 1) * ((32 * (part + 1)) / C - (32 * part) / C) * XR);
    mbar_init_fence();
  }
  __syncthreads();
  // relaxed: fence.mbarrier_init above publishes the inits
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (tid == 0) pdl_trigger();
  const i64 tok0 = (i64)b * L;
  const int W1 = p.window - 1;
  const int lo = max(0, seq - W1), npa = seq - lo;
  const int per = (npa + C + 31) / C;
  const int u0 = min(npa, part * per), u1 = lastp ? npa : min(npa, (part + 1) * per);
  const int npast = u1 - u0, pos0 = lo + u0;
  const int ntot = npast + (lastp ? L : 0), ntile = (ntot + 63) >> 6;
  const int e0 = (32 * part) / C, nown = 32 % C ? (32 * (part + 1)) / C - e0 : 32 / C;
  const int oi = tid >> 3, t8 = tid & 7;
  const bool own = oi < 4 * nown;
  const int ow = own ? oi / nown : 0, oe = e0 + (own ? oi % nown : 0);
  const int orow = 16 * (2 * (oe >> 3) + (ow >> 1)) + 8 * (ow & 1) + (oe & 7);
  uint4 gv{};
  const u32 qsa = smem_u32(qxs + (h * 128 + tl) * 8);
  TR tr;
  if (ldr) {
    const int r = tid >> 3, c8 = tid & 7, lr = lane & 7, prow = 4 * warp + (lr & 3) + 32 * (lr >> 2);
    int ent[T];
    const int* ptb = p.page_table + (i64)b * p.max_pages + pos0;
    for (int k = 0; k < T; ++k) {
      const int u = 64 * k + prow;
      ent[k] = ldg_s32p(ptb + u, u < npast);
    }
    const u8* kb = p.k_cache + c8 * 16;
    const u8* vb = p.v_cache + c8 * 16;
    const u32 d0 = rb + r * 128 + ((c8 ^ (r & 7)) << 4);
#pragma unroll
    for (int k = T - 1; k >= 0; --k) {
      if (k < ntile) {
        for (int hh = 0; hh < 2; ++hh) {
          const int e = shfl(ent[k], (lane >> 3) + 4 * hh), u = 64 * k + r + 32 * hh;
          const bool ok = u < npast;
          if (!(u < ntot && !ok)) {
            const i64 off = ok ? (i64)e * 128 : 0;
            cp16(d0 + k * STG + 4096 * hh, kb + off, ok ? 16 : 0);
            cp16(d0 + k * STG + 4096 * hh + 8192, vb + off, ok ? 16 : 0);
          }
        }
        mbar_cp_arrive(mb0 + 8 * k);
      }
    }
    pdl_wait();
    bar_sync(5, 512);
    tr = *trs;
  } else {
    pdl_wait();
    tr = tree_rows(p, b);
    uint2 qx[2][4];
    q_load(qx, p, tok0, h, mt, g, tig);
    const int nt_ = tid - 256, ni = nt_ >> 5, nhh = (nt_ >> 4) & 1, nq4 = nt_ & 15;
    uint2 kraw{}, vraw{};
    i64 crow = 0;
    if (lastp) new_raw(p, tok0 + ni, nhh, nq4, kraw, vraw, crow);
    rtab[nt_] = rope_cs(seq + ni, p.ifq[lane]);
    bar_sync(8, 256);
    u32 qa[2][4];
    q_frag(qa, qx, rtab, tr_byte(tr.d, 2 * mt) & 7, tr_byte(tr.d, 2 * mt + 1) & 7, tig);
    for (int i = 0; i < 4; ++i)
      *(uint2*)(qxs + (h * 128 + tl) * 8 + 2 * i) = {qa[i >> 1][2 * (i & 1)], qa[i >> 1][2 * (i & 1) + 1]};
    if (tid == 256) *trs = tr;
    bar_arrive(5, 512);
    if (lastp) {
      const int u = npast + ni, kt = u >> 6, rw = u & 63;
      for (int k = 0; k < T; ++k)
        if (k < ntile && k != kt) mbar_arrive(mb0 + 8 * k);
      u32 kq, vq;
      new_kv(kraw, vraw, rtab + (tr_byte(tr.d, ni) & 7) * 32 + nq4 * 2, kq, vq);
      kv_put(p, crow, nhh, nq4, kq, vq);
      const int nc = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
      u8* kvs = stg + kt * STG + rw * 128 + ((nc ^ (rw & 7)) << 4) + wi;
      *(u32*)kvs = kq;
      *(u32*)(kvs + 8192) = vq;
      fence_async();
      mbar_arrive(mb0 + 8 * kt);
    }
  }
  if (own) gv = ldq_u4(p.qkv + (tok0 + (orow >> 4)) * p.qkv_stride + 20 * HD + (orow & 15) * HD + 8 * t8);
  WGS ws;
  WGC wc;
  wg_setup(wc, rb, smem_u32(vtb) + wgi * 2 * VTBUF, h, mt, lane);
  const int wlo = seq - W1 - pos0;
  u32 an[2];
  int dj[2];
  for (int rh = 0; rh < 2; ++rh) {
    an[rh] = lastp ? (u32)tr_byte(tr.a, 2 * mt + rh) : 0u;
    dj[rh] = tr_byte(tr.d, 2 * mt + rh);
  }
  u32 vpar = 0;
#pragma unroll 1
  for (int k = ntile - 2 + s; k >= 0; k -= 2) {
    mbar_wait(mb0 + 8 * k, 0);
    // keys from seq + depth - W1 to the part's end, then (last part) ancestor new keys
    const bool m0 = (64 * k + 64 > npast) || (wlo + L - 1 - 64 * k > 0);
    int lo2[2], hi2[2];
    for (int rh = 0; rh < 2; ++rh) {
      lo2[rh] = wlo + dj[rh] - 64 * k;
      hi2[rh] = npast - 64 * k;
    }
    tile(ws, wc, qsa, k * STG, vpar * VTBUF, tig, 1 + wgi, m0, lo2, hi2, p.sc, an, k + 2 < ntile);
    vpar ^= 1u;
  }
  u8* ar = vtb + wgi * 2 * VTBUF;
  for (int rh = 0; rh < 2; ++rh)
    ws.l[rh] = quad_sum(ws.l[rh]);
  bar_sync(1 + wgi, 128);
#define OSEL(i, r) ((r) ? ws.o[(i) + 2] : ws.o[(i)])
  const bool s1 = s != 0;
  for (int k = 0; k < 4; ++k)
    *(float4*)(ar + k * 2048 + tl * 16) = {
        OSEL(8 * k, !s1), OSEL(8 * k + 4, !s1), OSEL(8 * k + 1, !s1), OSEL(8 * k + 5, !s1)};
  *(float2*)(ar + 8192 + tl * 8) = {!s1 ? ws.m[1] : ws.m[0], !s1 ? ws.l[1] : ws.l[0]};
  bar_sync(6 + h, 256);
  float mo[16], Mm, Ls;
  {
    const u8* gp = vtb + (wgi ^ 2) * 2 * VTBUF;
    const float2 mlp = *(const float2*)(gp + 8192 + tl * 8);
    const float m0 = s1 ? ws.m[1] : ws.m[0], l0 = s1 ? ws.l[1] : ws.l[0], mp = mlp.x, lp = mlp.y;
    const float mn = fmaxf(m0, mp), ms = mn == NEG_INF ? 0.f : mn;
    const float w0 = l0 > 0.f ? ex2((m0 - ms) * p.sc) : 0.f, wp = lp > 0.f ? ex2((mp - ms) * p.sc) : 0.f;
    Ls = w0 * l0 + wp * lp;
    Mm = mn * p.sc;
    const float il = rcp0(Ls);
    for (int k = 0; k < 4; ++k) {
      const float4 x = *(const float4*)(gp + k * 2048 + tl * 16);
      mo[4 * k] = (OSEL(8 * k, s1) * w0 + x.x * wp) * il;
      mo[4 * k + 1] = (OSEL(8 * k + 4, s1) * w0 + x.y * wp) * il;
      mo[4 * k + 2] = (OSEL(8 * k + 1, s1) * w0 + x.z * wp) * il;
      mo[4 * k + 3] = (OSEL(8 * k + 5, s1) * w0 + x.w * wp) * il;
    }
  }
#undef OSEL
  bar_sync(6 + h, 256);
  {
    u8* ep = ar + (8 * mt + g) * XR;
    for (int k = 0; k < 4; ++k)
      *(float4*)(ep + (16 * k + 4 * tig) * 4) = as_f4(mo + 4 * k);
    if (tig == 0) *(float2*)(ep + 256) = {Mm, Ls};
    fence_async();
  }
  if (RCV) {
    bar_sync(1 + wgi, 128);
    cl_wait();
  } else {
    if (tid == 0) mbar_expect(mbr, 4 * (C - 1) * nown * XR);
    cl_wait();
    cl_sync();
  }
  if (tl == 0) {
    const u32 src0 = smem_u32(ar);
#pragma unroll 1
    for (int q = 0; q < C; ++q) {
      if (q == part) continue;
      const int qe0 = (32 * q) / C, qn = (32 * (q + 1)) / C - qe0;
      const u32 dst = RCV ? smem_u32(rcv) + (u32)((wgi * (C - 1) + (part < q ? part : part - 1)) * qn) * XR
                          : rb + (u32)((wgi * C + part) * qn) * XR;
      bulk_push(mapa(dst, q), src0 + (u32)qe0 * XR, (u32)qn * XR, mapa(mbr, q));
    }
  }
  if (RCV) __syncthreads();
  // the pushes signal only the receivers' mbarriers (no bulk group to wait on): a CTA exits after every CTA has its
  // bytes
  mbar_wait(mbr, 0);
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (32 * warp < 32 * nown) {
    const u8* rv = RCV ? rcv + (ow * (C - 1) * nown + (oe - e0)) * XR : stg + (ow * C * nown + (oe - e0)) * XR;
    auto rs = [&](int x) { return rv + (RCV && x > part ? x - 1 : x) * nown * XR; };
    const u8* lv = vtb + ow * 2 * VTBUF + oe * XR;
#define ML_OF(x) ((x) < C ? *(const float2*)(((x) == part ? lv : rs(x)) + 256) : make_float2(NEG_INF, 0.f))
    float2 ml = ML_OF(t8), ml2 = ML_OF(t8 + 8);
    float Mx = quad_max(fmaxf(ml.x, ml2.x));
    Mx = fmaxf(Mx, shfl_xor(Mx, 4));
    const float ms = Mx == NEG_INF ? 0.f : Mx;
    const float wl = ml.y > 0.f ? ex2(ml.x - ms) * ml.y : 0.f, wl2 = ml2.y > 0.f ? ex2(ml2.x - ms) * ml2.y : 0.f;
    float acc[8] = {}, lt = 0.f;
#pragma unroll
    for (int src = 0; src < C; ++src) {
      const float4* xp = (const float4*)((src == part ? lv : rs(src)) + 32 * t8);
      const float4 x0 = xp[0], x1 = xp[1];
      const float w = shfl(src < 8 ? wl : wl2, (lane & ~7) | (src & 7));
      lt += w;
      acc[0] += w * x0.x;
      acc[1] += w * x0.y;
      acc[2] += w * x0.z;
      acc[3] += w * x0.w;
      acc[4] += w * x1.x;
      acc[5] += w * x1.y;
      acc[6] += w * x1.z;
      acc[7] += w * x1.w;
    }
    const float rsn = bf_rms<8>(acc, rcp0(lt));
    const u32 gw[4] = {gv.x, gv.y, gv.z, gv.w};
    u32 ov[4];
    for (int d2 = 0; d2 < 4; ++d2)
      ov[d2] = gate2(acc[2 * d2], acc[2 * d2 + 1], rsn, gw[d2]);
    if (own)
      asm volatile(
          "st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(
              (PS ? prow(p, tok0 + (orow >> 4)) + (orow & 15) * HD : p.out + ((i64)b * 128 + orow) * HD) + 8 * t8),
          "r"(ov[0]),
          "r"(ov[1]),
          "r"(ov[2]),
          "r"(ov[3])
          : "memory");
  }
  cl_wait();
}
#define VS_INST(C, T)                                                        \
  extern "C" __global__ void __launch_bounds__(512, 1) vs_c##C(const AP p) { \
    vs<C, T>(p);                                                             \
  }
VS_INST(8, 5)
VS_INST(6, 6) VS_INST(5, 7) VS_INST(4, 9) VS_INST(3, 11)
// VF (full layers): stream-K over request units (past + NEWW + OVH); CTA c's part of b -> slot c + b. Stage s:
// loaders arrive lb, V^T warps wait lb / arrive fb, both head warpgroups wait fb / arrive eb, loaders wait eb.
#define NEWW 512
#define OVH 768
#define SLOT (128 * 64 * 4 + 128 * 8)
#define PIN32(a)               \
  for (int x = 0; x < 32; ++x) \
  asm volatile("" : "+f"(a[x])::"memory")
    DEVI int units_scan(const AP& p, int lane, int s, int& u) {
  u = lane < p.B ? s + NEWW + OVH : 0;
  return wscan(u, lane);
}
DEVI void cp_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
DEVI void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
DEVI void fin16(const AP& p, int b, int r, int q, float (&v)[16]) {
  const float rs = bf_rms<4>(v, 1.f);
  const i64 tok = (i64)b * 8 + (r >> 4);
  const u16* gp = p.qkv + tok * p.qkv_stride + 20 * HD + (r & 15) * HD + 4 * q;
  u16* op = ODST(tok) + (r & 15) * HD + 4 * q;
  for (int m = 0; m < 4; ++m) {
    const uint2 gr = ldq_u2(gp + 16 * m);
    *(uint2*)(op + 16 * m) = {gate2(v[4 * m], v[4 * m + 1], rs, gr.x), gate2(v[4 * m + 2], v[4 * m + 3], rs, gr.y)};
  }
}
#define VF_NST 6
#define VF_VTB 2048
#define VF_STG (8192 + 4 * VF_VTB)
#define VF_E 5
#define VF_MAXSEG 34
#define VF_VS (VF_NST * VF_STG)
#define VF_X (VF_VS + VF_NST * 8192)
#define VF_Q (VF_X + 9 * 256 * 16)
#define VF_PT (VF_Q + 16384)
#define VF_SEG (VF_PT + VF_NST * 256)
#define VF_BAR (VF_SEG + VF_MAXSEG * 16)
DEVI int tile_ord(int t, int J) {
  return t < J ? J - 1 - t : J;
}
DEVI void cp4(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
#define CPA(i, j, e) \
  "mul.wide.u32 a" #i ", %" #e ", 128;\nadd.s64 a" #j ", a" #i ", %13;\nadd.s64 a" #i ", a" #i ", %12;\n"
#define CPG(i, j, n)                                                                                         \
  "cp.async.cg.shared.global [%" #i "], [a" #i "], 16, %" #n ";\ncp.async.cg.shared.global [%" #j "], [a" #j \
  "], 16, %" #n ";\n"
template <int ACC>
DEVI void wg_qk_ss(float (&d)[32], u64 da, u64 db) {
  if constexpr (ACC)
    WGMMA(1, "+f", "%32, %33", "l"(da), "l"(db));
  else
    WGMMA(0, "=f", "%32, %33", "l"(da), "l"(db));
}
extern "C" __global__ void __launch_bounds__(640, 1) vf_main(const AP p) {
  constexpr int L = 8, TT = 64;
  extern __shared__ __align__(1024) u8 smem[];
  const u32 sb0 = smem_u32(smem), rb = (sb0 + 1023u) & ~1023u;
  u8* base = smem + (rb - sb0);
  float4* xch = (float4*)(base + VF_X);
  int4* seg = (int4*)(base + VF_SEG);
  const u32 vs0 = rb + VF_VS, pt0 = rb + VF_PT;
  const u32 fb0 = rb + VF_BAR, eb0 = fb0 + 8 * VF_NST;
  const u32 lb0 = eb0 + 8 * VF_NST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int B = p.B, G = gridDim.x, c = blockIdx.x;
  __shared__ int pre[40], sq[32];
  const int ps = warp == 0 && lane < B ? past(p, lane) : 0;
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < VF_NST; ++s) {
      mbar_init(fb0 + 8 * s, 2);
      mbar_init(eb0 + 8 * s, 2);
    }
#pragma unroll 1
    for (int s = 0; s < VF_NST; ++s)
      mbar_init(lb0 + 8 * s, 64);
    mbar_init_fence();
  }
  if (warp == 0) {
    int u;
    const int x = units_scan(p, lane, ps, u);
    if (lane < B) {
      pre[lane] = x - u;
      sq[lane] = ps;
    }
    if (lane == 31) pre[B] = x;
  }
  __syncthreads();
  if (tid == 0) {
    const i64 NU = pre[B];
    const int c0 = (int)((i64)c * NU / G), c1 = (int)((i64)(c + 1) * NU / G);
    int b = 0, ns = 0, q = 0;
    for (int i = 1; i <= B; ++i)
      b += pre[i] <= c0;
    for (; b < B && pre[b] < c1 && ns < VF_MAXSEG - 1; ++b, ++ns) {
      const int P0 = pre[b], seqb = sq[b], NUb = seqb + NEWW;
      const int u0 = max(c0 - P0 - OVH, 0), u1 = min(c1 - P0 - OVH, NUb);
      const int npast = max(0, min(u1, seqb) - u0);
      const bool has_new = u1 == NUb, complete = c0 <= P0 && has_new;
      seg[ns] = make_int4(b | (has_new ? 256 : 0) | (complete ? 512 : 0), u0, npast, q);
      q += (npast + TT - 1) / TT + (has_new ? 1 : 0);
    }
    seg[ns] = make_int4(-1, 0, 0, q);
    seg[ns + 1] = make_int4(-1, 0, 0, 1 << 30);
    pre[33] = ns;
  }
  __syncthreads();
  pdl_trigger();
  const int nseg = pre[33], Q = seg[nseg].w;
  if (warp < 4) {
    nreg_dec<64>();
    if (warp < 2) {
      const int cc = tid & 7, ro = tid >> 3, er = (lane & 3) + 4 * warp + 8 * (lane >> 2);
      const u32 eslot = ((er & 7) * 8 + (er >> 3)) * 4;
      bool waited = false;
      int jn = -1, qn_end = 0, nrn = 0, ncnt = 0, sn = 0;
      const int* pn = p.page_table;
      auto ent = [&](int qn) {
        while (qn >= qn_end) {
          const int4 a = seg[++jn];
          qn_end = seg[jn + 1].w;
          ncnt = (a.z + TT - 1) / TT;
          pn = p.page_table + (i64)(a.x & 255) * p.max_pages + a.y + er + TT * ncnt;
          nrn = a.z - er - TT * ncnt;
        }
        if (ncnt-- > 0) {
          pn -= TT;
          nrn += TT;
        } else
          nrn = 0;
        cp4(pt0 + sn * 256 + eslot, nrn > 0 ? pn : p.page_table, nrn > 0 ? 4 : 0);
        if (++sn == VF_NST) sn = 0;
      };
#pragma unroll 1
      for (int q = 0; q < VF_E; ++q)
        ent(q);
      cp_commit();
      cp_wait<0>();
      int jc = -1, qc_end = 0, nvr = 0, nv0 = 0, ccnt = 0, bx = 0, st = 0, ph = 0;
      u32 ready = 0;
#pragma unroll 1
      for (int q = 0; q < Q; ++q) {
        while (q >= qc_end) {
          const int4 a = seg[++jc];
          qc_end = seg[jc + 1].w;
          ccnt = (a.z + TT - 1) / TT;
          nvr = nv0 = a.z - ro - TT * ccnt;
          bx = a.x;
        }
        nvr = ccnt-- > 0 ? nvr + TT : nv0;
        const u32 kd = rb + st * VF_STG, vd = vs0 + st * 8192, lbar = lb0 + 8 * st;
        if (!ready) mbar_wait(eb0 + 8 * st, ph ^ 1);
        if ((bx & 256) && nvr + ro <= 0) {
          if (!waited) {
            pdl_wait();
            waited = true;
          }
          const int b = bx & 255;
#pragma unroll 1
          for (int ps = 0; ps < 4; ++ps) {
            const int item = 64 * ps + tid, ni = item >> 5, nhh = (item >> 4) & 1, nq4 = item & 15;
            const i64 tok = (i64)b * L + ni;
            uint2 kraw, vraw;
            i64 crow;
            new_raw(p, tok, nhh, nq4, kraw, vraw, crow);
            u32 kq, vq;
            new_kv(kraw, vraw, nullptr, kq, vq);
            kv_put(p, crow, nhh, nq4, kq, vq);
            const u32 off = ni * 128 + (((nhh * 4 + (nq4 >> 2)) ^ ni) << 4) + (nq4 & 3) * 4;
            sts32(kd + off, kq);
            sts32(vd + off, vq);
          }
          for (int z = tid; z < 56 * 8; z += 64)
            sts4(vd + 1024 + 16 * z, uint4{});
          fence_async();
          mbar_arrive(lbar);
        } else {
          cp_wait<VF_E - 1>();
          __syncwarp();
          const u32 pts = pt0 + st * 256, d0 = ro * 128 + ((cc ^ ro) << 4);
          const u8* kb = p.k_cache + 16 * cc;
          const u8* vb = p.v_cache + 16 * cc;
          for (int h4 = 0; h4 < 2; ++h4) {
            u32 e[4];
            lds4(e, pts + 32 * ro + 16 * h4);
            const u32 dk = kd + d0 + 4096 * h4, dv = vd + d0 + 4096 * h4;
            asm volatile("{\n.reg .b64 a<8>;\n" CPA(0, 1, 8) CPA(2, 3, 9) CPA(4, 5, 10) CPA(6, 7, 11) CPG(0, 1, 14)
                             CPG(2, 3, 15) CPG(4, 5, 16) CPG(6, 7, 17) "}" ::"r"(dk),
                         "r"(dv),
                         "r"(dk + 1024),
                         "r"(dv + 1024),
                         "r"(dk + 2048),
                         "r"(dv + 2048),
                         "r"(dk + 3072),
                         "r"(dv + 3072),
                         "r"(e[0]),
                         "r"(e[1]),
                         "r"(e[2]),
                         "r"(e[3]),
                         "l"(kb),
                         "l"(vb),
                         "r"(32 * h4 < nvr ? 16 : 0),
                         "r"(32 * h4 + 8 < nvr ? 16 : 0),
                         "r"(32 * h4 + 16 < nvr ? 16 : 0),
                         "r"(32 * h4 + 24 < nvr ? 16 : 0)
                         : "memory");
          }
          mbar_cp_arrive(shfl(lbar + (tid >> 10), 0));
        }
        ent(q + VF_E);
        cp_commit();
        if (++st == VF_NST) {
          st = 0;
          ph ^= 1;
        }
        ready = mbar_test(eb0 + 8 * st, ph ^ 1);
      }
      if (!waited) pdl_wait();
    } else {
      const int tw = shfl(warp - 2, 0), k8 = lane >> 3, r8 = lane & 7;
      int st = 0, ph = 0;
#pragma unroll 1
      for (int q = 0; q < Q; ++q) {
        mbar_wait(lb0 + 8 * st, ph);
        const u32 sto = rb + st * VF_STG, vsrc = vs0 + st * 8192;
        for (int half = 0; half < 2; ++half) {
          const int pw = 2 * tw + half;
          u32 x[4][4];
          for (int hn = 0; hn < 4; ++hn)
            ldsm4t(vsrc + (32 * (hn & 1) + 8 * k8 + r8) * 128 + (((4 * (hn >> 1) + pw) ^ r8) << 4), x[hn]);
          for (int hn = 0; hn < 4; ++hn)
            stsm4t(sto + 8192 + hn * VF_VTB + (2 * pw + (k8 & 1)) * 256 + (k8 >> 1) * 128 + r8 * 16, x[hn]);
        }
        fence_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(fb0 + 8 * st);
        if (++st == VF_NST) {
          st = 0;
          ph ^= 1;
        }
      }
    }
    return;
  }
  nreg_inc<104>();
  const int ct = tid - 128, li = ct >> 8, h = (ct >> 7) & 1, mt = (ct >> 5) & 3, tl = ct & 127, tt = ct & 255;
  const int g = lane >> 2, tig = lane & 3, cw = shfl(warp - 4, 0), uli = cw >> 3, uh = (cw >> 2) & 1;
  const u32 urb = shfl(rb + (tid >> 10), 0), qwb = rb + VF_Q + (li * 2 + h) * 4096;
  const u32 qlo = ((urb + VF_Q + (uli * 2 + uh) * 4096) >> 4) | (1u << 16), klo0 = ((urb + uh * 64) >> 4) | (1u << 16);
  const u32 vlo0 = ((urb + 8192 + 2 * uh * VF_VTB) >> 4) | (8u << 16);
  auto mkd = [](u32 lo, u32 hi) { return ((u64)hi << 32) | lo; };
  const float sc = p.sc;
  bool first = true;
#pragma unroll 1
  for (int k = 0; k < nseg; ++k) {
    int4 sg = seg[k];
    sg.x = shfl(sg.x, 0);
    sg.z = shfl(sg.z, 0);
    sg.w = shfl(sg.w, 0);
    const int b = sg.x & 255, npast = sg.z, q0 = sg.w;
    const bool has_new = sg.x & 256, complete = sg.x & 512;
    const int J = (npast + TT - 1) / TT, nt = J + (has_new ? 1 : 0);
    const int n = nt > uli ? (nt - uli + 1) >> 1 : 0;
    const i64 tok0 = (i64)b * L;
    if (first) {
      pdl_wait();
      first = false;
    }
    u32 an2 = 0;
    {
      uint2 qx[2][4];
      q_load(qx, p, tok0, h, mt, g, tig);
      u32 anv = ((2u << (2 * mt)) - 1u) | (((4u << (2 * mt)) - 1u) << 8);
      if (p.anc) {
        u16 x;
        asm volatile("ld.global.u16 %0, [%1];" : "=h"(x) : "l"(p.anc + 8 * b + 2 * mt) : "memory");
        anv = x;
      }
      u32 qa[2][4];
      q_frag(qa, qx, nullptr, 0, 0, tig);
      for (int ks = 0; ks < 2; ++ks)
        for (int w4 = 0; w4 < 4; ++w4) {
          const int mrow = 16 * mt + 8 * (w4 & 1) + g, c = 2 * ks + (w4 >> 1);
          sts32(qwb + (mrow >> 3) * 512 + (mrow & 7) * 64 + ((c ^ ((mrow >> 1) & 3)) << 4) + 4 * tig, qa[ks][w4]);
        }
      fence_async();
      bar_sync(3 + li * 2 + h, 128);
      an2 = shfl(anv, 0);
    }
    float o[32] = {}, m[2] = {NEG_INF, NEG_INF}, mb[2] = {-8.f, -8.f}, l[2] = {}, al[2] = {1.f, 1.f};
    float s[32];
    int cst = (q0 + uli) % VF_NST, cph = ((q0 + uli) / VF_NST) & 1, pst = 0;
    auto qk = [&]() {
      mbar_wait(fb0 + 8 * cst, cph);
      const u32 kl = klo0 + cst * (VF_STG >> 4);
      wg_fence();
      wg_qk_ss<0>(s, mkd(qlo, 0x80000020u), mkd(kl, 0x40000040u));
      wg_qk_ss<1>(s, mkd(qlo + 2, 0x80000020u), mkd(kl + 2, 0x40000040u));
      wg_commit();
    };
#pragma unroll 1
    for (int j = 0; j < n; ++j) {
      const int hp = npast - TT * tile_ord(uli + 2 * j, J);
      qk();
      wg_wait<0>();
      PIN32(s);
      PIN32(o);
      if (j > 0 && tl == 0) mbar_arrive(eb0 + 8 * pst);
      if (hp < TT) {
        const u32 sh = an2 >> (2 * tig);
        for (int x = 0; x < 32; ++x) {
          const int key = 8 * (x >> 2) + 2 * tig + (x & 1);
          if (key >= (x < 4 && hp <= 0 && ((sh >> (8 * ((x >> 1) & 1) + (x & 1))) & 1u) ? TT : max(hp, 0)))
            s[x] = NEG_INF;
        }
      }
      float mx[2];
      row_max(s, mx);
      for (int rh = 0; rh < 2; ++rh)
        mx[rh] = fmaxf(m[rh], mx[rh]);
      if (__any_sync(~0u, mx[0] != m[0] || mx[1] != m[1])) {
        for (int rh = 0; rh < 2; ++rh)
          sm_step(m[rh], l[rh], al[rh], mb[rh], mx[rh], sc);
        if (j > 0)
          for (int x = 0; x < 32; ++x)
            o[x] *= al[(x >> 1) & 1];
      }
      const u32 vl = vlo0 + cst * (VF_STG >> 4);
      pst = cst;
      cst += 2;
      if (cst >= VF_NST) {
        cst -= VF_NST;
        cph ^= 1;
      }
      float u[2][4];
      for (int nb = 0; nb < 2; ++nb) {
        float f[16];
        for (int e = 0; e < 16; ++e) {
          const int x = 16 * nb + e, rh = (x >> 1) & 1;
          f[e] = ex2(fmaf(s[x], sc, -mb[rh]));
          u[rh][(x >> 2) & 3] = x & 17 ? u[rh][(x >> 2) & 3] + f[e] : f[e];
        }
        u32 pa[4];
        p_frag(pa, f);
        wg_fence();
        wg_pv(o, pa, mkd(vl + nb * (VF_VTB >> 4), 0x10u));
        wg_commit();
      }
      for (int rh = 0; rh < 2; ++rh)
        l[rh] += (u[rh][0] + u[rh][1]) + (u[rh][2] + u[rh][3]);
    }
    wg_wait<0>();
    PIN32(o);
    if (n > 0 && tl == 0) mbar_arrive(eb0 + 8 * pst);
    for (int rh = 0; rh < 2; ++rh)
      l[rh] = quad_sum(l[rh]);
    bar_sync(1, 512);
    if (li == 1) {
      for (int x = 0; x < 8; ++x)
        xch[x * 256 + tt] = as_f4(o + 4 * x);
      xch[8 * 256 + tt] = {m[0], m[1], l[0], l[1]};
    }
    bar_sync(1, 512);
    if (li == 0) {
      const float4 ml = xch[8 * 256 + tt];
      const float m1[2] = {ml.x, ml.y}, l1[2] = {ml.z, ml.w};
      float a0[2], a1[2], M[2], Ls[2];
      for (int rh = 0; rh < 2; ++rh) {
        M[rh] = fmaxf(m[rh], m1[rh]);
        const float ms = M[rh] == NEG_INF ? 0.f : M[rh];
        a0[rh] = ex2((m[rh] - ms) * sc);
        a1[rh] = ex2((m1[rh] - ms) * sc);
        Ls[rh] = l[rh] * a0[rh] + l1[rh] * a1[rh];
      }
      for (int x = 0; x < 8; ++x) {
        const float4 v = xch[x * 256 + tt];
        const float vx[4] = {v.x, v.y, v.z, v.w};
        for (int e = 0; e < 4; ++e)
          o[4 * x + e] = o[4 * x + e] * a0[e >> 1] + vx[e] * a1[e >> 1];
      }
      u8* dst = p.ws + (i64)(c + b) * SLOT;
      for (int rh = 0; rh < 2; ++rh) {
        const int r = (2 * mt + rh) * 16 + h * 8 + g;
        const float inv = rcp0(Ls[rh]);
        float v[16];
        for (int e = 0; e < 16; ++e)
          v[e] = o[8 * (e >> 2) + 4 * (e & 1) + 2 * rh + ((e >> 1) & 1)] * inv;
        if (complete)
          fin16(p, b, r, tig, v);
        else {
          for (int x = 0; x < 4; ++x)
            *(float4*)(dst + (r * HD + 16 * x + 4 * tig) * 4) = as_f4(v + 4 * x);
          if (tig == 0) *(float2*)(dst + 32768 + 8 * r) = {M[rh] * sc, Ls[rh]};
        }
      }
    }
  }
}
DEVI void vfc_range(const AP& p, int b, int lane, int& cf, int& P) {
  const int B = p.B;
  int u;
  const int x = units_scan(p, lane, lane < B ? past(p, lane) : 0, u);
  const i64 NU = shfl(x, 31), G = p.G;
  const i64 pb0 = shfl(x - u, b), pb1 = shfl(x, b);
  cf = B == 1 ? 0 : (int)(((pb0 + 1) * G - 1) / NU);
  P = B == 1 ? (int)G : (int)((pb1 * G - 1) / NU) - cf + 1;
}
template <int RPC>
DEVI void vf_comb(const AP& p) {
  constexpr int NPG = 32 / RPC, NB = 8;
  __shared__ float red[NPG > 1 ? NPG / 2 : 1][RPC][16][6];
  const int tid = threadIdx.x, lane = tid & 31, dg = tid & 15, r = (tid >> 4) % RPC, pg = (tid >> 4) / RPC;
  const int b = blockIdx.y, R = blockIdx.x * RPC + r;
  if (tid == 0) pdl_trigger();
  int cf, P;
  vfc_range(p, b, lane, cf, P);
  if (P <= 1) return;
  const i64 tok = (i64)b * 8 + (R >> 4);
  pdl_wait();
  uint2 gr{};
  float M = NEG_INF, Lw = 0.f, a[4] = {};
#pragma unroll 1
  for (int k0 = pg; k0 < P; k0 += NB * NPG) {
    float2 ml[NB];
    float4 ok[NB];
    for (int i = 0; i < NB; ++i) {
      const int k = k0 + i * NPG;
      ml[i] = {NEG_INF, 0.f};
      ok[i] = {};
      if (k < P) {
        const u8* sl = p.ws + (i64)(cf + k + b) * SLOT;
        ml[i] = __ldcg((const float2*)(sl + 32768 + 8 * R));
        ok[i] = __ldcg((const float4*)(sl + (R * HD + 4 * dg) * 4));
      }
    }
    if (k0 == pg) gr = ldq_u2(p.qkv + tok * p.qkv_stride + 20 * HD + (R & 15) * HD + 4 * dg);
    for (int i = 0; i < NB; ++i) {
      if (ml[i].y > 0.f) {
        const float mn = fmaxf(M, ml[i].x);
        const float sa = ex2(M - mn), w = ml[i].y * ex2(ml[i].x - mn);
        Lw = Lw * sa + w;
        a[0] = a[0] * sa + w * ok[i].x;
        a[1] = a[1] * sa + w * ok[i].y;
        a[2] = a[2] * sa + w * ok[i].z;
        a[3] = a[3] * sa + w * ok[i].w;
        M = mn;
      }
    }
  }
  for (int hf = NPG / 2; hf >= 1; hf >>= 1) {
    float* rr = red[pg & (hf - 1)][r][dg];
    if (pg >= hf && pg < 2 * hf) {
      rr[0] = M;
      rr[1] = Lw;
      for (int e = 0; e < 4; ++e)
        rr[2 + e] = a[e];
    }
    __syncthreads();
    if (pg < hf && rr[1] > 0.f) {
      const float mn = fmaxf(M, rr[0]);
      const float sa = ex2(M - mn), w = ex2(rr[0] - mn);
      Lw = Lw * sa + w * rr[1];
      for (int e = 0; e < 4; ++e)
        a[e] = a[e] * sa + w * rr[2 + e];
      M = mn;
    }
    __syncthreads();
  }
  if (pg) return;
  const float rs = bf_rms<16>(a, rcp0(Lw));
  *(uint2*)(ODST(tok) + (R & 15) * HD + 4 * dg) = {gate2(a[0], a[1], rs, gr.x), gate2(a[2], a[3], rs, gr.y)};
}
#define VFC_INST(RPC)                                                          \
  extern "C" __global__ void __launch_bounds__(512) vf_comb##RPC(const AP p) { \
    vf_comb<RPC>(p);                                                           \
  }
VFC_INST(1) VFC_INST(2) VFC_INST(4) VFC_INST(8) VFC_INST(16) VFC_INST(32)
