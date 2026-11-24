// Fused NVFP4 W4A16 MoE of SP batches >= 2049 tokens (SM90). decode() and the weight layout are moe_w4a16.cuh's,
// adapted from https://github.com/gauravjain14/muse-okc-submissions (muse_moe_w4a16; Apache-2.0).
#include "ptx.cuh"
namespace pfm {
// NG guests: other ranks' experts, pulled per batch
constexpr int NL = 32, NG = 2, NE = NL + NG, TOPK = 8, WBLK = 35840;
constexpr int NSLOT = 4, WOFF = 32768, SOFF = 36864, SLOTB = 38912;
constexpr int R2MAX = 96, WOFF_R = 16384, SOFF_R1 = 24576, SOFF_R2 = 32768;
constexpr int EWS = 3328;
constexpr int STG = NSLOT * SLOTB, BAR = STG + 69632;
constexpr int C_QUEUE = 96, C_CDONE = 97, C_LDONE = 98, C_EXIT = 99;
constexpr int C_GW = 104, C_DONE = 128, C_READY = 1152;
constexpr int GW13 = 16 * 32, GW2 = 16 * 16;
constexpr int C_TOT = 3200, C_LB = 3240, C_PRE = 3280, C_WORDS = 3400;
// pull claim word: own 128 B line, apart from the producer's C_QUEUE
constexpr int C_GQUEUE = 3328;
constexpr int K_G1 = 0, K_G2 = 1, K_CNT = 2, K_CMB = 4, K_PLAN = 5, K_LIST = 7, K_NOP = 8;
constexpr int MAXLAG = 16, MAXSEG = 2 * (NE + 1 + MAXLAG);
// item widths: 32 .. 256 step 32; 240 = an expert's last 256-wide tile with >= 16 padding rows
constexpr int NW = 9;
DEVI int widx(int n) {
  return n & 31 ? 8 : (n >> 5) - 1;
}
struct alignas(64) Tmap {
  u64 v[16];
};
struct Params {
  Tmap wm[6], wsg[6];
  const Tmap *th, *tx2, *th2, *txh, *thh;
  const int* ids;
  const u8 *w13, *w2;
  const float *gs13, *gs2, *nw;
  u16* xt;
  int* ctr;
  int *rev, *ucnt;
  int lag, M, cap, first, ne, R;
  const u16* xs[8];
  u16* os[8];
  int cl, pre;
  u8 *sw13, *sw2;
  float* sgs;
  const int* gsrc;
  const u64* peer;
};
struct Plan {
  int nwork, nlu, u0, u1, pre;
  int seg0[MAXSEG + 1], sege[MAXSEG];
  int ord[NE], rk[NE];
  int cnt[NE], nt[NE], q[NE], r[NE], off[NE], tb[NE], sh[NE], cut[NE];
};
DEVI void tma_load3(u32 dst, const Tmap* m, int c1, int c2, u32 bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::
          "r"(dst),
      "l"(m),
      "r"(0),
      "r"(c1),
      "r"(c2),
      "r"(bar)
      : "memory");
}
DEVI void tma_load4(u32 dst, const Tmap* m, int c1, int c2, int c3, u32 bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], "
      "[%6];" ::"r"(dst),
      "l"(m),
      "r"(0),
      "r"(c1),
      "r"(c2),
      "r"(c3),
      "r"(bar)
      : "memory");
}
DEVI void tmap_acquire(const Tmap* m) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"(m) : "memory");
}
DEVI void tma_store(const Tmap* m, int c0, int c1, u32 src) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(m), "r"(c0), "r"(c1), "r"(src)
      : "memory");
}
DEVI void wait_geq(const int* p, int v) {
  while (ld_acquire(p) < v)
    __nanosleep(128);
  fence_async_global();
}
DEVI uint4 ld_na(const uint4* a) {
  uint4 v;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(a));
  return v;
}
DEVI void stmatrix_t2(u32 a, u32 r0, u32 r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};" ::"r"(a), "r"(r0), "r"(r1) : "memory");
}
DEVI float silu(float v) {
  return __fdividef(v, 1.f + ex2(v * -1.4426950216293334961f));
}
DEVI int tile_of_row(const Plan& pl, int e, int row) {
  const int q = pl.q[e], r = pl.r[e], v = (row + (row >= pl.cut[e] ? pl.sh[e] : 0)) >> 5;
  return pl.tb[e] + (v < r * (q + 1) ? v / (q + 1) : r + (v - r * (q + 1)) / q);
}
constexpr int B_F = 0, B_E = 8 * NSLOT, B_IF = 16 * NSLOT, B_IE = B_IF + 16, B_PF = B_IE + 16, B_PE = B_PF + 8,
              B_PL = B_PE + 8;
DEVI void count_unit(const Params& p, int u, int q, int lane, int ctid, int* hist) {
  if (ctid < 64) hist[ctid] = 0;
  bar_sync(1, 256);
  const int t = u * 128 + q * 16 + (lane >> 1);
  if (t < p.M) {
    const int4 v = AS(const int4, p.ids)[2 * t + (lane & 1)];
    const int e4[4] = {v.x, v.y, v.z, v.w};
    for (int k = 0; k < 4; ++k)
      if (e4[k] >= 0 && e4[k] < p.ne) atomicAdd(hist + e4[k], 1);
  }
  bar_sync(1, 256);
  if (ctid < 32) {
    const int c = hist[ctid], cg = hist[NL + ctid];
    int s = c + cg;
    for (int o = 16; o; o >>= 1)
      s += shfl_xor(s, o);
    p.ucnt[u * 64 + ctid] = c;
    p.ucnt[u * 64 + NL + ctid] = ctid == 31 ? s : cg;
    if (c) atomicAdd(p.ctr + C_TOT + ctid, c);
    if (cg) atomicAdd(p.ctr + C_TOT + NL + ctid, cg);
    if (ctid == 0 && s) atomicAdd(p.ctr + C_TOT + NE, s);
  }
  bar_sync(1, 256);
  if (ctid == 0) {
    __threadfence();
    red_release(p.ctr + C_CDONE, 1);
  }
}
DEVI void cp_async4(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
DEVI u32 tid_x() {
  u32 r;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r));
  return r;
}
DEVI u32 lds32(u32 a) {
  u32 v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
DEVI void arrive_peer(u32 bar, u32 cta) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(mapa(bar, cta)) : "memory");
}
DEVI void st_peer(u32 addr, u32 cta, int v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(mapa(addr, cta)), "r"(v) : "memory");
}
DEVI bool mbar_test_cl(u32 b, u32 ph) {
  u32 ok;
  asm volatile(
      "{ .reg .pred p; mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(ok)
      : "r"(b), "r"(ph)
      : "memory");
  return ok != 0;
}
DEVI void mbar_wait_cl(u32 b, u32 ph) {
  asm volatile(
      "{\n.reg .pred P;\nW_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1;\n@!P bra "
      "W_%=;\n}\n" ::"r"(b),
      "r"(ph)
      : "memory");
}
DEVI void tma_load_mc(u32 dst, const Tmap* m, int c0, int c1, u32 bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, "
      "%3}], [%4], %5;" ::"r"(dst),
      "l"(m),
      "r"(c0),
      "r"(c1),
      "r"(bar),
      "h"((u16)3)
      : "memory");
}
DEVI void release_slot(u32 bar, u32 crank) {
  mbar_arrive(bar);
  // default .release orders the slot's reads
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(mapa(bar, crank ^ 1)) : "memory");
}
template <int N>
DEVI void wgmma_rs(float* d, const u32* a, u64 desc);
// @WGMMA_RS@
DEVI u32 word4(const uint4& w, int j) {
  return j == 0 ? w.x : j == 1 ? w.y : j == 2 ? w.z : w.w;
}
DEVI void decode(const uint4& w, int j, u32 R, u32 (&a)[4]) {
  constexpr u32 M = 0x81C081C0u;
  const u32 q = word4(w, j), qp = word4(w, j ^ 1);
  const u32 F = j & 1 ? __byte_perm(qp, q, 0x6341) : __byte_perm(q, qp, 0x6341);
  u32 o1, o2, o3;
  if (!(j & 1)) {
    o1 = ((q << 2) ^ F) & M;
    o2 = (q << 6) & M;
    o3 = rotl(q, 12) & M;
  } else {
    o1 = (q << 4) & M;
    o2 = rotl(q, 10) & M;
    o3 = (rotl(q, 14) ^ F) & M;
  }
  a[0] = bmul2s<0>(q & M, R);
  a[1] = bmul2s<1>(o1, R);
  a[2] = bmul2s<0>(o2, R);
  a[3] = bmul2s<1>(o3, R);
}
template <int KK>
DEVI u32 scale_pair(u32 odd, u32 w0, u32 w1, u32 w2) {
  constexpr u32 M6 = 0x03F003F0u, NB = 0x00F000F0u, C = 0x78007800u;
  u32 v;
  if (KK == 3)
    v = odd ? (((w1 << 4) & NB) | ((w2 << 6) & ~NB)) : (((w0 << 4) & NB) | ((w2 << 8) & ~NB));
  else
    v = (KK == 0 ? w0 : KK == 1 ? w1 : w2) >> (odd ? 6 : 0);
  return (v & M6) | C;
}
DEVI u32 scale_kk(int kk, u32 odd, u32 w0, u32 w1, u32 w2) {
  return kk == 0   ? scale_pair<0>(odd, w0, w1, w2)
         : kk == 1 ? scale_pair<1>(odd, w0, w1, w2)
         : kk == 2 ? scale_pair<2>(odd, w0, w1, w2)
                   : scale_pair<3>(odd, w0, w1, w2);
}
DEVI int halves(const Plan& pl, int e) {
  const int q = pl.q[e], r = pl.r[e];
  return r * (32 * (q + 1) <= R2MAX ? 1 : 2) + (pl.nt[e] - r) * (32 * q <= R2MAX ? 1 : 2);
}
DEVI void make_plan(const Params& p, Plan& pl, int lane) {
  const int npos = NE + 1 + p.lag;
  const int u0 = pl.u0, nu = (p.M + 127) >> 7, budget = p.cap - 1088;
  const bool gl = lane < NG;
  int u1 = nu, c = 0, cg = 0;
  const int* src = p.ctr + (pl.pre ? C_PRE : p.cl ? C_LB : C_TOT);
  if (p.cl || __ldcg(src + NE) <= budget) {
    c = __ldcg(src + lane);
    if (gl) cg = __ldcg(src + NL + lane);
  } else {
    int run = 0;
    for (int b = u0; b < nu; b += 32) {
      const int inc = wscan(b + lane < nu ? __ldcg(p.ucnt + (b + lane) * 64 + 63) : 0, lane);
      const u32 over = ballot(b + lane < nu && run + inc > budget);
      if (over) {
        u1 = b + __ffs(over) - 1;
        break;
      }
      run += shfl(inc, 31);
    }
    for (int u = u0; u < u1; u += 16) {
      int v[16], w[16];
      for (int k = 0; k < 16; ++k)
        v[k] = u + k < u1 ? __ldcg(p.ucnt + (u + k) * 64 + lane) : 0;
      for (int k = 0; k < 16; ++k)
        w[k] = u + k < u1 && gl ? __ldcg(p.ucnt + (u + k) * 64 + NL + lane) : 0;
      for (int k = 0; k < 16; ++k)
        if (u + k < u1) {
          c += v[k];
          cg += w[k];
        }
    }
  }
  const int nt = (c + 255) >> 8, u = (c + 31) >> 5, ntg = (cg + 255) >> 8, ug = (cg + 31) >> 5;
  // u <= 8 nt: the quotient is 0..8, counted without a division
  auto qdiv = [](int u, int nt) {
    int q = 0;
    for (int k = 1; k <= 8; ++k)
      q += u >= k * nt;
    return nt ? q : 0;
  };
  auto shr = [&](int c, int nt, int u) {
    const int q = qdiv(u, nt), r = u - q * nt;
    return (r ? q == 7 : q == 8) ? (32 * u - c) & 16 : 0;
  };
  const int sh = shr(c, nt, u), shg = shr(cg, ntg, ug);
  const int off = wscan(32 * u - sh, lane), tb = wscan(nt, lane);
  const int offg = wscan(32 * ug - shg, lane) + shfl(off, 31), tbg = wscan(ntg, lane) + shfl(tb, 31);
  int pos = 0, posg = NL;
#pragma unroll 4
  for (int k = 0; k < NL; ++k) {
    const int ck = shfl(c, k);
    pos += ck > c || (ck == c && k < lane);
  }
  for (int k = 0; k < NG; ++k) {
    const int ck = shfl(cg, k);
    posg += ck > cg || (ck == cg && k < lane);
  }
  auto put = [&](int e, int c, int nt, int u, int off, int tb, int pos, int sh) {
    const int q = qdiv(u, nt), r = u - q * nt, s = r ? r - 1 : nt - 1;
    pl.cnt[e] = c;
    pl.nt[e] = nt;
    pl.q[e] = q;
    pl.r[e] = r;
    pl.off[e] = off - (32 * u - sh);
    pl.tb[e] = tb - nt;
    pl.ord[pos] = e;
    pl.rk[e] = pos;
    pl.sh[e] = sh;
    pl.cut[e] = 32 * (s * q + min(s, r)) + 256 - sh;
  };
  put(lane, c, nt, u, off, tb, pos, sh);
  if (gl) put(NL + lane, cg, ntg, ug, offg, tbg, posg, shg);
  if (lane == 0) {
    pl.u1 = u1;
    pl.nlu = p.cl || pl.pre ? 0 : u1 - u0;
  }
  __syncwarp();
  int base = 0;
  for (int i0 = 0; i0 < npos; i0 += 32) {
    const int i = i0 + lane;
    int sz[2] = {0, 0}, ek[2] = {0, 0};
    if (i >= 1 && i - 1 < NE) {
      ek[0] = pl.ord[i - 1];
      sz[0] = 32 * halves(pl, ek[0]);
    }
    if (i >= 1 + p.lag && i - 1 - p.lag < NE) {
      ek[1] = pl.ord[i - 1 - p.lag];
      sz[1] = 16 * halves(pl, ek[1]);
    }
    sz[1] += sz[1] & 1;
    const int own = sz[0] + sz[1];
    const int tot = wscan(own, lane);
    if (i < npos) {
      int st = base + tot - own;
      for (int k = 0; k < 2; ++k) {
        pl.seg0[2 * i + k] = st;
        pl.sege[2 * i + k] = ek[k] | (k << 8);
        st += sz[k];
      }
    }
    base += shfl(tot, 31);
  }
  if (lane == 0) {
    pl.seg0[2 * npos] = base;
    pl.nwork = base;
  }
  __syncwarp();
}
constexpr int B_MF = 224, B_ME = 232, B_MBOX = 240;
DEVI void prefetch_chunk(const u8* b, int hf) {
  if (hf == 1)
    prefetch_l2(b + 16384, 16384 + 3072);
  else
    prefetch_l2(b, WBLK);
}
template <int KPS, bool RG2>
DEVI void load_item(
    u32 ring,
    u32 bars,
    const Tmap* tx,
    const Tmap* tw,
    const u8* wb,
    int n,
    int row0,
    int blk0,
    int nrb,
    int hf,
    int& slot,
    u32& ph,
    u32 crank,
    int* queue,
    u32 mph,
    int& nxt) {
  constexpr int WB = RG2 ? 8192 : 4096, SB = RG2 ? 3072 : 1536;
  constexpr int WO = RG2 ? WOFF_R : WOFF, SO = RG2 ? (KPS == 2 ? SOFF_R2 : SOFF_R1) : SOFF;
#pragma unroll 4
  for (int kb = 0; kb < 64; kb += KPS) {
    const int kk = kb & 3, blk = blk0 + (kb >> 2) * nrb;
    if (kk == 0 && kb < 60) prefetch_chunk(wb + (size_t)(blk + nrb) * WBLK, hf);
    if (kb == 60 && crank == 0) {
      nxt = atomicAdd(queue, 2);
      mbar_wait_cl(bars + B_ME, mph ^ 1);
      st_peer(bars + B_MBOX, 1, nxt);
      arrive_peer(bars + B_MF, 1);
    }
    if (kb == 62 && crank == 1 && mbar_test_cl(bars + B_MF, mph)) {
      nxt = lds32(bars + B_MBOX) + 1;
      arrive_peer(bars + B_ME, 0);
    }
    mbar_wait(bars + B_E + 8 * slot, ph ^ 1);
    const u32 fb = bars + B_F + 8 * slot, st = ring + slot * SLOTB;
    mbar_expect(fb, KPS * (n * 128 + WB) + (kk ? 0 : SB));
    if (KPS == 2)
      tma_load3(st, tx, row0, kb, fb);
    else
      tma_load_mc(st + crank * n * 64, tx, kb * 64, row0 + crank * (n >> 1), fb);
    tma_load4(st + WO, tw, kk, RG2 ? 0 : hf * 8, blk, fb);
    if (!kk) bulk_g2s(st + SO, wb + (size_t)blk * WBLK + 32768 + (RG2 ? 0 : hf * 1536), SB, fb);
    advance<NSLOT>(slot, ph);
  }
}
DEVI void producer(const Params& p, u8* sm, Plan& pl, u32 bars, int* iq) {
  int slot = 0, seg = 0, stage = 0, gready = 0, nxt = -1;
  u32 ph = 0, qs = 0, qph = 0, mph = 0;
  const u32 crank = cl_rank();
  const int ncu = pl.pre ? 0 : ((p.M + 127) >> 7) - pl.u0;
  const u32 ring = smem_u32(sm);
  auto qget = [&] {
    mbar_wait(bars + B_IE + 8 * qs, qph ^ 1);
    return iq + 8 * qs;
  };
  auto qput = [&] {
    mbar_arrive(bars + B_IF + 8 * qs);
    advance<2>(qs, qph);
  };
  auto plan = [&] {
    wait_geq(p.ctr + C_CDONE, ncu);
    *qget() = K_PLAN;
    qput();
    mbar_wait(bars + B_PL, 0);
    stage = 1;
  };
  if (pl.pre) plan();
  for (;;) {
    int it;
    if (crank == 0) {
      if (nxt >= 0) {
        it = nxt;
        nxt = -1;
      } else {
        it = atomicAdd(p.ctr + C_QUEUE, 2);
        mbar_wait_cl(bars + B_ME, mph ^ 1);
        st_peer(bars + B_MBOX, 1, it);
        arrive_peer(bars + B_MF, 1);
      }
    } else {
      if (nxt >= 0) {
        it = nxt;
        nxt = -1;
      } else {
        mbar_wait_cl(bars + B_MF, mph);
        it = *AS(volatile int, sm + BAR + B_MBOX) + 1;
        arrive_peer(bars + B_ME, 0);
      }
    }
    mph ^= 1;
    int kind = -1, a = 0, b = 0, row0 = 0, n = 0, tile = 0;
    if (it < ncu) {
      kind = K_CNT;
      a = pl.u0 + it;
    } else {
      if (stage < 1) plan();
      int w = it - ncu;
      if (w < pl.nlu) {
        kind = K_LIST;
        a = pl.u0 + w;
      } else {
        if (stage < 2) {
          wait_geq(p.ctr + C_LDONE, pl.nlu);
          stage = 2;
        }
        w -= pl.nlu + ((ncu + pl.nlu) & 1);
        if (w < 0) {
          kind = K_NOP;
        } else if (w < pl.nwork) {
          while (pl.seg0[seg + 1] <= w)
            ++seg;
          const int e = pl.sege[seg] & 255, li = w - pl.seg0[seg];
          kind = pl.sege[seg] >> 8;
          a = e;
          const int q = pl.q[e], r = pl.r[e], s1 = kind == K_G1 ? 64 : 32;
          const int p1 = 32 * (q + 1) <= R2MAX ? s1 / 2 : s1, p0 = 32 * q <= R2MAX ? s1 / 2 : s1;
          const int l1 = __ffs(p1) - 1, l0 = __ffs(p0) - 1, r1 = r * p1;
          const int tl = li < r1 ? li >> l1 : r + ((li - r1) >> l0);
          b = li < r1 ? li & (p1 - 1) : (li - r1) & (p0 - 1);
          const int sv = r ? r - 1 : pl.nt[e] - 1, shv = pl.sh[e];
          n = 32 * (q + (tl < r)) - (tl == sv ? shv : 0);
          row0 = pl.off[e] + 32 * (tl * q + min(tl, r)) - (tl > sv ? shv : 0);
          tile = pl.tb[e] + tl;
        }
      }
    }
    if (kind == K_NOP) continue;
    if (kind < 0 && stage < 3) {
      stage = 3;
      *qget() = K_CMB;
      qput();
    }
    const int gb = kind >= 0 && kind <= K_G2 && a >= NL ? 1 << (2 * (a - NL) + kind) : 0;
    if (gb & ~gready) {
      wait_geq(p.ctr + C_GW + 2 * (a - NL) + kind, kind ? GW2 : GW13);
      gready |= gb;
    }
    int* d = qget();
    d[0] = kind;
    d[1] = a;
    d[2] = b;
    d[3] = row0;
    d[4] = n;
    d[5] = tile;
    qput();
    if (kind < 0) break;
    if (kind > K_G2) continue;
    const int e = a;
    const bool rg2 = n <= R2MAX, stg = e >= NL;
    const int nrb = kind ? 16 : 32, blk0 = (stg ? e - NL : e) * 16 * nrb + (rg2 ? b : b >> 1);
    const u8* wb = stg ? (kind ? p.sw2 : p.sw13) : kind ? p.w2 : p.w13;
    const Tmap* tw = (stg ? p.wsg : p.wm) + kind;
    const int hf = rg2 ? 2 : b & 1;
    prefetch_chunk(wb + (size_t)blk0 * WBLK, hf);
    if (kind == K_G1)
      wait_geq(p.ctr + C_READY + tile, min(n, pl.cnt[e] - (row0 - pl.off[e])));
    else
      wait_geq(p.ctr + C_DONE + tile, 64);
    const Tmap* txh = (kind ? p.thh : p.txh) + widx(n);
#define LOAD_ITEM(KPS, RG2, TX, TW) \
  load_item<KPS, RG2>(ring, bars, TX, TW, wb, n, row0, blk0, nrb, hf, slot, ph, crank, p.ctr + C_QUEUE, mph, nxt)
    if (rg2 && n <= 64)
      LOAD_ITEM(2, true, (kind ? p.th2 : p.tx2) + widx(n), tw + 4);
    else if (rg2)
      LOAD_ITEM(1, true, txh, tw + 2);
    else
      LOAD_ITEM(1, false, txh, tw);
#undef LOAD_ITEM
  }
}
DEVI void publisher(const Params& p, u8* sm, u32 bars, const int* pq, const Plan& pl, int lane) {
  const u32 stg = smem_u32(sm + STG);
  if (lane == 0)
    for (int i = 0; i < NW; ++i)
      tmap_acquire(p.th + i);
  u32 ph = 0;
  for (;;) {
    mbar_wait(bars + B_PF, ph);
    ph ^= 1;
    const int kind = pq[0], bi = pq[1], row0 = pq[2], n = pq[3], tile = pq[4], e = pq[5];
    __syncwarp();
    if (kind < 0) break;
    const bool rg2 = n <= R2MAX;
    if (kind == K_G1) {
      const int wq = lane >> 2, wa = lane & 3;
      for (int T = 0; T < (rg2 ? 2 : 1); ++T)
        for (int tok = 0; tok < n; ++tok) {
          const u32 row = stg + T * 16384 + tok * 128, x = tok & 7;
          const u32 v = lds32(row + ((wq ^ x) << 4) + 4 * wa);
          __syncwarp();
          sts32(row + (((4 * (wq & 1) + wa) ^ x) << 4) + 4 * (wq >> 1), v);
          __syncwarp();
        }
      fence_async();
      __syncwarp();
      if (lane == 0) {
        tma_store(p.th + widx(n), bi * (rg2 ? 128 : 64), row0, stg);
        if (rg2) tma_store(p.th + widx(n), bi * 128 + 64, row0, stg + 16384);
        bulk_commit_wait_read();
        mbar_arrive(bars + B_PE);
        bulk_wait();
        fence_async_global();
        red_release(p.ctr + C_DONE + tile, rg2 ? 2 : 1);
      }
      __syncwarp();
    } else {
      const int r0 = row0 - pl.off[e], nr = min(n, pl.cnt[e] - r0), rs = rg2 ? 528 : 272, cb = rg2 ? 512 : 256;
      for (int i = lane; i < nr; i += 32) {
        const int tj = __ldcg(p.rev + (size_t)e * p.M + r0 + i), t = tj >> 3, o = t / p.R;
        bulk_s2g(AS(u8, p.os[o]) + ((size_t)(tj & 7) * p.R + t - o * p.R) * 8192 + bi * cb, stg + i * rs, cb);
      }
      bulk_commit_wait_read();
      __syncwarp();
      if (lane == 0) mbar_arrive(bars + B_PE);
    }
  }
  bulk_wait();
  fence_async_global();
}
template <int N>
DEVI void
item(const Params& p, u8* sm, u32 bars, int* pq, const int* d, int q, int lane, int& slot, u32& ph, u32& pph) {
  constexpr bool R2 = N <= R2MAX;
  constexpr int KPS = R2 && N <= 64 ? 2 : 1, NT = R2 ? 2 : 1;
  constexpr bool DP = KPS == 2;
  const int g = lane >> 2;
  const u32 odd = (tid_x() >> 5) & 1, crank = cl_rank();
  float acc[128];
  for (int i = 0; i < N / 2; ++i) {
    acc[i] = 0.f;
    if (R2) acc[64 + i] = 0.f;
  }
  // epilogue-only inputs go to shared memory: no registers held through the main loop
  const u8* ew = sm + BAR + EWS + 160 * q;
  const u32 ews = smem_u32(ew);
  {
    const int kind = d[0], e = d[1], bi = d[2];
    if (lane == 0) {
      if (e >= NL)
        cpa16(ews, p.sgs);
      else
        cp_async4(ews, (kind == K_G1 ? p.gs13 : p.gs2) + e);
    }
    if (kind != K_G1 && (lane & 3) == 0)
      for (int T = 0; T < NT; ++T) {
        const int o = (R2 ? 2 * bi + T : bi) * 128 + 16 * q + g;
        cp_async4(ews + 16 + (g * NT + T) * 8, p.nw + o);
        cp_async4(ews + 20 + (g * NT + T) * 8, p.nw + o + 8);
      }
  }
  u32 s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, af[DP ? 4 : 2][4];
  int prev = -1;
  constexpr u32 RGS = DP ? 1024 : 512, SO = DP ? SOFF_R2 : R2 ? SOFF_R1 : SOFF;
  const u32 ring = smem_u32(sm), wo = (R2 ? WOFF_R : WOFF) + q * RGS + lane * 16, so = SO + (q >> 1) * 384 + lane * 12;
  constexpr int KU = 4 * KPS;
#pragma unroll 1
  for (int kc = 0; kc < 64 / KU; ++kc) {
#pragma unroll
    for (int ku = 0; ku < KU; ++ku) {
      const int kk = ku & 3, kq = KPS == 2 ? (ku & 1) : 0;
      if (kq == 0) mbar_wait(bars + B_F + 8 * slot, ph);
      const u32 st = ring + slot * SLOTB;
      const uint4 wq = lds128(st + wo + kq * 512);
      uint4 wq2 = wq;
      if (R2) wq2 = lds128(st + wo + 8 * RGS + kq * 512);
      if (kk == 0) {
        s0 = lds32(st + so);
        s1 = lds32(st + so + 4);
        s2 = lds32(st + so + 8);
        if (R2) {
          s3 = lds32(st + so + 1536);
          s4 = lds32(st + so + 1540);
          s5 = lds32(st + so + 1544);
        }
      }
      const u32 R = scale_kk(kk, odd, s0, s1, s2), Rb = !R2 ? 0 : scale_kk(kk, odd, s3, s4, s5);
      const u64 desc = sw128_desc(st + kq * N * 128);
      for (int j = 0; j < 4; ++j) {
        const int b0 = DP ? 2 * (j & 1) : R2 ? 0 : j & 1;
        decode(wq, j, R, af[b0]);
        if constexpr (DP) decode(wq2, j, Rb, af[b0 + 1]);
        wg_fence();
        wgmma_rs<N>(acc, af[b0], desc + 2 * j);
        if constexpr (DP) wgmma_rs<N>(acc + 64, af[b0 + 1], desc + 2 * j);
        wg_commit();
        wg_wait<1>();
        if (j == 0 && kq == 0 && prev >= 0) {
          __syncwarp();
          if (lane == 0) release_slot(bars + B_E + 8 * prev, crank);
        }
        if constexpr (R2 && !DP) {
          decode(wq2, j, Rb, af[1]);
          wg_fence();
          wgmma_rs<N>(acc + 64, af[1], desc + 2 * j);
          wg_commit();
          wg_wait<1>();
        }
      }
      if (kq == KPS - 1) {
        prev = slot;
        advance<NSLOT>(slot, ph);
      }
    }
  }
  wg_wait<0>();
  __syncwarp();
  if (lane == 0) release_slot(bars + B_E + 8 * prev, crank);
  const int kind = d[0], bi = d[2], row0 = d[3], tile = d[5];
  u8* stg = sm + STG;
  mbar_wait(bars + B_PE, pph ^ 1);
  pph ^= 1;
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncwarp();
  const int gso = d[1] >= NL ? (d[0] == K_G1 ? 0 : NG) + d[1] - NL : 0;
  const float cs = AS(const float, ew)[gso] * 16384.f;
  float nwv[NT][2];
  for (int T = 0; T < NT; ++T)
    for (int u = 0; u < 2; ++u)
      nwv[T][u] = AS(const float, ew + 16 + (g * NT + T) * 8)[u];
  {
    const u32 tx = tid_x();
    const int q = (tx >> 5) - 4, lane = tx & 31;
#pragma unroll
    for (int T = 0; T < NT; ++T) {
      const int ao = 64 * T;
      if (kind == K_G1) {
        const u32 sbh = smem_u32(stg + T * 16384);
        const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
        for (int nb = 0; nb < N / 8; nb += 4) {
          u32 r[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int nj = nb + i;
            if (nj < N / 8)
              r[i] =
                  bf2(silu(acc[ao + 4 * nj] * cs) * (acc[ao + 4 * nj + 2] * cs),
                      silu(acc[ao + 4 * nj + 1] * cs) * (acc[ao + 4 * nj + 3] * cs));
          }
          const int tok = 8 * (nb + mi) + rr;
          const u32 a = sbh + tok * 128 + ((q ^ (tok & 7)) << 4);
          if (nb + 4 <= N / 8)
            stmatrix_t(a, r);
          else
            stmatrix_t2(a, r[0], r[1]);
        }
      } else {
        const float n0 = nwv[T][0], n8 = nwv[T][1];
        constexpr int RS = R2 ? 528 : 272;
        const int m = lane >> 3, rr = lane & 7, c = 2 * (q & 3) + (m & 1);
        const u32 sb = smem_u32(stg) + (8 * (m >> 1) + rr) * RS + (R2 ? T * 256 : 0) + (q >> 2) * 128 + (c << 4);
        for (int j2 = 0; j2 < N / 16; ++j2) {
          u32 r[4];
          for (int u = 0; u < 2; ++u) {
            const int ai = ao + 4 * (2 * j2 + u);
            r[2 * u] = bf2(acc[ai] * cs * n0, acc[ai + 1] * cs * n0);
            r[2 * u + 1] = bf2(acc[ai + 2] * cs * n8, acc[ai + 3] * cs * n8);
          }
          stmatrix_t(sb + 16 * RS * j2, r);
        }
      }
    }
  }
  fence_async();
  bar_sync(1, 256);
  if (q == 0 && lane == 0) {
    pq[0] = kind;
    pq[1] = bi;
    pq[2] = row0;
    pq[3] = N;
    pq[4] = tile;
    pq[5] = d[1];
    mbar_arrive(bars + B_PF);
  }
}
DEVI void transpose4(uint4& v, bool b0, bool b1) {
  u32 s0 = b1 ? v.x : v.z, s1 = b1 ? v.y : v.w;
  s0 = shfl_xor(s0, 2);
  s1 = shfl_xor(s1, 2);
  if (b1) {
    v.x = s0;
    v.y = s1;
  } else {
    v.z = s0;
    v.w = s1;
  }
  s0 = b0 ? v.x : v.y;
  s1 = b0 ? v.z : v.w;
  s0 = shfl_xor(s0, 1);
  s1 = shfl_xor(s1, 1);
  if (b0) {
    v.x = s0;
    v.z = s1;
  } else {
    v.y = s0;
    v.w = s1;
  }
}
// rev of one unit; with p.cl it is also the counting pass
DEVI void list_unit(const Params& p, int u, int ctid, int* hist) {
  if (ctid < NE) hist[ctid] = 0;
  bar_sync(1, 256);
  const int t = 128 * u + ctid;
  int e8[8], lr[8];
  if (ctid < 128 && t < p.M) {
    const int4 a = AS(const int4, p.ids)[2 * t], c = AS(const int4, p.ids)[2 * t + 1];
    e8[0] = a.x;
    e8[1] = a.y;
    e8[2] = a.z;
    e8[3] = a.w;
    e8[4] = c.x;
    e8[5] = c.y;
    e8[6] = c.z;
    e8[7] = c.w;
    for (int j = 0; j < TOPK; ++j)
      if (e8[j] >= 0 && e8[j] < p.ne) lr[j] = atomicAdd(hist + e8[j], 1);
  }
  bar_sync(1, 256);
  if (ctid < NE) hist[NE + ctid] = hist[ctid] ? atomicAdd(p.ctr + C_LB + ctid, hist[ctid]) : 0;
  bar_sync(1, 256);
  if (ctid < 128 && t < p.M)
    for (int j = 0; j < TOPK; ++j)
      if (e8[j] >= 0 && e8[j] < p.ne) {
        const int r = hist[NE + e8[j]] + lr[j];
        p.rev[(size_t)e8[j] * p.M + r] = t * 8 + j;
        p.rev[(size_t)NE * p.M + t * 8 + j] = r;
      }
  bar_sync(1, 256);
  if (ctid == 0) {
    __threadfence();
    red_release(p.ctr + (p.cl ? C_CDONE : C_LDONE), 1);
  }
}
DEVI void stage_block(const Params& p, const Plan& pl, int w, int lane) {
  int j = 0;
  for (; j < NG - 1; ++j) {
    if (!pl.cnt[NL + j]) continue;
    if (w < GW13 + GW2) break;
    w -= GW13 + GW2;
  }
  const int e = __ldg(p.gsrc + j), hs = e % NL, k2 = w >= GW13, b = k2 ? w - GW13 : w, nb = k2 ? GW2 : GW13;
  const u64* pt = p.peer + 4 * (e / NL);
  const uint4* src = AS(const uint4, AS(const u8, __ldg(pt + k2)) + ((size_t)hs * nb + b) * WBLK) + lane;
  uint4* dst = AS(uint4, (k2 ? p.sw2 : p.sw13) + ((size_t)j * nb + b) * WBLK) + lane;
  constexpr int NV = WBLK / 16 / 32, NB = 7;
#pragma unroll 1
  for (int i0 = 0; i0 < NV; i0 += NB) {
    uint4 v[NB];
    for (int i = 0; i < NB; ++i)
      v[i] = ld_na(src + 32 * (i0 + i));
    for (int i = 0; i < NB; ++i)
      dst[32 * (i0 + i)] = v[i];
  }
  if (b == 0 && lane == 0) p.sgs[k2 * NG + j] = AS(const float, __ldg(pt + 2 + k2))[hs];
  __syncwarp();
  if (lane == 0) red_release(p.ctr + C_GW + 2 * j + k2, 1);
}
// Every gather row (and guest staging block) is claimed from C_GQUEUE, so the rows, and the G1
// items waiting on them, never depend on a CTA that is not resident: pfm_main completes with any
// subset of its CTAs resident (SM partitions, MPS limits, other work on the device).
DEVI void helper_gather(const Params& p, const Plan& pl, int lane) {
  if (lane == 0) wait_geq(p.ctr + (p.cl ? C_CDONE : C_LDONE), p.cl ? pl.u1 - pl.u0 : pl.nlu);
  __syncwarp();
  const int c = lane & 7, lo = 8 * (lane >> 3);
  int pos = 0, base = 0, rloc = 0, nwb = 0;
  for (int i = 0; i < NL; ++i)
    rloc += pl.cnt[pl.ord[i]];
  for (int j = 0; j < NG; ++j)
    nwb += pl.cnt[NL + j] ? GW13 + GW2 : 0;
  constexpr int END = -(1 << 30);
  auto locate = [&](int k) {
    if (nwb && k >= rloc) {
      if (k < rloc + nwb) return -1 - (k - rloc);
      k -= nwb;
    }
    while (pos < NE && base + pl.cnt[pl.ord[pos]] <= k)
      base += pl.cnt[pl.ord[pos++]];
    return pos >= NE ? END : pl.ord[pos] << 20 | (k - base);
  };
  auto claim_k = [&] {
    int k = 0;
    if (lane == 0) k = atomicAdd(p.ctr + C_GQUEUE, 1);
    return shfl(k, 0);
  };
  // asm volatile keeps the next route's load ahead of this row's data loads
  auto route = [&](int x) {
    int r = 0;
    if (x >= 0)
      asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(r) : "l"(p.rev + (size_t)(x >> 20) * p.M + (x & 0xFFFFF)));
    return r;
  };
  int x = locate(claim_k()), tj = route(x);
  while (x != END) {
    if (x < 0) {
      stage_block(p, pl, -1 - x, lane);
      x = locate(claim_k());
      tj = route(x);
      continue;
    }
    const int e = x >> 20, t = tj >> 3, rt = t / p.R;
    const int le = lane < TOPK ? __ldcg(p.ids + 8 * t + lane) : -1;
    int fr = lane < TOPK ? __ldcg(p.rev + (size_t)NE * p.M + 8 * t + lane) : 0;
    const int xn = locate(claim_k());
    const bool v = le >= 0 && le < p.ne;
    if (__reduce_min_sync(~0u, v ? pl.rk[le] * TOPK + lane : NE * TOPK) != pl.rk[e] * TOPK + (tj & 7)) {
      x = xn;
      tj = route(x);
      continue;
    }
    const u32 dm = ballot(v);
    if (!v) fr = 0;
    const int drow = v ? pl.off[le] + fr : 0;
    const uint4* src = AS(const uint4, p.xs[rt] + (size_t)(t - rt * p.R) * 4096) + lo + 2 * (c & 3) + (c >> 2);
    const int tjn = route(xn);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint4 w[8];
      for (int k2 = 0; k2 < 8; ++k2)
        w[k2] = ld_na(src + 32 * (8 * h + k2));
      for (int k2 = 0; k2 < 8; ++k2)
        transpose4(w[k2], lane & 1, lane & 2);
      for (u32 m = dm; m; m &= m - 1) {
        uint4* dst = AS(uint4, p.xt + (size_t)shfl(drow, __ffs(m) - 1) * 4096) + lo + c;
        for (int k2 = 0; k2 < 8; ++k2)
          dst[32 * (8 * h + k2)] = w[k2];
      }
    }
    __syncwarp();
    if (v) red_release(p.ctr + C_READY + tile_of_row(pl, le, fr), 1);
    x = xn;
    tj = tjn;
  }
}
DEVI void helper(const Params& p, const Plan& pl, u32 bars, int lane, bool first) {
  if (first) mbar_wait(bars + B_PL, 0);
  helper_gather(p, pl, lane);
}
extern "C" __global__ void __launch_bounds__(384, 1) pfm_main(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) u8 sm[];
  const u32 bars = smem_u32(sm + BAR);
  int* iq = AS(int, sm + BAR + 128);
  int* pq = iq + 16;
  Plan& pl = *AS(Plan, sm + BAR + 256);
  const int tid = threadIdx.x;
  pdl_trigger();  // the next pfm_main reads nothing before pdl_wait
  pdl_wait();
  const int u0 = p.first ? 0 : __ldcg(p.ctr + C_WORDS);
  if (u0 >= (p.M + 127) >> 7) return;
  if (tid == 0) {
    pl.u0 = u0;
    const int npre = __ldcg(p.ctr + C_PRE + NE);
    pl.pre = p.pre && p.first && npre > 0 && npre <= p.cap - 1088;
    for (int i = 0; i < NSLOT; ++i) {
      mbar_init(bars + B_F + 8 * i, 1);
      mbar_init(bars + B_E + 8 * i, 16);
    }
    mbar_init(bars + B_MF, 1);
    mbar_init(bars + B_ME, 1);
    for (int i = 0; i < 2; ++i) {
      mbar_init(bars + B_IF + 8 * i, 1);
      mbar_init(bars + B_IE + 8 * i, 8);
    }
    for (int b = B_PF; b <= B_PL; b += 8)
      mbar_init(bars + b, 1);
    mbar_init_fence();
    for (int i = 0; i < NW; ++i) {
      tmap_acquire(p.txh + i);
      tmap_acquire(p.thh + i);
      if (i < 2) {
        tmap_acquire(p.tx2 + i);
        tmap_acquire(p.th2 + i);
      }
    }
  }
  cl_sync();
  if (tid < 128) {
    nreg_dec<56>();
    if (tid == 0)
      producer(p, sm, pl, bars, iq);
    else if ((tid >> 5) == 1)
      publisher(p, sm, bars, pq, pl, tid & 31);
    else if (tid >= 64)
      helper(p, pl, bars, tid & 31, true);
  } else {
    nreg_inc<224>();
    const int ctid = tid - 128, q = ctid >> 5, lane = ctid & 31;
    int* hist = AS(int, sm + STG);
    int slot = 0;
    u32 ph = 0, qn = 0, pph = 0;
    for (;;) {
      const u32 qs = qn & 1, qph = (qn >> 1) & 1;
      mbar_wait(bars + B_IF + 8 * qs, qph);
      const int* d = iq + 8 * qs;
      const int kind = d[0];
      if (kind == K_CNT) {
        if (p.cl)
          list_unit(p, d[1], ctid, hist);
        else
          count_unit(p, d[1], q, lane, ctid, hist);
      } else if (kind == K_PLAN) {
        if (q == 0) {
          make_plan(p, pl, lane);
          if (lane == 0) mbar_arrive(bars + B_PL);
        }
      } else if (kind == K_CMB)
        helper(p, pl, bars, lane, false);
      else if (kind == K_LIST)
        list_unit(p, d[1], ctid, hist);
      else if (kind >= 0) {
        switch (d[4]) {
#define PFM_CASE(W)                                      \
  case W:                                                \
    item<W>(p, sm, bars, pq, d, q, lane, slot, ph, pph); \
    break;
          PFM_CASES
#undef PFM_CASE
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bars + B_IE + 8 * qs);
      ++qn;
      if (kind < 0) break;
    }
    if (ctid == 0) {
      mbar_wait(bars + B_PE, pph ^ 1);
      pq[0] = -1;
      mbar_arrive(bars + B_PF);
    }
  }
  // last CTA zeroes counters, sets the batch cursor
  asm volatile("barrier.sync 2, 384;" ::: "memory");
  if (tid < 32) {
    int last = 0;
    if (tid == 0) {
      __threadfence();
      last = atomicAdd(p.ctr + C_EXIT, 1) == (int)gridDim.x - 1;
    }
    if (shfl(last, 0)) {
      __threadfence();
      for (int i = tid; i < C_WORDS; i += 32)
        p.ctr[i] = 0;
      if (tid == 0) p.ctr[C_WORDS] = pl.u1;
    }
  }
  cl_sync();
}
}  // namespace pfm
extern "C" __global__ void __launch_bounds__(256, 2)
    moe_w4a16_sp_fin(const u16* recv, const float* tw, u16* out, int R, int rows) {
  extern __shared__ __align__(16) u8 fsm[];
  uint4* ys = AS(uint4, fsm);
  float* cs = AS(float, fsm + 65536);
  const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
  const uint4* rj = AS(const uint4, recv) + (size_t)j * R * 512 + lane;
  int r = blockIdx.x;
  uint4 v[16];
  auto load = [&](int r) {
    for (int i = 0; i < 16; ++i)
      v[i] = __ldcs(rj + (size_t)r * 512 + i * 32);
  };
  load(r);
  for (;;) {
    float ss = 0.f;
    for (int i = 0; i < 16; ++i) {
      for (int h = 0; h < 4; ++h) {
        const float lo = bf_lo(pfm::word4(v[i], h)), hi = bf_hi(pfm::word4(v[i], h));
        ss = fmaf(lo, lo, ss);
        ss = fmaf(hi, hi, ss);
      }
      ys[j * 512 + i * 32 + lane] = v[i];
    }
    for (int o = 16; o; o >>= 1)
      ss += shfl_xor(ss, o);
    if (lane == 0) cs[j] = tw[r * 8 + j] * rsqrtf(ss * (1.f / 4096.f) + 1e-8f);
    const int rn = r + gridDim.x;
    if (rn < rows) load(rn);
    __syncthreads();
    float c[8];
    for (int k = 0; k < 8; ++k)
      c[k] = cs[k];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int ch = 256 * u + threadIdx.x;
      float o[8];
      for (int h = 0; h < 8; ++h)
        o[h] = 0.f;
      for (int k = 0; k < 8; ++k) {
        const uint4 w = ys[k * 512 + ch];
        for (int h = 0; h < 4; ++h) {
          o[2 * h] = fmaf(c[k], bf_lo(pfm::word4(w, h)), o[2 * h]);
          o[2 * h + 1] = fmaf(c[k], bf_hi(pfm::word4(w, h)), o[2 * h + 1]);
        }
      }
      AS(uint4, out)
      [(size_t)r * 512 + ch] = make_uint4(bf2(o[0], o[1]), bf2(o[2], o[3]), bf2(o[4], o[5]), bf2(o[6], o[7]));
    }
    if (rn >= rows) break;
    r = rn;
    __syncthreads();
  }
}
// cnt [expert] += routes of ids (zero between calls); cnt[257 + r] = 1024 x dur[r] / mean: slow ranks cost more
extern "C" __global__ void __launch_bounds__(256)
    moe_w4a16_split_cnt(const int* ids, int* cnt, int n, const int* dur, int* ctr) {
  __shared__ int h[256];
  pdl_trigger();  // dsplit only preloads ids before its pdl_wait
  if (ctr && blockIdx.x == 0 && threadIdx.x <= pfm::NE) ctr[pfm::C_PRE + threadIdx.x] = 0;
  if (blockIdx.x == 0 && threadIdx.x < 8) {
    long long s = 0, z = 0;
    for (int q = 0; q < 8; ++q) {
      s += dur[q];
      z |= dur[q] <= 0;
    }
    cnt[257 + threadIdx.x] = z ? 1024 : (int)min(4095LL, max(1LL, (8192LL * dur[threadIdx.x] + s / 2) / s));
  }
  h[threadIdx.x] = 0;
  __syncthreads();
  for (int i = blockIdx.x * 2048 + threadIdx.x; i < min(n, blockIdx.x * 2048 + 2048); i += 256)
    atomicAdd(h + ids[i], 1);
  __syncthreads();
  if (h[threadIdx.x]) atomicAdd(cnt + threadIdx.x, h[threadIdx.x]);
}
// ids per CTA; expert cost = routes + ITEM per tile (128 rows up to 2048 tokens, else 256), measured
constexpr int DSPLIT_IDS = 1024, DSPLIT_ITEM_SMALL = 61, DSPLIT_ITEM_BIG = 45;
// Every CTA runs the same greedy on cost x rank weight: while it lowers the max, move a whole expert of the busiest
// rank to a rank with a free guest slot, minimizing (new max, expert, rank). slots: <= 1 guest, into an idle slot.
// rev: also list me's routes for pfm_main (p.pre), counts at C_PRE (split_cnt clears them)
extern "C" __global__ void __launch_bounds__(256) moe_w4a16_dsplit(
    const int* __restrict__ ids,
    int* cnt,
    int* __restrict__ out,
    int* gsrc,
    int n,
    int me,
    int slots,
    int* __restrict__ rev,
    int* ctr) {
  constexpr int G = pfm::NG;
  __shared__ int tot[256], ld0[8], gls[8][G], last;
  __shared__ unsigned wt[8];
  __shared__ signed char own[256], gix[256];
  const int tid = threadIdx.x, lane = tid & 31, i0 = blockIdx.x * DSPLIT_IDS + tid;
  int x[DSPLIT_IDS / 256];
#pragma unroll
  for (int k = 0; k < DSPLIT_IDS / 256; ++k)
    x[k] = i0 + 256 * k < n ? ids[i0 + 256 * k] : 0;
  pdl_wait();  // cnt / ctr are split_cnt's
  if (tid < 8) {
    wt[tid] = __ldcg(cnt + 257 + tid);
    for (int j = 0; j < G; ++j)
      gls[tid][j] = -1;
  }
  const int c = __ldcg(cnt + tid), s = !c              ? 0
                                       : n <= 8 * 2048 ? c + DSPLIT_ITEM_SMALL * ((c + 127) >> 7)
                                                       : c + DSPLIT_ITEM_BIG * ((c + 255) >> 8);
  tot[tid] = s;
  own[tid] = tid >> 5;
  gix[tid] = -1;
  const int ws = __reduce_add_sync(~0u, s);
  if (lane == 0) ld0[tid >> 5] = ws;
  __syncthreads();
  if (tid == 0) last = atomicAdd(cnt + 256, 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (last) {
    cnt[tid] = 0;
    if (tid == 0) cnt[256] = 0;
  }
  if (tid < 32) {
    int load[8], ng[8], fr[8];
    u32 w[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      load[q] = ld0[q];
      ng[q] = 0;
      w[q] = wt[q];
      fr[q] = slots ? min(1, __popc(ballot(tot[32 * q + lane] == 0))) : G;
    }
    for (int step = 0; step < 8 * G; ++step) {
      int r = 0, lr = load[0];
      u32 wr = w[0], tr = lr * wr;
#pragma unroll
      for (int q = 1; q < 8; ++q)
        if (load[q] * w[q] > tr) {
          lr = load[q];
          wr = w[q];
          tr = lr * wr;
          r = q;
        }
      const int e = 32 * r + lane, u = own[e] == r ? tot[e] : 0;
      u32 bm = ~0u, bl = ~0u;
      if (u > 0)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const u32 m = max((lr - u) * wr, (load[q] + u) * w[q]), l = e << 3 | q;
          if (q != r && ng[q] < fr[q] && m < tr && (m < bm || (m == bm && l < bl))) {
            bm = m;
            bl = l;
          }
        }
      const u32 mm = __reduce_min_sync(~0u, bm), bc = __reduce_min_sync(~0u, bm == mm ? bl : ~0u);
      if (mm == ~0u) break;
      const int q = bc & 7, e2 = bc >> 3, u2 = tot[e2];
      int gq = 0;
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        if (x == r) load[x] -= u2;
        if (x == q) {
          load[x] += u2;
          gq = ng[x]++;
        }
      }
      if (lane == 0) {
        own[e2] = q;
        gls[q][gq] = e2;
      }
      __syncwarp();
    }
  }
  __syncthreads();
  if (tid < G && gls[me][tid] >= 0) gix[gls[me][tid]] = tid;
  if (blockIdx.x == 0 && tid < G) gsrc[tid] = gls[me][tid];
  if (tid <= pfm::NE) tot[tid] = 0;
  __syncthreads();
  int lo[DSPLIT_IDS / 256], lr[DSPLIT_IDS / 256];
#pragma unroll
  for (int k = 0; k < DSPLIT_IDS / 256; ++k) {
    const int o = own[x[k]];
    lo[k] = o != me || i0 + 256 * k >= n ? -1 : o == x[k] >> 5 ? x[k] & 31 : 32 + gix[x[k]];
    if (i0 + 256 * k < n) out[i0 + 256 * k] = lo[k];
    if (rev && lo[k] >= 0) {
      lr[k] = atomicAdd(tot + lo[k], 1);
      atomicAdd(tot + pfm::NE, 1);
    }
  }
  if (!rev) return;
  __syncthreads();
  if (tid <= pfm::NE && tot[tid]) tot[64 + tid] = atomicAdd(ctr + pfm::C_PRE + tid, tot[tid]);
  __syncthreads();
  for (int k = 0; k < DSPLIT_IDS / 256; ++k)
    if (lo[k] >= 0) {
      const int r = tot[64 + lo[k]] + lr[k];
      rev[(size_t)lo[k] * (n >> 3) + r] = i0 + 256 * k;
      rev[(size_t)pfm::NE * (n >> 3) + i0 + 256 * k] = r;
    }
}
