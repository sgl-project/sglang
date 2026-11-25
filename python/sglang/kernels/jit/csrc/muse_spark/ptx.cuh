// Inline-PTX helpers pasted at #include "ptx.cuh" into the Muse Spark NVRTC sources: moe_w4a16*.cuh,
// verify_attention.cuh, extend_attention.cuh, dflash_tree.cuh and muse_spark_sp_experts.py.
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
typedef long long i64;
#define DEVI __device__ __forceinline__
#define AS(T, p) reinterpret_cast<T*>(p)
template <class T>
DEVI T shfl(T v, int src) {
  return __shfl_sync(~0u, v, src);
}
template <class T>
DEVI T shfl_xor(T v, int m) {
  return __shfl_xor_sync(~0u, v, m);
}
template <class T>
DEVI T shfl_up(T v, int d) {
  return __shfl_up_sync(~0u, v, d);
}
DEVI u32 ballot(bool p) {
  return __ballot_sync(~0u, p);
}
// inclusive prefix sum over the warp
DEVI int wscan(int v, int lane) {
  for (int o = 1; o < 32; o <<= 1) {
    const int a = shfl_up(v, o);
    if (lane >= o) v += a;
  }
  return v;
}
template <int N, class T>
DEVI void advance(T& s, u32& ph) {
  if (++s == N) {
    s = 0;
    ph ^= 1;
  }
}
// sgl_kernel/ptx/barrier.cuh has to_shared / cluster_ctarank / mapa_shared, but NVRTC opens no standard header
// and it needs sgl_kernel/utils.cuh (<bit>, tvm-ffi) and C++20 concepts; hence smem_u32 / cl_rank / mapa here.
DEVI u32 smem_u32(const void* p) {
  return static_cast<u32>(__cvta_generic_to_shared(p));
}
DEVI void mbar_init(u32 bar, u32 n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
DEVI void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
DEVI void mbar_arrive(u32 bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
DEVI void mbar_expect(u32 bar, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEVI void mbar_wait(u32 bar, u32 ph) {
  asm volatile(
      "{ .reg .pred P1; W%=: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1; @!P1 bra W%=; }" ::"r"(bar), "r"(ph)
      : "memory");
}
DEVI bool mbar_test(u32 bar, u32 ph) {
  u32 ok;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok)
               : "r"(bar), "r"(ph)
               : "memory");
  return ok != 0;
}
DEVI u32 cl_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEVI u32 mapa(u32 a, u32 rank) {
  u32 d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}
DEVI void cl_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEVI void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEVI void bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
DEVI void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
DEVI void stmatrix_t(u32 a, const u32 (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(a),
               "r"(r[0]),
               "r"(r[1]),
               "r"(r[2]),
               "r"(r[3])
               : "memory");
}
DEVI void sts32(u32 a, u32 v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
DEVI void sts4(u32 a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI uint4 lds128(u32 a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
DEVI void fence_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// other CTAs' global writes (acquired) before bulk copies read them
DEVI void fence_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src),
               "r"(bytes),
               "r"(bar)
               : "memory");
}
DEVI void bulk_s2g(void* dst, u32 src, u32 bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes)
               : "memory");
}
DEVI void prefetch_l2(const void* p, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
DEVI void cpa16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
DEVI void bulk_push(u32 dst, u32 src, u32 n, u32 mbar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
      "r"(src),
      "r"(n),
      "r"(mbar)
      : "memory");
}
DEVI void bulk_commit_wait_read() {
  asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;" ::: "memory");
}
DEVI void bulk_wait() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
DEVI void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
DEVI void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
DEVI int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
DEVI void red_release(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
DEVI u64 policy_evict_first() {
  u64 p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
DEVI u64 policy_evict_last() {
  u64 p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
template <int kRegs>
DEVI void nreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegs) : "memory");
}
template <int kRegs>
DEVI void nreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kRegs) : "memory");
}
DEVI void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
DEVI void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int kPending>
DEVI void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(kPending) : "memory");
}
// keeps accumulator accesses after wg_wait
template <int kN>
DEVI void acc_pin(float (&x)[kN]) {
#pragma unroll
  for (int i = 0; i < kN; ++i)
    asm volatile("" : "+f"(x[i])::"memory");
}
DEVI u64 sw128_desc(u32 a) {
  return (u64)((a & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)64 << 32) | ((u64)1 << 62);
}
// predecessor outputs after pdl_wait()
DEVI uint4 ldq_u4(const void* a) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(a)
               : "memory");
  return v;
}
DEVI u64 wg_desc(u32 a, u32 lbo, u32 sbo, u32 mode) {
  return (u64)((a & 0x3FFFFu) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | ((u64)mode << 62);
}
#define R4(c, a, i) c(a[i]), c(a[i + 1]), c(a[i + 2]), c(a[i + 3])
#define R16(c, a, i) R4(c, a, i), R4(c, a, i + 4), R4(c, a, i + 8), R4(c, a, i + 12)
#define OPS32                                                                                                          \
  "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30," \
  "%31"
#define NEG_INF __int_as_float(0xff800000)
DEVI u64 dadd(u64 d, u32 off) {
  u64 r;
  asm("{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, %1;\nadd.u32 lo, lo, %2;\nmov.b64 %0, {lo, hi};\n}"
      : "=l"(r)
      : "l"(d), "r"(off));
  return r;
}
// m64n64k32 E4M3 wgmma into d
#define WGMMA(ACC, C, A, ...)                                                                               \
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, " #ACC                                                     \
               ", 0;\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {" OPS32 "}, " A ", p, 1, 1;\n}" \
               : R16(C, d, 0), R16(C, d, 16)                                                                \
               : __VA_ARGS__)
template <int ACC>
DEVI void wg_qk(float (&d)[32], const u32 (&a)[4], u64 desc) {
  if constexpr (ACC)
    WGMMA(1, "+f", "{%32,%33,%34,%35}, %36", R4("r", a, 0), "l"(desc));
  else
    WGMMA(0, "=f", "{%32,%33,%34,%35}, %36", R4("r", a, 0), "l"(desc));
}
DEVI void wg_pv(float (&d)[32], const u32 (&a)[4], u64 desc) {
  wg_qk<1>(d, a, desc);
}
DEVI float rcp0(float l) {
  return l > 0.f ? __fdividef(1.f, l) : 0.f;
}
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DEVI float sigm(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));
}
DEVI float quad_sum(float x) {
  x += shfl_xor(x, 1);
  return x + shfl_xor(x, 2);
}
DEVI float quad_max(float x) {
  x = fmaxf(x, shfl_xor(x, 1));
  return fmaxf(x, shfl_xor(x, 2));
}
// row maxima (rows g, g + 8) of an m64nN accumulator fragment over the quad
template <int kN>
DEVI void row_max(const float (&s)[kN], float (&mx)[2]) {
  float y[2][4];
  for (int k = 0; k < 4; ++k) {
    y[0][k] = fmaxf(s[4 * k], s[4 * k + 1]);
    y[1][k] = fmaxf(s[4 * k + 2], s[4 * k + 3]);
  }
  for (int n8 = 4; n8 < kN / 4; ++n8) {
    y[0][n8 & 3] = fmaxf(y[0][n8 & 3], fmaxf(s[4 * n8], s[4 * n8 + 1]));
    y[1][n8 & 3] = fmaxf(y[1][n8 & 3], fmaxf(s[4 * n8 + 2], s[4 * n8 + 3]));
  }
  for (int rh = 0; rh < 2; ++rh)
    mx[rh] = quad_max(fmaxf(fmaxf(y[rh][0], y[rh][1]), fmaxf(y[rh][2], y[rh][3])));
}
DEVI u16 f2bf(float f) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return r;
}
DEVI float bfr(float x) {
  return __uint_as_float((u32)f2bf(x) << 16);
}
DEVI float bf_lo(u32 v) {
  return __uint_as_float(v << 16);
}
DEVI float bf_hi(u32 v) {
  return __uint_as_float(v & 0xFFFF0000u);
}
DEVI u32 bf2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI u32 gate2(float a, float b, float rs, u32 g) {
  return bf2(a * rs * sigm(bf_lo(g)), b * rs * sigm(bf_hi(g)));
}
DEVI u32 e4x2(float lo, float hi) {
  u16 r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI u32 e4m3x4(float a, float b, float c, float d) {
  return e4x2(a, b) | (e4x2(c, d) << 16);
}
DEVI void p_frag(u32 (&a)[4], const float* f) {
  a[0] = e4m3x4(f[0], f[1], f[4], f[5]);
  a[1] = e4m3x4(f[2], f[3], f[6], f[7]);
  a[2] = e4m3x4(f[8], f[9], f[12], f[13]);
  a[3] = e4m3x4(f[10], f[11], f[14], f[15]);
}
DEVI u32 rotl(u32 x, u32 r) {
  return __funnelshift_l(x, x, r);
}
DEVI u32 prmt(u32 a, u32 b, u32 s) {
  u32 d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}
// bf16x2 a times the low (H = 0) or high bf16 of b
#define BMUL2S(X)                                                                                                  \
  asm("{ .reg .b16 l, h; .reg .b32 t; mov.b32 {l, h}, %2; mov.b32 t, {" #X ", " #X "}; mul.rn.bf16x2 %0, %1, t; }" \
      : "=r"(d)                                                                                                    \
      : "r"(a), "r"(b))
template <int H>
DEVI u32 bmul2s(u32 a, u32 b) {
  u32 d;
  if (H == 0)
    BMUL2S(l);
  else
    BMUL2S(h);
  return d;
}
#undef BMUL2S
