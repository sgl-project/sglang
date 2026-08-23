// NVFP4 W4A4 skinny GEMM for sm_120a (RTX PRO 6000 Blackwell, 188 SMs).
//
// out[M,N] = alpha * sum_k deq(A[m,k]) * deq(B[n,k]),  M <= 16.
//   A: [M, K/2] uint8, packed E2M1, row-major, low nibble = even k.
//   B: [N, K/2] uint8, packed E2M1, row-major storage (given as .t() view).
//   SFA: swizzled 128x4-atom UE4M3 scales, logical [128(pad M), K/16].
//   SFB: swizzled 128x4-atom UE4M3 scales, logical [N, K/16].
//   Swizzle atom (K-major): off(mn, kb) = (mn/128)*512*(SFK/4) + (kb/4)*512
//                                          + 16*(mn%32) + 4*((mn/32)%4) + kb%4.
//
// Design: persistent stream-K over "units" (one unit = NTILE columns x 128 K
// elements) with a cp.async (LDGSTS) multi-stage smem pipeline, non-warp-
// specialized. Work = NT*KT units in (ntile-major, ktile-minor) order; CTA c
// of G owns the contiguous range [c*U/G, (c+1)*U/G). The pipeline runs
// seamlessly across tile boundaries. Tiles fully inside one range are stored
// directly (alpha-scaled bf16). Tiles split across ranges accumulate raw f32
// partials into ws[rank] and the last contributor (ticket via sem[nt])
// reduces and stores. Classic scheduling is the special case G = NT*SPLIT.
//
// Every warp computes 16 columns for all M<=16 rows with
// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
// (SASS: OMMA.SF.16864). Operands are consumed from smem as plain u32 reads;
// the 128-row NTILE keeps scale-factor atoms verbatim-copyable.
//
// Compile-time parameters (-D at NVRTC time):
//   KDIM, NDIM : GEMM K and N
//   STAGES     : smem pipeline depth
//   NWARPS     : warps per CTA (1/2/4/8/16); NTILE = 16*NWARPS
//   PREFETCH   : 1 to add .L2::256B hint on weight cp.async
//   SPLIT      : >0 static aligned schedule (host sets G == NT*SPLIT);
//                0 generalized stream-K walk
//   L2POLICY   : 1 applies an L2 eviction-policy descriptor (L2DESC) to
//                weight copies; dodges part of the benchmark's dirty-L2
//                writeback tax on the gate_up stream
//   L2DESC     : the descriptor constant (fractional evict_first)
//   POLB1      : 0 leaves the second B copy unhinted
//   MEMONLY    : dev ablation - skip MMAs, keep all memory traffic
//
// grid = (G, 1), block = 32*NWARPS. G <= NT*KT chosen by the host.

#ifndef MEMONLY
#define MEMONLY 0
#endif
#ifndef L2POLICY
#define L2POLICY 0
#endif
#ifndef POLSFB
#define POLSFB 1  // 1: SFB copies also use the L2 policy path
#endif
#ifndef POLSFB4
#define POLSFB4 0  // 1: hinted SFB sub-atom copies on the NWARPS<8 path
#endif
#ifndef STOREWT
#define STOREWT 0  // 1: write-through output stores (no L2 dirty allocation)
#endif
#ifndef NORED
#define NORED 0  // dev ablation - direct-store partial tiles (wrong results)
#endif
#ifndef SPLIT
#define SPLIT 0  // >0: static aligned schedule (G == NT*SPLIT); 0: stream-K
#endif
#ifndef SPINRED
#define SPINRED \
  1  // 1: parallel spin-reduce (contributors co-resident);
     // 0: serial ticket-reduce (no residency requirement)
#endif
#ifndef FENCEREL
#define FENCEREL \
  0  // 1: release/acquire sem handshake in the spin-reduce
     // epilogue instead of full __threadfence()s
#endif
#ifndef KROT
#define KROT \
  0  // 1: rotate each CTA's k-walk start by c % KT_SPLIT to
     // decorrelate the wave-synchronized DRAM address phase
     // (static path only; reorders the k accumulation)
#endif
#ifndef POLB1
#define POLB1 1  // 1: cache-policy hint also on the second B copy
#endif
#ifndef ROWPAIR
#define ROWPAIR 0  // 1: thread copies two adjacent B rows (2r, 2r+1)
#endif
#ifndef L2DESC
#define L2DESC 0x12F0000000000000ULL  // fractional evict_first, fraction 1.0
#endif
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;

#define NTILE (16 * NWARPS)
#define THREADS (32 * NWARPS)
#define KBYTES (KDIM / 2)    // packed bytes per row
#define SFK (KDIM / 16)      // scale blocks per row
#define KTILES (KDIM / 128)  // k-units per tile
#define NT (NDIM / NTILE)    // column tiles
#define UTOT (NT * KTILES)   // total work units

static_assert(NDIM % NTILE == 0, "NTILE must divide NDIM");
static_assert(KDIM % 128 == 0, "KDIM must be a multiple of 128");
static_assert(NWARPS == 1 || NWARPS == 2 || NWARPS == 4 || NWARPS == 8 || NWARPS == 16, "NTILE in {16,32,64,128,256}");
#if NWARPS == 1 && !SPLIT
// The NW1 SFB source math folds the tile's 32-row half of the swizzle atom
// into loop-invariant producer state; only the static schedule keeps ntp
// constant.
#error "NWARPS=1 supports only the static aligned schedule (SPLIT >= 1)"
#endif

// Stage layout in u32 words: B rows pitch 20 (80 B), then A, SFA, SFB chunks.
// SFB holds 2 scale words (2x4 UE4M3 blocks) per tile row: NTILE*2 words.
// NTILE>=128: verbatim 512B swizzle-atom chunks. NTILE<128: packed sub-atom
// chunks, layout [block b][row r%32] (8B/row for NTILE=64, 4B for NTILE=32).
#define B_WORDS (NTILE * 20)
#define A_OFF B_WORDS              // 16 rows * 20
#define SFA_OFF (A_OFF + 16 * 20)  // 2 chunks * 128
#define SFB_OFF (SFA_OFF + 256)
#define STAGE_WORDS (SFB_OFF + NTILE * 2)

__device__ __forceinline__ u32 smem_addr(const void* p) {
  u32 r;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
  return r;
}

#ifndef L1SFA
#define L1SFA 0  // 1: A/SFA copies use .ca (allocate in L1 for cross-CTA reuse)
#endif

__device__ __forceinline__ void cp16(u32 dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
#if L1SFA
  // A and SFA lines are re-read by every CTA that lands on the SM; .ca lets
  // them survive in L1 across sequential CTAs (weights stay .cg, L1-bypass).
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
#endif
}

__device__ __forceinline__ void cp8(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst), "l"(src));
}

// Policy-hinted 8-byte copy (.ca only: cp.async.cg is 16-byte-only).
__device__ __forceinline__ void cp8w(u32 dst, const void* src, u64 pol) {
#if L2POLICY > 0
  asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
#else
  cp8(dst, src);
#endif
}

__device__ __forceinline__ void cp4(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ u64 l2_policy() {
#if L2POLICY == 1
  // == createpolicy.fractional.L2::evict_first.b64 pol, 1.0. ptxas 13.0
  // constant-folds createpolicy to this descriptor, but its lowering inside
  // this kernel leaves UR0/UR1 uninitialized (illegal instruction at run
  // time), so materialize the descriptor as a plain constant instead.
  // (frac f: high word = (0x120 | (16f-1)) << 20; 1.0=0x12F, 0.5=0x127.)
  return L2DESC;
#else
  return 0;
#endif
}

__device__ __forceinline__ void cp16w(u32 dst, const void* src, u64 pol) {
#if L2POLICY > 0 && PREFETCH == 2
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
#elif L2POLICY > 0 && PREFETCH
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
#elif L2POLICY > 0
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
#elif PREFETCH == 2
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
#elif PREFETCH
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
#endif
}

// Prefetch-hinted B copy without a cache policy (encoding-safe everywhere).
__device__ __forceinline__ void cp16p(u32 dst, const void* src) {
#if PREFETCH == 2
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
#elif PREFETCH
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
#endif
}

#ifndef PFL2
#define PFL2 0  // k-iters of explicit L2 prefetch lookahead for B (0 = off)
#endif
#ifndef PFL2BULK
#define PFL2BULK 0  // 1: bulk prefetch carrying the cache-policy descriptor
#endif

// Explicit L2 prefetch of one 128B B line, PFL2 k-iters ahead of the
// producer. Deepens the DRAM request queue without consuming smem stages.
// Bulk form rides the evict_first descriptor (uniform-datapath UBLKPF);
// plain form is a per-thread CCTL.E.PF2 without a policy.
__device__ __forceinline__ void pfl2(const void* src, u64 pol) {
#if PFL2BULK && L2POLICY > 0
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], 128, %1;\n" ::"l"(src), "l"(pol));
#elif PFL2BULK
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;\n" ::"l"(src));
#else
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(src));
#endif
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ u32 pack_bf16x2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

#ifndef STORECS
#define STORECS 1
#endif

// Output is write-once, never re-read here: streaming store keeps it from
// evicting L2 lines the harness flush left dirty.
__device__ __forceinline__ void st_out(u32* p, u32 v) {
#if STOREWT
  // Write-through: no dirty L2 line, so the store never forces a dirty
  // harness line out (the harness flush leaves L2 100% dirty).
  asm volatile("st.global.wt.b32 [%0], %1;" ::"l"(p), "r"(v));
#elif STORECS
  asm volatile("st.global.cs.b32 [%0], %1;" ::"l"(p), "r"(v));
#else
  *p = v;
#endif
}

__device__ __forceinline__ void
mma_nvfp4(float d[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1, u32 sfa, u32 sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0),
        "r"(a1),
        "r"(a2),
        "r"(a3),
        "r"(b0),
        "r"(b1),
        "r"(sfa),
        "h"((u16)0),
        "h"((u16)0),
        "r"(sfb),
        "h"((u16)0),
        "h"((u16)0));
}

#ifndef KROTP
#define KROTP 0  // >0: delayed rotation — identity on [0,KROTP), rotated suffix
#endif
#ifndef KROTMUL
#define KROTMUL 1  // suffix phase multiplier (coprime to KT_SPLIT-KROTP)
#endif

#ifndef IKET
#define IKET 0  // 1 (dev only): per-CTA %globaltimer event trace to g_iket
#endif
#if IKET
// Dev-only in-kernel event trace: tid0 buffers GPU-global nanosecond
// timestamps in registers and dumps one 64B record per CTA at kernel exit
// (read back via cuModuleGetGlobal). Layout per CTA c:
//   [0] smid<<32 | rank   [1] entry   [2] first stage ready
//   [3] mainloop exit     [4] arrive (pre-release)   [5] spin exit
//   [6] exit (pre-dump)   [7] sentinel 0x1CE7C0DE
// The whole block vanishes at IKET=0: default cubins are byte-identical.
extern "C" __device__ u64 g_iket[4096 * 8];
#define IKET_TS(v) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)::"memory")
#endif

extern "C" __global__ void __launch_bounds__(THREADS) fp4gemm(
    const u32* __restrict__ input,      // [M, KBYTES/4]
    const u32* __restrict__ weight,     // [NDIM, KBYTES/4]
    const u32* __restrict__ input_sf,   // swizzled, 128 rows
    const u32* __restrict__ weight_sf,  // swizzled, NDIM rows
    const float* __restrict__ alpha,
    u32* __restrict__ out,   // bf16 [M, NDIM] as u32 pairs
    float* __restrict__ ws,  // [MAXC, 16, NDIM] partials
    u32* __restrict__ sem,   // [NT] tickets
    int M) {
  extern __shared__ u32 smem[];
  __shared__ u32 ticket;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const u32 c = blockIdx.x;
#if IKET
  u64 ik_e0 = 0, ik_e1 = 0, ik_e2 = 0, ik_e3 = 0, ik_e4 = 0;
  u32 ik_smid = 0;
  if (tid == 0) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ik_smid)::"memory");
    IKET_TS(ik_e0);
  }
#endif

#if SPLIT
  // Static aligned schedule: CTA c owns tile c/SPLIT, k-slice c%SPLIT.
  // Same unit ranges as stream-K with G == NT*SPLIT, but every producer
  // pointer is loop-invariant and the epilogue runs once, out of the loop.
#define KT_SPLIT (KTILES / SPLIT)
  static_assert(KTILES % SPLIT == 0, "SPLIT must divide KTILES");
  const int ntc0 = (int)(c / SPLIT);
  const int kt00 = (int)(c % SPLIT) * KT_SPLIT;
#if KROT
  // PM-sampling timelines show wave-synchronized CTAs (all at the same
  // k-offset of a constant-stride tile grid) hold DRAM at ~92% until natural
  // drift decorrelates them (~15us). Rotating each CTA's k-walk start phase
  // decorrelates the address stream from cycle 0. Pure reordering of the
  // k accumulation (deterministic per CTA -> still bit-stable per launch).
#if KROTP > 0
  // Aligned-prologue delayed rotation: the first KROTP slots stay identity —
  // the pipeline-fill burst hits slice-start addresses coalesced across CTAs
  // like the base schedule (the round-23 per-CTA probe measured full-slice
  // rotation paying a +0.7us cold-fill head tax for zero body benefit in
  // those slots) — while the suffix [KROTP, KT_SPLIT) rotates by a per-tile
  // phase. ntc0 steps by 1 across the 40-CTA shared-A group (c steps by
  // SPLIT), so a tile-indexed phase covers evenly for any KROTP; KROTMUL
  // (coprime to KT_SPLIT-KROTP) additionally de-phases tile-adjacent CTAs.
  const int krot0 = (int)(((u32)ntc0 * (u32)KROTMUL) % (u32)(KT_SPLIT - KROTP));
#define KROT_MAP(kt) ((kt) < KROTP ? (kt) : (kt) + krot0 - (((kt) + krot0 >= KT_SPLIT) ? (KT_SPLIT - KROTP) : 0))
#else
  const int krot0 = (int)(c % (u32)KT_SPLIT);
#define KROT_MAP(kt) ((kt) + krot0 - (((kt) + krot0 >= KT_SPLIT) ? KT_SPLIT : 0))
#endif
#else
#define KROT_MAP(kt) (kt)
#endif
#else
  const u32 G = gridDim.x;
  const u32 beg = (u32)(((u64)c * UTOT) / G);
  const u32 end = (u32)(((u64)(c + 1) * UTOT) / G);
  const int rlen = (int)(end - beg);
#endif

  const int q = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3
  const int mA = q + ((lane & 1) << 3);

  const u32 smem_base = smem_addr(smem);
#if L2POLICY > 0
  // ptxas 13.0 folds uniform smem-address parts into hinted LDGSTS as
  // [R+UR+imm]; that encoding cannot also carry the cache-policy descriptor
  // and the load reads uninitialized URs as its descriptor instead (illegal
  // instruction at run time). Launder the base used for hinted destinations
  // through SHFL so it is never provably uniform and those addresses stay in
  // plain [R+imm] form.
  const u32 smem_hint = __shfl_sync(0xffffffffu, smem_base, lane);
#else
  const u32 smem_hint = smem_base;
#endif
  const u64 pol = l2_policy();
#if !SPLIT
  const float alp = *alpha;
#endif

  // ---- producer thread roles (per stage) ----
  // B: all THREADS threads, 2 chunks each. A/SFA/SFB: thread subsets.
#if ROWPAIR
  const u32 bdst0 = smem_hint + (((tid >> 2) * 2) * 20 + (tid & 3) * 4) * 4;
  const u32 bdst1 = bdst0 + 80;
#else
  const u32 bdst0 = smem_hint + ((tid >> 2) * 20 + (tid & 3) * 4) * 4;
  // Tie bdst1 to bdst0 so ptxas reuses one materialized base register and
  // keeps hinted LDGSTS in the [R+imm] form (see cp16w note).
  const u32 bdst1 = bdst0 + (NTILE / 2) * 80;
#endif
  const int ar = tid >> 2;  // A row (tid < 64)
  const bool apred = tid < 64 && ar < M;
  const u32 adst = smem_base + (A_OFF + ar * 20 + (tid & 3) * 4) * 4;
  const int sfi = tid & 63;         // chunk copy index
  const int sfj = sfi >> 5;         // chunk 0/1
  const int sfo = (sfi & 31) * 16;  // byte offset within 512B chunk
  const u32 sfadst = smem_base + SFA_OFF * 4 + sfj * 512 + sfo;
#if NWARPS >= 8
  // SFB verbatim 512B-chunk copies: threads [128, 128+64*(NTILE/128)).
  const int sfh = (tid - 128) >> 6;  // SFB atom index
  const u32 sfbdst = smem_hint + SFB_OFF * 4 + sfh * 1024 + sfj * 512 + sfo;
#elif NWARPS == 1
  // SFB packed quarter-atom copies for NTILE=16: 32 chunks of 4B; chunk
  // (b=tid>>4, w=tid&15) covers tile row w of scale block b. The 16 tile
  // rows occupy one 32-row half ((ntp&1)<<4) and one 4-column group
  // ((ntp>>1)&3) of the 128-row swizzle atom; both fold into SFB_SRC.
  const int sfb_b = tid >> 4;
  const int sfb_r2 = tid & 15;
  const u32 sfbdst = smem_base + SFB_OFF * 4 + sfb_b * (NTILE * 4) + sfb_r2 * 4;
  // Second A row (32 producer threads cover 8 rows per cp16 pass).
  const bool apred8 = (ar + 8) < M;
#else
  // SFB packed sub-atom copies: 64 chunks (8B for NTILE=64, 4B for NTILE=32),
  // chunk c covers rows (c&31) [+32] of scale block b = c>>5.
  const int sfc = (NWARPS == 2) ? tid : (tid - 64);
  const int sfb_b = sfc >> 5;
  const int sfb_r2 = sfc & 31;
#if POLSFB4 && L2POLICY > 0
  // Launder the FINAL destination address: tying sfbdst to bdst0 is
  // impossible (non-constant delta across threads), and a hinted LDGSTS whose
  // address ptxas decomposes to [R+UR+imm] silently drops the descriptor and
  // reads uninitialized URs (see cp16w note). An opaque per-thread value can
  // only stay in the [R+imm] form.
  const u32 sfbdst =
      __shfl_sync(0xffffffffu, smem_base + SFB_OFF * 4 + sfb_b * (NTILE * 4) + sfb_r2 * (NTILE / 8), lane);
#else
  const u32 sfbdst = smem_base + SFB_OFF * 4 + sfb_b * (NTILE * 4) + sfb_r2 * (NTILE / 8);
#endif
#endif

  // Producer walk state: unit (ntp, ktp). Static path sets it once; the
  // stream-K path advances it once per PRODUCE.
#if SPLIT
  const int ntp = ntc0;
  const int ktp = kt00;
#else
  int ntp = (int)(beg / KTILES);
  int ktp = (int)(beg % KTILES);
#endif
  const char* wsrc0;
  const char* wsrc1;
  const char* asrc;
  const char* sfasrc;
  const char* sfbsrc;
#if NWARPS == 1
  const char* asrc8;
#define SET_ASRC8() (asrc8 = asrc + 8 * (u64)KBYTES)
#else
#define SET_ASRC8() ((void)0)
#endif

#if NWARPS >= 8
#define SFB_SRC() \
  ((const char*)weight_sf + (u64)(ntp * (NTILE / 128) + sfh) * (SFK / 4) * 512 + sfj * 512 + sfo + (u64)ktp * 1024)
#elif NWARPS == 1
// Atom row-group (ntp*16)/128, 32-row half (ntp&1), sub-atom column
// 4*((ntp/2)%4); block b at 512B, tile row w at 16B within the half.
#define SFB_SRC()                                                                                         \
  ((const char*)weight_sf + (u64)((ntp * NTILE) >> 7) * (SFK / 4) * 512 + (u64)ktp * 1024 + sfb_b * 512 + \
   16 * (((ntp & 1) << 4) + sfb_r2) + 4 * (((ntp * NTILE) >> 5) & 3))
#else
// Atom row-group (ntp*NTILE)/128, sub-atom column 4*((ntp*NTILE/32)%4),
// block b at 512B, row chunk at 16B.
#define SFB_SRC()                                                                                         \
  ((const char*)weight_sf + (u64)((ntp * NTILE) >> 7) * (SFK / 4) * 512 + (u64)ktp * 1024 + sfb_b * 512 + \
   16 * sfb_r2 + 4 * (((ntp * NTILE) >> 5) & 3))
#endif

#if ROWPAIR
#define B_ROW0 ((tid >> 2) * 2)
#define B_ROW_STEP 1
#else
#define B_ROW0 (tid >> 2)
#define B_ROW_STEP (NTILE / 2)
#endif

#define SET_PROD_PTRS()                                                                                  \
  do {                                                                                                   \
    wsrc0 = (const char*)weight + (u64)(ntp * NTILE + B_ROW0) * KBYTES + (tid & 3) * 16 + (u64)ktp * 64; \
    wsrc1 = wsrc0 + (u64)B_ROW_STEP * KBYTES;                                                            \
    asrc = (const char*)input + (u64)(ar < 16 ? ar : 0) * KBYTES + (tid & 3) * 16 + (u64)ktp * 64;       \
    SET_ASRC8();                                                                                         \
    sfasrc = (const char*)input_sf + sfj * 512 + sfo + (u64)ktp * 1024;                                  \
    sfbsrc = SFB_SRC();                                                                                  \
  } while (0)

  SET_PROD_PTRS();

#if NWARPS == 1
// 32 threads stage everything: A rows ar and ar+8 (2x cp16), both SFA 512B
// chunks (2x cp16), one 4B SFB chunk.
#define SFAB_COPIES(sofs, ko, sko)                         \
  do {                                                     \
    cp16(adst + sofs, asrc + (ko), apred);                 \
    cp16(adst + 640 + sofs, asrc8 + (ko), apred8);         \
    cp16(sfadst + sofs, sfasrc + (sko), true);             \
    cp16(sfadst + 512 + sofs, sfasrc + 512 + (sko), true); \
    cp4(sfbdst + sofs, sfbsrc + (sko));                    \
  } while (0)
#elif NWARPS == 2
#define SFAB_COPIES(sofs, ko, sko)             \
  do {                                         \
    cp16(adst + sofs, asrc + (ko), apred);     \
    cp16(sfadst + sofs, sfasrc + (sko), true); \
    cp4(sfbdst + sofs, sfbsrc + (sko));        \
  } while (0)
#elif NWARPS == 4
#if POLSFB4 && L2POLICY > 0
#define CP_SFB(d, s) cp8w(d, s, pol)
#else
#define CP_SFB(d, s) cp8(d, s)
#endif
#define SFAB_COPIES(sofs, ko, sko)               \
  do {                                           \
    if (tid < 64) {                              \
      cp16(adst + sofs, asrc + (ko), apred);     \
    } else {                                     \
      cp16(sfadst + sofs, sfasrc + (sko), true); \
      CP_SFB(sfbdst + sofs, sfbsrc + (sko));     \
    }                                            \
  } while (0)
#else
#define SFAB_COPIES(sofs, ko, sko)                 \
  do {                                             \
    if (tid < 64) {                                \
      cp16(adst + sofs, asrc + (ko), apred);       \
    } else if (tid < 128) {                        \
      cp16(sfadst + sofs, sfasrc + (sko), true);   \
    } else if (tid < 128 + 64 * (NTILE / 128)) {   \
      if (POLSFB)                                  \
        cp16w(sfbdst + sofs, sfbsrc + (sko), pol); \
      else                                         \
        cp16(sfbdst + sofs, sfbsrc + (sko), true); \
    }                                              \
  } while (0)
#endif

#if POLB1
#define CP_B1(d, s) cp16w(d, s, pol)
#else
#define CP_B1(d, s) cp16p(d, s)
#endif

// Stream-K producer: pointer-advancing walk across units.
#define PRODUCE(idx, slot)                       \
  do {                                           \
    if ((idx) < rlen) {                          \
      const u32 sofs = (slot) * STAGE_WORDS * 4; \
      cp16w(bdst0 + sofs, wsrc0, pol);           \
      CP_B1(bdst1 + sofs, wsrc1);                \
      SFAB_COPIES(sofs, 0, 0);                   \
      if (++ktp == KTILES) {                     \
        ktp = 0;                                 \
        ++ntp;                                   \
        SET_PROD_PTRS();                         \
      } else {                                   \
        wsrc0 += 64;                             \
        wsrc1 += 64;                             \
        asrc += 64;                              \
        sfasrc += 1024;                          \
        sfbsrc += 1024;                          \
      }                                          \
    }                                            \
    cp_commit();                                 \
  } while (0)

// Static producer: loop-invariant bases plus a k-relative offset.
#if PFL2 > 0
// One 128B-line prefetch per B row (both halves), issued on even k-iters at
// distance PFL2 (even) ahead; predicated off near the k-slice end.
#define PF_B(ktl, koff)                                                  \
  do {                                                                   \
    if (((ktl) & 1) == 0 && (tid & 3) == 0 && (ktl) + PFL2 < KT_SPLIT) { \
      pfl2(wsrc0 + (koff) + (u64)PFL2 * 64, pol);                        \
      pfl2(wsrc1 + (koff) + (u64)PFL2 * 64, pol);                        \
    }                                                                    \
  } while (0)
#else
#define PF_B(ktl, koff) \
  do {                  \
  } while (0)
#endif
#define PRODUCE_S(ktl, slot)                        \
  do {                                              \
    if ((ktl) < KT_SPLIT) {                         \
      const u64 koff = (u64)KROT_MAP(ktl) * 64;     \
      const u64 sfkoff = (u64)KROT_MAP(ktl) * 1024; \
      const u32 sofs = (slot) * STAGE_WORDS * 4;    \
      cp16w(bdst0 + sofs, wsrc0 + koff, pol);       \
      CP_B1(bdst1 + sofs, wsrc1 + koff);            \
      SFAB_COPIES(sofs, koff, sfkoff);              \
      PF_B(KROT_MAP(ktl), koff);                    \
    }                                               \
    cp_commit();                                    \
  } while (0)

  // ---- consumer per-thread word indices (within a stage) ----
  const int aw0 = A_OFF + q * 20 + t;        // row q
  const int aw1 = A_OFF + (q + 8) * 20 + t;  // row q+8
  const int w0 = warp * 16 + q;              // B row, group 0
  const int w1 = warp * 16 + 8 + q;          // B row, group 1
  const int bw0 = w0 * 20 + t;
  const int bw1 = w1 * 20 + t;
  const int sfaw = SFA_OFF + 4 * mA;
#if NWARPS >= 8
  const int sfbw0 = SFB_OFF + (w0 >> 7) * 256 + 4 * (w0 & 31) + ((w0 >> 5) & 3);
  const int sfbw1 = SFB_OFF + (w1 >> 7) * 256 + 4 * (w1 & 31) + ((w1 >> 5) & 3);
#define SFB_JSTEP 128
#elif NWARPS == 4
  const int sfbw0 = SFB_OFF + 2 * (w0 & 31) + (w0 >> 5);
  const int sfbw1 = SFB_OFF + 2 * (w1 & 31) + (w1 >> 5);
#define SFB_JSTEP 64
#elif NWARPS == 2
  const int sfbw0 = SFB_OFF + w0;
  const int sfbw1 = SFB_OFF + w1;
#define SFB_JSTEP 32
#else
  // NWARPS==1: stage layout [block b][tile row w], 4B per row.
  const int sfbw0 = SFB_OFF + w0;
  const int sfbw1 = SFB_OFF + w1;
#define SFB_JSTEP 16
#endif

  const int r0 = q;
  const int r1 = q + 8;
  const bool p0 = r0 < M;
  const bool p1 = r1 < M;

  float acc0[4] = {0.f, 0.f, 0.f, 0.f};
  float acc1[4] = {0.f, 0.f, 0.f, 0.f};

#define CONSUME(st)                               \
  _Pragma("unroll") for (int j = 0; j < 2; ++j) { \
    const u32 a0 = (st)[aw0 + j * 8];             \
    const u32 a2 = (st)[aw0 + j * 8 + 4];         \
    const u32 a1 = (st)[aw1 + j * 8];             \
    const u32 a3 = (st)[aw1 + j * 8 + 4];         \
    const u32 sfa = (st)[sfaw + j * 128];         \
    const u32 b00 = (st)[bw0 + j * 8];            \
    const u32 b01 = (st)[bw0 + j * 8 + 4];        \
    const u32 sfb0 = (st)[sfbw0 + j * SFB_JSTEP]; \
    const u32 b10 = (st)[bw1 + j * 8];            \
    const u32 b11 = (st)[bw1 + j * 8 + 4];        \
    const u32 sfb1 = (st)[sfbw1 + j * SFB_JSTEP]; \
    MMA_OR_MEM();                                 \
  }
#if MEMONLY
#define MMA_OR_MEM()                                                \
  do {                                                              \
    acc0[0] += (float)(a0 + a1 + a2 + a3 + b00 + b01 + sfa + sfb0); \
    acc1[0] += (float)(b10 + b11 + sfb1);                           \
  } while (0)
#else
#define MMA_OR_MEM()                                      \
  do {                                                    \
    mma_nvfp4(acc0, a0, a1, a2, a3, b00, b01, sfa, sfb0); \
    mma_nvfp4(acc1, a0, a1, a2, a3, b10, b11, sfa, sfb1); \
  } while (0)
#endif

#if SPLIT
  // ================= static aligned schedule =================
  // ---- prologue ----
#if L2POLICY > 0 && NWARPS >= 8
  // Unrolled prologue slots are compile-time constants; on the NW8 register
  // schedule ptxas re-derives a uniform smem component and folds hinted
  // LDGSTS to [R+UR+imm], dropping the descriptor (phantom desc[UR1]).
  // Runtime slots keep the addresses in laundered [R+imm] form.
#pragma unroll 1
#else
#pragma unroll
#endif
  for (int i = 0; i < STAGES - 1; ++i)
    PRODUCE_S(i, i % STAGES);

  // ---- main loop (compile-time trip count, wrap-counter slots) ----
  {
    int ps = (STAGES - 1) % STAGES;  // produce slot
    int cs = 0;                      // consume slot
    for (int i = 0; i < KT_SPLIT; ++i) {
      cp_wait<STAGES - 2>();
      __syncthreads();
#if IKET
      if (tid == 0 && i == 0) IKET_TS(ik_e1);
#endif
      PRODUCE_S(i + STAGES - 1, ps);
      const u32* st = smem + cs * STAGE_WORDS;
      CONSUME(st);
      if (++ps == STAGES) ps = 0;
      if (++cs == STAGES) cs = 0;
    }
  }
#if IKET
  if (tid == 0) IKET_TS(ik_e2);
#endif

  // ---- epilogue (once, out of the loop) ----
  {
    const int n_base = ntc0 * NTILE + warp * 16;
    const int c0 = n_base + 2 * t;
    const int c1 = n_base + 8 + 2 * t;
#if SPLIT == 1
    const float alp = *alpha;
    if (p0) {
      st_out(out + (((u64)r0 * NDIM + c0) >> 1), pack_bf16x2(alp * acc0[0], alp * acc0[1]));
      st_out(out + (((u64)r0 * NDIM + c1) >> 1), pack_bf16x2(alp * acc1[0], alp * acc1[1]));
    }
    if (p1) {
      st_out(out + (((u64)r1 * NDIM + c0) >> 1), pack_bf16x2(alp * acc0[2], alp * acc0[3]));
      st_out(out + (((u64)r1 * NDIM + c1) >> 1), pack_bf16x2(alp * acc1[2], alp * acc1[3]));
    }
#if IKET
    if (tid == 0) {
      u64 ik_e5;
      IKET_TS(ik_e5);
      u64* ib = g_iket + (u64)c * 8;
      ib[0] = (u64)ik_smid << 32;
      ib[1] = ik_e0;
      ib[2] = ik_e1;
      ib[3] = ik_e2;
      ib[4] = ik_e3;
      ib[5] = ik_e4;
      ib[6] = ik_e5;
      ib[7] = 0x1CE7C0DEULL;
    }
#endif
#else
    // Split-K: slotted f32x2 workspace, then ALL SPLIT contributors spin on
    // the arrival ticket and reduce disjoint column slices in parallel
    // (requires every contributor to be co-resident; holds for the shipped
    // down config, G=160 on 188 SMs). sem[nt] counts arrivals, sem[NT+nt]
    // counts finished reducers; the last reducer resets both. SPINRED=0
    // instead uses the serial ticket reduction (last arriver reduces all
    // slices alone) which has no residency requirement — required for
    // G > 1 wave.
    const int rank = (int)(c % SPLIT);
    float2* wp0 = (float2*)(ws + ((u64)rank * 16 + r0) * NDIM);
    float2* wp1 = (float2*)(ws + ((u64)rank * 16 + r1) * NDIM);
    wp0[c0 >> 1] = {acc0[0], acc0[1]};
    wp0[c1 >> 1] = {acc1[0], acc1[1]};
    wp1[c0 >> 1] = {acc0[2], acc0[3]};
    wp1[c1 >> 1] = {acc1[2], acc1[3]};
    __syncthreads();
#if !SPINRED
    if (tid == 0) {
      __threadfence();
      ticket = atomicAdd(&sem[ntc0], 1u);
    }
    __syncthreads();
    if (ticket == SPLIT - 1) {
      __threadfence();
      const float alp = *alpha;
      for (int idx = tid; idx < (NTILE / 2) * M; idx += THREADS) {
        const int row = idx / (NTILE / 2);
        const int cp = idx % (NTILE / 2);
        const u64 col = (u64)ntc0 * NTILE + 2 * cp;
        float v0 = 0.f, v1 = 0.f;
#pragma unroll
        for (int s = 0; s < SPLIT; ++s) {
          const float2 w = *(const float2*)(ws + ((u64)s * 16 + row) * NDIM + col);
          v0 += w.x;
          v1 += w.y;
        }
        st_out(out + (((u64)row * NDIM + col) >> 1), pack_bf16x2(alp * v0, alp * v1));
      }
      if (tid == 0) sem[ntc0] = 0u;
    }
#else
#if FENCEREL
    // Release/acquire message passing instead of full __threadfence()s:
    // the release-arrival (after the barrier) publishes every thread's ws
    // stores; the acquire-spin + barrier makes all contributors' stores
    // visible CTA-wide. Drops 3x MEMBAR.SC.GPU (+CCTL.IVALL) from the tail.
    if (tid == 0) {
#if IKET
      IKET_TS(ik_e3);
#endif
      u32 _a;
      asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_a) : "l"(sem + ntc0) : "memory");
      u32 seen;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(sem + ntc0));
      } while (seen < SPLIT);
#if IKET
      IKET_TS(ik_e4);
#endif
    }
    __syncthreads();
#else
    if (tid == 0) {
#if IKET
      IKET_TS(ik_e3);
#endif
      __threadfence();
      atomicAdd(&sem[ntc0], 1u);
      u32 seen;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(sem + ntc0));
      } while (seen < SPLIT);
#if IKET
      IKET_TS(ik_e4);
#endif
    }
    __syncthreads();
    __threadfence();  // order all contributors' ws stores before our reads
#endif
    {
      const float alp = *alpha;
      const int work = (NTILE / 2) * M;
      const int lo = work * rank / SPLIT;
      const int hi = work * (rank + 1) / SPLIT;
      for (int idx = lo + tid; idx < hi; idx += THREADS) {
        const int row = idx / (NTILE / 2);
        const int cp = idx % (NTILE / 2);
        const u64 col = (u64)ntc0 * NTILE + 2 * cp;
        float v0 = 0.f, v1 = 0.f;
#pragma unroll
        for (int s = 0; s < SPLIT; ++s) {
          const float2 w = *(const float2*)(ws + ((u64)s * 16 + row) * NDIM + col);
          v0 += w.x;
          v1 += w.y;
        }
        st_out(out + (((u64)row * NDIM + col) >> 1), pack_bf16x2(alp * v0, alp * v1));
      }
    }
    __syncthreads();
    if (tid == 0) {
#if FENCEREL
      u32 done;
      asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(done) : "l"(sem + NT + ntc0) : "memory");
#else
      __threadfence();
      const u32 done = atomicAdd(&sem[NT + ntc0], 1u);
#endif
      if (done == SPLIT - 1) {
        sem[ntc0] = 0u;
        sem[NT + ntc0] = 0u;
      }
    }
#if IKET
    if (tid == 0) {
      u64 ik_e5;
      IKET_TS(ik_e5);
      u64* ib = g_iket + (u64)c * 8;
      ib[0] = ((u64)ik_smid << 32) | (u64)rank;
      ib[1] = ik_e0;
      ib[2] = ik_e1;
      ib[3] = ik_e2;
      ib[4] = ik_e3;
      ib[5] = ik_e4;
      ib[6] = ik_e5;
      ib[7] = 0x1CE7C0DEULL;
    }
#endif
#endif  // SPINRED
#endif
  }
#else
  // ================= generalized stream-K =================
  // Consumer walk state.
  int ntc = (int)(beg / KTILES);
  int ktc = (int)(beg % KTILES);
  int ktlo = ktc;  // k-unit where this CTA entered the current tile

  // ---- prologue ----
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i)
    PRODUCE(i, i % STAGES);

  // ---- main loop ----
  for (int i = 0; i < rlen; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    PRODUCE(i + STAGES - 1, (i + STAGES - 1) % STAGES);
    const u32* st = smem + (i % STAGES) * STAGE_WORDS;
    CONSUME(st);

    // ---- tile flush at tile boundary or range end ----
    if (ktc == KTILES - 1 || i == rlen - 1) {
      const int n_base = ntc * NTILE + warp * 16;
      const int c0 = n_base + 2 * t;
      const int c1 = n_base + 8 + 2 * t;
      if (NORED || (ktlo == 0 && ktc == KTILES - 1)) {
        // Full tile owned by this CTA: direct store.
        if (p0) {
          st_out(out + (((u64)r0 * NDIM + c0) >> 1), pack_bf16x2(alp * acc0[0], alp * acc0[1]));
          st_out(out + (((u64)r0 * NDIM + c1) >> 1), pack_bf16x2(alp * acc1[0], alp * acc1[1]));
        }
        if (p1) {
          st_out(out + (((u64)r1 * NDIM + c0) >> 1), pack_bf16x2(alp * acc0[2], alp * acc0[3]));
          st_out(out + (((u64)r1 * NDIM + c1) >> 1), pack_bf16x2(alp * acc1[2], alp * acc1[3]));
        }
      } else {
        // Partial tile: slotted workspace + ticket reduction.
        const u64 tile_beg = (u64)ntc * KTILES;
        const u32 cfirst = (u32)((G * (tile_beg + 1) - 1) / UTOT);
        const u32 clast = (u32)((G * (tile_beg + KTILES) - 1) / UTOT);
        const u32 nexp = clast - cfirst + 1;
        const u32 rank = c - cfirst;
        float* wp0 = ws + ((u64)rank * 16 + r0) * NDIM;
        float* wp1 = ws + ((u64)rank * 16 + r1) * NDIM;
        wp0[c0] = acc0[0];
        wp0[c0 + 1] = acc0[1];
        wp0[c1] = acc1[0];
        wp0[c1 + 1] = acc1[1];
        wp1[c0] = acc0[2];
        wp1[c0 + 1] = acc0[3];
        wp1[c1] = acc1[2];
        wp1[c1 + 1] = acc1[3];
        __syncthreads();
        if (tid == 0) {
          __threadfence();
          ticket = atomicAdd(&sem[ntc], 1u);
        }
        __syncthreads();
        if (ticket == nexp - 1) {
          __threadfence();
          for (int idx = tid; idx < (NTILE / 2) * M; idx += THREADS) {
            const int row = idx / (NTILE / 2);
            const int cp = idx % (NTILE / 2);
            const u64 col = (u64)ntc * NTILE + 2 * cp;
            float v0 = 0.f, v1 = 0.f;
            for (u32 s = 0; s < nexp; ++s) {
              const float* wsr = ws + ((u64)s * 16 + row) * NDIM + col;
              v0 += wsr[0];
              v1 += wsr[1];
            }
            st_out(out + (((u64)row * NDIM + col) >> 1), pack_bf16x2(alp * v0, alp * v1));
          }
          if (tid == 0) sem[ntc] = 0u;
        }
      }
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        acc0[z] = 0.f;
        acc1[z] = 0.f;
      }
    }
    if (++ktc == KTILES) {
      ktc = 0;
      ++ntc;
      ktlo = 0;
    }
  }
#endif
}
