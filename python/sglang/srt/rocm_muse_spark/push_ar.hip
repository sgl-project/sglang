// Generic bf16 push all-reduce for gfx1250, TP NR ranks.
//
// Data is its own flag (Lamport): inbox records start as the BF16 -0.0 pattern (0x8000); senders map -0.0 to +0.0;
// readers spin with system-scope volatile loads until no lane holds the sentinel.
//   one-shot (small): every rank pushes its whole input to every peer, each rank sums the NR copies in rank order.
//   two-shot (large): rank j owns chunk j (n16 / NR packs); others push chunk j to j, j sums in rank order and pushes
//                     the sum to every peer; every rank writes the full result.
// Every call launches the same fixed grid G, so all blocks advance their own call counter in lockstep; call c uses
// inbox slot c % 3 and resets (back to the sentinel) what the previous call left in this rank's inbox. No peer can
// write slot (c + 2) % 3 = prev again before this rank's next kernel, which starts only after this one completed.
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef NR
#define NR 8
#endif
#define MAX_NR 8
#define SENT2 0x80008000u
#define BLK 256

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __amdgpu_buffer_rsrc_t rsrc_t;
// gfx12 cache policy: TH [2:0], SCOPE [4:3] (3 = SYS); bit 31 = volatile
#define CPOL_ST (3u << 3)
#define CPOL_LD ((3u << 3) | (1u << 31))

struct Args {
    const uint4* in;      // [n16] bf16 packs
    uint4* out;           // [n16]
    uint8_t* inbox[MAX_NR];
    int32_t* state;       // [G][4]: call count, prev n16, prev mode (0 none, 1 oneshot, 2 twoshot)
    uint32_t n16;
    uint32_t max16;       // inbox capacity in packs per record region
    int32_t rank;
    uint32_t oneshot;
};
static_assert(sizeof(Args) == 104, "Args is mirrored by a ctypes struct");

__device__ __forceinline__ rsrc_t make_rsrc(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, 0, 0x7fffffff, 0x00020000);
}
__device__ __forceinline__ void bst(rsrc_t r, uint32_t pack, uint4 v) {
    u32x4 t = {v.x, v.y, v.z, v.w};
    __builtin_amdgcn_raw_buffer_store_b128(t, r, pack * 16u, 0, CPOL_ST);
}
__device__ __forceinline__ uint4 bld(rsrc_t r, uint32_t pack) {
    u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, pack * 16u, 0, CPOL_LD);
    asm volatile("" ::: "memory");
    return make_uint4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ float bf2f(uint32_t b16) { return __uint_as_float(b16 << 16); }
__device__ __forceinline__ uint32_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t nz16(uint32_t b) { return b == 0x8000u ? 0u : b; }
__device__ __forceinline__ uint32_t nz32(uint32_t w) { return nz16(w & 0xffffu) | (nz16(w >> 16) << 16); }
__device__ __forceinline__ uint4 nzp(uint4 v) { return make_uint4(nz32(v.x), nz32(v.y), nz32(v.z), nz32(v.w)); }
__device__ __forceinline__ bool has_sent(uint4 v) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    bool hit = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) hit |= ((w[i] & 0xffffu) == 0x8000u) | ((w[i] >> 16) == 0x8000u);
    return hit;
}
__device__ __forceinline__ uint4 wait_ld(rsrc_t r, uint32_t pack) {
    uint4 t;
    do { t = bld(r, pack); } while (has_sent(t));
    return t;
}
__device__ __forceinline__ void add(uint4 v, float* acc, bool first) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = bf2f(w[i] & 0xffffu), hi = bf2f(w[i] >> 16);
        acc[2 * i] = first ? lo : acc[2 * i] + lo;
        acc[2 * i + 1] = first ? hi : acc[2 * i + 1] + hi;
    }
}
__device__ __forceinline__ uint4 pk(const float* a) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = nz16(f2bf(a[2 * i])) | (nz16(f2bf(a[2 * i + 1])) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}
// inbox (packs): s1 [3][NR][max16] then s2 [3][max16]
__device__ __forceinline__ uint32_t s1(const Args& a, uint32_t slot, int src, uint32_t p) {
    return (slot * NR + src) * a.max16 + p;
}
__device__ __forceinline__ uint32_t s2(const Args& a, uint32_t slot, uint32_t p) {
    return 3u * NR * a.max16 + slot * a.max16 + p;
}

extern "C" __global__ void __launch_bounds__(BLK) push_allreduce(Args a) {
    const int b = blockIdx.x, tid = threadIdx.x;
    const uint32_t stride = gridDim.x * BLK, start = b * BLK + tid;
    int32_t* st = a.state + 4 * b;
    const uint32_t call = (uint32_t)st[0], pn16 = (uint32_t)st[1];
    const int pmode = st[2];
    const uint32_t slot = call % 3u, prev = (call + 2u) % 3u;
    const int me = a.rank;
    const rsrc_t my = make_rsrc(a.inbox[me]);
    rsrc_t peer[NR];
#pragma unroll
    for (int j = 0; j < NR; ++j) peer[j] = make_rsrc(a.inbox[j]);

    if (a.oneshot) {
        for (uint32_t p = start; p < a.n16; p += stride) {
            const uint4 v = nzp(a.in[p]);
#pragma unroll
            for (int k = 1; k < NR; ++k) bst(peer[(me + k) % NR], s1(a, slot, me, p), v);
        }
        for (uint32_t p = start; p < a.n16; p += stride) {
            float acc[8];
#pragma unroll
            for (int src = 0; src < NR; ++src) {
                const uint4 t = src == me ? a.in[p] : wait_ld(my, s1(a, slot, src, p));
                add(t, acc, src == 0);
            }
            a.out[p] = pk(acc);
        }
    } else {
        const uint32_t C = a.n16 / NR;
        for (uint32_t q = start; q < C; q += stride) {
#pragma unroll
            for (int k = 1; k < NR; ++k) {
                const int j = (me + k) % NR;
                bst(peer[j], s1(a, slot, me, j * C + q), nzp(a.in[j * C + q]));
            }
        }
        for (uint32_t q = start; q < C; q += stride) {
            const uint32_t idx = me * C + q;
            float acc[8];
#pragma unroll
            for (int src = 0; src < NR; ++src) {
                const uint4 t = src == me ? a.in[idx] : wait_ld(my, s1(a, slot, src, idx));
                add(t, acc, src == 0);
            }
            const uint4 sum = pk(acc);
#pragma unroll
            for (int k = 1; k < NR; ++k) bst(peer[(me + k) % NR], s2(a, slot, idx), sum);
            a.out[idx] = sum;
        }
#pragma unroll
        for (int k = 1; k < NR; ++k) {
            const int j = (me + k) % NR;
            for (uint32_t q = start; q < C; q += stride) a.out[j * C + q] = wait_ld(my, s2(a, slot, j * C + q));
        }
    }
    // reset what the previous call left in this rank's inbox (slot prev)
    const uint4 sent = make_uint4(SENT2, SENT2, SENT2, SENT2);
    if (pmode == 1) {
        for (uint32_t p = start; p < pn16; p += stride)
#pragma unroll
            for (int k = 1; k < NR; ++k) bst(my, s1(a, prev, (me + k) % NR, p), sent);
    } else if (pmode == 2) {
        const uint32_t C = pn16 / NR;
        for (uint32_t q = start; q < C; q += stride) {
#pragma unroll
            for (int k = 1; k < NR; ++k) {
                const int j = (me + k) % NR;
                bst(my, s1(a, prev, j, me * C + q), sent);
                bst(my, s2(a, prev, j * C + q), sent);
            }
        }
    }
    __syncthreads();  // every thread has read st[] before it advances
    if (tid == 0) {
        st[0] = (int32_t)(call + 1u);
        st[1] = (int32_t)a.n16;
        st[2] = a.oneshot ? 1 : 2;
    }
}

struct FillArgs {
    uint4* p;
    uint64_t n16;
};
extern "C" __global__ void push_allreduce_fill(FillArgs a) {
    const uint4 s = make_uint4(SENT2, SENT2, SENT2, SENT2);
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < a.n16; i += (uint64_t)gridDim.x * blockDim.x)
        a.p[i] = s;
}
