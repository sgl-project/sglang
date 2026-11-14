// Muse Spark MoE finalize fused with the TP4 all-reduce of its output (gfx950, xGMI peers).
//
// One 256-thread block per token row r of an M-row batch (M a multiple of 4, hidden 4096, top-8):
//   1. finalize: the per-expert input-scale RMSNorm and route-weighted sum of the row's local-expert slots, with the
//      thread mapping, reduction order and rounding points of aiter's FlyDSL muse_moe_finalize_vector_flydsl;
//   2. the row's partial goes to its owner rank r / (M / 4), which sums the four partials in the order of aiter's
//      cross_device_reduce_2stage (owner first, then owner+1, ...; FP32, one BF16 rounding) and pushes the sum to
//      the other ranks; every rank writes the row to `out`.
// So the result is bitwise the FlyDSL finalize followed by the aiter two-stage all-reduce, except that a -0.0 can
// come out as +0.0.
//
// The data is its own flag (Lamport): inbox records start as the BF16 -0.0 pattern (0x8000), senders map -0.0 to
// +0.0, readers spin with cache-bypassing volatile loads until no lane holds the sentinel. Each row keeps a call
// counter and the owner of its previous call (advanced identically on every rank because every rank runs the same
// shapes); row-call c uses inbox slot c % 3 and resets the records the previous row-call left in this rank's inbox
// (slot (c + 2) % 3, placed by that call's owner), which no peer can write again before this rank's next call.
// Inbox accesses are raw buffer ops with the sc0|sc1 cache policy on coarse-grained IPC memory.
#include <hip/hip_runtime.h>
#include <cstdint>

#define NR 4
#define HID 4096
#define TOPK 8
#define ROW_BYTES (HID * 2)
#define PACKS_PER_ROW (ROW_BYTES / 16)
#define SENT2 0x80008000u

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __amdgpu_buffer_rsrc_t rsrc_t;
#define CPOL_SYS 17
#define CPOL_SYS_VOL (17 | (1u << 31))

struct Peers {
    uint8_t* inbox[NR];
};

struct Args {
    const uint16_t* y;       // [M * 8, 4096] bf16 per-slot expert rows (only local slots are read)
    const uint16_t* gamma;   // [4096] bf16 post-expert norm weight
    const float* scores;     // [M, 8]
    const int32_t* ids;      // [M, 8] global expert ids
    uint4* out;              // [M, 4096] bf16
    Peers peers;
    int32_t* row_state;      // [max_rows][2]: row-call count, owner of the previous row-call (-1 before the first)
    int32_t local_start;
    int32_t num_local;
    float eps;
    float gain;
    int32_t rank;
    int32_t rows;
    uint32_t max_rows;
    uint32_t pad;
};
static_assert(sizeof(Args) == 112, "Args is mirrored by a ctypes struct");

__device__ __forceinline__ rsrc_t make_rsrc(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, 0, 0x7fffffff, 0x00020000);
}
__device__ __forceinline__ void bst_x4(rsrc_t r, uint32_t off, uint4 v) {
    u32x4 t = {v.x, v.y, v.z, v.w};
    __builtin_amdgcn_raw_buffer_store_b128(t, r, off, 0, CPOL_SYS);
}
__device__ __forceinline__ uint4 bld_x4(rsrc_t r, uint32_t off) {
    u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, CPOL_SYS_VOL);
    asm volatile("" ::: "memory");
    return make_uint4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void wg_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

__device__ __forceinline__ float bf2f(uint32_t b16) { return __uint_as_float(b16 << 16); }
__device__ __forceinline__ uint32_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t no_neg_zero(uint32_t b16) { return b16 == 0x8000u ? 0u : b16; }
__device__ __forceinline__ bool has_sentinel(uint4 v) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    bool hit = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) hit |= ((w[i] & 0xffffu) == 0x8000u) | ((w[i] >> 16) == 0x8000u);
    return hit;
}
__device__ __forceinline__ void add_pack(uint4 v, float* acc, bool first) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = bf2f(w[i] & 0xffffu), hi = bf2f(w[i] >> 16);
        acc[2 * i] = first ? lo : acc[2 * i] + lo;
        acc[2 * i + 1] = first ? hi : acc[2 * i + 1] + hi;
    }
}
__device__ __forceinline__ uint4 pack_acc(const float* acc) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = no_neg_zero(f2bf(acc[2 * i])) | (no_neg_zero(f2bf(acc[2 * i + 1])) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// Inbox of each rank: stage-1 records [3 slots][NR senders][max_rows][ROW_BYTES], then stage-2 records
// [3 slots][max_rows][ROW_BYTES].
__device__ __forceinline__ uint32_t s1_off(const Args& a, uint32_t slot, int sender, int row) {
    return ((slot * NR + sender) * a.max_rows + row) * ROW_BYTES;
}
__device__ __forceinline__ uint32_t s2_off(const Args& a, uint32_t slot, int row) {
    return (3u * NR * a.max_rows + slot * a.max_rows + row) * ROW_BYTES;
}

extern "C" __global__ void __launch_bounds__(256) muse_finalize_allreduce_tp4(Args a) {
    const int r = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const uint32_t call = (uint32_t)a.row_state[2 * r];
    const int prev_owner = a.row_state[2 * r + 1];
    const uint32_t slot = call % 3u, prev = (call + 2u) % 3u;
    const int owner = r / (a.rows / NR);
    __shared__ float sums[TOPK * 4];
    __shared__ uint16_t rowbuf[HID];

    // ---- finalize (FlyDSL muse_moe_finalize_vector_flydsl numerics: H 4096, 256 threads, vec 4) ----
    float scl[16], st[16];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint2 g = *(const uint2*)(a.gamma + tid * 4 + c * 1024);
        const uint32_t gw[4] = {g.x & 0xffffu, g.x >> 16, g.y & 0xffffu, g.y >> 16};
#pragma unroll
        for (int j = 0; j < 4; ++j) scl[c * 4 + j] = bf2f(f2bf(bf2f(gw[j]) + a.gain));
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) st[k] = 0.f;
    for (int s = 0; s < TOPK; ++s) {
        const int slot_idx = r * TOPK + s;
        const int e = a.ids[slot_idx];
        if (e >= a.local_start && e < a.local_start + a.num_local) {
            float sc = a.scores[slot_idx];
            const uint16_t* yrow = a.y + (size_t)slot_idx * HID;
            float v[16];
            float ss = 0.f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint2 raw = *(const uint2*)(yrow + tid * 4 + c * 1024);
                const uint32_t yw[4] = {raw.x & 0xffffu, raw.x >> 16, raw.y & 0xffffu, raw.y >> 16};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float val = bf2f(f2bf(bf2f(yw[j]) * scl[c * 4 + j]));
                    v[c * 4 + j] = val;
                    ss = ss + val * val;
                }
            }
#pragma unroll
            for (int sh = 32; sh >= 1; sh >>= 1) ss = ss + __shfl_xor(ss, sh, 64);
            if (lane == 0) sums[s * 4 + wave] = ss;
            __syncthreads();
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < 4; ++w) total = total + sums[s * 4 + w];
            const float mean = total * (1.0f / HID) + a.eps;
            sc = sc * (1.0f / MUSE_SQRT(mean));
#pragma unroll
            for (int k = 0; k < 16; ++k) st[k] = st[k] + sc * v[k];
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint2 o;
        o.x = no_neg_zero(f2bf(st[c * 4])) | (no_neg_zero(f2bf(st[c * 4 + 1])) << 16);
        o.y = no_neg_zero(f2bf(st[c * 4 + 2])) | (no_neg_zero(f2bf(st[c * 4 + 3])) << 16);
        *(uint2*)(rowbuf + tid * 4 + c * 1024) = o;
    }
    __syncthreads();
    uint4 mine[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) mine[i] = ((const uint4*)rowbuf)[tid + i * 256];

    const rsrc_t my = make_rsrc(a.peers.inbox[a.rank]);
    uint4* outrow = a.out + (size_t)r * PACKS_PER_ROW;
    if (owner != a.rank) {
        // ---- push the partial to the owner, then wait for the owner's sum ----
        const rsrc_t dst = make_rsrc(a.peers.inbox[owner]);
        const uint32_t base = s1_off(a, slot, a.rank, r);
#pragma unroll
        for (int i = 0; i < 2; ++i) bst_x4(dst, base + (tid + i * 256) * 16, mine[i]);
        const uint32_t rbase = s2_off(a, slot, r);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint4 t;
            do { t = bld_x4(my, rbase + (tid + i * 256) * 16); } while (has_sentinel(t));
            outrow[tid + i * 256] = t;
        }
    } else {
        // ---- owner: sum the four partials (owner first), push the sum to the other ranks ----
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = tid + i * 256;
            float acc[8];
            add_pack(mine[i], acc, true);
#pragma unroll
            for (int k = 1; k < NR; ++k) {
                const int src = (a.rank + k) % NR;
                uint4 t;
                do { t = bld_x4(my, s1_off(a, slot, src, r) + p * 16); } while (has_sentinel(t));
                add_pack(t, acc, false);
            }
            const uint4 sum = pack_acc(acc);
#pragma unroll
            for (int k = 1; k < NR; ++k) {
                const int d = (a.rank + k) % NR;
                bst_x4(make_rsrc(a.peers.inbox[d]), s2_off(a, slot, r) + p * 16, sum);
            }
            outrow[p] = sum;
        }
    }
    const uint4 sent = make_uint4(SENT2, SENT2, SENT2, SENT2);
    if (prev_owner == a.rank) {
#pragma unroll
        for (int k = 1; k < NR; ++k) {
            const uint32_t cbase = s1_off(a, prev, (a.rank + k) % NR, r);
#pragma unroll
            for (int i = 0; i < 2; ++i) bst_x4(my, cbase + (tid + i * 256) * 16, sent);
        }
    } else if (prev_owner >= 0) {
        const uint32_t cbase = s2_off(a, prev, r);
#pragma unroll
        for (int i = 0; i < 2; ++i) bst_x4(my, cbase + (tid + i * 256) * 16, sent);
    }
    wg_barrier();  // every wave has read row_state[r] before it advances
    if (tid == 0) {
        a.row_state[2 * r] = (int32_t)(call + 1u);
        a.row_state[2 * r + 1] = owner;
    }
}

struct FillArgs {
    uint4* p;
    uint64_t n16;
};
static_assert(sizeof(FillArgs) == 16, "FillArgs is mirrored by a ctypes struct");
extern "C" __global__ void muse_finalize_allreduce_fill(FillArgs a) {
    const uint4 s = make_uint4(SENT2, SENT2, SENT2, SENT2);
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < a.n16; i += (uint64_t)gridDim.x * blockDim.x)
        a.p[i] = s;
}
