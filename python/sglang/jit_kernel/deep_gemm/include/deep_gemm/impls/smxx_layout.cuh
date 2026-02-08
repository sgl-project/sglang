#pragma once

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K,
          uint32_t PADDED_SF_K = SF_K + (1 - (SF_K % 2))>
__global__ void transpose_fp32(const float* sf, float* out, const uint32_t mn) {
    typedef typename Vectorized<sizeof(float) * SF_K>::vec_t in_vec_t;
    constexpr static uint32_t kNumElemsPerVec = sizeof(in_vec_t) / sizeof(float);
    constexpr static uint32_t SF_VEC_K = SF_K / kNumElemsPerVec;

    // Shapes and strides
    extern __shared__ float smem_buffer[];
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(float));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = align<uint32_t>(mn, kNumTMAAlignedElems);

    // Shift into the block
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * SF_K;
    const auto& local_sf = reinterpret_cast<const in_vec_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));

    // Load
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_VEC_K; i += kNumThreads) {
        auto in_vec = __ldg(local_sf + i);
        const auto& in_values = reinterpret_cast<float*>(&in_vec);

        const auto& row = i / SF_VEC_K, col = (i % SF_VEC_K) * kNumElemsPerVec;
        #pragma unroll
        for (uint32_t j = 0; j < kNumElemsPerVec; ++ j)
            smem_buffer[row * PADDED_SF_K + col + j] = in_values[j];
    }
    __syncthreads();

    // Store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_K; i += kNumThreads) {
        const auto& sf_k_idx = i / in_block_mn, mn_idx = i % in_block_mn;
        const auto& global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx;
        out[sf_k_idx * tma_aligned_mn + global_mn_idx] = ld_shared(smem_buffer + mn_idx * PADDED_SF_K + sf_k_idx);
    }
}

// NOTES: the two kernels below always pack the K dimension

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K>
__global__ void transpose_and_pack_fp32_into_ue8m0(float* sf, uint32_t* out, const uint32_t mn) {
    extern __shared__ uint32_t smem_buffer[];

    // Shapes and strides
    constexpr auto kNumPackedSFK = constexpr_ceil_div(SF_K, 4u);
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(int));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = align<uint64_t>(mn, kNumTMAAlignedElems);

    // Shift into the group
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * kNumPackedSFK;

    // Load FP32 SFs
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block size");
    const auto local_sf = reinterpret_cast<uint32_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));
    const auto num_values = in_block_mn * SF_K;
    const auto num_uint4 = num_values / 4;
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < num_uint4; i += kNumThreads) {
        const auto& [x, y, z, w] = __ldg(reinterpret_cast<uint4*>(local_sf) + i);
        st_shared(reinterpret_cast<uint4*>(smem_buffer) + i, x, y, z, w);
    }

    // Fill unaligned values as well
    if (const auto unaligned_idx = num_uint4 * 4 + threadIdx.x; unaligned_idx < num_values)
        st_shared(smem_buffer + unaligned_idx, __ldg(local_sf + unaligned_idx));
    __syncthreads();

    // Pack into UE8M0 and store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < (kNumPackedSFK * BLOCK_MN); i += kNumThreads) {
        const auto sf_k_pack_idx = i / BLOCK_MN, mn_idx = i % BLOCK_MN;

        // Load shared memory
        uint32_t values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            const auto sf_k_idx = sf_k_pack_idx * 4 + j;
            values[j] = sf_k_idx < SF_K ? ld_shared(smem_buffer + mn_idx * SF_K + sf_k_idx) : 0;
        }

        // Pack and store
        uint32_t packed = 0;
        packed |= (values[0] >> 23u);
        packed |= (values[1] >> 15u);
        packed |= (values[2] >>  7u);
        packed |= (values[3] <<  1u);
        if (const auto global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx; global_mn_idx < mn)
            out[sf_k_pack_idx * tma_aligned_mn + global_mn_idx] = packed;
    }
}

template <uint32_t kNumGroups, uint32_t kNumThreads,
          uint32_t BLOCK_MN, uint32_t BLOCK_PACKED_SF_K, bool kTransposed = true>
__global__ void pack_fp32_into_ue8m0(float* sf, uint32_t* out, uint32_t* ks,
                                     const uint32_t mn, uint32_t sf_k, const uint32_t packed_sf_k) {
    // Always packing the K dimension
    // NOTES: should also assert `mn % 4 == 0` at launch
    DG_STATIC_ASSERT(kTransposed, "Currently only support transposed SFs (MN-major)");
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_PACKED_SF_K == kNumThreads / 32, "Invalid block sizes");

    // Shapes and strides
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto in_block_mn_uint4 = in_block_mn / 4;
    const auto in_block_packed_sf_k = min(BLOCK_PACKED_SF_K, packed_sf_k - blockIdx.y * BLOCK_PACKED_SF_K);

    // Shift into the right block along MN
    sf += blockIdx.x * BLOCK_MN;
    out += blockIdx.x * BLOCK_MN;

    // Each warp is responsible for a packed row
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = get_lane_idx();
    const auto packed_sf_k_idx = static_cast<uint64_t>(blockIdx.y) * BLOCK_PACKED_SF_K + warp_idx;
    if (warp_idx >= in_block_packed_sf_k)
        return;

    // Make an offset on the input
    uint32_t input_offset = 0;
    if constexpr (kNumGroups > 1) {
        // Load each group's size
        DG_STATIC_ASSERT(kNumGroups <= 128, "Too many groups");
        uint32_t group_ks[4];
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++ i) {
            const auto group_idx = lane_idx * 4 + i;
            group_ks[i] = group_idx < kNumGroups ? __ldg(ks + group_idx) : 0;
        }
        __syncwarp();

        // Make the offset
        sf_k = 0;
        auto sum_packed_sf_k = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumGroups; ++ i) {
            const auto sf_k_in_group = __shfl_sync(0xffffffff, group_ks[i % 4] / 128, i / 4);
            sf_k += sf_k_in_group;
            sum_packed_sf_k += ceil_div(sf_k_in_group, 4u);
            if (packed_sf_k_idx < sum_packed_sf_k)
                break;
            if (const auto remainder = sf_k_in_group % 4; remainder > 0)
                input_offset += 4 - remainder;
        }
    }

    for (uint32_t mn_idx = get_lane_idx(); mn_idx < in_block_mn_uint4; mn_idx += 32) {
        // Load
        uint4 values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            values[j] = make_uint4(0, 0, 0, 0);
            if (const auto sf_k_idx = packed_sf_k_idx * 4 + j - input_offset; sf_k_idx < sf_k)
                values[j] = __ldg(reinterpret_cast<uint4*>(sf + sf_k_idx * mn) + mn_idx);
        }

        // Pack and store
        uint4 packed;
        packed.x = (values[0].x >> 23u) | (values[1].x >> 15u) | (values[2].x >> 7u) | (values[3].x << 1u);
        packed.y = (values[0].y >> 23u) | (values[1].y >> 15u) | (values[2].y >> 7u) | (values[3].y << 1u);
        packed.z = (values[0].z >> 23u) | (values[1].z >> 15u) | (values[2].z >> 7u) | (values[3].z << 1u);
        packed.w = (values[0].w >> 23u) | (values[1].w >> 15u) | (values[2].w >> 7u) | (values[3].w << 1u);
        reinterpret_cast<uint4*>(out + packed_sf_k_idx * mn)[mn_idx] = packed;
    }
}

} // namespace deep_gemm
