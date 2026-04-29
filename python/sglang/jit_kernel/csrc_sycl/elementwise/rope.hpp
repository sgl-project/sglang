/**
 * RoPE (Rotary Position Embedding) SYCL Kernel for SGLang
 * 
 * SYCL implementation of fused rotary position embedding kernels
 * Supports both GPT-NeoX and interleaved (GPT-J) styles
 * 
 * Algorithm:
 * - GPT-J (interleaved): pairs of elements (x,y) rotated by (cos, sin)
 *   out_x = x * cos - y * sin
 *   out_y = x * sin + y * cos
 * - GPT-NeoX: first half and second half rotated separately
 *   x elements in first half, y elements in second half
 * 
 * Formula: 
 * RoPE(x, pos) = x * cos(pos * θ) - y * sin(pos * θ)
 * where θ = 10000^(-2i/d) for dimension i
 */

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace sgl {
namespace sycl_kernel {

// ============================================================================
// Vector Type Helpers
// ============================================================================

template<typename T, int N>
struct alignas(N * sizeof(T)) aligned_vector {
    T data[N];
    
    aligned_vector() = default;
    
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
};

// Packed type for half-precision pairs (bf16/fp16)
struct packed2_t {
    float x, y;
};

// Cast from aligned_vector<T, 2> to packed2_t for processing
template<typename T>
inline packed2_t cast_to_fp32x2(const aligned_vector<T, 2>& v) {
    return packed2_t{static_cast<float>(v[0]), static_cast<float>(v[1])};
}

// Cast from packed2_t back to aligned_vector<T, 2>
template<typename T>
inline aligned_vector<T, 2> cast_from_fp32x2(const packed2_t& v) {
    aligned_vector<T, 2> result;
    result[0] = static_cast<T>(v.x);
    result[1] = static_cast<T>(v.y);
    return result;
}

// ============================================================================
// RoPE Kernel Parameters
// ============================================================================

template<typename DType, typename IdType>
struct FusedRopeParams {
    DType* __restrict__ q_ptr;
    DType* __restrict__ k_ptr;  // Pre-offset in host code
    const float* __restrict__ cos_sin_cache_ptr;
    const IdType* __restrict__ positions;
    int64_t q_stride;
    int64_t k_stride;
    int64_t head_stride;
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t num_tokens;
};

template<typename DType, typename IdType>
struct FusedRopeStoreParams {
    FusedRopeParams<DType, IdType> base_params;
    DType* __restrict__ v_ptr;
    DType* __restrict__ k_cache;
    DType* __restrict__ v_cache;
    const IdType* __restrict__ out_loc;
    int64_t v_stride;
    int64_t cache_stride;
};

// ============================================================================
// RoPE Kernel - Interleaved Style (GPT-J)
// ============================================================================

template<bool kIsNeox, int64_t kRopeDim, typename DType, typename IdType, uint32_t kWorkThreads>
class FusedRopeKernel {
public:
    using DType2 = aligned_vector<DType, 2>;
    
    // Calculate vector size for efficient memory access
    static constexpr int64_t kVecSize = []() {
        uint32_t power = 1;
        uint32_t factor = 2 * kWorkThreads * (1 + kIsNeox);
        while (power * factor < kRopeDim) power *= 2;
        return power;
    }();
    
    static constexpr int64_t kDimPerThread = kVecSize * 2 * (1 + kIsNeox);
    static constexpr uint32_t kLaneCount = kRopeDim / kDimPerThread;
    
    FusedRopeKernel(const FusedRopeParams<DType, IdType>& params) : params_(params) {}
    
    void operator()(::sycl::nd_item<1> item) const {
        const uint32_t lane_id = item.get_local_id(0) % kWorkThreads;
        if constexpr (kLaneCount < kWorkThreads) {
            if (lane_id >= kLaneCount) return;
        }
        
        constexpr uint32_t kBlockSize = 128;
        constexpr uint32_t kWorkersPerBlock = kBlockSize / kWorkThreads;
        const auto num_blks = item.get_group_range(0);
        const auto num_workers = num_blks * kWorkersPerBlock;
        const auto num_q_and_k_heads = params_.num_qo_heads + params_.num_kv_heads;
        const auto num_works = num_q_and_k_heads * params_.num_tokens;
        const auto start_worker_id = (item.get_group(0) * kBlockSize + item.get_local_id(0)) / kWorkThreads;
        
        const float* cos_cache_ptr = params_.cos_sin_cache_ptr;
        const float* sin_cache_ptr = cos_cache_ptr + (kRopeDim / 2);
        
        for (auto idx = start_worker_id; idx < num_works; idx += num_workers) {
            const int64_t token_id = idx / num_q_and_k_heads;
            const int64_t head_id = idx % num_q_and_k_heads;
            const auto pos = params_.positions[token_id];
            const auto load_q = head_id < params_.num_qo_heads;
            const int64_t kv_head_id = head_id - params_.num_qo_heads;
            
            DType* input = load_q 
                ? params_.q_ptr + token_id * params_.q_stride + head_id * params_.head_stride
                : params_.k_ptr + token_id * params_.k_stride + kv_head_id * params_.head_stride;
            
            const float* cos_ptr = cos_cache_ptr + pos * kRopeDim;
            const float* sin_ptr = sin_cache_ptr + pos * kRopeDim;
            
            if constexpr (kIsNeox) {
                // GPT-NeoX style: first half and second half
                DType* input_x = input;
                DType* input_y = input + (kRopeDim / 2);
                DType2* input_vec_x_ptr = reinterpret_cast<DType2*>(input_x);
                DType2* input_vec_y_ptr = reinterpret_cast<DType2*>(input_y);
                const aligned_vector<float, 2>* cos_ptr_vec = reinterpret_cast<const aligned_vector<float, 2>*>(cos_ptr);
                const aligned_vector<float, 2>* sin_ptr_vec = reinterpret_cast<const aligned_vector<float, 2>*>(sin_ptr);
                
                #pragma unroll 4
                for (int64_t i = 0; i < kVecSize; ++i) {
                    const int64_t vec_idx = static_cast<int64_t>(lane_id) * kVecSize + i;
                    DType2 input_vec_x = input_vec_x_ptr[vec_idx];
                    DType2 input_vec_y = input_vec_y_ptr[vec_idx];
                    aligned_vector<float, 2> cos_pair = cos_ptr_vec[vec_idx];
                    aligned_vector<float, 2> sin_pair = sin_ptr_vec[vec_idx];
                    
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        float x = static_cast<float>(input_vec_x[j]);
                        float y = static_cast<float>(input_vec_y[j]);
                        float cos_val = cos_pair[j];
                        float sin_val = sin_pair[j];
                        
                        float out_x = x * cos_val - y * sin_val;
                        float out_y = x * sin_val + y * cos_val;
                        
                        input_vec_x[j] = static_cast<DType>(out_x);
                        input_vec_y[j] = static_cast<DType>(out_y);
                    }
                    
                    input_vec_x_ptr[vec_idx] = input_vec_x;
                    input_vec_y_ptr[vec_idx] = input_vec_y;
                }
            } else {
                // Interleaved style (GPT-J): pairs (x, y) adjacent
                DType2* input_vec_ptr = reinterpret_cast<DType2*>(input);
                const float* cos_vec_ptr = cos_ptr;
                const float* sin_vec_ptr = sin_ptr;
                
                #pragma unroll 4
                for (int64_t i = 0; i < kVecSize; ++i) {
                    const int64_t vec_idx = static_cast<int64_t>(lane_id) * kVecSize + i;
                    DType2 input_vec = input_vec_ptr[vec_idx];
                    float cos_val = cos_vec_ptr[vec_idx];
                    float sin_val = sin_vec_ptr[vec_idx];
                    
                    packed2_t xy = cast_to_fp32x2(input_vec);
                    float out_x = xy.x * cos_val - xy.y * sin_val;
                    float out_y = xy.x * sin_val + xy.y * cos_val;
                    
                    input_vec_ptr[vec_idx] = cast_from_fp32x2<DType>({out_x, out_y});
                }
            }
        }
    }
    
private:
    FusedRopeParams<DType, IdType> params_;
};

// ============================================================================
// RoPE + KV Cache Store Kernel
// ============================================================================

template<bool kIsNeox, int64_t kRopeDim, typename DType, typename IdType, uint32_t kWorkThreads>
class FusedRopeStoreKernel {
public:
    using DType2 = aligned_vector<DType, 2>;
    
    static constexpr int64_t kVecSize = kRopeDim / (2 * kWorkThreads * (1 + kIsNeox));
    static constexpr int64_t kDimPerThread = kVecSize * 2 * (1 + kIsNeox);
    
    FusedRopeStoreKernel(const FusedRopeStoreParams<DType, IdType>& params) : params_(params) {}
    
    void operator()(::sycl::nd_item<1> item) const {
        constexpr uint32_t kBlockSize = 128;
        constexpr uint32_t kWorkersPerBlock = kBlockSize / kWorkThreads;
        const auto& base = params_.base_params;
        const auto num_blks = item.get_group_range(0);
        const auto num_workers = num_blks * kWorkersPerBlock;
        const auto num_q_and_k_heads = base.num_qo_heads + base.num_kv_heads;
        const auto num_works = num_q_and_k_heads * base.num_tokens;
        const auto num_extra_works = base.num_kv_heads * base.num_tokens;
        const auto start_worker_id = (item.get_group(0) * kBlockSize + item.get_local_id(0)) / kWorkThreads;
        const auto lane_id = item.get_local_id(0) % kWorkThreads;
        
        const float* cos_cache_ptr = base.cos_sin_cache_ptr;
        const float* sin_cache_ptr = cos_cache_ptr + (kRopeDim / 2);
        
        // Phase 1: RoPE + K cache store
        auto idx = start_worker_id;
        for (; idx < num_works; idx += num_workers) {
            const int64_t token_id = idx / num_q_and_k_heads;
            const int64_t head_id = idx % num_q_and_k_heads;
            const auto pos = base.positions[token_id];
            const auto loc = params_.out_loc[token_id];
            const auto load_q = head_id < base.num_qo_heads;
            const int64_t kv_head_id = load_q ? head_id : (head_id - base.num_qo_heads);
            
            DType* input = load_q 
                ? base.q_ptr + token_id * base.q_stride + head_id * base.head_stride
                : base.k_ptr + token_id * base.k_stride + kv_head_id * base.head_stride;
            
            const float* cos_ptr = cos_cache_ptr + pos * kRopeDim;
            const float* sin_ptr = sin_cache_ptr + pos * kRopeDim;
            
            if constexpr (kIsNeox) {
                DType* input_x = input;
                DType* input_y = input + (kRopeDim / 2);
                DType2* input_vec_x_ptr = reinterpret_cast<DType2*>(input_x);
                DType2* input_vec_y_ptr = reinterpret_cast<DType2*>(input_y);
                const aligned_vector<float, 2>* cos_ptr_vec = reinterpret_cast<const aligned_vector<float, 2>*>(cos_ptr);
                const aligned_vector<float, 2>* sin_ptr_vec = reinterpret_cast<const aligned_vector<float, 2>*>(sin_ptr);
                
                #pragma unroll 4
                for (int64_t i = 0; i < kVecSize; ++i) {
                    const int64_t vec_idx = static_cast<int64_t>(lane_id) * kVecSize + i;
                    DType2 input_vec_x = input_vec_x_ptr[vec_idx];
                    DType2 input_vec_y = input_vec_y_ptr[vec_idx];
                    aligned_vector<float, 2> cos_pair = cos_ptr_vec[vec_idx];
                    aligned_vector<float, 2> sin_pair = sin_ptr_vec[vec_idx];
                    
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        float x = static_cast<float>(input_vec_x[j]);
                        float y = static_cast<float>(input_vec_y[j]);
                        float cos_val = cos_pair[j];
                        float sin_val = sin_pair[j];
                        
                        float out_x = x * cos_val - y * sin_val;
                        float out_y = x * sin_val + y * cos_val;
                        
                        input_vec_x[j] = static_cast<DType>(out_x);
                        input_vec_y[j] = static_cast<DType>(out_y);
                    }
                    
                    input_vec_x_ptr[vec_idx] = input_vec_x;
                    input_vec_y_ptr[vec_idx] = input_vec_y;
                    
                    // Store to K cache if this is a K head
                    if (!load_q) {
                        DType* k_out = params_.k_cache + loc * params_.cache_stride + kv_head_id * base.head_stride;
                        DType* k_out_y = k_out + (kRopeDim / 2);
                        reinterpret_cast<DType2*>(k_out)[vec_idx] = input_vec_x;
                        reinterpret_cast<DType2*>(k_out_y)[vec_idx] = input_vec_y;
                    }
                }
            } else {
                DType2* input_vec_ptr = reinterpret_cast<DType2*>(input);
                const float* cos_vec_ptr = cos_ptr;
                const float* sin_vec_ptr = sin_ptr;
                
                #pragma unroll 4
                for (int64_t i = 0; i < kVecSize; ++i) {
                    const int64_t vec_idx = static_cast<int64_t>(lane_id) * kVecSize + i;
                    DType2 input_vec = input_vec_ptr[vec_idx];
                    float cos_val = cos_vec_ptr[vec_idx];
                    float sin_val = sin_vec_ptr[vec_idx];
                    
                    packed2_t xy = cast_to_fp32x2(input_vec);
                    float out_x = xy.x * cos_val - xy.y * sin_val;
                    float out_y = xy.x * sin_val + xy.y * cos_val;
                    
                    DType2 result = cast_from_fp32x2<DType>({out_x, out_y});
                    input_vec_ptr[vec_idx] = result;
                    
                    // Store to K cache if this is a K head
                    if (!load_q) {
                        DType* k_out = params_.k_cache + loc * params_.cache_stride + kv_head_id * base.head_stride;
                        reinterpret_cast<DType2*>(k_out)[vec_idx] = result;
                    }
                }
            }
        }
        
        // Synchronize within work-group
        ::sycl::group_barrier(item.get_group());
        
        // Phase 2: V cache store
        for (idx = start_worker_id; idx < num_extra_works; idx += num_workers) {
            const int64_t token_id = idx / base.num_kv_heads;
            const int64_t head_id = idx % base.num_kv_heads;
            const auto loc = params_.out_loc[token_id];
            
            const DType* v_input = params_.v_ptr + token_id * params_.v_stride + head_id * base.head_stride;
            DType* v_output = params_.v_cache + loc * params_.cache_stride + head_id * base.head_stride;
            
            // Copy V to cache (each thread handles elements at stride kWorkThreads)
            for (int64_t i = lane_id; i < base.head_stride; i += kWorkThreads) {
                v_output[i] = v_input[i];
            }
        }
    }
    
private:
    FusedRopeStoreParams<DType, IdType> params_;
};

// ============================================================================
// Launcher Functions
// ============================================================================

template<bool kIsNeox, int64_t kRopeDim, typename DType, typename IdType>
void fused_rope_launcher(
    ::sycl::queue& queue,
    DType* q_ptr,
    DType* k_ptr,
    const float* cos_sin_cache_ptr,
    const IdType* positions,
    int64_t q_stride,
    int64_t k_stride,
    int64_t head_stride,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t num_tokens
) {
    constexpr uint32_t kBlockSize = 128;
    
    // Calculate work threads based on rope_dim
    constexpr uint32_t kDimPerThread = 16 / sizeof(DType);
    constexpr uint32_t kWorkThreads = []() {
        uint32_t power = 1;
        while (power * kDimPerThread < kRopeDim) power *= 2;
        return power;
    }();
    
    constexpr uint32_t kWorkersPerBlock = kBlockSize / kWorkThreads;
    const uint32_t num_q_and_k_heads = num_qo_heads + num_kv_heads;
    const uint32_t num_works = num_q_and_k_heads * num_tokens;
    const uint32_t num_blocks = (num_works + kWorkersPerBlock - 1) / kWorkersPerBlock;
    
    // Prepare parameters. K ptr stays at allocated buffer start; indexing handles offset.
    FusedRopeParams<DType, IdType> params{
        q_ptr,
        k_ptr,
        cos_sin_cache_ptr,
        positions,
        q_stride,
        k_stride,
        head_stride,
        num_qo_heads,
        num_kv_heads,
        num_tokens
    };
    
    queue.submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>(
                ::sycl::range<1>(num_blocks * kBlockSize),
                ::sycl::range<1>(kBlockSize)
            ),
            FusedRopeKernel<kIsNeox, kRopeDim, DType, IdType, kWorkThreads>(params)
        );
    });
}

template<bool kIsNeox, int64_t kRopeDim, typename DType, typename IdType>
void fused_rope_store_launcher(
    ::sycl::queue& queue,
    DType* q_ptr,
    DType* k_ptr,
    DType* v_ptr,
    DType* k_cache,
    DType* v_cache,
    const float* cos_sin_cache_ptr,
    const IdType* positions,
    const IdType* out_loc,
    int64_t q_stride,
    int64_t k_stride,
    int64_t v_stride,
    int64_t head_stride,
    int64_t cache_stride,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t num_tokens
) {
    constexpr uint32_t kBlockSize = 128;
    
    // Calculate work threads
    constexpr uint32_t kDimPerThread = 16 / sizeof(DType);
    constexpr uint32_t kWorkThreads = []() {
        uint32_t power = 1;
        while (power * kDimPerThread < kRopeDim) power *= 2;
        return power;
    }();
    
    constexpr uint32_t kWorkersPerBlock = kBlockSize / kWorkThreads;
    const uint32_t num_q_and_k_heads = num_qo_heads + num_kv_heads;
    const uint32_t num_total_works = (num_q_and_k_heads + num_kv_heads) * num_tokens;
    const uint32_t num_blocks = (num_total_works + kWorkersPerBlock - 1) / kWorkersPerBlock;
    
    FusedRopeParams<DType, IdType> base_params{
        q_ptr,
        k_ptr,
        cos_sin_cache_ptr,
        positions,
        q_stride,
        k_stride,
        head_stride,
        num_qo_heads,
        num_kv_heads,
        num_tokens
    };
    
    FusedRopeStoreParams<DType, IdType> params{
        base_params,
        v_ptr,
        k_cache,
        v_cache,
        out_loc,
        v_stride,
        cache_stride
    };
    
    queue.submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>(
                ::sycl::range<1>(num_blocks * kBlockSize),
                ::sycl::range<1>(kBlockSize)
            ),
            FusedRopeStoreKernel<kIsNeox, kRopeDim, DType, IdType, kWorkThreads>(params)
        );
    });
}

// ============================================================================
// C API Export Macros
// ============================================================================

#define _DEFINE_ROPE_API(IS_NEOX, ROPE_DIM, DTYPE_SUFFIX, DTYPE, IDTYPE_SUFFIX, IDTYPE) \
extern "C" void fused_rope_##IS_NEOX##_##ROPE_DIM##_##DTYPE_SUFFIX##_##IDTYPE_SUFFIX( \
    void* queue_ptr, \
    void* q_ptr, \
    void* k_ptr, \
    const void* cos_sin_cache_ptr, \
    const void* positions, \
    int64_t q_stride, \
    int64_t k_stride, \
    int64_t head_stride, \
    uint32_t num_qo_heads, \
    uint32_t num_kv_heads, \
    uint32_t num_tokens \
) { \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr); \
    fused_rope_launcher<IS_NEOX, ROPE_DIM, DTYPE, IDTYPE>( \
        queue, \
        static_cast<DTYPE*>(q_ptr), \
        static_cast<DTYPE*>(k_ptr), \
        static_cast<const float*>(cos_sin_cache_ptr), \
        static_cast<const IDTYPE*>(positions), \
        q_stride, k_stride, head_stride, \
        num_qo_heads, num_kv_heads, num_tokens \
    ); \
} \
\
extern "C" void fused_rope_store_##IS_NEOX##_##ROPE_DIM##_##DTYPE_SUFFIX##_##IDTYPE_SUFFIX( \
    void* queue_ptr, \
    void* q_ptr, \
    void* k_ptr, \
    void* v_ptr, \
    void* k_cache, \
    void* v_cache, \
    const void* cos_sin_cache_ptr, \
    const void* positions, \
    const void* out_loc, \
    int64_t q_stride, \
    int64_t k_stride, \
    int64_t v_stride, \
    int64_t head_stride, \
    int64_t cache_stride, \
    uint32_t num_qo_heads, \
    uint32_t num_kv_heads, \
    uint32_t num_tokens \
) { \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr); \
    fused_rope_store_launcher<IS_NEOX, ROPE_DIM, DTYPE, IDTYPE>( \
        queue, \
        static_cast<DTYPE*>(q_ptr), \
        static_cast<DTYPE*>(k_ptr), \
        static_cast<DTYPE*>(v_ptr), \
        static_cast<DTYPE*>(k_cache), \
        static_cast<DTYPE*>(v_cache), \
        static_cast<const float*>(cos_sin_cache_ptr), \
        static_cast<const IDTYPE*>(positions), \
        static_cast<const IDTYPE*>(out_loc), \
        q_stride, k_stride, v_stride, head_stride, cache_stride, \
        num_qo_heads, num_kv_heads, num_tokens \
    ); \
}

#define DEFINE_ROPE_API(IS_NEOX, ROPE_DIM, DTYPE_SUFFIX, DTYPE, IDTYPE_SUFFIX, IDTYPE) \
    _DEFINE_ROPE_API(IS_NEOX, ROPE_DIM, DTYPE_SUFFIX, DTYPE, IDTYPE_SUFFIX, IDTYPE)

// Generate APIs for common rope dimensions and types
#define DEFINE_ROPE_ALL_COMBOS(ROPE_DIM) \
    DEFINE_ROPE_API(true, ROPE_DIM, fp16, ::sycl::half, i32, int32_t) \
    DEFINE_ROPE_API(true, ROPE_DIM, fp16, ::sycl::half, i64, int64_t) \
    DEFINE_ROPE_API(true, ROPE_DIM, bf16, ::sycl::ext::oneapi::bfloat16, i32, int32_t) \
    DEFINE_ROPE_API(true, ROPE_DIM, bf16, ::sycl::ext::oneapi::bfloat16, i64, int64_t) \
    DEFINE_ROPE_API(false, ROPE_DIM, fp16, ::sycl::half, i32, int32_t) \
    DEFINE_ROPE_API(false, ROPE_DIM, fp16, ::sycl::half, i64, int64_t) \
    DEFINE_ROPE_API(false, ROPE_DIM, bf16, ::sycl::ext::oneapi::bfloat16, i32, int32_t) \
    DEFINE_ROPE_API(false, ROPE_DIM, bf16, ::sycl::ext::oneapi::bfloat16, i64, int64_t)

// Common rope dimensions (including non-power-of-2 like 80, 96)
#ifdef SGL_ROPE_DIM
  // JIT mode: compile only requested dimension
  DEFINE_ROPE_ALL_COMBOS(SGL_ROPE_DIM)
#else
  // AOT mode: pre-compile common dimensions
  DEFINE_ROPE_ALL_COMBOS(64)
  DEFINE_ROPE_ALL_COMBOS(128)
  DEFINE_ROPE_ALL_COMBOS(256)
#endif

#undef DEFINE_ROPE_API
#undef _DEFINE_ROPE_API
#undef DEFINE_ROPE_ALL_COMBOS

} // namespace sycl_kernel
} // namespace sgl
