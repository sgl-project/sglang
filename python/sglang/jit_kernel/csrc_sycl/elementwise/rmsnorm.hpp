/**
 * RMSNorm SYCL Kernel for SGLang
 * 
 * Optimized SYCL implementation with vectorization and sub-group optimizations
 * Based on strategies from sgl-kernel-xpu while remaining JIT-compatible
 * 
 * Key optimizations:
 * - Vectorized memory access (vec4/vec8) for better bandwidth utilization
 * - Sub-group reductions for faster parallel reduction
 * - Automatic kernel selection (vectorized vs simple) based on size
 * - Loop unrolling for better instruction pipelining
 * 
 * Formula: y = x / sqrt(mean(x^2) + eps) * weight
 */

#pragma once

#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

// Sub-group size matching sgl-kernel-xpu's NUM_REDUCE_STAGES
static constexpr int kSubGroupSize = 16;

// ============================================================================
// Vector Type Helpers (for vectorized memory access)
// ============================================================================

template<typename T, int N>
struct alignas(N * sizeof(T)) aligned_vector {
    T data[N];
    
    aligned_vector() = default;
    
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
};

// Determine optimal vector size based on hidden size and type
template<typename T, int64_t kHiddenSize>
constexpr int get_vec_size() {
    constexpr int max_vec = 8;  // Maximum vectorization
    constexpr int base_vec = (kHiddenSize % 8 == 0) ? 8 : 
                             (kHiddenSize % 4 == 0) ? 4 : 
                             (kHiddenSize % 2 == 0) ? 2 : 1;
    return (base_vec <= max_vec) ? base_vec : max_vec;
}

// ============================================================================
// RMSNorm Kernel with Vectorization and Sub-group Optimizations
// ============================================================================

template <typename T, int64_t kHiddenSize, int kVecSize = get_vec_size<T, kHiddenSize>()>
class RMSNormKernel {
public:
    using Vec = aligned_vector<T, kVecSize>;
    static constexpr int64_t kVectorizedSize = kHiddenSize / kVecSize;
    
    RMSNormKernel(
        const T* input,
        const T* weight,
        T* output,
        int64_t input_stride,
        int64_t output_stride,
        uint32_t num_tokens,
        float eps
    ) : input_(input),
        weight_(weight),
        output_(output),
        input_stride_(input_stride),
        output_stride_(output_stride),
        num_tokens_(num_tokens),
        eps_(eps) {}

    [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
        const size_t token_idx = item.get_group(0);
        if (token_idx >= num_tokens_) return;
        
        const size_t tid = item.get_local_id(0);
        const size_t num_threads = item.get_local_range(0);
        
        // Pointers for this token
        const T* input_ptr = input_ + token_idx * input_stride_;
        T* output_ptr = output_ + token_idx * output_stride_;
        const Vec* input_vec_ptr = reinterpret_cast<const Vec*>(input_ptr);
        const Vec* weight_vec_ptr = reinterpret_cast<const Vec*>(weight_);
        Vec* output_vec_ptr = reinterpret_cast<Vec*>(output_ptr);
        
        // Step 1: Vectorized sum of squares computation
        float thread_sum_sq = 0.0f;
        
        // Process vectorized elements
        #pragma unroll 4
        for (int64_t i = tid; i < kVectorizedSize; i += num_threads) {
            Vec in_vec = input_vec_ptr[i];
            #pragma unroll
            for (int v = 0; v < kVecSize; ++v) {
                float val = static_cast<float>(in_vec[v]);
                thread_sum_sq += val * val;
            }
        }
        
        // Step 2: Full work-group reduction (matches AOT reduce_over_group(group))
        float total_sum_sq = ::sycl::reduce_over_group(item.get_group(), thread_sum_sq, ::sycl::plus<float>());
        
        // Compute RMS scale
        float rms_scale = ::sycl::rsqrt(total_sum_sq / static_cast<float>(kHiddenSize) + eps_);
        
        // Step 4: Vectorized normalization and weight application
        #pragma unroll 4
        for (int64_t i = tid; i < kVectorizedSize; i += num_threads) {
            Vec in_vec = input_vec_ptr[i];
            Vec weight_vec = weight_vec_ptr[i];
            Vec out_vec;
            
            #pragma unroll
            for (int v = 0; v < kVecSize; ++v) {
                float normalized = static_cast<float>(in_vec[v]) * rms_scale;
                float weighted = normalized * static_cast<float>(weight_vec[v]);
                out_vec[v] = static_cast<T>(weighted);
            }
            
            output_vec_ptr[i] = out_vec;
        }
    }

private:
    const T* input_;
    const T* weight_;
    T* output_;
    int64_t input_stride_;
    int64_t output_stride_;
    uint32_t num_tokens_;
    float eps_;
};

// ============================================================================
// Fallback Kernel for Non-Vectorizable Sizes
// ============================================================================

template <typename T, int64_t kHiddenSize>
class RMSNormKernelFallback {
public:
    RMSNormKernelFallback(
        const T* input,
        const T* weight,
        T* output,
        int64_t input_stride,
        int64_t output_stride,
        uint32_t num_tokens,
        float eps
    ) : input_(input),
        weight_(weight),
        output_(output),
        input_stride_(input_stride),
        output_stride_(output_stride),
        num_tokens_(num_tokens),
        eps_(eps) {}

    [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
        const size_t token_idx = item.get_group(0);
        if (token_idx >= num_tokens_) return;
        
        const size_t tid = item.get_local_id(0);
        const size_t num_threads = item.get_local_range(0);
        
        const T* input_ptr = input_ + token_idx * input_stride_;
        T* output_ptr = output_ + token_idx * output_stride_;
        
        // Compute sum of squares
        float thread_sum_sq = 0.0f;
        for (size_t i = tid; i < kHiddenSize; i += num_threads) {
            float val = static_cast<float>(input_ptr[i]);
            thread_sum_sq += val * val;
        }
        
        // Full work-group reduction
        float total_sum_sq = ::sycl::reduce_over_group(item.get_group(), thread_sum_sq, ::sycl::plus<float>());
        float rms_scale = ::sycl::rsqrt(total_sum_sq / static_cast<float>(kHiddenSize) + eps_);
        
        // Apply normalization
        for (size_t i = tid; i < kHiddenSize; i += num_threads) {
            float normalized = static_cast<float>(input_ptr[i]) * rms_scale;
            float weighted = normalized * static_cast<float>(weight_[i]);
            output_ptr[i] = static_cast<T>(weighted);
        }
    }

private:
    const T* input_;
    const T* weight_;
    T* output_;
    int64_t input_stride_;
    int64_t output_stride_;
    uint32_t num_tokens_;
    float eps_;
};

// ============================================================================
// Launcher - Selects vectorized kernel or fallback based on size
// ============================================================================

template <int64_t kHiddenSize, typename DType>
void rmsnorm_launcher(
    ::sycl::queue& queue,
    const void* input,
    const void* weight,
    void* output,
    int64_t num_tokens,
    int64_t input_stride,
    int64_t output_stride,
    float eps
) {
    constexpr size_t kThreadsPerBlock = 256;
    const size_t num_blocks = num_tokens;
    
    const DType* input_ptr = static_cast<const DType*>(input);
    const DType* weight_ptr = static_cast<const DType*>(weight);
    DType* output_ptr = static_cast<DType*>(output);
    
    // Use vectorized kernel for aligned sizes, fallback otherwise
    constexpr bool use_vectorized = (kHiddenSize % 4 == 0) && (kHiddenSize >= 512);
    
    queue.submit([&](::sycl::handler& cgh) {
        if constexpr (use_vectorized) {
            cgh.parallel_for(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(num_blocks * kThreadsPerBlock),
                    ::sycl::range<1>(kThreadsPerBlock)
                ),
                RMSNormKernel<DType, kHiddenSize>(
                    input_ptr, weight_ptr, output_ptr,
                    input_stride, output_stride,
                    static_cast<uint32_t>(num_tokens),
                    eps
                )
            );
        } else {
            cgh.parallel_for(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(num_blocks * kThreadsPerBlock),
                    ::sycl::range<1>(kThreadsPerBlock)
                ),
                RMSNormKernelFallback<DType, kHiddenSize>(
                    input_ptr, weight_ptr, output_ptr,
                    input_stride, output_stride,
                    static_cast<uint32_t>(num_tokens),
                    eps
                )
            );
        }
    });
}

// ============================================================================
// C API for Python Binding
// ============================================================================

// Two-level macro to ensure macro arguments are expanded before token pasting.
#define _DEFINE_RMSNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HIDDEN_SIZE)               \
extern "C" void rmsnorm_forward_##DTYPE_SUFFIX##_##HIDDEN_SIZE(                 \
    void* queue_ptr,                                                            \
    const void* input,                                                          \
    const void* weight,                                                         \
    void* output,                                                               \
    int64_t num_tokens,                                                         \
    int64_t input_stride,                                                       \
    int64_t output_stride,                                                      \
    float eps                                                                   \
) {                                                                             \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                      \
    rmsnorm_launcher<HIDDEN_SIZE, DTYPE>(                                       \
        queue, input, weight, output, num_tokens,                               \
        input_stride, output_stride, eps                                        \
    );                                                                          \
}
#define DEFINE_RMSNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HIDDEN_SIZE)                \
    _DEFINE_RMSNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HIDDEN_SIZE)

// Generate C API for all supported hidden sizes and dtypes
#define DEFINE_RMSNORM_ALL_DTYPES(HIDDEN_SIZE)                                  \
    DEFINE_RMSNORM_FORWARD(fp32, float, HIDDEN_SIZE)                            \
    DEFINE_RMSNORM_FORWARD(fp16, ::sycl::half, HIDDEN_SIZE)                     \
    DEFINE_RMSNORM_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, HIDDEN_SIZE)

// When SGL_RMSNORM_HIDDEN_SIZE is defined, compile only the requested variant.
// Otherwise compile all variants (for testing or pre-compilation).
#ifdef SGL_RMSNORM_HIDDEN_SIZE
  #if defined(SGL_RMSNORM_DTYPE_fp32)
    DEFINE_RMSNORM_FORWARD(fp32, float, SGL_RMSNORM_HIDDEN_SIZE)
  #elif defined(SGL_RMSNORM_DTYPE_fp16)
    DEFINE_RMSNORM_FORWARD(fp16, ::sycl::half, SGL_RMSNORM_HIDDEN_SIZE)
  #elif defined(SGL_RMSNORM_DTYPE_bf16)
    DEFINE_RMSNORM_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, SGL_RMSNORM_HIDDEN_SIZE)
  #else
    DEFINE_RMSNORM_ALL_DTYPES(SGL_RMSNORM_HIDDEN_SIZE)
  #endif
#else
DEFINE_RMSNORM_ALL_DTYPES(64)
DEFINE_RMSNORM_ALL_DTYPES(128)
DEFINE_RMSNORM_ALL_DTYPES(256)
DEFINE_RMSNORM_ALL_DTYPES(512)
DEFINE_RMSNORM_ALL_DTYPES(1024)
DEFINE_RMSNORM_ALL_DTYPES(1536)
DEFINE_RMSNORM_ALL_DTYPES(2048)
DEFINE_RMSNORM_ALL_DTYPES(2304)
DEFINE_RMSNORM_ALL_DTYPES(2560)
DEFINE_RMSNORM_ALL_DTYPES(3072)
DEFINE_RMSNORM_ALL_DTYPES(4096)
DEFINE_RMSNORM_ALL_DTYPES(5120)
DEFINE_RMSNORM_ALL_DTYPES(6144)
DEFINE_RMSNORM_ALL_DTYPES(7168)
DEFINE_RMSNORM_ALL_DTYPES(8192)
DEFINE_RMSNORM_ALL_DTYPES(12288)
DEFINE_RMSNORM_ALL_DTYPES(16384)
#endif

#undef DEFINE_RMSNORM_FORWARD
#undef _DEFINE_RMSNORM_FORWARD
#undef DEFINE_RMSNORM_ALL_DTYPES

} // namespace sycl_kernel
} // namespace sgl
