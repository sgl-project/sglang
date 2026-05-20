/**
 * QKNorm SYCL Kernel for SGLang
 *
 * Fused in-place QK normalization for Intel XPU.
 * Applies RMSNorm independently to each Q and K head:
 *   q[i] = q[i] / sqrt(mean(q[i]^2) + eps) * q_weight
 *   k[i] = k[i] / sqrt(mean(k[i]^2) + eps) * k_weight
 *
 * Layout:
 *   q: [num_tokens, num_qo_heads, head_dim]
 *   k: [num_tokens, num_kv_heads, head_dim]
 *   q_weight, k_weight: [head_dim]
 *
 * Optimizations:
 * - Vectorized memory access (vec4/vec8)
 * - Sub-group reductions
 * - One work-group per (token, head) pair
 */

#pragma once

#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

static constexpr int kQKNormSubGroupSize = 16;

// ============================================================================
// Vector Type Helpers
// ============================================================================

template<typename T, int N>
struct alignas(N * sizeof(T)) qknorm_aligned_vector {
    T data[N];

    qknorm_aligned_vector() = default;

    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
};

template<typename T, int64_t kSize>
constexpr int qknorm_get_vec_size() {
    return (kSize % 8 == 0) ? 8 :
           (kSize % 4 == 0) ? 4 :
           (kSize % 2 == 0) ? 2 : 1;
}

// ============================================================================
// QKNorm Kernel — one work-group per (token, head) pair
// ============================================================================

template <typename T, int64_t kHeadDim,
          int kVecSize = qknorm_get_vec_size<T, kHeadDim>()>
class QKNormKernel {
public:
    using Vec = qknorm_aligned_vector<T, kVecSize>;
    static constexpr int64_t kVectorizedSize = kHeadDim / kVecSize;

    QKNormKernel(
        T* q,
        T* k,
        const T* q_weight,
        const T* k_weight,
        int64_t q_stride,
        int64_t k_stride,
        uint32_t num_qo_heads,
        uint32_t num_kv_heads,
        uint32_t num_tokens,
        float eps
    ) : q_(q), k_(k), q_weight_(q_weight), k_weight_(k_weight),
        q_stride_(q_stride), k_stride_(k_stride),
        num_qo_heads_(num_qo_heads), num_kv_heads_(num_kv_heads),
        num_tokens_(num_tokens), eps_(eps) {}

    [[sycl::reqd_sub_group_size(kQKNormSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
        const size_t work_id = item.get_group(0);
        const uint32_t num_q_and_k = num_qo_heads_ + num_kv_heads_;
        const uint32_t total_works = num_q_and_k * num_tokens_;
        if (work_id >= total_works) return;

        const uint32_t token_id = work_id / num_q_and_k;
        const uint32_t head_id = work_id % num_q_and_k;
        const bool is_q = head_id < num_qo_heads_;

        T* head_ptr;
        const T* weight_ptr;
        if (is_q) {
            head_ptr = q_ + token_id * q_stride_ + head_id * kHeadDim;
            weight_ptr = q_weight_;
        } else {
            const uint32_t k_head_id = head_id - num_qo_heads_;
            head_ptr = k_ + token_id * k_stride_ + k_head_id * kHeadDim;
            weight_ptr = k_weight_;
        }

        const size_t tid = item.get_local_id(0);
        const size_t num_threads = item.get_local_range(0);

        const Vec* input_vec = reinterpret_cast<const Vec*>(head_ptr);
        const Vec* weight_vec = reinterpret_cast<const Vec*>(weight_ptr);
        Vec* output_vec = reinterpret_cast<Vec*>(head_ptr);

        // Step 1: Compute sum of squares (vectorized)
        float thread_sum_sq = 0.0f;
        #pragma unroll 4
        for (int64_t i = tid; i < kVectorizedSize; i += num_threads) {
            Vec v = input_vec[i];
            #pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
                float val = static_cast<float>(v[j]);
                thread_sum_sq += val * val;
            }
        }

        // Step 2: Full work-group reduction
        float total_sum_sq = ::sycl::reduce_over_group(item.get_group(), thread_sum_sq, ::sycl::plus<float>());
        float rms_scale = ::sycl::rsqrt(total_sum_sq / static_cast<float>(kHeadDim) + eps_);

        // Step 4: Apply normalization with weight (vectorized, in-place)
        #pragma unroll 4
        for (int64_t i = tid; i < kVectorizedSize; i += num_threads) {
            Vec in_v = input_vec[i];
            Vec w_v = weight_vec[i];
            Vec out_v;
            #pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
                float normalized = static_cast<float>(in_v[j]) * rms_scale;
                out_v[j] = static_cast<T>(normalized * static_cast<float>(w_v[j]));
            }
            output_vec[i] = out_v;
        }
    }

private:
    T* q_;
    T* k_;
    const T* q_weight_;
    const T* k_weight_;
    int64_t q_stride_;
    int64_t k_stride_;
    uint32_t num_qo_heads_;
    uint32_t num_kv_heads_;
    uint32_t num_tokens_;
    float eps_;
};

// ============================================================================
// Fallback Kernel (scalar, for non-vectorizable head_dims)
// ============================================================================

template <typename T, int64_t kHeadDim>
class QKNormKernelFallback {
public:
    QKNormKernelFallback(
        T* q,
        T* k,
        const T* q_weight,
        const T* k_weight,
        int64_t q_stride,
        int64_t k_stride,
        uint32_t num_qo_heads,
        uint32_t num_kv_heads,
        uint32_t num_tokens,
        float eps
    ) : q_(q), k_(k), q_weight_(q_weight), k_weight_(k_weight),
        q_stride_(q_stride), k_stride_(k_stride),
        num_qo_heads_(num_qo_heads), num_kv_heads_(num_kv_heads),
        num_tokens_(num_tokens), eps_(eps) {}

    [[sycl::reqd_sub_group_size(kQKNormSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
        const size_t work_id = item.get_group(0);
        const uint32_t num_q_and_k = num_qo_heads_ + num_kv_heads_;
        const uint32_t total_works = num_q_and_k * num_tokens_;
        if (work_id >= total_works) return;

        const uint32_t token_id = work_id / num_q_and_k;
        const uint32_t head_id = work_id % num_q_and_k;
        const bool is_q = head_id < num_qo_heads_;

        T* head_ptr;
        const T* weight_ptr;
        if (is_q) {
            head_ptr = q_ + token_id * q_stride_ + head_id * kHeadDim;
            weight_ptr = q_weight_;
        } else {
            const uint32_t k_head_id = head_id - num_qo_heads_;
            head_ptr = k_ + token_id * k_stride_ + k_head_id * kHeadDim;
            weight_ptr = k_weight_;
        }

        const size_t tid = item.get_local_id(0);
        const size_t num_threads = item.get_local_range(0);

        // Scalar sum of squares
        float thread_sum_sq = 0.0f;
        for (size_t i = tid; i < kHeadDim; i += num_threads) {
            float val = static_cast<float>(head_ptr[i]);
            thread_sum_sq += val * val;
        }

        // Full work-group reduction
        float total_sum_sq = ::sycl::reduce_over_group(item.get_group(), thread_sum_sq, ::sycl::plus<float>());
        float rms_scale = ::sycl::rsqrt(total_sum_sq / static_cast<float>(kHeadDim) + eps_);

        for (size_t i = tid; i < kHeadDim; i += num_threads) {
            float normalized = static_cast<float>(head_ptr[i]) * rms_scale;
            head_ptr[i] = static_cast<T>(normalized * static_cast<float>(weight_ptr[i]));
        }
    }

private:
    T* q_;
    T* k_;
    const T* q_weight_;
    const T* k_weight_;
    int64_t q_stride_;
    int64_t k_stride_;
    uint32_t num_qo_heads_;
    uint32_t num_kv_heads_;
    uint32_t num_tokens_;
    float eps_;
};

// ============================================================================
// Launcher
// ============================================================================

template <int64_t kHeadDim, typename DType>
void qknorm_launcher(
    ::sycl::queue& queue,
    void* q,
    void* k,
    const void* q_weight,
    const void* k_weight,
    int64_t q_stride,
    int64_t k_stride,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t num_tokens,
    float eps
) {
    constexpr size_t kThreadsPerBlock = 256;
    const size_t num_works = (num_qo_heads + num_kv_heads) * num_tokens;

    DType* q_ptr = static_cast<DType*>(q);
    DType* k_ptr = static_cast<DType*>(k);
    const DType* qw_ptr = static_cast<const DType*>(q_weight);
    const DType* kw_ptr = static_cast<const DType*>(k_weight);

    constexpr bool use_vectorized = (kHeadDim % 4 == 0) && (kHeadDim >= 64);

    queue.submit([&](::sycl::handler& cgh) {
        if constexpr (use_vectorized) {
            cgh.parallel_for(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(num_works * kThreadsPerBlock),
                    ::sycl::range<1>(kThreadsPerBlock)
                ),
                QKNormKernel<DType, kHeadDim>(
                    q_ptr, k_ptr, qw_ptr, kw_ptr,
                    q_stride, k_stride,
                    num_qo_heads, num_kv_heads, num_tokens,
                    eps
                )
            );
        } else {
            cgh.parallel_for(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(num_works * kThreadsPerBlock),
                    ::sycl::range<1>(kThreadsPerBlock)
                ),
                QKNormKernelFallback<DType, kHeadDim>(
                    q_ptr, k_ptr, qw_ptr, kw_ptr,
                    q_stride, k_stride,
                    num_qo_heads, num_kv_heads, num_tokens,
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
#define _DEFINE_QKNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HEAD_DIM)                   \
extern "C" void qknorm_forward_##DTYPE_SUFFIX##_##HEAD_DIM(                     \
    void* queue_ptr,                                                            \
    void* q,                                                                    \
    void* k,                                                                    \
    const void* q_weight,                                                       \
    const void* k_weight,                                                       \
    int64_t q_stride,                                                           \
    int64_t k_stride,                                                           \
    uint32_t num_qo_heads,                                                      \
    uint32_t num_kv_heads,                                                      \
    uint32_t num_tokens,                                                        \
    float eps                                                                   \
) {                                                                             \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                      \
    qknorm_launcher<HEAD_DIM, DTYPE>(                                           \
        queue, q, k, q_weight, k_weight,                                        \
        q_stride, k_stride,                                                     \
        num_qo_heads, num_kv_heads, num_tokens, eps                             \
    );                                                                          \
}
#define DEFINE_QKNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HEAD_DIM)                    \
    _DEFINE_QKNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HEAD_DIM)

#define DEFINE_QKNORM_ALL_DTYPES(HEAD_DIM)                                      \
    DEFINE_QKNORM_FORWARD(fp16, ::sycl::half, HEAD_DIM)                         \
    DEFINE_QKNORM_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, HEAD_DIM)

// When SGL_QKNORM_HEAD_DIM is defined, compile only the requested variant.
#ifdef SGL_QKNORM_HEAD_DIM
  #if defined(SGL_QKNORM_DTYPE_fp16)
    DEFINE_QKNORM_FORWARD(fp16, ::sycl::half, SGL_QKNORM_HEAD_DIM)
  #elif defined(SGL_QKNORM_DTYPE_bf16)
    DEFINE_QKNORM_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, SGL_QKNORM_HEAD_DIM)
  #else
    DEFINE_QKNORM_ALL_DTYPES(SGL_QKNORM_HEAD_DIM)
  #endif
#else
DEFINE_QKNORM_ALL_DTYPES(64)
DEFINE_QKNORM_ALL_DTYPES(128)
DEFINE_QKNORM_ALL_DTYPES(256)
DEFINE_QKNORM_ALL_DTYPES(512)
DEFINE_QKNORM_ALL_DTYPES(1024)
#endif

#undef DEFINE_QKNORM_FORWARD
#undef _DEFINE_QKNORM_FORWARD
#undef DEFINE_QKNORM_ALL_DTYPES

} // namespace sycl_kernel
} // namespace sgl
