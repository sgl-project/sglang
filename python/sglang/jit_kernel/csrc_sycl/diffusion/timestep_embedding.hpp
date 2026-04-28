/**
 * Timestep Embedding SYCL Kernel for SGLang
 *
 * Implements sinusoidal timestep embeddings for diffusion models on Intel XPU.
 * Based on the CUDA implementation and optimized for SYCL/oneAPI.
 *
 * Formula:
 *   For each timestep t and dimension i:
 *   freq = scale * t * exp(neg_log_max_period * i)
 *   embedding = [cos(freq), sin(freq)] or [sin(freq), cos(freq)]
 *
 * Layout:
 *   t: [batch_size] - input timesteps
 *   output: [batch_size, dim] - output embeddings (always float32)
 *
 * Optimizations:
 * - Vectorized memory access (vec4 for float32)
 * - One work-group per batch element
 * - Each work-item processes multiple dimension elements
 */

#pragma once

#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

// ============================================================================
// Vector Type Helpers
// ============================================================================

template<typename T, int N>
struct alignas(N * sizeof(T)) timestep_aligned_vector {
    T data[N];

    timestep_aligned_vector() = default;

    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
};

using float4 = timestep_aligned_vector<float, 4>;

// ============================================================================
// Timestep Embedding Kernel
// ============================================================================

template <typename TIn, bool kFlipSinToCos>
class TimestepEmbeddingKernel {
public:
    TimestepEmbeddingKernel(
        const TIn* t_ptr,
        float* output_ptr,
        int dim,
        float neg_log_max_period,
        float scale,
        int batch_size
    ) : t_ptr_(t_ptr),
        output_ptr_(output_ptr),
        dim_(dim),
        neg_log_max_period_(neg_log_max_period),
        scale_(scale),
        batch_size_(batch_size) {}

    void operator()(::sycl::nd_item<1> item) const {
        const size_t row_idx = item.get_group(0);
        if (row_idx >= static_cast<size_t>(batch_size_)) return;

        // Read timestep value and convert to float
        const float t_val = static_cast<float>(t_ptr_[row_idx]);
        float* output_batch_base_ptr = output_ptr_ + row_idx * dim_;

        const int half_dim = dim_ / 2;
        const size_t tid = item.get_local_id(0);
        const size_t num_threads = item.get_local_range(0);

        // Process in chunks of 4 (vectorized)
        for (size_t thread_offset = tid; thread_offset * 4 < static_cast<size_t>(half_dim); thread_offset += num_threads) {
            float4* top_half;
            float4* bottom_half;
            
            if constexpr (!kFlipSinToCos) {
                bottom_half = reinterpret_cast<float4*>(output_batch_base_ptr + thread_offset * 4);
                top_half = reinterpret_cast<float4*>(output_batch_base_ptr + half_dim + thread_offset * 4);
            } else {
                top_half = reinterpret_cast<float4*>(output_batch_base_ptr + thread_offset * 4);
                bottom_half = reinterpret_cast<float4*>(output_batch_base_ptr + half_dim + thread_offset * 4);
            }

            // Compute frequencies
            float4 vals;
            vals[0] = scale_ * t_val * ::sycl::exp(neg_log_max_period_ * static_cast<float>(thread_offset * 4 + 0));
            vals[1] = scale_ * t_val * ::sycl::exp(neg_log_max_period_ * static_cast<float>(thread_offset * 4 + 1));
            vals[2] = scale_ * t_val * ::sycl::exp(neg_log_max_period_ * static_cast<float>(thread_offset * 4 + 2));
            vals[3] = scale_ * t_val * ::sycl::exp(neg_log_max_period_ * static_cast<float>(thread_offset * 4 + 3));

            // Compute cos values
            float4 cos_vals;
            cos_vals[0] = ::sycl::cos(vals[0]);
            cos_vals[1] = ::sycl::cos(vals[1]);
            cos_vals[2] = ::sycl::cos(vals[2]);
            cos_vals[3] = ::sycl::cos(vals[3]);
            *top_half = cos_vals;

            // Compute sin values
            float4 sin_vals;
            sin_vals[0] = ::sycl::sin(vals[0]);
            sin_vals[1] = ::sycl::sin(vals[1]);
            sin_vals[2] = ::sycl::sin(vals[2]);
            sin_vals[3] = ::sycl::sin(vals[3]);
            *bottom_half = sin_vals;
        }
    }

private:
    const TIn* t_ptr_;
    float* output_ptr_;
    int dim_;
    float neg_log_max_period_;
    float scale_;
    int batch_size_;
};

// ============================================================================
// Launcher
// ============================================================================

template <typename TIn>
void timestep_embedding_launcher(
    ::sycl::queue& queue,
    const void* t,
    void* output,
    int dim,
    bool flip_sin_to_cos,
    float downscale_freq_shift,
    float scale,
    int max_period,
    int batch_size
) {
    const int half_dim = dim / 2;
    
    // Similar thread configuration to CUDA version
    constexpr int kMaxThreadsPerBlock = 1024;
    
    const int num_threads_per_row = std::min(kMaxThreadsPerBlock, half_dim / 4);
    
    const size_t threads_per_group = num_threads_per_row;
    const size_t num_groups = batch_size;
    
    const float neg_log_max_period =
        ::sycl::log(static_cast<float>(max_period)) * (-1.0f) / (static_cast<float>(half_dim) - downscale_freq_shift);
    
    const TIn* t_ptr = static_cast<const TIn*>(t);
    float* output_ptr = static_cast<float*>(output);
    
    queue.submit([&](::sycl::handler& cgh) {
        if (flip_sin_to_cos) {
            cgh.parallel_for(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(num_groups * threads_per_group),
                    ::sycl::range<1>(threads_per_group)
                ),
                TimestepEmbeddingKernel<TIn, true>(
                    t_ptr, output_ptr, dim,
                    neg_log_max_period, scale, batch_size
                )
            );
        } else {
            cgh.parallel_for(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(num_groups * threads_per_group),
                    ::sycl::range<1>(threads_per_group)
                ),
                TimestepEmbeddingKernel<TIn, false>(
                    t_ptr, output_ptr, dim,
                    neg_log_max_period, scale, batch_size
                )
            );
        }
    });
}

// ============================================================================
// C API for Python Binding
// ============================================================================

#define DEFINE_TIMESTEP_EMBEDDING_FORWARD(DTYPE_SUFFIX, DTYPE)                  \
extern "C" void timestep_embedding_forward_##DTYPE_SUFFIX(                      \
    void* queue_ptr,                                                            \
    const void* t,                                                              \
    void* output,                                                               \
    int dim,                                                                    \
    bool flip_sin_to_cos,                                                       \
    float downscale_freq_shift,                                                 \
    float scale,                                                                \
    int max_period,                                                             \
    int batch_size                                                              \
) {                                                                             \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                      \
    timestep_embedding_launcher<DTYPE>(                                         \
        queue, t, output, dim, flip_sin_to_cos,                                \
        downscale_freq_shift, scale, max_period, batch_size                    \
    );                                                                          \
}

// Instantiate for supported input types
DEFINE_TIMESTEP_EMBEDDING_FORWARD(fp32, float)
DEFINE_TIMESTEP_EMBEDDING_FORWARD(fp16, ::sycl::half)
DEFINE_TIMESTEP_EMBEDDING_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)

#undef DEFINE_TIMESTEP_EMBEDDING_FORWARD

} // namespace sycl_kernel
} // namespace sgl
