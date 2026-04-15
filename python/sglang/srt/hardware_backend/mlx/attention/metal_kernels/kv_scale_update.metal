#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

#define DIV_CONST 240.0f

template <typename T>
[[kernel]] void kv_scale_update(const device T *k [[buffer(0)]],
                                const device T *v [[buffer(1)]],
                                device atomic<float> *k_scale [[buffer(2)]],
                                device atomic<float> *v_scale [[buffer(3)]],
                                constant long &num_elements [[buffer(4)]],
                                uint gid [[thread_position_in_grid]],
                                uint grid_size [[threads_per_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tg_size [[threads_per_threadgroup]],
                                threadgroup float *shared_k [[threadgroup(0)]],
                                threadgroup float *shared_v
                                [[threadgroup(1)]]) {

  // Per-thread local maxima
  float local_max_k = 0.0f;
  float local_max_v = 0.0f;

  // Strided loop covering entire array
  for (long idx = gid; idx < num_elements; idx += grid_size) {
    float avk = abs(static_cast<float>(k[idx]));
    float avv = abs(static_cast<float>(v[idx]));
    local_max_k = max(local_max_k, avk);
    local_max_v = max(local_max_v, avv);
  }

  // Store per-thread maxima to shared memory
  shared_k[tid] = local_max_k;
  shared_v[tid] = local_max_v;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Parallel reduction in shared memory to find block maxima
  for (uint s = tg_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_k[tid] = max(shared_k[tid], shared_k[tid + s]);
      shared_v[tid] = max(shared_v[tid], shared_v[tid + s]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Thread 0 of block updates global scales atomically
  if (tid == 0) {
    float candidate_k_scale = shared_k[0] / DIV_CONST;
    float candidate_v_scale = shared_v[0] / DIV_CONST;

    // Atomic max update for k_scale
    if (candidate_k_scale > 0.0f) {
      float current = atomic_load_explicit(k_scale, memory_order_relaxed);
      while (candidate_k_scale > current) {
        if (atomic_compare_exchange_weak_explicit(
                k_scale, &current, candidate_k_scale, memory_order_relaxed,
                memory_order_relaxed)) {
          break;
        }
      }
    }

    // Atomic max update for v_scale
    if (candidate_v_scale > 0.0f) {
      float current = atomic_load_explicit(v_scale, memory_order_relaxed);
      while (candidate_v_scale > current) {
        if (atomic_compare_exchange_weak_explicit(
                v_scale, &current, candidate_v_scale, memory_order_relaxed,
                memory_order_relaxed)) {
          break;
        }
      }
    }
  }
}

#define instantiate_kv_scale_update(type)                                      \
  template [[host_name("kv_scale_update_" #type)]] [[kernel]] void             \
  kv_scale_update<type>(const device type *k [[buffer(0)]],                    \
                        const device type *v [[buffer(1)]],                    \
                        device atomic<float> *k_scale [[buffer(2)]],           \
                        device atomic<float> *v_scale [[buffer(3)]],           \
                        constant long &num_elements [[buffer(4)]],             \
                        uint gid [[thread_position_in_grid]],                  \
                        uint grid_size [[threads_per_grid]],                   \
                        uint tid [[thread_position_in_threadgroup]],           \
                        uint tg_size [[threads_per_threadgroup]],              \
                        threadgroup float *shared_k [[threadgroup(0)]],        \
                        threadgroup float *shared_v [[threadgroup(1)]]);

instantiate_kv_scale_update(float);
instantiate_kv_scale_update(bfloat16_t);
instantiate_kv_scale_update(half);