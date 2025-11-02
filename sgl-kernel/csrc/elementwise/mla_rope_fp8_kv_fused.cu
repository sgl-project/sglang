/* Copyright 2024 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// MLA RoPE + FP8 Quantization + KV Cache Write Fusion Kernel
// Fuses RoPE application, FP8 quantization, and direct KV cache write

#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <ATen/ATen.h>
#include <torch/types.h>
#endif

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_fp8.h>
#endif

// TODO: Use pytorch_extension_utils.h when it's available in sgl-kernel/include
#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be " #d "D")
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), #a " must equal " #b)
#define CHECK_LAST_DIM_CONTIGUOUS(x) TORCH_CHECK(x.stride(x.dim() - 1) == 1, #x " last dim must be contiguous")

namespace {

template <typename T>
struct Vec2Traits;

template <>
struct Vec2Traits<__half> {
  using v2 = __half2;
  __device__ static inline float2 to_float2(v2 h2) {
    return __half22float2(h2);
  }
  __device__ static inline float to_float(const __half& h) {
    return __half2float(h);
  }
};

template <>
struct Vec2Traits<nv_bfloat16> {
  using v2 = nv_bfloat162;
  __device__ static inline float2 to_float2(v2 h2) {
    return __bfloat1622float2(h2);
  }
  __device__ static inline float to_float(const nv_bfloat16& h) {
    return __bfloat162float(h);
  }
};

__device__ inline uint8_t float_to_e4m3fn_byte(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __nv_fp8_storage_t byte = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  return static_cast<uint8_t>(byte);
#else
  x = fmaxf(-448.0f, fminf(448.0f, x));
  union {
    float f;
    uint32_t u;
  } conv;
  conv.f = x;
  uint32_t sign = (conv.u >> 31) & 0x1;
  if (x == 0.0f) return 0;
  int exp = ((conv.u >> 23) & 0xFF) - 127;
  exp = max(-6, min(8, exp));
  uint32_t mant = (conv.u >> 20) & 0x7;
  uint8_t result = (sign << 7) | ((exp + 7) << 3) | mant;
  return result;
#endif
}

__device__ inline uint32_t pack4(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
  return (uint32_t)a0 | ((uint32_t)a1 << 8) | ((uint32_t)a2 << 16) | ((uint32_t)a3 << 24);
}

__device__ inline void rope_rotate(float& xr, float& xi, float c, float s) {
  float xr_new = xr * c - xi * s;
  float xi_new = xr * s + xi * c;
  xr = xr_new;
  xi = xi_new;
}

template <int WARPS_PER_CTA, typename T>
__global__ void FusedRopeQuantizeKernelVec(
    const T* __restrict__ q_nope,
    const T* __restrict__ q_rope,
    int64_t qn_stride_tok,
    int64_t qn_stride_head,
    int64_t qr_stride_tok,
    int64_t qr_stride_head,
    const T* __restrict__ k_nope,
    const T* __restrict__ k_rope,
    int64_t kn_stride_tok,
    int64_t kr_stride_tok,
    const float* __restrict__ cos_sin,
    const int64_t* __restrict__ pos_ids,
    int nnz,
    int num_heads,
    int Dn,
    int Dr,
    bool is_neox,
    uint8_t* __restrict__ q_out_fp8,
    int64_t qout_stride_tok_bytes,
    int64_t qout_stride_head_bytes,
    uint8_t* __restrict__ k_nope_out_fp8,
    uint8_t* __restrict__ k_rope_out_fp8,
    uint8_t* __restrict__ kv_buffer_bytes,
    int64_t kv_stride_row_bytes,
    const int64_t* __restrict__ kv_cache_loc) {
  constexpr int WARP_SIZE = 32;
  int warp_in_block = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x & (WARP_SIZE - 1);

  int global_row = blockIdx.x * WARPS_PER_CTA + warp_in_block;
  if (global_row >= nnz * num_heads) return;

  int token_id = global_row / num_heads;
  int head_id = global_row % num_heads;

  const T* qn = q_nope + size_t(token_id) * qn_stride_tok + size_t(head_id) * qn_stride_head;
  const T* qr = q_rope + size_t(token_id) * qr_stride_tok + size_t(head_id) * qr_stride_head;
  const T* kn = k_nope + size_t(token_id) * kn_stride_tok;
  const T* kr = k_rope + size_t(token_id) * kr_stride_tok;

  uint8_t* qdst = q_out_fp8 + size_t(token_id) * qout_stride_tok_bytes + size_t(head_id) * qout_stride_head_bytes;
  uint8_t* kndst = k_nope_out_fp8 ? (k_nope_out_fp8 + size_t(token_id) * Dn) : nullptr;
  uint8_t* krdst = k_rope_out_fp8 ? (k_rope_out_fp8 + size_t(token_id) * Dr) : nullptr;

  int pos = static_cast<int>(pos_ids[token_id]);

  uint8_t* kvdst = nullptr;
  if (kv_buffer_bytes && kv_cache_loc) {
    int64_t flat_row = kv_cache_loc[token_id];
    kvdst = kv_buffer_bytes + flat_row * kv_stride_row_bytes;
  }

  const float* cos_ptr = cos_sin + size_t(pos) * Dr;
  const float* sin_ptr = cos_ptr + (Dr / 2);

  using V2 = typename Vec2Traits<T>::v2;

  // Process Q_nope: vectorized quantize + write
  for (int c = lane * 4; c < Dn; c += WARP_SIZE * 4) {
    V2 h0 = *reinterpret_cast<const V2*>(qn + c + 0);
    V2 h1 = *reinterpret_cast<const V2*>(qn + c + 2);
    float2 f0 = Vec2Traits<T>::to_float2(h0);
    float2 f1 = Vec2Traits<T>::to_float2(h1);

    uint32_t packed = pack4(
        float_to_e4m3fn_byte(f0.x), float_to_e4m3fn_byte(f0.y), float_to_e4m3fn_byte(f1.x), float_to_e4m3fn_byte(f1.y));
    *reinterpret_cast<uint32_t*>(qdst + c) = packed;
  }

  for (int c = lane * 4; c < Dr; c += WARP_SIZE * 4) {
    V2 h0 = *reinterpret_cast<const V2*>(qr + c + 0);
    V2 h1 = *reinterpret_cast<const V2*>(qr + c + 2);
    float2 f0 = Vec2Traits<T>::to_float2(h0);
    float2 f1 = Vec2Traits<T>::to_float2(h1);

    int base0 = (c + 0) >> 1;
    int base1 = (c + 2) >> 1;
    float c0 = cos_ptr[base0], s0 = sin_ptr[base0];
    float c1 = cos_ptr[base1], s1 = sin_ptr[base1];

    rope_rotate(f0.x, f0.y, c0, s0);
    rope_rotate(f1.x, f1.y, c1, s1);

    uint32_t packed = pack4(
        float_to_e4m3fn_byte(f0.x), float_to_e4m3fn_byte(f0.y), float_to_e4m3fn_byte(f1.x), float_to_e4m3fn_byte(f1.y));
    *reinterpret_cast<uint32_t*>(qdst + Dn + c) = packed;
  }

  if (head_id == 0) {
    for (int c = lane * 4; c < Dn; c += WARP_SIZE * 4) {
      V2 h0 = *reinterpret_cast<const V2*>(kn + c + 0);
      V2 h1 = *reinterpret_cast<const V2*>(kn + c + 2);
      float2 f0 = Vec2Traits<T>::to_float2(h0);
      float2 f1 = Vec2Traits<T>::to_float2(h1);

      uint32_t packed = pack4(
          float_to_e4m3fn_byte(f0.x),
          float_to_e4m3fn_byte(f0.y),
          float_to_e4m3fn_byte(f1.x),
          float_to_e4m3fn_byte(f1.y));

      if (kndst) *reinterpret_cast<uint32_t*>(kndst + c) = packed;
      if (kvdst) *reinterpret_cast<uint32_t*>(kvdst + c) = packed;
    }

    for (int c = lane * 4; c < Dr; c += WARP_SIZE * 4) {
      V2 h0 = *reinterpret_cast<const V2*>(kr + c + 0);
      V2 h1 = *reinterpret_cast<const V2*>(kr + c + 2);
      float2 f0 = Vec2Traits<T>::to_float2(h0);
      float2 f1 = Vec2Traits<T>::to_float2(h1);

      int base0 = (c + 0) >> 1;
      int base1 = (c + 2) >> 1;
      float c0 = cos_ptr[base0], s0 = sin_ptr[base0];
      float c1 = cos_ptr[base1], s1 = sin_ptr[base1];

      rope_rotate(f0.x, f0.y, c0, s0);
      rope_rotate(f1.x, f1.y, c1, s1);

      uint32_t packed = pack4(
          float_to_e4m3fn_byte(f0.x),
          float_to_e4m3fn_byte(f0.y),
          float_to_e4m3fn_byte(f1.x),
          float_to_e4m3fn_byte(f1.y));

      if (krdst) *reinterpret_cast<uint32_t*>(krdst + c) = packed;
      if (kvdst) *reinterpret_cast<uint32_t*>(kvdst + Dn + c) = packed;
    }
  }
}

// ============================================================================
// Scalar fallback kernel: for dimensions not divisible by 4
// Template supports both FP16 (__half) and BF16 (nv_bfloat16)
// ============================================================================
template <int BLOCK_THREADS, typename T>
__global__ void FusedRopeQuantizeKernelScalar(
    const T* __restrict__ q_nope,
    const T* __restrict__ q_rope,
    int64_t qn_stride_tok,
    int64_t qn_stride_head,
    int64_t qr_stride_tok,
    int64_t qr_stride_head,
    const T* __restrict__ k_nope,
    const T* __restrict__ k_rope,
    int64_t kn_stride_tok,  // NEW: K_nope stride(0) in elements
    int64_t kr_stride_tok,  // NEW: K_rope stride(0) in elements
    const float* __restrict__ cos_sin,
    const int64_t* __restrict__ pos_ids,
    int nnz,
    int num_heads,
    int Dn,
    int Dr,
    bool is_neox,
    uint8_t* __restrict__ q_out_fp8,
    int64_t qout_stride_tok_bytes,
    int64_t qout_stride_head_bytes,
    uint8_t* __restrict__ k_nope_out_fp8,
    uint8_t* __restrict__ k_rope_out_fp8,
    uint8_t* __restrict__ kv_buffer_bytes,
    int64_t kv_stride_row_bytes,  // 2D: row stride in bytes
    const int64_t* __restrict__ kv_cache_loc) {
  for (int global_row = blockIdx.x * BLOCK_THREADS + threadIdx.x; global_row < nnz * num_heads;
       global_row += gridDim.x * BLOCK_THREADS) {
    int token_id = global_row / num_heads;
    int head_id = global_row % num_heads;

    int pos = static_cast<int>(pos_ids[token_id]);
    const float* cos_ptr = cos_sin + size_t(pos) * Dr;
    const float* sin_ptr = cos_ptr + (Dr / 2);

    {
      const T* qn = q_nope + size_t(token_id) * qn_stride_tok + size_t(head_id) * qn_stride_head;
      const T* qr = q_rope + size_t(token_id) * qr_stride_tok + size_t(head_id) * qr_stride_head;
      uint8_t* qdst = q_out_fp8 + size_t(token_id) * qout_stride_tok_bytes + size_t(head_id) * qout_stride_head_bytes;

      for (int i = 0; i < Dn; ++i) {
        qdst[i] = float_to_e4m3fn_byte(Vec2Traits<T>::to_float(qn[i]));
      }

      if (!is_neox) {
        for (int i = 0; i < Dr; i += 2) {
          int base = i >> 1;
          float xr = Vec2Traits<T>::to_float(qr[i + 0]);
          float xi = (i + 1 < Dr) ? Vec2Traits<T>::to_float(qr[i + 1]) : 0.0f;
          rope_rotate(xr, xi, cos_ptr[base], sin_ptr[base]);
          qdst[Dn + i] = float_to_e4m3fn_byte(xr);
          if (i + 1 < Dr) qdst[Dn + i + 1] = float_to_e4m3fn_byte(xi);
        }
      } else {
        int half = Dr / 2;
        for (int i = 0; i < half; ++i) {
          float xr = Vec2Traits<T>::to_float(qr[i]);
          float xi = Vec2Traits<T>::to_float(qr[i + half]);
          rope_rotate(xr, xi, cos_ptr[i], sin_ptr[i]);
          qdst[Dn + i] = float_to_e4m3fn_byte(xr);
          qdst[Dn + i + half] = float_to_e4m3fn_byte(xi);
        }
      }
    }

    const T* kn = k_nope + size_t(token_id) * kn_stride_tok;
    const T* kr = k_rope + size_t(token_id) * kr_stride_tok;

    if (head_id == 0) {
      if (k_nope_out_fp8) {
        uint8_t* knd = k_nope_out_fp8 + size_t(token_id) * Dn;
        for (int i = 0; i < Dn; ++i) {
          knd[i] = float_to_e4m3fn_byte(Vec2Traits<T>::to_float(kn[i]));
        }
      }
      if (k_rope_out_fp8) {
        uint8_t* krd = k_rope_out_fp8 + size_t(token_id) * Dr;
        if (!is_neox) {
          for (int i = 0; i < Dr; i += 2) {
            int base = i >> 1;
            float xr = Vec2Traits<T>::to_float(kr[i]);
            float xi = (i + 1 < Dr) ? Vec2Traits<T>::to_float(kr[i + 1]) : 0.0f;
            rope_rotate(xr, xi, cos_ptr[base], sin_ptr[base]);
            krd[i] = float_to_e4m3fn_byte(xr);
            if (i + 1 < Dr) krd[i + 1] = float_to_e4m3fn_byte(xi);
          }
        } else {
          int half = Dr / 2;
          for (int i = 0; i < half; ++i) {
            float xr = Vec2Traits<T>::to_float(kr[i]);
            float xi = Vec2Traits<T>::to_float(kr[i + half]);
            rope_rotate(xr, xi, cos_ptr[i], sin_ptr[i]);
            krd[i] = float_to_e4m3fn_byte(xr);
            krd[i + half] = float_to_e4m3fn_byte(xi);
          }
        }
      }

      if (kv_buffer_bytes && kv_cache_loc) {
        int64_t flat_row = kv_cache_loc[token_id];
        uint8_t* dst = kv_buffer_bytes + flat_row * kv_stride_row_bytes;
        for (int i = 0; i < Dn; ++i) {
          dst[i] = float_to_e4m3fn_byte(Vec2Traits<T>::to_float(kn[i]));
        }
        if (!is_neox) {
          for (int i = 0; i < Dr; i += 2) {
            int base = i >> 1;
            float xr = Vec2Traits<T>::to_float(kr[i]);
            float xi = (i + 1 < Dr) ? Vec2Traits<T>::to_float(kr[i + 1]) : 0.0f;
            rope_rotate(xr, xi, cos_ptr[base], sin_ptr[base]);
            dst[Dn + i] = float_to_e4m3fn_byte(xr);
            if (i + 1 < Dr) dst[Dn + i + 1] = float_to_e4m3fn_byte(xi);
          }
        } else {
          int half = Dr / 2;
          for (int i = 0; i < half; ++i) {
            float xr = Vec2Traits<T>::to_float(kr[i]);
            float xi = Vec2Traits<T>::to_float(kr[i + half]);
            rope_rotate(xr, xi, cos_ptr[i], sin_ptr[i]);
            dst[Dn + i] = float_to_e4m3fn_byte(xr);
            dst[Dn + i + half] = float_to_e4m3fn_byte(xi);
          }
        }
      }
    }
  }
}

}  // namespace

void mla_rope_quantize_fp8_fused(
    at::Tensor q_nope,
    at::Tensor q_rope,
    at::Tensor k_nope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool is_neox,
    at::Tensor q_out,
    c10::optional<at::Tensor> k_nope_out,
    c10::optional<at::Tensor> k_rope_out,
    c10::optional<at::Tensor> kv_buffer,
    c10::optional<at::Tensor> kv_cache_loc) {
  CHECK_INPUT(q_nope);
  CHECK_INPUT(q_rope);
  CHECK_INPUT(k_nope);
  CHECK_INPUT(k_rope);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  CHECK_INPUT(q_out);

  auto device = q_nope.device();
  CHECK_EQ(q_rope.device(), device);
  CHECK_EQ(k_nope.device(), device);
  CHECK_EQ(k_rope.device(), device);
  CHECK_EQ(cos_sin_cache.device(), device);
  CHECK_EQ(pos_ids.device(), device);
  CHECK_EQ(q_out.device(), device);

  TORCH_CHECK(q_nope.dim() == 2 || q_nope.dim() == 3, "q_nope must be 2D or 3D");
  TORCH_CHECK(q_rope.dim() == 2 || q_rope.dim() == 3, "q_rope must be 2D or 3D");
  CHECK_DIM(2, k_nope);
  CHECK_DIM(2, k_rope);
  CHECK_DIM(1, pos_ids);
  CHECK_DIM(2, cos_sin_cache);

  // Determine dimensions and strides based on Q shape
  int nnz_tokens, num_heads, Dn, Dr;
  int64_t qn_stride_tok, qn_stride_head, qr_stride_tok, qr_stride_head;
  int64_t qout_stride_tok_bytes, qout_stride_head_bytes;

  if (q_nope.dim() == 3) {
    nnz_tokens = q_nope.size(0);
    num_heads = q_nope.size(1);
    Dn = q_nope.size(2);
    Dr = q_rope.size(2);

    CHECK_EQ(q_rope.size(0), nnz_tokens);
    CHECK_EQ(q_rope.size(1), num_heads);
    CHECK_EQ(q_out.dim(), 3);
    CHECK_EQ(q_out.size(0), nnz_tokens);
    CHECK_EQ(q_out.size(1), num_heads);
    CHECK_EQ(q_out.size(2), Dn + Dr);

    qn_stride_tok = q_nope.stride(0);
    qn_stride_head = q_nope.stride(1);
    qr_stride_tok = q_rope.stride(0);
    qr_stride_head = q_rope.stride(1);
    qout_stride_tok_bytes = q_out.stride(0);
    qout_stride_head_bytes = q_out.stride(1);
  } else {
    nnz_tokens = q_nope.size(0);
    Dn = q_nope.size(1);
    Dr = q_rope.size(1);
    num_heads = 1;

    CHECK_EQ(q_rope.size(0), nnz_tokens);
    CHECK_EQ(q_out.dim(), 2);
    CHECK_EQ(q_out.size(0), nnz_tokens);
    CHECK_EQ(q_out.size(1), Dn + Dr);

    qn_stride_tok = q_nope.stride(0);
    qn_stride_head = 0;
    qr_stride_tok = q_rope.stride(0);
    qr_stride_head = 0;
    qout_stride_tok_bytes = q_out.stride(0);
    qout_stride_head_bytes = 0;
  }

  int nnz_k = k_rope.size(0);
  CHECK_EQ(k_nope.size(0), nnz_k);
  CHECK_EQ(k_nope.size(1), Dn);
  CHECK_EQ(k_rope.size(0), nnz_k);
  CHECK_EQ(k_rope.size(1), Dr);
  CHECK_EQ(nnz_k, nnz_tokens);

  int64_t kn_stride_tok = k_nope.stride(0);
  int64_t kr_stride_tok = k_rope.stride(0);

  CHECK_LAST_DIM_CONTIGUOUS(k_nope);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope);
  CHECK_LAST_DIM_CONTIGUOUS(q_nope);
  CHECK_LAST_DIM_CONTIGUOUS(q_rope);
  CHECK_LAST_DIM_CONTIGUOUS(q_out);

  uint8_t* k_nope_out_ptr = nullptr;
  uint8_t* k_rope_out_ptr = nullptr;
  if (k_nope_out.has_value()) {
    auto t = k_nope_out.value();
    CHECK_INPUT(t);
    CHECK_DIM(2, t);
    CHECK_EQ(t.size(0), nnz_k);
    CHECK_EQ(t.size(1), Dn);
    k_nope_out_ptr = reinterpret_cast<uint8_t*>(t.data_ptr());
  }
  if (k_rope_out.has_value()) {
    auto t = k_rope_out.value();
    CHECK_INPUT(t);
    CHECK_DIM(2, t);
    CHECK_EQ(t.size(0), nnz_k);
    CHECK_EQ(t.size(1), Dr);
    k_rope_out_ptr = reinterpret_cast<uint8_t*>(t.data_ptr());
  }

  uint8_t* kv_buf_ptr = nullptr;
  int64_t kv_stride_row_bytes = 0;
  const int64_t* kv_loc_ptr = nullptr;
  if (kv_buffer.has_value() || kv_cache_loc.has_value()) {
    TORCH_CHECK(kv_buffer.has_value() && kv_cache_loc.has_value(), "kv_buffer and kv_cache_loc must be both provided");
    auto kv = kv_buffer.value();
    auto loc = kv_cache_loc.value();
    CHECK_INPUT(kv);
    CHECK_INPUT(loc);
    CHECK_DIM(1, loc);

    TORCH_CHECK(kv.dim() == 2 || (kv.dim() == 3 && kv.size(1) == 1), "kv_buffer must be 2D or 3D with middle dim=1");

    int kv_dim_actual = (kv.dim() == 3) ? kv.size(2) : kv.size(1);
    CHECK_EQ(kv_dim_actual, Dn + Dr);
    CHECK_EQ(loc.size(0), nnz_k);
    CHECK_LAST_DIM_CONTIGUOUS(kv);

    kv_buf_ptr = reinterpret_cast<uint8_t*>(kv.data_ptr());
    kv_stride_row_bytes = kv.stride(0) * kv.element_size();
    kv_loc_ptr = loc.data_ptr<int64_t>();
  }

  const float* cs_ptr = cos_sin_cache.data_ptr<float>();
  const int64_t* pos_ptr = pos_ids.data_ptr<int64_t>();
  uint8_t* q_out_ptr = reinterpret_cast<uint8_t*>(q_out.data_ptr());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  int total_rows = nnz_tokens * num_heads;

  bool can_vectorize = ((Dn & 3) == 0) && ((Dr & 3) == 0) && !is_neox;
  if (can_vectorize) {
    bool strides_aligned =
        (qout_stride_tok_bytes % 4 == 0) && (num_heads > 1 ? (qout_stride_head_bytes % 4 == 0) : true);
    if (!strides_aligned) {
      can_vectorize = false;
    }
  }

  auto dtype = q_nope.scalar_type();

  if (dtype == at::kHalf) {
    const __half* qn_ptr = reinterpret_cast<const __half*>(q_nope.data_ptr<at::Half>());
    const __half* qr_ptr = reinterpret_cast<const __half*>(q_rope.data_ptr<at::Half>());
    const __half* kn_ptr = reinterpret_cast<const __half*>(k_nope.data_ptr<at::Half>());
    const __half* kr_ptr = reinterpret_cast<const __half*>(k_rope.data_ptr<at::Half>());

    if (can_vectorize) {
      constexpr int WARPS_PER_CTA = 4;
      dim3 vecBlock(WARPS_PER_CTA * 32);
      dim3 vecGrid((total_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA);

      FusedRopeQuantizeKernelVec<WARPS_PER_CTA, __half><<<vecGrid, vecBlock, 0, stream>>>(
          qn_ptr,
          qr_ptr,
          qn_stride_tok,
          qn_stride_head,
          qr_stride_tok,
          qr_stride_head,
          kn_ptr,
          kr_ptr,
          kn_stride_tok,
          kr_stride_tok,
          cs_ptr,
          pos_ptr,
          nnz_tokens,
          num_heads,
          Dn,
          Dr,
          is_neox,
          q_out_ptr,
          qout_stride_tok_bytes,
          qout_stride_head_bytes,
          k_nope_out_ptr,
          k_rope_out_ptr,
          kv_buf_ptr,
          kv_stride_row_bytes,
          kv_loc_ptr);
    } else {
      constexpr int BLOCK_THREADS = 256;
      dim3 grid((total_rows + BLOCK_THREADS - 1) / BLOCK_THREADS);

      FusedRopeQuantizeKernelScalar<BLOCK_THREADS, __half><<<grid, BLOCK_THREADS, 0, stream>>>(
          qn_ptr,
          qr_ptr,
          qn_stride_tok,
          qn_stride_head,
          qr_stride_tok,
          qr_stride_head,
          kn_ptr,
          kr_ptr,
          kn_stride_tok,
          kr_stride_tok,
          cs_ptr,
          pos_ptr,
          nnz_tokens,
          num_heads,
          Dn,
          Dr,
          is_neox,
          q_out_ptr,
          qout_stride_tok_bytes,
          qout_stride_head_bytes,
          k_nope_out_ptr,
          k_rope_out_ptr,
          kv_buf_ptr,
          kv_stride_row_bytes,
          kv_loc_ptr);
    }
  } else if (dtype == at::kBFloat16) {
    const nv_bfloat16* qn_ptr = reinterpret_cast<const nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
    const nv_bfloat16* qr_ptr = reinterpret_cast<const nv_bfloat16*>(q_rope.data_ptr<at::BFloat16>());
    const nv_bfloat16* kn_ptr = reinterpret_cast<const nv_bfloat16*>(k_nope.data_ptr<at::BFloat16>());
    const nv_bfloat16* kr_ptr = reinterpret_cast<const nv_bfloat16*>(k_rope.data_ptr<at::BFloat16>());

    if (can_vectorize) {
      constexpr int WARPS_PER_CTA = 4;
      dim3 vecBlock(WARPS_PER_CTA * 32);
      dim3 vecGrid((total_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA);

      FusedRopeQuantizeKernelVec<WARPS_PER_CTA, nv_bfloat16><<<vecGrid, vecBlock, 0, stream>>>(
          qn_ptr,
          qr_ptr,
          qn_stride_tok,
          qn_stride_head,
          qr_stride_tok,
          qr_stride_head,
          kn_ptr,
          kr_ptr,
          kn_stride_tok,
          kr_stride_tok,
          cs_ptr,
          pos_ptr,
          nnz_tokens,
          num_heads,
          Dn,
          Dr,
          is_neox,
          q_out_ptr,
          qout_stride_tok_bytes,
          qout_stride_head_bytes,
          k_nope_out_ptr,
          k_rope_out_ptr,
          kv_buf_ptr,
          kv_stride_row_bytes,
          kv_loc_ptr);
    } else {
      constexpr int BLOCK_THREADS = 256;
      dim3 grid((total_rows + BLOCK_THREADS - 1) / BLOCK_THREADS);

      FusedRopeQuantizeKernelScalar<BLOCK_THREADS, nv_bfloat16><<<grid, BLOCK_THREADS, 0, stream>>>(
          qn_ptr,
          qr_ptr,
          qn_stride_tok,
          qn_stride_head,
          qr_stride_tok,
          qr_stride_head,
          kn_ptr,
          kr_ptr,
          kn_stride_tok,
          kr_stride_tok,
          cs_ptr,
          pos_ptr,
          nnz_tokens,
          num_heads,
          Dn,
          Dr,
          is_neox,
          q_out_ptr,
          qout_stride_tok_bytes,
          qout_stride_head_bytes,
          k_nope_out_ptr,
          k_rope_out_ptr,
          kv_buf_ptr,
          kv_stride_row_bytes,
          kv_loc_ptr);
    }
  } else {
    TORCH_CHECK(false, "Unsupported dtype for fused kernel. Only FP16 and BF16 are supported.");
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "mla_rope_quantize_fp8_fused",
      &mla_rope_quantize_fp8_fused,
      "Fused MLA RoPE + FP8 quantization with optional KV cache write",
      py::arg("q_nope"),
      py::arg("q_rope"),
      py::arg("k_nope"),
      py::arg("k_rope"),
      py::arg("cos_sin_cache"),
      py::arg("pos_ids"),
      py::arg("is_neox"),
      py::arg("q_out"),
      py::arg("k_nope_out") = py::none(),
      py::arg("k_rope_out") = py::none(),
      py::arg("kv_buffer") = py::none(),
      py::arg("kv_cache_loc") = py::none());
}
#endif
