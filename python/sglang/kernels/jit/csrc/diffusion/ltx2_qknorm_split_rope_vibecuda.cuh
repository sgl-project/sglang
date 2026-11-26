#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <limits>

namespace sglang {
namespace ltx2_qknorm_split_rope_vibecuda {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 1000) && (__CUDA_ARCH__ != 1030)
#error "VibeCUDA LTX2 QKNorm split-RoPE supports only sm_100a and sm_103a"
#endif

struct alignas(8) BF16x4 {
  bf16_t values[4];
};

struct Table4 {
  const bf16_t* ptr;
  int64_t s0;
  int64_t s1;
  int64_t s2;
  int64_t s3;
};

struct RowContext {
  const bf16_t* x;
  const bf16_t* weight;
  bf16_t* out;
  Table4 cos;
  Table4 sin;
};

SGL_DEVICE float to_float(bf16_t x) {
  return __bfloat162float(x);
}

SGL_DEVICE float4 load4(const bf16_t* p) {
  BF16x4 value = *reinterpret_cast<const BF16x4*>(p);
  return make_float4(
      to_float(value.values[0]), to_float(value.values[1]), to_float(value.values[2]), to_float(value.values[3]));
}

SGL_DEVICE void store4(bf16_t* p, float a, float b, float c, float d) {
  BF16x4 value;
  value.values[0] = __float2bfloat16(a);
  value.values[1] = __float2bfloat16(b);
  value.values[2] = __float2bfloat16(c);
  value.values[3] = __float2bfloat16(d);
  *reinterpret_cast<BF16x4*>(p) = value;
}

SGL_DEVICE void convert2(uint32_t word, float& lo, float& hi) {
  lo = __uint_as_float(word << 16);
  hi = __uint_as_float(word & 0xffff0000u);
}

SGL_DEVICE uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&packed);
}

SGL_DEVICE void load8(const bf16_t* p, float* values) {
  uint4 packed = *reinterpret_cast<const uint4*>(p);
  convert2(packed.x, values[0], values[1]);
  convert2(packed.y, values[2], values[3]);
  convert2(packed.z, values[4], values[5]);
  convert2(packed.w, values[6], values[7]);
}

SGL_DEVICE void store8(bf16_t* p, const float* values) {
  uint4 packed;
  packed.x = pack2(values[0], values[1]);
  packed.y = pack2(values[2], values[3]);
  packed.z = pack2(values[4], values[5]);
  packed.w = pack2(values[6], values[7]);
  *reinterpret_cast<uint4*>(p) = packed;
}

SGL_DEVICE void load8_8B(const bf16_t* p, float* values) {
  uint2 a = *reinterpret_cast<const uint2*>(p);
  uint2 b = *reinterpret_cast<const uint2*>(p + 4);
  convert2(a.x, values[0], values[1]);
  convert2(a.y, values[2], values[3]);
  convert2(b.x, values[4], values[5]);
  convert2(b.y, values[6], values[7]);
}

// The reduction order matches aten's 128-thread vectorized RMSNorm kernel.
template <bool kVectorX, bool kVectorTable>
__global__ void __launch_bounds__(128, 12) ltx2_qknorm_rope_kernel(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ k,
    bf16_t* __restrict__ q_out,
    bf16_t* __restrict__ k_out,
    const bf16_t* __restrict__ q_weight,
    const bf16_t* __restrict__ k_weight,
    Table4 q_cos,
    Table4 q_sin,
    Table4 k_cos,
    Table4 k_sin,
    int32_t hidden,
    int32_t hidden_vec,
    int32_t rows_q,
    int64_t seq_q,
    int64_t seq_k,
    int32_t half,
    float eps,
    int32_t pair8,
    int32_t table16) {
  const int32_t row = blockIdx.x;
  RowContext ctx;
  int64_t batch;
  int64_t seq;
  if (row < rows_q) {
    batch = row / seq_q;
    seq = row % seq_q;
    ctx.x = q + static_cast<int64_t>(row) * hidden;
    ctx.weight = q_weight;
    ctx.out = q_out + static_cast<int64_t>(row) * hidden;
    ctx.cos = q_cos;
    ctx.sin = q_sin;
  } else {
    const int64_t row_k = static_cast<int64_t>(row) - rows_q;
    batch = row_k / seq_k;
    seq = row_k % seq_k;
    ctx.x = k + row_k * hidden;
    ctx.weight = k_weight;
    ctx.out = k_out + row_k * hidden;
    ctx.cos = k_cos;
    ctx.sin = k_sin;
  }

  const int32_t thread = threadIdx.x + threadIdx.y * 32;
  constexpr int32_t kThreads = 128;
  constexpr int32_t kStep = kThreads * 4;
  float acc = 0.0f;
  const bf16_t* x_ptr = ctx.x + static_cast<int64_t>(thread) * 4;
  for (int32_t i = thread; i < hidden_vec; i += kThreads, x_ptr += kStep) {
    float4 value;
    if constexpr (kVectorX) {
      value = load4(x_ptr);
    } else {
      value = make_float4(to_float(x_ptr[0]), to_float(x_ptr[1]), to_float(x_ptr[2]), to_float(x_ptr[3]));
    }
    acc = fmaf(value.x, value.x, acc);
    acc = fmaf(value.y, value.y, acc);
    acc = fmaf(value.z, value.z, acc);
    acc = fmaf(value.w, value.w, acc);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }

  __shared__ float warp_sums[4];
  for (int offset = 2; offset > 0; offset >>= 1) {
    if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
      warp_sums[threadIdx.y - offset] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < offset) {
      acc += warp_sums[threadIdx.y];
    }
    __syncthreads();
  }
  __shared__ float mean_square;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    mean_square = acc / static_cast<float>(hidden);
  }
  __syncthreads();
  const float rstd = rsqrtf(mean_square + eps);

  const int32_t head_width = 2 * half;
  const int64_t cos_base = batch * ctx.cos.s0 + seq * ctx.cos.s2;
  const int64_t sin_base = batch * ctx.sin.s0 + seq * ctx.sin.s2;

  if (pair8) {
    // Each thread owns an eight-element first-half group and its RoPE partner.
    const int32_t pairs_per_head = half >> 3;
    const int32_t pairs_per_row = hidden_vec >> 2;
    const int32_t pair_step = kThreads % pairs_per_head;
    const int32_t head_step = kThreads / pairs_per_head;
    int32_t pair = thread % pairs_per_head;
    int32_t head = thread / pairs_per_head;
    int64_t cos_head = static_cast<int64_t>(head) * ctx.cos.s1;
    int64_t sin_head = static_cast<int64_t>(head) * ctx.sin.s1;
    const int64_t cos_step = static_cast<int64_t>(head_step) * ctx.cos.s1;
    const int64_t sin_step = static_cast<int64_t>(head_step) * ctx.sin.s1;
    for (int32_t p = thread; p < pairs_per_row; p += kThreads) {
      const int64_t offset = static_cast<int64_t>(head) * head_width + pair * 8;
      const bf16_t* x_first = ctx.x + offset;
      const bf16_t* x_second = x_first + half;
      const bf16_t* weight_first = ctx.weight + offset;
      const bf16_t* weight_second = weight_first + half;
      const bf16_t* cos_ptr = ctx.cos.ptr + cos_base + cos_head + static_cast<int64_t>(pair * 8) * ctx.cos.s3;
      const bf16_t* sin_ptr = ctx.sin.ptr + sin_base + sin_head + static_cast<int64_t>(pair * 8) * ctx.sin.s3;
      float xf[8];
      float xs[8];
      float gf[8];
      float gs[8];
      float cv[8];
      float sv[8];
      load8(x_first, xf);
      load8(x_second, xs);
      load8(weight_first, gf);
      load8(weight_second, gs);
      if (table16) {
        load8(cos_ptr, cv);
        load8(sin_ptr, sv);
      } else if constexpr (kVectorTable) {
        load8_8B(cos_ptr, cv);
        load8_8B(sin_ptr, sv);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          cv[i] = to_float(cos_ptr[i * ctx.cos.s3]);
          sv[i] = to_float(sin_ptr[i * ctx.sin.s3]);
        }
      }
      float out_first[8];
      float out_second[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float norm_first = gf[i] * (rstd * xf[i]);
        const float norm_second = gs[i] * (rstd * xs[i]);
        out_first[i] = fmaf(-sv[i], norm_second, norm_first * cv[i]);
        out_second[i] = fmaf(sv[i], norm_first, norm_second * cv[i]);
      }
      bf16_t* out_ptr = ctx.out + offset;
      store8(out_ptr, out_first);
      store8(out_ptr + half, out_second);
      head += head_step;
      cos_head += cos_step;
      sin_head += sin_step;
      pair += pair_step;
      if (pair >= pairs_per_head) {
        pair -= pairs_per_head;
        head += 1;
        cos_head += ctx.cos.s1;
        sin_head += ctx.sin.s1;
      }
    }
    return;
  }

  const int32_t head_step = kStep / head_width;
  const int32_t dim_step = kStep - head_step * head_width;
  const int32_t partner_slots = half >> 2;
  int32_t dim = (thread * 4) % head_width;
  int64_t cos_head = static_cast<int64_t>((thread * 4) / head_width) * ctx.cos.s1;
  int64_t sin_head = static_cast<int64_t>((thread * 4) / head_width) * ctx.sin.s1;
  const int64_t cos_step = static_cast<int64_t>(head_step) * ctx.cos.s1;
  const int64_t sin_step = static_cast<int64_t>(head_step) * ctx.sin.s1;
  const bf16_t* x1_ptr = ctx.x + static_cast<int64_t>(thread) * 4;
  const bf16_t* weight1_ptr = ctx.weight + thread * 4;
  bf16_t* out_ptr = ctx.out + static_cast<int64_t>(thread) * 4;

  for (int32_t i = thread; i < hidden_vec; i += kThreads, x1_ptr += kStep, weight1_ptr += kStep, out_ptr += kStep) {
    const bool first_half = dim < half;
    const int32_t table_dim = first_half ? dim : dim - half;
    const int32_t partner_slot = first_half ? i + partner_slots : i - partner_slots;
    const bf16_t* x2_ptr = ctx.x + static_cast<int64_t>(partner_slot) * 4;
    const bf16_t* weight2_ptr = ctx.weight + partner_slot * 4;
    const bf16_t* cos_ptr = ctx.cos.ptr + cos_base + cos_head + static_cast<int64_t>(table_dim) * ctx.cos.s3;
    const bf16_t* sin_ptr = ctx.sin.ptr + sin_base + sin_head + static_cast<int64_t>(table_dim) * ctx.sin.s3;

    float4 x1;
    float4 x2;
    float4 weight1;
    float4 weight2;
    float4 cos_value;
    float4 sin_value;
    if constexpr (kVectorX) {
      x1 = load4(x1_ptr);
      x2 = load4(x2_ptr);
      weight1 = load4(weight1_ptr);
      weight2 = load4(weight2_ptr);
    } else {
      x1 = make_float4(to_float(x1_ptr[0]), to_float(x1_ptr[1]), to_float(x1_ptr[2]), to_float(x1_ptr[3]));
      x2 = make_float4(to_float(x2_ptr[0]), to_float(x2_ptr[1]), to_float(x2_ptr[2]), to_float(x2_ptr[3]));
      weight1 = make_float4(
          to_float(weight1_ptr[0]), to_float(weight1_ptr[1]), to_float(weight1_ptr[2]), to_float(weight1_ptr[3]));
      weight2 = make_float4(
          to_float(weight2_ptr[0]), to_float(weight2_ptr[1]), to_float(weight2_ptr[2]), to_float(weight2_ptr[3]));
    }
    if constexpr (kVectorTable) {
      cos_value = load4(cos_ptr);
      sin_value = load4(sin_ptr);
    } else {
      cos_value = make_float4(
          to_float(cos_ptr[0]),
          to_float(cos_ptr[ctx.cos.s3]),
          to_float(cos_ptr[2 * ctx.cos.s3]),
          to_float(cos_ptr[3 * ctx.cos.s3]));
      sin_value = make_float4(
          to_float(sin_ptr[0]),
          to_float(sin_ptr[ctx.sin.s3]),
          to_float(sin_ptr[2 * ctx.sin.s3]),
          to_float(sin_ptr[3 * ctx.sin.s3]));
    }
    const float4 norm1 = make_float4(
        weight1.x * (rstd * x1.x), weight1.y * (rstd * x1.y), weight1.z * (rstd * x1.z), weight1.w * (rstd * x1.w));
    const float4 norm2 = make_float4(
        weight2.x * (rstd * x2.x), weight2.y * (rstd * x2.y), weight2.z * (rstd * x2.z), weight2.w * (rstd * x2.w));
    const float sign = first_half ? -1.0f : 1.0f;
    const float out0 = fmaf(sign * sin_value.x, norm2.x, norm1.x * cos_value.x);
    const float out1 = fmaf(sign * sin_value.y, norm2.y, norm1.y * cos_value.y);
    const float out2 = fmaf(sign * sin_value.z, norm2.z, norm1.z * cos_value.z);
    const float out3 = fmaf(sign * sin_value.w, norm2.w, norm1.w * cos_value.w);
    if constexpr (kVectorX) {
      store4(out_ptr, out0, out1, out2, out3);
    } else {
      out_ptr[0] = __float2bfloat16(out0);
      out_ptr[1] = __float2bfloat16(out1);
      out_ptr[2] = __float2bfloat16(out2);
      out_ptr[3] = __float2bfloat16(out3);
    }
    dim += dim_step;
    cos_head += cos_step;
    sin_head += sin_step;
    if (dim >= head_width) {
      dim -= head_width;
      cos_head += ctx.cos.s1;
      sin_head += ctx.sin.s1;
    }
  }
}

inline Table4 make_table(tvm::ffi::TensorView tensor) {
  return {
      static_cast<const bf16_t*>(tensor.data_ptr()),
      tensor.stride(0),
      tensor.stride(1),
      tensor.stride(2),
      tensor.stride(3),
  };
}

inline bool aligned(const tvm::ffi::TensorView& tensor, int64_t bytes) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0;
}

struct VibeCUDALTX2QKNormSplitRopeKernel {
  static void
  run(tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView q_cos,
      tvm::ffi::TensorView q_sin,
      tvm::ffi::TensorView q_weight,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView k_cos,
      tvm::ffi::TensorView k_sin,
      tvm::ffi::TensorView k_weight,
      double eps,
      int64_t num_heads,
      int64_t head_dim) {
    using namespace host;
    CHECK_HOST(num_heads > 0 && head_dim > 0) << "LTX2: heads and head_dim must be positive";
    CHECK_HOST(head_dim % 8 == 0) << "VibeCUDA LTX2: head_dim must be divisible by 8";
    CHECK_HOST(num_heads * head_dim <= std::numeric_limits<int32_t>::max())
        << "VibeCUDA LTX2: hidden size exceeds int32 indexing";

    auto batch = SymbolicSize{"batch"};
    auto q_tokens = SymbolicSize{"q_tokens"};
    auto k_tokens = SymbolicSize{"k_tokens"};
    auto heads = SymbolicSize{"heads"};
    auto hidden = SymbolicSize{"hidden"};
    auto half = SymbolicSize{"half_head_dim"};
    auto device = SymbolicDevice{};
    heads.set_value(num_heads);
    hidden.set_value(num_heads * head_dim);
    half.set_value(head_dim / 2);
    device.set_options<kDLCUDA>();

    TensorMatcher({batch, q_tokens, hidden}).with_dtype<bf16_t>().with_device(device).verify(q).verify(q_out);
    TensorMatcher({batch, k_tokens, hidden}).with_dtype<bf16_t>().with_device(device).verify(k).verify(k_out);
    TensorMatcher({batch, heads, q_tokens, half})
        .with_strides({-1, -1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_cos)
        .verify(q_sin);
    TensorMatcher({batch, heads, k_tokens, half})
        .with_strides({-1, -1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(k_cos)
        .verify(k_sin);
    TensorMatcher({hidden}).with_dtype<bf16_t>().with_device(device).verify(q_weight).verify(k_weight);

    const int64_t rows_q = batch.unwrap() * q_tokens.unwrap();
    const int64_t rows_k = batch.unwrap() * k_tokens.unwrap();
    const int64_t total_rows = rows_q + rows_k;
    CHECK_HOST(total_rows <= std::numeric_limits<int32_t>::max())
        << "VibeCUDA LTX2: row count exceeds the CUDA grid limit";
    if (total_rows == 0) return;

    const bool vector_x = aligned(q, 8) && aligned(k, 8) && aligned(q_out, 8) && aligned(k_out, 8) &&
                          aligned(q_weight, 8) && aligned(k_weight, 8);
    const auto vector_table = [](tvm::ffi::TensorView tensor) {
      return tensor.stride(2) % 4 == 0 && tensor.stride(1) % 4 == 0 && tensor.stride(0) % 4 == 0 && aligned(tensor, 8);
    };
    const bool vector_tables = vector_table(q_cos) && vector_table(q_sin) && vector_table(k_cos) && vector_table(k_sin);
    const bool pair8 = vector_x && num_heads * head_dim % 16 == 0 && head_dim / 2 % 8 == 0 && aligned(q, 16) &&
                       aligned(k, 16) && aligned(q_out, 16) && aligned(k_out, 16) && aligned(q_weight, 16) &&
                       aligned(k_weight, 16);
    const auto table16 = [](tvm::ffi::TensorView tensor) {
      return tensor.stride(2) % 8 == 0 && tensor.stride(1) % 8 == 0 && tensor.stride(0) % 8 == 0 && aligned(tensor, 16);
    };
    const bool tables16 =
        pair8 && vector_tables && table16(q_cos) && table16(q_sin) && table16(k_cos) && table16(k_sin);

    const int32_t hidden_i32 = static_cast<int32_t>(hidden.unwrap());
    const dim3 grid(static_cast<uint32_t>(total_rows));
    const dim3 block(32, 4);
    const Table4 q_cos_table = make_table(q_cos);
    const Table4 q_sin_table = make_table(q_sin);
    const Table4 k_cos_table = make_table(k_cos);
    const Table4 k_sin_table = make_table(k_sin);

#define LTX2_VIBECUDA_LAUNCH(VECTOR_X, VECTOR_TABLE)   \
  LaunchKernel(grid, block, device.unwrap())(          \
      ltx2_qknorm_rope_kernel<VECTOR_X, VECTOR_TABLE>, \
      static_cast<const bf16_t*>(q.data_ptr()),        \
      static_cast<const bf16_t*>(k.data_ptr()),        \
      static_cast<bf16_t*>(q_out.data_ptr()),          \
      static_cast<bf16_t*>(k_out.data_ptr()),          \
      static_cast<const bf16_t*>(q_weight.data_ptr()), \
      static_cast<const bf16_t*>(k_weight.data_ptr()), \
      q_cos_table,                                     \
      q_sin_table,                                     \
      k_cos_table,                                     \
      k_sin_table,                                     \
      hidden_i32,                                      \
      hidden_i32 / 4,                                  \
      static_cast<int32_t>(rows_q),                    \
      q_tokens.unwrap(),                               \
      k_tokens.unwrap(),                               \
      static_cast<int32_t>(head_dim / 2),              \
      static_cast<float>(eps),                         \
      static_cast<int32_t>(pair8),                     \
      static_cast<int32_t>(tables16))

    if (vector_x) {
      if (vector_tables) {
        LTX2_VIBECUDA_LAUNCH(true, true);
      } else {
        LTX2_VIBECUDA_LAUNCH(true, false);
      }
    } else {
      if (vector_tables) {
        LTX2_VIBECUDA_LAUNCH(false, true);
      } else {
        LTX2_VIBECUDA_LAUNCH(false, false);
      }
    }
#undef LTX2_VIBECUDA_LAUNCH
  }
};

}  // namespace ltx2_qknorm_split_rope_vibecuda
}  // namespace sglang
