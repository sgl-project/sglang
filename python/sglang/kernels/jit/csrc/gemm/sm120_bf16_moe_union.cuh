// All original BF16 expert weights and selected routes. One owner per expert.
#include "hopper_bf16_gemv.cuh"
namespace sglang {
__global__ void bf16_union_plan(const int* ids, int* map, unsigned M) {
  unsigned route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= M * 22) return;
  int expert = ids[route], first = M * 22;
  for (unsigned t = 0; t < 4; ++t) {
    int found = -1;
    if (t < M)
      for (unsigned s = 0; s < 22; ++s)
        if (ids[t * 22 + s] == expert) {
          found = t * 22 + s;
          break;
        }
    map[route * 5 + t] = found;
    if (found >= 0 && found < first) first = found;
  }
  map[route * 5 + 4] = first == route;
}
template <unsigned N, unsigned K, unsigned Warps, bool FC1>
__device__ __forceinline__ void bf16_union_load(
    bf16_t* ws,
    bf16_t* sx,
    const bf16_t* w,
    const bf16_t* x,
    const int* routes,
    unsigned expert,
    unsigned tile,
    unsigned col,
    unsigned lane) {
#pragma unroll
  for (unsigned j = 0; j < 8; ++j) {
    unsigned row = 2 * j + lane / 16, offset = (lane % 16) * 8;
    unsigned bytes = col < K ? 16 : 0;
    const bf16_t* src = w + (static_cast<size_t>(expert) * N + tile + row) * K + (bytes ? col + offset : 0);
    unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(ws + row * 136 + offset));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
  }
  for (unsigned i = threadIdx.x; i < 64; i += Warps * 32) {
    unsigned token = i / 16;
    int route = routes[token];
    unsigned bytes = col < K && route >= 0 ? 16 : 0;
    unsigned inputrow = bytes ? (FC1 ? token : route) : 0;
    const bf16_t* src = x + inputrow * K + (bytes ? col + (i % 16) * 8 : 0);
    unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(sx + i * 8));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <unsigned N, unsigned K, unsigned Warps, bool FC1>
__global__ __launch_bounds__(Warps * 32) void bf16_union_mm(
    const bf16_t* x, const bf16_t* w, const int* ids, const int* map, bf16_t* out) {
  const int* routes = map + blockIdx.y * 5;
  if (!routes[4]) return;
  __shared__ __align__(16) bf16_t sw[Warps][2][16 * 136];
  __shared__ __align__(16) bf16_t sx[2][4][128];
  unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32, tile = (blockIdx.x * Warps + warp) * 16,
           expert = ids[blockIdx.y];
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  for (unsigned stage = 0; stage < 2; ++stage)
    bf16_union_load<N, K, Warps, FC1>(sw[warp][stage], &sx[stage][0][0], w, x, routes, expert, tile, stage * 128, lane);
  for (unsigned base = 0; base < K; base += 128) {
    unsigned buf = (base / 128) % 2;
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (unsigned inner = 0; inner < 128; inner += 16) {
      unsigned addr =
          static_cast<unsigned>(__cvta_generic_to_shared(sw[warp][buf] + (lane % 16) * 136 + inner + (lane / 16) * 8));
      unsigned a0, a1, a2, a3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                   : "r"(addr)
                   : "memory");
      unsigned token = (lane / 4) % 4;
      unsigned b0 = *reinterpret_cast<unsigned*>(sx[buf][token] + inner + (lane % 4) * 2),
               b1 = *reinterpret_cast<unsigned*>(sx[buf][token] + inner + (lane % 4) * 2 + 8);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
    bf16_union_load<N, K, Warps, FC1>(sw[warp][buf], &sx[buf][0][0], w, x, routes, expert, tile, base + 256, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (lane % 4 < 2) {
    unsigned ct = 2 * (lane % 4), row = tile + lane / 4;
    float vals[4] = {d0, d1, d2, d3};
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      int route = routes[ct + i % 2];
      if (route >= 0) {
        float v = vals[i];
        if constexpr (FC1) {
          v = fmaxf(v, 0.f);
          v = v * v;
        }
        out[route * N + row + (i / 2) * 8] = cast<bf16_t>(v);
      }
    }
  }
}
__global__ void bf16_union_reduce(const bf16_t* partial, const float* weights, bf16_t* out, unsigned M) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * 1024) return;
  unsigned token = i / 1024, col = i % 1024;
  float value = 0;
  for (unsigned s = 0; s < 22; ++s)
    value += static_cast<float>(partial[(token * 22 + s) * 1024 + col]) * weights[token * 22 + s];
  out[i] = cast<bf16_t>(value);
}
template <unsigned Warps>
struct Bf16MoeUnionRaw {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView w1,
      tvm::ffi::TensorView w2,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView map,
      tvm::ffi::TensorView z,
      tvm::ffi::TensorView partial,
      tvm::ffi::TensorView out) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto rows = SymbolicSize{"M"};
    TensorMatcher({rows, 1024}).with_dtype<bf16_t>().with_device(dev).verify(x).verify(out);
    TensorMatcher({512, 2688, 1024}).with_dtype<bf16_t>().with_device(dev).verify(w1);
    TensorMatcher({512, 1024, 2688}).with_dtype<bf16_t>().with_device(dev).verify(w2);
    unsigned M = rows.unwrap();
    CHECK_HOST(M >= 1 && M <= 4);
    TensorMatcher({M, 22}).with_dtype<int32_t>().with_device(dev).verify(ids);
    TensorMatcher({M, 22}).with_dtype<float>().with_device(dev).verify(weights);
    TensorMatcher({M * 22, 5}).with_dtype<int32_t>().with_device(dev).verify(map);
    TensorMatcher({M * 22, 2688}).with_dtype<bf16_t>().with_device(dev).verify(z);
    TensorMatcher({M * 22, 1024}).with_dtype<bf16_t>().with_device(dev).verify(partial);
    LaunchKernel(1, 128, dev.unwrap())(
        bf16_union_plan, static_cast<const int*>(ids.data_ptr()), static_cast<int*>(map.data_ptr()), M);
    LaunchKernel({2688 / (Warps * 16), M * 22}, Warps * 32, dev.unwrap())(
        bf16_union_mm<2688, 1024, Warps, true>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w1.data_ptr()),
        static_cast<const int*>(ids.data_ptr()),
        static_cast<const int*>(map.data_ptr()),
        static_cast<bf16_t*>(z.data_ptr()));
    LaunchKernel({1024 / (Warps * 16), M * 22}, Warps * 32, dev.unwrap())(
        bf16_union_mm<1024, 2688, Warps, false>,
        static_cast<const bf16_t*>(z.data_ptr()),
        static_cast<const bf16_t*>(w2.data_ptr()),
        static_cast<const int*>(ids.data_ptr()),
        static_cast<const int*>(map.data_ptr()),
        static_cast<bf16_t*>(partial.data_ptr()));
    LaunchKernel((M * 1024 + 255) / 256, 256, dev.unwrap())(
        bf16_union_reduce,
        static_cast<const bf16_t*>(partial.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        M);
  }
};
}  // namespace sglang
