// Adapted from https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/utils.cuh

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ void st_na_global_v1(const int* ptr, int v) {
  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ void st_na_global_v2(const int2* ptr, const int2& v) {
  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void st_na_global_v4(const int4* ptr, const int4& v) {
  asm volatile(
      "st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
      : "memory");
}

__device__ __forceinline__ int ld_na_global_v1(const int* ptr) {
  int r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
#endif
  return r;
}

__device__ __forceinline__ int2 ld_na_global_v2(const int2* ptr) {
  int2 r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
#endif
  return r;
}

__device__ __forceinline__ int4 ld_na_global_v4(const int4* ptr) {
  int4 r;
#ifdef USE_L2_HINT
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(ptr));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(ptr));
#endif
  return r;
}

__device__ __forceinline__ void prefetch_L2(const void* p) {
#if defined(ENABLE_L2_PREFETCH)
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#endif
}
