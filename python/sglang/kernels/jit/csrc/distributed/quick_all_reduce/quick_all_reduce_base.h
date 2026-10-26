// JIT copy of sgl-kernel's kernels/aot/csrc/allreduce/quick_all_reduce_base.h, cut to what the INT4 / INT8
// two-shot of quick_all_reduce.cuh uses: fp16 arithmetic on bf16 data cast at load and store. Keep the
// arithmetic in step with the AOT copy: the outputs are bit-identical to it.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#define __quickreduce_device_inline__ __device__ __forceinline__
#define __quickreduce_launch_bounds_two_shot__ __launch_bounds__(256, 4)

namespace quickreduce {

typedef __hip_bfloat162 nv_bfloat162;

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

#if !defined(QR_MAX_BLOCKS)
#error "QR_MAX_BLOCKS is set by the JIT loader"
#endif

// We use a workgroup of 256 threads
static constexpr int kBlockSize = 256;
static constexpr int kAtomStride = kBlockSize;

// Max number of resident blocks; also sizes the flag buffer.
static constexpr int kMaxNumBlocks = QR_MAX_BLOCKS;

// Standard CDNA wavefront size.
static constexpr int kWavefront = 64;

// 256 thread, 4 wavefronts.
static dim3 constexpr kBlockTwoShot = {kWavefront, kBlockSize / kWavefront, 1};

// Number of threads in a group for quantization
// It corresponds to 32 F16 elements in quantization block
static constexpr int kThreadGroupSize = 8;

// Methods
__quickreduce_device_inline__ __host__ unsigned long divceil(unsigned long x, unsigned long y) {
  return ((x + y - 1) / y);
}

union BufferResource {
  __quickreduce_device_inline__ constexpr BufferResource() : config(0x00020000U) {}

  __quickreduce_device_inline__ constexpr BufferResource(void* buffer_address, uint32_t buffer_size)
      : address(buffer_address), range(buffer_size), config(0x00020000U) {}

  int32x4_t descriptor;
  struct {
    void* address;  // 8B, out of which first 48b is address, and 16b is stride
    // (unused)
    uint32_t range;   // Byte range for the buffer resource
    uint32_t config;  // Constant, DFMT=32b
  };
};

__quickreduce_device_inline__ static int32x4_t buffer_load_dwordx4(
    int32x4_t srsrc, int32_t voffset, int32_t soffset, int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__quickreduce_device_inline__ static void
buffer_store_dwordx4(int32x4_t data, int32x4_t srsrc, int32_t voffset, int32_t soffset, int32_t aux) __asm(
    "llvm.amdgcn.raw.buffer.store.v4i32");
// MODE.FP16_OVFL clamps overflowing fp16 results to +/-MAX_FP16 instead of inf, the f32 -> f16
// conversion (v_cvt_pk_f16_f32) included. The memory clobber is load-bearing: without it the
// compiler may hoist a conversion above the s_setreg, and that conversion still produces inf.
__quickreduce_device_inline__ static void set_fp16_ovfl(bool const value) {
#if defined(__gfx942__) || defined(__gfx950__)
  if (value) {
    asm volatile("s_setreg_imm32_b32 0x5c1, 1;" ::: "memory");
  } else {
    asm volatile("s_setreg_imm32_b32 0x5c1, 0;" ::: "memory");
  }
#endif
}
template <typename T>
__quickreduce_device_inline__ void packed_assign_add(int32x4_t* A, int32x4_t* B);

template <>
__quickreduce_device_inline__ void packed_assign_add<half>(int32x4_t* A, int32x4_t* B) {
  int32x4_t& tR_fragment = A[0];
  int32x4_t& tA_fragment = B[0];

  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(tR_fragment[0]) : "v"(tR_fragment[0]), "v"(tA_fragment[0]));
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(tR_fragment[1]) : "v"(tR_fragment[1]), "v"(tA_fragment[1]));
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(tR_fragment[2]) : "v"(tR_fragment[2]), "v"(tA_fragment[2]));
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(tR_fragment[3]) : "v"(tR_fragment[3]), "v"(tA_fragment[3]));
}

template <typename T>
__quickreduce_device_inline__ int packed_max(int a, int b);

template <>
__quickreduce_device_inline__ int packed_max<half>(int a, int b) {
  int result;
  asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <typename T>
__quickreduce_device_inline__ int packed_min(int a, int b);

template <>
__quickreduce_device_inline__ int packed_min<half>(int a, int b) {
  int result;
  asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <typename T>
__quickreduce_device_inline__ int packed_abs_max(int a, int b);

template <>
__quickreduce_device_inline__ int packed_abs_max<half>(int a, int b) {
  half2 wmaxh2 = __builtin_bit_cast(half2, a);
  half2 wminh2 = __builtin_bit_cast(half2, b);
  half2 wblockmaxh2;

  wblockmaxh2.x = __hgt(__habs(wmaxh2.x), __habs(wminh2.x)) ? wmaxh2.x : wminh2.x;
  wblockmaxh2.y = __hgt(__habs(wmaxh2.y), __habs(wminh2.y)) ? wmaxh2.y : wminh2.y;
  return __builtin_bit_cast(int, wblockmaxh2);
}

template <typename T>
__quickreduce_device_inline__ int packed_add(int a, int b);

template <>
__quickreduce_device_inline__ int packed_add<half>(int a, int b) {
  int result;
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__quickreduce_device_inline__ int packed_add<int16_t>(int a, int b) {
  int result;
  asm volatile("v_pk_add_i16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <typename T>
__quickreduce_device_inline__ int packed_mul(int a, int b);

template <>
__quickreduce_device_inline__ int packed_mul<half>(int a, int b) {
  int result;
  asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <typename T>
__quickreduce_device_inline__ int packed_rcp(int a);

template <>
__quickreduce_device_inline__ int packed_rcp<half>(int a) {
  return __builtin_bit_cast(int, h2rcp(__builtin_bit_cast(half2, a)));
}

// changes dtype
__quickreduce_device_inline__ float T2float_cast(half a) {
  return __half2float(a);
}

// Max and min over each 8-lane group, combined by packed_abs_max, in every lane of the group.
// DPP lane moves (pairs, quads, then the two halves of each 8-lane half-row) instead of LDS
// permutes; max/min are exact, so the result equals the AOT reduce-to-leader-and-broadcast.
template <typename T>
__quickreduce_device_inline__ int group_abs_max(int32x4_t atom) {
  int a = packed_max<T>(atom[0], atom[1]);
  int b = packed_max<T>(atom[2], atom[3]);
  int wmax = packed_max<T>(a, b);
  a = packed_min<T>(atom[0], atom[1]);
  b = packed_min<T>(atom[2], atom[3]);
  int wmin = packed_min<T>(a, b);
  // DPP controls: quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_half_mirror.
  wmax = packed_max<T>(wmax, __builtin_amdgcn_mov_dpp(wmax, 0xB1, 0xF, 0xF, false));
  wmin = packed_min<T>(wmin, __builtin_amdgcn_mov_dpp(wmin, 0xB1, 0xF, 0xF, false));
  wmax = packed_max<T>(wmax, __builtin_amdgcn_mov_dpp(wmax, 0x4E, 0xF, 0xF, false));
  wmin = packed_min<T>(wmin, __builtin_amdgcn_mov_dpp(wmin, 0x4E, 0xF, 0xF, false));
  wmax = packed_max<T>(wmax, __builtin_amdgcn_mov_dpp(wmax, 0x141, 0xF, 0xF, false));
  wmin = packed_min<T>(wmin, __builtin_amdgcn_mov_dpp(wmin, 0x141, 0xF, 0xF, false));
  return packed_abs_max<T>(wmax, wmin);
}

// Global-address-space view of a communication buffer (local or peer: both plain GPU VAs). FLAT
// accesses also count in lgkmcnt, and their waits are imprecise: with remote FLAT stores in
// flight, an LDS wait or a wait for an earlier load also waits for those stores.
#define QR_GLOBAL(T, p) ((__attribute__((address_space(1))) T*)(p))

// s_waitcnt vmcnt(0) (gfx9 encoding; expcnt and lgkmcnt left at their maxima). The flag release on
// one lane covers only its own wave's stores, so every wave waits for its own before the barrier.
__quickreduce_device_inline__ void wait_own_stores() {
  __builtin_amdgcn_s_waitcnt(0x0F70);
}

__quickreduce_device_inline__ void set_sync_flag(uint32_t* flag_ptr, uint32_t flag) {
  __atomic_store_n(QR_GLOBAL(uint32_t, flag_ptr), flag, __ATOMIC_RELEASE);
}

__quickreduce_device_inline__ void wait_sync_flag(uint32_t* flag_ptr, uint32_t flag) {
  while (__atomic_load_n(QR_GLOBAL(uint32_t, flag_ptr), __ATOMIC_RELAXED) != flag) {
  }
}

}  // namespace quickreduce
