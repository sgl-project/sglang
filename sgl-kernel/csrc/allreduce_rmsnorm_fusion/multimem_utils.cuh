#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace vllm {

/* 
* ********************************************************** *
* Code copied from Pytorch codebase:                         *
* https://github.com/pytorch/pytorch/blob/f6275bf0fe198f7f27569776ec221eb040a4cfa2/torch/csrc/distributed/c10d/CUDASymmetricMemoryOps.cu#L129                     *
* Multimem All Reduce Meta kernel                            *
* ********************************************************** *
*/
template <typename T>
__inline__ size_t get_alignment(T ptr_or_size)
{
  auto val = reinterpret_cast<uintptr_t>(ptr_or_size);
  if (val % 16 == 0)
  {
    return 16;
  }
  else if (val % 8 == 0)
  {
    return 8;
  }
  else if (val % 4 == 0)
  {
    return 4;
  }
  else if (val % 2 == 0)
  {
    return 2;
  }
  else
  {
    return 1;
  }
}

template <>
__inline__ size_t get_alignment<size_t>(size_t size)
{
  return get_alignment(reinterpret_cast<void *>(size));
}

template <bool Value, class... Args>
inline constexpr bool dependent_bool_value = Value;

template <class... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

template <auto... Args>
inline constexpr bool dependent_false_nt =
    dependent_bool_value<false, decltype(Args)...>;

enum class MemOpSem
{
  Relaxed,
  Acquire,
  Release,
  AcqRel,
};

#define CAS_ASM(addr, compare, val, old_val, sem)                 \
asm volatile("atom.global" sem ".sys.cas.b32 %0, [%1], %2, %3;" \
              : "=r"(old_val)                                    \
              : "l"(addr), "r"(compare), "r"(val)                \
              : "memory");

template <MemOpSem Sem>
__device__ __forceinline__ uint32_t
cas(uint32_t *addr, uint32_t compare, uint32_t val)
{
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
  return 0;
#else
  uint32_t old_val;
  if constexpr (Sem == MemOpSem::Relaxed)
  {
    CAS_ASM(addr, compare, val, old_val, ".relaxed");
  }
  else if constexpr (Sem == MemOpSem::Acquire)
  {
    CAS_ASM(addr, compare, val, old_val, ".acquire");
  }
  else if constexpr (Sem == MemOpSem::Release)
  {
    CAS_ASM(addr, compare, val, old_val, ".release");
  }
  else
  {
    static_assert(dependent_false_nt<Sem>);
  }
  return old_val;
#endif
}

__device__ __forceinline__ void trap()
{
#if defined(USE_ROCM)
  assert(0);
#else
  __trap();
#endif
}

__device__ __forceinline__ size_t global_timer_ns()
{
#if defined(USE_ROCM)
  CUDA_KERNEL_ASSERT(false);
  return 0;
#else
  size_t val;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(val) : : "memory");
  return val;
#endif
}

constexpr size_t ns_per_ms = 1e6;

template <MemOpSem Sem>
__device__ __forceinline__ bool try_put_signal(
    uint32_t *addr,
    size_t timeout_ms)
{
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 0, 1) != 0)
  {
    if (timeout_ms != 0 && global_timer_ns() > deadline)
    {
      return false;
    }
  }
  return true;
}

template <MemOpSem Sem>
__device__ __forceinline__ bool try_wait_signal(
    uint32_t *addr,
    size_t timeout_ms)
{
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 1, 0) != 1)
  {
    if (timeout_ms != 0 && global_timer_ns() > deadline)
    {
      return false;
    }
  }
  return true;
}

template <MemOpSem Sem>
__device__ __forceinline__ void put_signal(uint32_t *addr)
{
  while (cas<Sem>(addr, 0, 1) != 0)
    ;
}

template <MemOpSem Sem>
__device__ __forceinline__ void wait_signal(uint32_t *addr)
{
  while (cas<Sem>(addr, 1, 0) != 1)
    ;
}

template <MemOpSem Sem>
__device__ __forceinline__ void sync_remote_blocks(
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size);

template <>
__device__ __forceinline__ void sync_remote_blocks<MemOpSem::Relaxed>(
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size)
{
  if (threadIdx.x < world_size)
  {
    auto target_rank = threadIdx.x;
    put_signal<MemOpSem::Relaxed>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<MemOpSem::Relaxed>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <>
__device__ __forceinline__ void sync_remote_blocks<MemOpSem::AcqRel>(
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size)
{
  if (threadIdx.x < world_size)
  {
    auto target_rank = threadIdx.x;
    put_signal<MemOpSem::Release>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<MemOpSem::Acquire>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <int Size>
union Vec;

template <>
union Vec<4>
{
  uint16_t u16[2];
  uint32_t u32, as_scalar;
  float f32;
};

template <>
union Vec<8>
{
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, as_scalar;
  float f32[2];
};

template <>
union alignas(16) Vec<16>
{
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  uint4 u128, as_scalar;
  float f32[4];
};

template <typename T>
struct MultimemLdReduce
{
  template <int Alignment>
  __device__ __inline__ Vec<Alignment> operator()(T *mc_ptr)
  {
    static_assert(dependent_false<T>);
  }
};

template <int Alignment, typename T>
__device__ __inline__ Vec<Alignment> multimem_ld_reduce_add(T *mc_ptr)
{
  MultimemLdReduce<T> functor;
  return functor.template operator()<Alignment>(mc_ptr);
}

#if defined(USE_ROCM) || !defined(NVCC_SUPPORTS_MULTICAST)
#define SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(type, asm_type, acc_prec) \
  template <>                                                          \
  struct MultimemLdReduce<type>                                        \
  {                                                                    \
    template <int Alignment>                                           \
    __device__ __inline__ Vec<Alignment> operator()(type *mc_ptr)      \
    {                                                                  \
      CUDA_KERNEL_ASSERT(false);                                       \
    }                                                                  \
  };
#else
#define SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(type, asm_type, acc_prec)    \
  template <>                                                             \
  struct MultimemLdReduce<type>                                           \
  {                                                                       \
    template <int Alignment>                                              \
    __device__ __inline__ Vec<Alignment> operator()(type *mc_ptr)         \
    {                                                                     \
      Vec<Alignment> vec;                                                 \
      if constexpr (Alignment == 16)                                      \
      {                                                                   \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec          \
            ".v4" asm_type " {%0,%1,%2,%3}, [%4];"                        \
            : "=r"(vec.u32[0]),                                           \
              "=r"(vec.u32[1]),                                           \
              "=r"(vec.u32[2]),                                           \
              "=r"(vec.u32[3])                                            \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      }                                                                   \
      else if constexpr (Alignment == 8)                                  \
      {                                                                   \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec          \
            ".v2" asm_type " {%0,%1}, [%2];"                              \
            : "=r"(vec.u32[0]), "=r"(vec.u32[1])                          \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      }                                                                   \
      else if constexpr (Alignment == 4)                                  \
      {                                                                   \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec asm_type \
            " %0, [%1];"                                                  \
            : "=r"(vec.u32)                                               \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      }                                                                   \
      return vec;                                                         \
    }                                                                     \
  };
#endif

  SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(at::BFloat16, ".bf16x2", ".acc::f32");
  SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(c10::Half, ".f16x2", ".acc::f32");
  SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(float, ".f32", "");

  template <int Alignment, typename T>
  __device__ __inline__ void multimem_st(T *mc_ptr, Vec<Alignment> &vec)
  {
#if defined(USE_ROCM) || !defined(NVCC_SUPPORTS_MULTICAST)
    CUDA_KERNEL_ASSERT(false);
#else
    if constexpr (Alignment == 16)
    {
      asm("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
          :
          : "l"(mc_ptr),
            "r"(vec.u32[0]),
            "r"(vec.u32[1]),
            "r"(vec.u32[2]),
            "r"(vec.u32[3])
          : "memory");
    }
    else if constexpr (Alignment == 8)
    {
      asm("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};"
          :
          : "l"(mc_ptr), "r"(vec.u32[0]), "r"(vec.u32[1])
          : "memory");
    }
    else if constexpr (Alignment == 4)
    {
      asm("multimem.st.relaxed.sys.global.f32 [%0], %1;"
          :
          : "l"(mc_ptr), "r"(vec.u32)
          : "memory");
    }
    else
    {
      static_assert(dependent_false<T>);
    }
#endif
  }
/* 
* ********************************************************** *
* Code copied from Pytorch codebase                          *
* End:: Multimem All Reduce Meta kernel                      *
* ********************************************************** *
*/

/* 
* ********************************************************* *
* BLOCK REDUCE SUM                                          *
* ********************************************************* *
*/
template <typename T, int NUM>
__inline__ __device__ T warpReduceSum(T *val)
{
#pragma unroll
  for (int i = 0; i < NUM; i++)
  {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSum(T *val)
{
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<T, NUM>(val);

  if (lane == 0)
  {
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++)
  {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSum<T, NUM>(val);
  return (T)0.0f;
}
}  // namespace vllm
