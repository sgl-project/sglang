/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/numeric_types.h"

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#include <cinttypes>
#endif

#include <cute/config.hpp>

#include <cute/arch/util.hpp>   // cute::cast_smem_ptr_to_uint
#include <cute/arch/config.hpp> // CUTE_ARCH_TMA_SMxx_ENABLED
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90.hpp>

#include <cute/container/alignment.hpp>
#include <cute/container/bit_field.hpp>
#include <cute/container/array.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Barriers are 64-bit of user-managed information used in broadly two types syncronization patterns
/// 1) arrive/wait on threads (usage: cp.async and warp-specialized kernels)
/// 2) transaction-based (usage: TMA transaction where a CTA issues one transaction)
//////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize barrier present in shared memory
CUTE_HOST_DEVICE
void
initialize_barrier(uint64_t& smem_barrier,                 // 64 bits user-manged barrier in smem
                   int thread_count = 1)                   // Thread count expected to arrive/wait on this barrier
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(thread_count));
#endif
}

// Set the number of bytes transfered per transaction and perform an arrive operation as well
CUTE_HOST_DEVICE
void
set_barrier_transaction_bytes(uint64_t& smem_barrier,      // 64 bits user-manged barrier in smem
                              uint32_t bytes)              // Number of bytes transfered by per TMA transaction
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(bytes));
#endif
}

// Barrier wait
CUTE_HOST_DEVICE
void
wait_barrier(uint64_t& smem_barrier,                       // 64 bits user-manged barrier in smem
             int phase_bit)                                // Current phase bit the barrier waiting to flip
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
    "{\n"
    ".reg .pred                P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
    "@P1                       bra DONE;\n"
    "bra                   LAB_WAIT;\n"
    "DONE:\n"
    "}\n"
    :: "r"(smem_int_ptr),
       "r"(phase_bit));

#endif
}

// Barrier arrive
CUTE_HOST_DEVICE
void
arrive_barrier(uint64_t& smem_barrier)                      // 64 bits user-manged barrier in smem
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
    "{\n"
    ".reg .b64 state; \n"
    "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
    "}\n"
    :: "r"(smem_int_ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TMA Descriptor and utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace TMA {

enum class SmemSwizzleBits : uint8_t {
  DISABLE = 0,
  B32 = 1,
  B64 = 2,
  B128 = 3,
};

enum class SmemSwizzleBase : uint8_t {
  SWIZZLE_BASE_16B         = 0,
  
  SWIZZLE_BASE_32B         = 1,
  SWIZZLE_BASE_32B_FLIP_8B = 2,
  SWIZZLE_BASE_64B         = 3,
  
};

enum class OOBFill : uint8_t {
  ZERO = 0,
  CONSTANT = 1,
};

CUTE_HOST_DEVICE char const* to_string(OOBFill const& t) {
  switch (t) {
    case OOBFill::ZERO:     return "ZERO";
    case OOBFill::CONSTANT: return "CONSTANT";
  }
  return nullptr;
}

enum class L2Promotion : uint8_t {
  DISABLE = 0,
  B64 = 1,
  B128 = 2,
  B256 = 3,
};

CUTE_HOST_DEVICE char const* to_string(L2Promotion const& t) {
  switch (t) {
    case L2Promotion::DISABLE: return "DISABLE";
    case L2Promotion::B64:     return "B64";
    case L2Promotion::B128:    return "B128";
    case L2Promotion::B256:    return "B256";
  }
  return nullptr;
}

// Aux parameters which are independent with the problem size
struct DescriptorAuxParams {
  OOBFill     oobfill_     = OOBFill::ZERO;
  L2Promotion l2promo_     = L2Promotion::DISABLE;
};

enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};


enum class CacheHintSm100 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};


#if (__CUDACC_VER_MAJOR__ >= 12)

#if !defined(__CUDACC_RTC__)
/// @return The TMA descriptor datatype enum corresponding to T.
template <class T>
inline CUtensorMapDataType
to_CUtensorMapDataType() {
  if constexpr (is_same_v<T,       int8_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T,      uint8_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T, float_e4m3_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T, float_e5m2_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T, float_ue8m0_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;   } else
  if constexpr (is_same_v<T, type_erased_dynamic_float8_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;} else 
  if constexpr (is_same_v<T,     uint16_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT16;   } else
  if constexpr (is_same_v<T,     uint32_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT32;   } else
  if constexpr (is_same_v<T,     uint64_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT64;   } else
  if constexpr (is_same_v<T,      int32_t>) { return CU_TENSOR_MAP_DATA_TYPE_INT32;    } else
  if constexpr (is_same_v<T,      int64_t>) { return CU_TENSOR_MAP_DATA_TYPE_INT64;    } else
  if constexpr (is_same_v<T,       half_t>) { return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;  } else
  if constexpr (is_same_v<T,        float>) { return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;  } else
  if constexpr (is_same_v<T,       double>) { return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;  } else
  if constexpr (is_same_v<T,   bfloat16_t>) { return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; } else
  if constexpr (is_same_v<T,   tfloat32_t>) { return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32; } else
  #if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ > 6)))
  if constexpr (is_same_v<T, float_e2m1_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;  } else
  if constexpr (is_same_v<T, float_e2m3_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, float_e3m2_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, type_erased_dynamic_float4_t>)    { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;  } else
  if constexpr (is_same_v<T, type_erased_dynamic_float6_t>)    { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::float_e2m1_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::float_e2m3_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::float_e3m2_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::type_erased_dynamic_float4_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::type_erased_dynamic_float6_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  #endif

  { static_assert(sizeof(T) < 0, "Unknown TMA Format!"); }
}

inline CUtensorMapSwizzle
to_CUtensorMapSwizzle(SmemSwizzleBits const& t, SmemSwizzleBase const& b) {
  switch (t) {
    default: throw std::runtime_error("Unsupported pair of SmemSwizzleBits and SmemSwizzleBase!");
    case SmemSwizzleBits::DISABLE: 
      assert((b == SmemSwizzleBase::SWIZZLE_BASE_16B) && "Expected 16B swizzle base for 0B swizzle bits.");
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case SmemSwizzleBits::B32:
      assert((b == SmemSwizzleBase::SWIZZLE_BASE_16B) && "Expected 16B swizzle base for 32B swizzle bits.");
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case SmemSwizzleBits::B64:
      assert((b == SmemSwizzleBase::SWIZZLE_BASE_16B) && "Expected 16B swizzle base for 64B swizzle bits.");
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case SmemSwizzleBits::B128:
      switch (b) {
        default: throw std::runtime_error("Unsupported pair of SmemSwizzleBits and SmemSwizzleBase!");
        case SmemSwizzleBase::SWIZZLE_BASE_16B: return CU_TENSOR_MAP_SWIZZLE_128B;
        
        #if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ > 6)))
        case SmemSwizzleBase::SWIZZLE_BASE_32B: return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
        case SmemSwizzleBase::SWIZZLE_BASE_64B: return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
        #endif
      }
  }
}

inline CUtensorMapFloatOOBfill
to_CUtensorMapFloatOOBfill(OOBFill const& t) {
  switch(t) {
    default:                throw std::runtime_error("Unknown OOBFill!");
    case OOBFill::ZERO:     return CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    case OOBFill::CONSTANT: return CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
  }
}

inline CUtensorMapL2promotion
to_CUtensorMapL2promotion(L2Promotion const& t) {
  switch(t) {
    default: throw std::runtime_error("Unknown L2Promotion!");
    case L2Promotion::DISABLE: return CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case L2Promotion::B64:     return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case L2Promotion::B128:    return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case L2Promotion::B256:    return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }
}

#endif // !defined(__CUDACC_RTC__)

#endif // (__CUDACC_VER_MAJOR__ >= 12)

} // end namespace TMA

#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;
  using Im2ColTmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };
  using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
/// Initiates a TensorMap Prefetch
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
prefetch_tma_descriptor(TmaDescriptor const* desc_ptr)
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state space: const or param)
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a TensorMap modification (by each field)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Replace tensor pointer directly in GMEM
CUTE_HOST_DEVICE
void
tma_descriptor_replace_addr_in_global_mem(TmaDescriptor const* desc_ptr,
                                          void const* const new_tensor_ptr)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint64_t const new_desc_addr = reinterpret_cast<uint64_t>(new_tensor_ptr);
  asm volatile (
    "tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
    :: "l"(gmem_int_desc), "l"(new_desc_addr));
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

// Replace tensor pointer by bringing the tensormap from GMEM into the shared memory
CUTE_HOST_DEVICE
void
tma_descriptor_replace_addr_in_shared_mem(TmaDescriptor& smem_desc,
                                          void const* const new_tensor_ptr)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const new_desc_addr = reinterpret_cast<uint64_t>(new_tensor_ptr);
  asm volatile (
    "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
    :: "r"(smem_int_desc), "l"(new_desc_addr));
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

// Replace tensor dims and strides for GEMMs by bringing the tensormap from GMEM into the shared memory
CUTE_HOST_DEVICE
void
tma_descriptor_replace_dims_strides_in_shared_mem(TmaDescriptor                 & smem_desc,
                                                  cute::array<uint32_t, 5> const& prob_shape,
                                                  cute::array<uint64_t, 5> const& prob_stride)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const smem_int64_desc = 0;
  asm volatile (
    "cvt.u64.u32 %0, %1;"
    :: "l"(smem_int64_desc), "r"(smem_int_desc));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[0]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[1]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 2, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[2]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 3, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[3]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 4, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[4]));
  // Strides must be a multiple of 16. Also, stride for the intermost dimension is implicitly 1
  #if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 5)))
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[1]));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 1, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[2]));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 2, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[3]));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 3, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[4]));
  #else
  // 4 LSBs are not included
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[1] >> 4));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 1, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[2] >> 4));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 2, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[3] >> 4));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 3, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[4] >> 4));
  #endif
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a fused copy and fence operation (needed when modifying tensormap in shared memory)
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
tma_descriptor_cp_fence_release(TmaDescriptor const* gmem_desc_ptr, TmaDescriptor& smem_desc)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
  asm volatile (
    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
    :: "l"(gmem_int_desc), "r"(smem_int_desc));
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a release fence operation (needed when modifying tensormap directly in GMEM)
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
tma_descriptor_fence_release()
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  asm volatile ("fence.proxy.tensormap::generic.release.gpu;");
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a acquire fence operation
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
tma_descriptor_fence_acquire(TmaDescriptor const* desc_ptr)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
    :
    : "l"(gmem_int_desc)
    : "memory");
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

///////////////////////////////////////////////////////////////////////////////

} // end namespace cute
