/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
    \brief Architecture-specific operators on memory
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/platform/platform.h"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The bytes of loading
    int LoadBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always
    >
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11)) &&                                  \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  #define CUTLASS_ENABLE_L2_PREFETCH 1
#else
  #define CUTLASS_ENABLE_L2_PREFETCH 0
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// The redundant mov PTX instruction is used to enforce the compiler to
// keep the initializing code before ld.global
template <typename AccessType>
struct global_load<AccessType,
                   32,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 *data = reinterpret_cast<uint4 *>(&D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %9, 0;\n"
        "  mov.b32 %0, %10;\n"
        "  mov.b32 %1, %11;\n"
        "  mov.b32 %2, %12;\n"
        "  mov.b32 %3, %13;\n"
        "  mov.b32 %4, %14;\n"
        "  mov.b32 %5, %15;\n"
        "  mov.b32 %6, %16;\n"
        "  mov.b32 %7, %17;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%8];\n"
        "  @p ld.global.L2::128B.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#else
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n"
        "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#endif
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w),
          "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data[0].x), "r"(data[0].y),
          "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y),
          "r"(data[1].z), "r"(data[1].w), "l"(((uint8_t *)ptr) + 16));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   32,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 *data = reinterpret_cast<uint4 *>(&D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %9, 0;\n"
        "  mov.b32 %0, %10;\n"
        "  mov.b32 %1, %11;\n"
        "  mov.b32 %2, %12;\n"
        "  mov.b32 %3, %13;\n"
        "  mov.b32 %4, %14;\n"
        "  mov.b32 %5, %15;\n"
        "  mov.b32 %6, %16;\n"
        "  mov.b32 %7, %17;\n"
        "  @p ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%8];\n"
        "  @p ld.global.lu.v4.u32 {%4, %5, %6, %7}, [%18];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w),
          "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data[0].x), "r"(data[0].y),
          "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y),
          "r"(data[1].z), "r"(data[1].w), "l"(((uint8_t *)ptr) + 16));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#else
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#endif
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   8,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint2 &data = reinterpret_cast<uint2 &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %3, 0;\n"
        "  mov.b32 %0, %4;\n"
        "  mov.b32 %1, %5;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.v2.u32 {%0, %1}, [%2];\n"
#else
        "  @p ld.global.v2.u32 {%0, %1}, [%2];\n"
#endif
        "}\n"
        : "=r"(data.x), "=r"(data.y)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   8,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint2 &data = reinterpret_cast<uint2 &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %3, 0;\n"
        "  mov.b32 %0, %4;\n"
        "  mov.b32 %1, %5;\n"
        "  @p ld.global.lu.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   4,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  unsigned &data = reinterpret_cast<unsigned &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  mov.b32 %0, %3;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.u32 %0, [%1];\n"
#else
        "  @p ld.global.u32 %0, [%1];\n"
#endif
        "}\n"
        : "=r"(data)
        : "l"(ptr), "r"((int)pred_guard), "r"(data));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   4,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  unsigned &data = reinterpret_cast<unsigned &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  mov.b32 %0, %3;\n"
        "  @p ld.global.lu.u32 %0, [%1];\n"
        "}\n"
        : "=r"(data)
        : "l"(ptr), "r"((int)pred_guard), "r"(data));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   2,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint16_t &data = reinterpret_cast<uint16_t &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  mov.b16 %0, %3;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.u16 %0, [%1];\n"
#else
        "  @p ld.global.u16 %0, [%1];\n"
#endif
        "}\n"
        : "=h"(data)
        : "l"(ptr), "r"((int)pred_guard), "h"(data));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   2,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint16_t &data = reinterpret_cast<uint16_t &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  mov.b16 %0, %3;\n"
        "  @p ld.global.lu.u16 %0, [%1];\n"
        "}\n"
        : "=h"(data)
        : "l"(ptr), "r"((int)pred_guard), "h"(data));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   1,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    if (pred_guard) D = *(reinterpret_cast<AccessType const *>(ptr));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes
    >
struct global_store;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////


template <typename AccessType>
struct global_store<AccessType, 64> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint4 const *data = reinterpret_cast<uint4 const *>(&D);

  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %5, 0;\n"
      "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
      "  @p st.global.v4.u32 [%6], {%7, %8, %9, %10};\n"
      "  @p st.global.v4.u32 [%11], {%12, %13, %14, %15};\n"
      "  @p st.global.v4.u32 [%16], {%17, %18, %19, %20};\n"
      "}\n"
      :
      : "l"(ptr), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
        "r"(data[0].w), "r"((int)pred_guard), "l"(((uint8_t *)ptr) + 16),
        "r"(data[1].x), "r"(data[1].y), "r"(data[1].z), "r"(data[1].w), 
        "l"(((uint8_t *)ptr) + 32),
        "r"(data[2].x), "r"(data[2].y), "r"(data[2].z), "r"(data[2].w),
        "l"(((uint8_t *)ptr) + 48),
        "r"(data[3].x), "r"(data[3].y), "r"(data[3].z), "r"(data[3].w));
  }
};


template <typename AccessType>
struct global_store<AccessType, 32> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint4 const *data = reinterpret_cast<uint4 const *>(&D);

  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %5, 0;\n"
      "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
      "  @p st.global.v4.u32 [%6], {%7, %8, %9, %10};\n"
      "}\n"
      :
      : "l"(ptr), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
        "r"(data[0].w), "r"((int)pred_guard), "l"(((uint8_t *)ptr) + 16),
        "r"(data[1].x), "r"(data[1].y), "r"(data[1].z), "r"(data[1].w));
  }
};

template <typename AccessType>
struct global_store<AccessType, 16> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint4 const &data = reinterpret_cast<uint4 const &>(D);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %5, 0;\n"
      "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
      "}\n"
      :
      : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w), "r"((int)pred_guard));
  }
};

template <typename AccessType>
struct global_store<AccessType, 8> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint2 const &data = reinterpret_cast<uint2 const &>(D);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %3, 0;\n"
      "  @p st.global.v2.u32 [%0], {%1, %2};\n"
      "}\n"
      :
      : "l"(ptr), "r"(data.x), "r"(data.y), "r"((int)pred_guard));
  }
};

template <typename AccessType>
struct global_store<AccessType, 4> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p st.global.u32 [%0], %1;\n"
      "}\n"
      :
      : "l"(ptr), "r"(data), "r"((int)pred_guard));
  }
};

template <typename AccessType>
struct global_store<AccessType, 2> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint16_t const &data = reinterpret_cast<uint16_t const &>(D);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p st.global.u16 [%0], %1;\n"
      "}\n"
      :
      : "l"(ptr), "h"(data), "r"((int)pred_guard));
  }
};

template <typename AccessType>
struct global_store<AccessType, 1> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    if (pred_guard) *(reinterpret_cast<AccessType *>(ptr)) = D;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// ld.shared
template <int Bytes>
CUTLASS_DEVICE
void shared_load(void *dst, uint32_t ptr);

/// ld.shared - 16b
template <>
CUTLASS_DEVICE
void shared_load<2>(void *dst, uint32_t ptr) {
  asm volatile("ld.shared.u16 %0, [%1];\n"
    : "=h"(*reinterpret_cast<uint16_t *>(dst))
    : "r"(ptr));
}

/// ld.shared - 32b
template <>
CUTLASS_DEVICE
void shared_load<4>(void *dst, uint32_t ptr) {
  asm volatile("ld.shared.u32 %0, [%1];\n"
    : "=r"(*reinterpret_cast<uint32_t *>(dst))
    : "r"(ptr));
}

/// ld.shared - 64b
template <>
CUTLASS_DEVICE
void shared_load<8>(void *dst, uint32_t ptr) {
  uint2 *dst_u64 = reinterpret_cast<uint2 *>(dst);
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n"
    :
      "=r"(dst_u64->x),
      "=r"(dst_u64->y)
    : "r"(ptr));
}

/// ld.shared - 128b
template <>
CUTLASS_DEVICE
void shared_load<16>(void *dst, uint32_t ptr) {
  uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// st.shared
template <int Bytes>
CUTLASS_DEVICE
void shared_store(uint32_t ptr, void const *src);

/// st.shared - 16b
template <>
CUTLASS_DEVICE
void shared_store<2>(uint32_t ptr, void const *src) {
  asm volatile("st.shared.u16 [%0], %1;\n"
    : :
    "r"(ptr),
    "h"(*reinterpret_cast<uint16_t const *>(src))
  );
}

/// st.shared - 32b
template <>
CUTLASS_DEVICE
void shared_store<4>(uint32_t ptr, void const *src) {
  asm volatile("st.shared.u32 [%0], %1;\n"
    : :
    "r"(ptr),
    "r"(*reinterpret_cast<uint32_t const  *>(src))
  );
}

/// st.shared - 64b
template <>
CUTLASS_DEVICE
void shared_store<8>(uint32_t ptr, void const *src) {
  uint2 const *dst_u64 = reinterpret_cast<uint2 const *>(src);
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
    : :
      "r"(ptr),
      "r"(dst_u64->x),
      "r"(dst_u64->y)
    );
}

/// st.shared - 128b
template <>
CUTLASS_DEVICE
void shared_store<16>(uint32_t ptr, void const *src) {
  uint4 const *dst_u128 = reinterpret_cast<uint4 const *>(src);
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
    : :
      "r"(ptr),
      "r"(dst_u128->x),
      "r"(dst_u128->y),
      "r"(dst_u128->z),
      "r"(dst_u128->w)
    );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/memory_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
