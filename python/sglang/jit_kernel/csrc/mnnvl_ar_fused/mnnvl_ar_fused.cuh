/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mnnvl_ar_fused_compat.cuh"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

struct AllReduceFusionParams {
  int nRanks;
  int rank;
  int numTokens;
  int tokenDim;
  void** bufferPtrsDev;
  void* bufferPtrLocal;
  void* multicastPtr;
  uint32_t* bufferFlags;
  bool rmsNormFusion;
  bool launchWithPdl;

  void const* input;
  void const* residualIn;
  void const* gamma;
  double epsilon;
  // 0 for standard RMSNorm (out = gamma * x * rsqrt(...)),
  // 1 for Gemma / Qwen3.5 (out = (1 + gamma) * x * rsqrt(...)).
  float weightBias = 0.f;

  void* residualOut;
  void* output;
  cudaStream_t stream = nullptr;
};

namespace utils {

constexpr uint16_t kNEGZERO_FP16 = 0x8000U;
constexpr uint32_t kNEGZERO_FP32 = 0x80000000U;

template <typename T>
union Fp16BitCast {
  T mFp;
  uint16_t mInt;

  constexpr Fp16BitCast() : mInt(0) {}

  constexpr Fp16BitCast(T val) : mFp(val) {}

  constexpr Fp16BitCast(uint16_t val) : mInt(val) {}
};

template <typename T>
inline __device__ float toFloat(T val) {
  return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <>
inline __device__ float toFloat<__nv_half>(__nv_half val) {
  return __half2float(val);
}

template <typename T>
inline __device__ T fromFloat(float val) {
  return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
inline __device__ __nv_half fromFloat<__nv_half>(float val) {
  return __float2half(val);
}

template <typename T>
static constexpr __device__ __host__ T negZero() {
  if constexpr (std::is_same_v<T, float>) {
    return -0.0F;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(kNEGZERO_FP16).mFp;
  } else {
    static_assert(sizeof(T) == 0, "negativeZero not specialized for this type");
  }
  return T{};  // Never reached, but needed for compilation
}

// WARNING: the Lamport sentinel is a *bit pattern* (fp32 -0.0 = 0x80000000;
// fp16/bf16 -0.0 = 0x8000). Always compare bit-exact -- do NOT fall back to
// `val == 0.F && signbit(val)`. nvcc emits `setp.eq.f32` with `.ftz=true`
//  which flushes fp32 subnormal operands to +/-0.0 *before*
// the equality while signbit() still reads bit 31, so any fp32 negative
// subnormal pattern (e.g. 0x80010000, which appears when bf16 negative
// subnormals 0x8001-0x807F land in the high half of a 4-byte poll load) would
// falsely match the sentinel and deadlock the polling loop.
template <typename T>
static inline __device__ bool isNegZero(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return __float_as_uint(val) == kNEGZERO_FP32;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(val).mInt == kNEGZERO_FP16;
  } else {
    static_assert(sizeof(T) == 0, "isNegZero not specialized for this type");
  }
  return false;  // Never reached, but needed for compilation
}

template <typename PackedType, typename T>
constexpr __device__ __host__ PackedType getPackedLamportInit() {
  static_assert(sizeof(PackedType) % sizeof(T) == 0, "PackedType size must be divisible by T size");
  constexpr int kNumElements = sizeof(PackedType) / sizeof(T);

  union PackedT {
    PackedType mPacked;
    std::array<T, kNumElements> mElements;

    constexpr PackedT() : mElements{} {
      for (int i = 0; i < kNumElements; i++) {
        mElements[i] = negZero<T>();
      }
    }
  };

  PackedT initValue{};
  return initValue.mPacked;
}

// A helper class to get the correct base pointer for a given layout
struct LamportBufferLayout {
  uint32_t numStages = 1;
  uint32_t bytesPerBuffer = 0;
  static constexpr uint32_t sNumLamportBuffers = 3;

  // Implicitly inlined
  [[nodiscard]] __device__ __host__ size_t getTotalBytes() const {
    return numStages * static_cast<size_t>(bytesPerBuffer / numStages) * sNumLamportBuffers;
  }

  // Implicitly inlined
  [[nodiscard]] __device__ __host__ void*
  getStagePtr(void* bufferBasePtr, uint32_t lamportIndex, uint32_t stageIndex) const {
    // Typecast to avoid warnings
    return reinterpret_cast<void*>(
        reinterpret_cast<char*>(bufferBasePtr) +
        static_cast<size_t>((lamportIndex * numStages + stageIndex) * static_cast<size_t>(bytesPerBuffer / numStages)));
  }
};
// Current Index
// Dirty Index
// bytes_per_buffer
// Dirty num_stages
// Dirty bytes_to_clear = {stage0, stage1, stage2, stage3}  # We fix this to 4 stages
// offset_access_ptr

namespace cg = cooperative_groups;

// PackedType is the one used in kernel for Lamport buffer (LDG.128 or LDG.64)
template <typename PackedType = float4>
__device__ struct __attribute__((aligned(32))) LamportFlags {
 public:
  __device__ explicit LamportFlags(uint32_t* bufferFlags, uint32_t numStages = 1)
      : mBufferFlagsPtr(bufferFlags), mFlagAccessPtr(&bufferFlags[8]) {
    mCurBufferLayout.numStages = numStages;
    uint4 flag = reinterpret_cast<uint4*>(bufferFlags)[0];
    mCurrentIndex = flag.x;
    mDirtyIndex = flag.y;
    // Buffer size is unchanged as the flag should be coupled to each buffer
    mCurBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.numStages = flag.w;
    *reinterpret_cast<uint4*>(&mBytesToClear) = reinterpret_cast<uint4*>(bufferFlags)[1];
  }

  // Return the base pointer of the lamport buffer indexed by mCurrentIndex and the stageIdx
  [[nodiscard]] __device__ void* getCurLamportBuf(void* bufferBasePtr, int stageIdx = 0) const {
    return mCurBufferLayout.getStagePtr(bufferBasePtr, mCurrentIndex, stageIdx);
  }

  // Fill the dirty lamport buffer with the init value; Use stageIdx to select the stage to clear,
  // -1 to clear all
  // FIXME: Current kernel may use less stages than the dirty numStages; How to guarantee the
  // correctness? CAUTION: This function requires all threads in the grid to participate and ASSUME
  // 1D thread block layout!
  __device__ void clearDirtyLamportBuf(void* bufferBasePtr, int stageIdx = -1) {
    // Rasterize the threads to 1D for flexible clearing

    uint32_t globalCtaIdx = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t globalTid = globalCtaIdx * blockDim.x + threadIdx.x;
    uint32_t numThreads = gridDim.x * gridDim.y * blockDim.x;

    if (stageIdx == -1) {
      // Clear all stages
      for (uint32_t i = 0; i < mDirtyBufferLayout.numStages; i++) {
        clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[i], mDirtyIndex, i);
      }
    } else if (stageIdx < mDirtyBufferLayout.numStages) {
      clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[stageIdx], mDirtyIndex, stageIdx);
    }
  }

  __device__ void ctaArrive() {
    int tid{0};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    cg::cluster_group cluster = cg::this_cluster();
    // We update the atomic counter per cluster
    tid = cluster.thread_rank();
    cluster.sync();
#else
    tid = threadIdx.x;
    __syncthreads();
#endif
    if (tid == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
      asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
      asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#else
      atomicAdd(mFlagAccessPtr, 1);
#endif
    }
  }

  __device__ void waitAndUpdate(uint4 bytesToClearPerStage) {
    bool isLastCtaT0{false};
    int targetCount{0};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cg::grid_group grid = cg::this_grid();
    // Use the first thread instead of the last thread as the last thread may exit early
    isLastCtaT0 = grid.thread_rank() == 0;
    targetCount = grid.num_clusters();
#else
    isLastCtaT0 = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0;
    targetCount = gridDim.x * gridDim.y;
#endif
    if (isLastCtaT0) {
      uint4* flagPtr = reinterpret_cast<uint4*>(mBufferFlagsPtr);
      while (*reinterpret_cast<uint32_t volatile*>(mFlagAccessPtr) < targetCount) {
      }
      // 'Current' becomes 'Dirty'
      flagPtr[0] = {
          (mCurrentIndex + 1) % 3,          // Current index
          mCurrentIndex,                    // Dirty index
          mCurBufferLayout.bytesPerBuffer,  // Buffer size
          mCurBufferLayout.numStages};      // Dirty - Number of stages
      flagPtr[1] = bytesToClearPerStage;
      *mFlagAccessPtr = 0;
    }
  }

 private:
  uint32_t* mBufferFlagsPtr;
  uint32_t* mFlagAccessPtr;

  uint32_t mCurrentIndex, mDirtyIndex;
  // So that we can access it with uint4
  alignas(16) std::array<uint32_t, 4> mBytesToClear;
  LamportBufferLayout mCurBufferLayout, mDirtyBufferLayout;

  inline __device__ void clearPackedBuf(
      void* bufferBasePtr,
      uint32_t globalTid,
      uint32_t numThreads,
      uint32_t bytesToClear,
      uint8_t dirtyIndex,
      uint8_t stageIdx) {
    // Round up to the float4 boundary
    uint32_t clearBoundary = ceil_div<uint32_t>(bytesToClear, sizeof(PackedType));
    for (uint32_t packedIdx = globalTid; packedIdx < clearBoundary; packedIdx += numThreads) {
      reinterpret_cast<PackedType*>(mDirtyBufferLayout.getStagePtr(bufferBasePtr, dirtyIndex, stageIdx))[packedIdx] =
          getPackedLamportInit<PackedType, float>();
    }
  }
};

template <typename PackedType, typename T>
union PackedVec {
  PackedType packed;
  T elements[sizeof(PackedType) / sizeof(T)];

  __device__ PackedVec& operator+=(PackedVec& other) {
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
      elements[i] += other.elements[i];
    }
    return *this;
  }

  __device__ PackedVec operator+(PackedVec& other) {
    PackedVec result;
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
      result.elements[i] = elements[i] + other.elements[i];
    }
    return result;
  }
};

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr) {
  return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr) {
  return *reinterpret_cast<PackedType const*>(ptr);
}

template <typename PackedType>
inline __device__ PackedType loadPackedVolatile(void const* ptr) {
  static_assert(sizeof(PackedType) == 0, "Not implemented");
  return PackedType{};
}

template <>
inline __device__ float4 loadPackedVolatile<float4>(void const* ptr) {
  float4 returnValue;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(returnValue.x), "=f"(returnValue.y), "=f"(returnValue.z), "=f"(returnValue.w)
               : "l"(ptr));
  return returnValue;
}

template <>
inline __device__ float2 loadPackedVolatile<float2>(void const* ptr) {
  float2 returnValue;
  asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(returnValue.x), "=f"(returnValue.y) : "l"(ptr));
  return returnValue;
}

template <typename T_IN>
inline __device__ void copyF4(T_IN* dst, T_IN const* src) {
  float4* dst4 = reinterpret_cast<float4*>(dst);
  float4 const* src4 = reinterpret_cast<float4 const*>(src);
  __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;
uint32_t constexpr kFINAL_MASK = 0xffffffff;

template <typename T>
inline __device__ T warpReduceSumFull(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(kFINAL_MASK, val, mask, kWARP_SIZE);
  }
  return val;
}

template <typename T>
inline __device__ T warpReduceSumPartial(T val) {
  int laneId = threadIdx.x & kLANE_ID_MASK;
  // We make sure only the last warp will call this function
  int warpSize = blockDim.x - (threadIdx.x & ~(kWARP_SIZE - 1));
  unsigned int active_mask = (1U << warpSize) - 1;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    int targetLane = laneId ^ mask;
    auto tmp = __shfl_xor_sync(active_mask, val, mask, kWARP_SIZE);
    val += targetLane < warpSize ? tmp : 0;
  }
  return val;
}

// SYNC:
//  - True: share the sum across all threads
//  - False: only thread 0 get the sum; Other thread's value is undefined.
template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val) {
  __shared__ T smem[kWARP_SIZE];
  int laneId = threadIdx.x & kLANE_ID_MASK;
  int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
  int warpNum = (blockDim.x + kWARP_SIZE - 1) >> kLOG2_WARP_SIZE;  // Ceiling division to include partial warps

  val = (warpId == warpNum - 1) ? warpReduceSumPartial(val) : warpReduceSumFull(val);
  if (laneId == 0) {
    smem[warpId] = val;
  }
  __syncthreads();

  if (warpId == 0) {
    val = (laneId < warpNum) ? smem[laneId] : (T)0.f;
    // Need to consider the corner case where we only have one warp and it is partial
    val = (warpNum == 1) ? warpReduceSumPartial(val) : warpReduceSumFull(val);

    if constexpr (SYNC) {
      if (laneId == 0) {
        smem[warpId] = val;
      }
    }
  }
  if constexpr (SYNC) {
    __syncthreads();
    val = smem[0];
  }
  return val;
}

template <typename T>
inline __device__ T blockReduceSumFull(T val) {
  __shared__ T smem[kWARP_SIZE];
  int lane_id = threadIdx.x & kLANE_ID_MASK;
  int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
  int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

  val = warpReduceSumFull(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();

  val = (lane_id < warp_num) ? smem[lane_id] : (T)0.f;
  val = warpReduceSumFull(val);

  return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSum(T val) {
  bool hasPartialWarp = (blockDim.x & kLANE_ID_MASK) != 0;
  if (hasPartialWarp) {
    return blockReduceSumPartial<T, SYNC>(val);
  } else {
    return blockReduceSumFull<T>(val);
  }
}
// A helper function to tune the grid configuration for fused oneshot and rmsnorm kernels
// Return (block_size, cluster_size, loads_per_thread)
std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread, int smVersionMajor) {
  // Start with preferred block_size and cluster_size
  int clusterSize = smVersionMajor >= 9 ? 8 : 1;
  int blockSize = 128;
  // ========================== Adjust the grid configuration ==========================
  int threadsNeeded = ceil_div(dim, eltsPerThread);
  int loadsPerThread = 1;

  blockSize = ceil_div(threadsNeeded, clusterSize);
  if (smVersionMajor >= 9) {
    while (threadsNeeded % clusterSize != 0 && clusterSize > 1) {
      clusterSize /= 2;
    }
    blockSize = ceil_div(threadsNeeded, clusterSize);
    while (blockSize < 128 && clusterSize >= 2) {
      blockSize *= 2;
      clusterSize /= 2;
    }
    int smCount = GetCudaMultiProcessorCount();
    while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512) {
      blockSize *= 2;
      clusterSize /= 2;
    }
  }
  // Trying to scale up use multiple loads or CGA
  while (blockSize > 1024) {
    if (smVersionMajor >= 9) {
      if (clusterSize < 8) {
        clusterSize = clusterSize << 1;
      } else {
        break;
      }
    } else {
      if (loadsPerThread < 8) {
        loadsPerThread += 1;
      } else {
        break;
      }
    }
    blockSize = ceil_div(threadsNeeded, clusterSize * loadsPerThread);
  }
  return {blockSize, clusterSize, loadsPerThread};
}
};  // namespace utils

using utils::blockReduceSum;
using utils::fromFloat;
using utils::isNegZero;
using utils::LamportFlags;
using utils::loadPacked;
using utils::loadPackedVolatile;
using utils::PackedVec;
using utils::toFloat;

template <uint8_t WorldSize, typename T, bool RMSNormFusion = false, typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshotAllreduceFusionKernel(
    T* outputPtr,
    T* prenormedPtr,
    T const* shardPtr,
    T const* residualInPtr,
    T const* gammaPtr,
    T** inputPtrs,
    T* mcastPtr,
    int const numTokens,
    int const tokenDim,
    float epsilon,
    float weightBias,
    int const rank,
    uint32_t* bufferFlags) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
  constexpr uint32_t kELT_SIZE = sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int packedIdx = cluster.thread_rank();
  int token = blockIdx.x;
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

  cudaGridDependencySynchronize();
#else
  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  // Offset w.r.t. the input shard
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;
#endif

  // We only use 1 stage for the oneshot allreduce
  LamportFlags<PackedType> flag(bufferFlags, 1);
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], 0));

  if (packedIdx * kELTS_PER_THREAD >= tokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
    return;
  }

  // ==================== Broadcast tokens to each rank =============================
  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) val.elements[i] = fromFloat<T>(0.f);
  }

  reinterpret_cast<PackedType*>(&stagePtrMcast[token * tokenDim * WorldSize + rank * tokenDim])[packedIdx] = val.packed;

  flag.ctaArrive();
  // ======================= Lamport Sync and clear the output buffer from previous iteration
  // =============================
  flag.clearDirtyLamportBuf(inputPtrs[rank], -1);

  PackedVec<PackedType, float> valuesLamport[WorldSize];
  while (1) {
    bool valid = true;
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
      valuesLamport[r].packed = loadPackedVolatile<PackedType>(
          &stagePtrLocal[token * tokenDim * WorldSize + r * tokenDim + packedIdx * kELTS_PER_THREAD]);

#pragma unroll
      for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++) {
        valid &= !isNegZero(valuesLamport[r].elements[i]);
      }
    }
    if (valid) {
      break;
    }
  }

  auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
  // ======================= Reduction =============================
  float accum[kELTS_PER_THREAD];
  PackedVec<PackedType, T> packedAccum;

#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = toFloat<T>(values[0].elements[i]);
  }

#pragma unroll
  for (int r = 1; r < WorldSize; r++) {
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      accum[i] += toFloat<T>(values[r].elements[i]);
    }
  }

#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    packedAccum.elements[i] = fromFloat<T>(accum[i]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  if constexpr (RMSNormFusion) {
    // =============================== Residual ===============================
    PackedVec<PackedType, T> residualIn;
    residualIn.packed = *reinterpret_cast<PackedType const*>(&residualInPtr[threadOffset]);
    packedAccum += residualIn;
    *reinterpret_cast<PackedType*>(&prenormedPtr[threadOffset]) = packedAccum.packed;
    // =============================== Rmsnorm ================================
    PackedVec<PackedType, T> gamma;
    gamma.packed = *reinterpret_cast<PackedType const*>(&gammaPtr[packedIdx * kELTS_PER_THREAD]);

    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      // FIXME: Use float square if accuracy issue
      threadSum += toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i]);
    }
    float blockSum = blockReduceSum<float, true>(threadSum);

    __shared__ float sharedVal[8];  // Temporary variable to share the sum within block
    float fullSum = blockSum;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1) {
      fullSum = 0.F;
      // Need to reduce over the entire cluster
      int const blockRank = cluster.block_rank();
      if (threadIdx.x < numBlocks) {
        cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
      }
      cluster.barrier_wait(cluster.barrier_arrive());
      for (int i = 0; i < numBlocks; ++i) {
        fullSum += sharedVal[i];
      }
    }
#endif
    float rcpRms = rsqrtf(fullSum / tokenDim + epsilon);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] =
          fromFloat<T>(toFloat<T>(packedAccum.elements[i]) * rcpRms * (weightBias + toFloat<T>(gamma.elements[i])));
    }
  }
  reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = packedAccum.packed;
  flag.waitAndUpdate({static_cast<uint32_t>(numTokens * tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

using utils::adjustGridConfig;

template <typename T>
cudaError_t oneshotAllreduceFusionDispatch(AllReduceFusionParams const& params) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const eltsPerThread = sizeof(float4) / sizeof(T);

  static const int kSMVersionMajor = GetCudaComputeCapability().first;

  auto [blockSize, clusterSize, loadsPerThread] = adjustGridConfig(numTokens, tokenDim, eltsPerThread, kSMVersionMajor);
  dim3 grid(numTokens, clusterSize, 1);

  FLASHINFER_LOG_DEBUG(
      "[MNNVL AllReduceOneShot] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: "
      "%d, "
      "loads_per_thread: %d, "
      "threads_needed: %d",
      numTokens,
      clusterSize,
      blockSize,
      clusterSize,
      loadsPerThread,
      ceil_div(tokenDim, eltsPerThread));

  FLASHINFER_CHECK(
      blockSize <= 1024 && loadsPerThread == 1,
      "Hidden Dimension %d exceeds the maximum supported hidden dimension (%d)",
      tokenDim,
      1024 * (kSMVersionMajor >= 9 ? 8 : 1) * eltsPerThread);

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = clusterSize;
  attrs[1].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{
      .gridDim = grid,
      .blockDim = blockSize,
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = attrs,
      .numAttrs = kSMVersionMajor >= 9 ? 2 : 1,
  };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, RMSNORM)         \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                   \
      &config,                                               \
      &oneshotAllreduceFusionKernel<WORLD_SIZE, T, RMSNORM>, \
      output,                                                \
      residualOut,                                           \
      input,                                                 \
      residualIn,                                            \
      gamma,                                                 \
      ucPtrs,                                                \
      mcPtr,                                                 \
      numTokens,                                             \
      tokenDim,                                              \
      static_cast<float>(params.epsilon),                    \
      params.weightBias,                                     \
      params.rank,                                           \
      params.bufferFlags));
#define DISPATCH_ALLREDUCE_KERNEL(WORLD_SIZE)   \
  if (params.rmsNormFusion) {                   \
    LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, true);  \
  } else {                                      \
    LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, false); \
  }

  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T* residualOut = reinterpret_cast<T*>(params.residualOut);
  T const* input = reinterpret_cast<T const*>(params.input);
  T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
  T const* gamma = reinterpret_cast<T const*>(params.gamma);

  switch (params.nRanks) {
      // FIXME: Do we need other world sizes?
    case 2:
      DISPATCH_ALLREDUCE_KERNEL(2);
      break;
    case 4:
      DISPATCH_ALLREDUCE_KERNEL(4);
      break;
    case 8:
      DISPATCH_ALLREDUCE_KERNEL(8);
      break;
    case 16:
      DISPATCH_ALLREDUCE_KERNEL(16);
      break;
    case 32:
      DISPATCH_ALLREDUCE_KERNEL(32);
      break;
    case 64:
      DISPATCH_ALLREDUCE_KERNEL(64);
      break;
    default:
      FLASHINFER_ERROR(
          "MNNVL AllReduce: unsupported world_size " + std::to_string(params.nRanks) +
          ". Supported sizes: {2, 4, 8, 16, 32, 64}");
      return cudaErrorInvalidValue;
  }
#undef LAUNCH_ALLREDUCE_KERNEL
  return cudaSuccess;
}

enum MNNVLTwoShotStage : uint8_t {
  SCATTER = 0,
  BROADCAST = 1,
  NUM_STAGES = 2,
};

template <uint8_t WorldSize, typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(
    T* outputPtr,
    T const* shardPtr,
    T** inputPtrs,
    T* mcastPtr,
    uint32_t const numTokens,
    uint32_t const tokenDim,
    uint32_t const rank,
    uint32_t* bufferFlags,
    bool const wait_for_results) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
  constexpr uint32_t kELT_SIZE = sizeof(T);

  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  // Offset w.r.t. the input shard
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

  int destRank = token % WorldSize;
  int destTokenOffset = token / WorldSize;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  LamportFlags<PackedType> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);

  T* scatterBufLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER));
  T* scatterBufDest = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
  T* broadcastBufW = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, MNNVLTwoShotStage::BROADCAST));
  T* broadcastBufR = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  // Make sure the clear function is called before OOB thread exits
  if (packedIdx * kELTS_PER_THREAD >= tokenDim) {
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
    return;
  }

  // =============================== Scatter ===============================

  // Load vectorized data
  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) {
      val.elements[i] = fromFloat<T>(0.F);
    }
  }

  // Store vectorized data
  reinterpret_cast<PackedType*>(&scatterBufDest[destTokenOffset * tokenDim * WorldSize + rank * tokenDim])[packedIdx] =
      val.packed;

  flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER);

  // =============================== Reduction and Broadcast ===============================

  if ((token % WorldSize) == rank) {
    int localToken = token / WorldSize;
    float accum[kELTS_PER_THREAD] = {0.F};

    // Use float as we only check each float value for validity
    PackedVec<PackedType, float> valuesLamport[WorldSize];
    while (1) {
      bool valid = true;
#pragma unroll
      for (int r = 0; r < WorldSize; r++) {
        valuesLamport[r].packed = loadPackedVolatile<PackedType>(
            &scatterBufLocal[localToken * tokenDim * WorldSize + r * tokenDim + packedIdx * kELTS_PER_THREAD]);

        // Check validity across all elements
#pragma unroll
        for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++) {
          valid &= !isNegZero(valuesLamport[r].elements[i]);
        }
      }
      if (valid) {
        break;
      }
    }

    // Now we view it as the value for reduction
    auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
#pragma unroll
      for (int i = 0; i < kELTS_PER_THREAD; i++) {
        accum[i] += toFloat<T>(values[r].elements[i]);
      }
    }

    // Store vectorized result
    PackedVec<PackedType, T> packedAccum;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fromFloat<T>(accum[i]);
    }
    reinterpret_cast<PackedType*>(&broadcastBufW[token * tokenDim])[packedIdx] = packedAccum.packed;
  }

  flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST);

  // Optionally wait for results if the next layer isn't doing the Lamport check
  if (wait_for_results) {
    // Update the atomic counter to indicate the block has read the offsets
    flag.ctaArrive();

    PackedVec<PackedType, float> valLamport;
    valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
    while (isNegZero(valLamport.elements[0])) {
      valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
    }
    if (outputPtr) {
      reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = valLamport.packed;
    }

    // Update the buffer flags
    flag.waitAndUpdate(
        {static_cast<uint32_t>(round_up(numTokens, WorldSize) * tokenDim * kELT_SIZE),  // Clear Size for scatter stage
         static_cast<uint32_t>(numTokens * tokenDim * kELT_SIZE),  // Clear Size for broadcast stage
         0,
         0});
    // If not wait for results, we will rely on the following kernel to update the buffer
  }
}

using utils::copyF4;
// This kernel works performant when loads_per_thread is 1.
// For this mode, we are able to support up to 1024 (threads) x 8 (elements) = 8192 hidden
// dimension. There are two options for further scaling up:
//      1. Use CGA if supported. It expands the hidden dimension to 8k x 8 = 64k.
//      2. Set loads_per_thread >1. Which can be used if CGA is not supported. Note that this will
//      be limited by the shared memory size and register count.
template <typename T_IN, typename T_OUT, int LoadsPerThread = 1>
__global__ __launch_bounds__(1024) void rmsNormLamport(
    T_IN* outputPreNorm,
    T_OUT* outputNorm,
    T_IN* bufferInput,
    T_IN const* gamma,
    float epsilon,
    float weightBias,
    T_IN const* residual,
    uint32_t numTokens,
    uint32_t dim,
    uint32_t worldSize,
    uint32_t* bufferFlags) {
  static_assert(std::is_same_v<T_IN, T_OUT>, "T_IN and T_OUT must be the same type");
  static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T_IN);

  uint32_t const token = blockIdx.x;
  uint32_t const blockSize = blockDim.x;
  uint32_t const threadOffset = threadIdx.x;

  uint32_t numThreads = blockSize;
  uint32_t clusterSize = 1;
  uint32_t blockOffset = 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  numThreads = cluster.num_threads();
  clusterSize = cluster.num_blocks();
  blockOffset = cluster.block_rank();
#endif
  uint32_t const dimPadded = round_up(dim, kELTS_PER_LOAD * numThreads);
  uint32_t const elemsPerThread = dimPadded / numThreads;
  uint32_t const loadStride = blockSize;

  extern __shared__ uint8_t smem[];
  float rInput[LoadsPerThread * kELTS_PER_LOAD];
  uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

  uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T_IN);
  T_IN* smemInput = (T_IN*)&smem[0];
  T_IN* smemResidual = (T_IN*)&smem[smemBufferSize];
  T_IN* smemGamma = (T_IN*)&smem[2 * smemBufferSize];

  LamportFlags<float4> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
  T_IN* input = reinterpret_cast<T_IN*>(
      flag.getCurLamportBuf(reinterpret_cast<void*>(bufferInput), MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  // The offset that current thread should load from. Note that the hidden dimension is split by CGA
  // size and each block loads a contiguous chunk; The size of chunk that each block processes
  uint32_t const blockChunkSize = ceil_div(dim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
  uint32_t const blockLoadOffset = token * dim + blockOffset * blockChunkSize;

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    // Each block load a contiguous chunk of tokens
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    offsets[i] = blockLoadOffset + threadLoadOffset;
  }

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemResidual[threadLoadOffset], &residual[blockLoadOffset + threadLoadOffset]);
    }
  }
  __pipeline_commit();
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemGamma[threadLoadOffset], &gamma[blockOffset * blockChunkSize + threadLoadOffset]);
    }
  }
  __pipeline_commit();

  flag.ctaArrive();
  bool valid = false;
  // ACQBLK if not lamport
  while (!valid) {
    valid = true;
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;

      if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
        float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
        float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

        float4 value = loadPackedVolatile<float4>(src4);
        // Assume that the 16B were written atomically, so we only need to check one value
        valid &= !isNegZero(value.x);
        *dst4 = value;
      }
    }
  }

  __pipeline_wait_prior(1);
  __syncthreads();

  float threadSum = 0.f;
#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    int threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> inp{.packed = loadPacked<float4>(&smemInput[threadLoadOffset])};
      PackedVec<float4, T_IN> res{.packed = loadPacked<float4>(&smemResidual[threadLoadOffset])};

      PackedVec<float4, T_IN> inp_plus_res = inp + res;
#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        rInput[i * kELTS_PER_LOAD + j] = toFloat<T_IN>(inp_plus_res.elements[j]);
        threadSum += toFloat<T_IN>(inp_plus_res.elements[j] * inp_plus_res.elements[j]);
      }

      *reinterpret_cast<float4*>(&outputPreNorm[blockLoadOffset + threadLoadOffset]) = inp_plus_res.packed;
    }
  }

  __pipeline_wait_prior(0);

  float blockSum = blockReduceSum<float, true>(threadSum);

  float fullSum = blockSum;
  __shared__ float sharedVal[8];
  // Use CGA Reduction if supported
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  int const numBlocks = cluster.num_blocks();
  if (numBlocks > 1) {
    fullSum = 0.F;
    // Need to reduce over the entire cluster
    int const blockRank = cluster.block_rank();
    if (threadIdx.x < numBlocks) {
      cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
    }
    cluster.barrier_wait(cluster.barrier_arrive());
    for (int i = 0; i < numBlocks; ++i) {
      fullSum += sharedVal[i];
    }
  }
#endif

  float rcpRms = rsqrtf(fullSum / dim + epsilon);

#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    PackedVec<float4, T_OUT> r_out;
    uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> gamma = {.packed = loadPacked<float4>(&smemGamma[threadLoadOffset])};

#pragma unroll
      for (uint32_t j = 0; j < kELTS_PER_LOAD; j++) {
        r_out.elements[j] =
            fromFloat<T_OUT>((weightBias + toFloat<T_IN>(gamma.elements[j])) * rInput[i * kELTS_PER_LOAD + j] * rcpRms);
      }

      *reinterpret_cast<float4*>(&outputNorm[blockLoadOffset + threadLoadOffset]) = r_out.packed;
    }
  }
  constexpr int kELTS_SIZE = sizeof(T_IN);

  // Assume the previous kernel does not modify the buffer_flags.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  // Update the buffer pointers
  flag.waitAndUpdate(
      {static_cast<uint32_t>(round_up(numTokens, worldSize) * dim * kELTS_SIZE),
       static_cast<uint32_t>(numTokens * dim * kELTS_SIZE),
       0,
       0});
}

template <typename T>
cudaError_t twoshotAllreduceFusionDispatch(AllReduceFusionParams const& params) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const numEltsPerThread = sizeof(float4) / sizeof(T);
  FLASHINFER_CHECK(
      tokenDim % numEltsPerThread == 0, "[MNNVL AllReduceTwoShot] token_dim must be divisible by %d", numEltsPerThread);

  int const arNumThreads = ceil_div(tokenDim, numEltsPerThread);
  int const arNumBlocksPerToken = ceil_div(arNumThreads, 128);

  dim3 arGrid(numTokens, arNumBlocksPerToken);

  cudaLaunchAttribute arAttrs[1];
  arAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  arAttrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;

  cudaLaunchConfig_t arConfig{
      .gridDim = arGrid,
      .blockDim = 128,
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = arAttrs,
      .numAttrs = 1,
  };

  FLASHINFER_LOG_DEBUG(
      "[MNNVL AllReduceTwoShot] Dispatch: grid size: (%d, %d, 1), block_size: 128", numTokens, arNumBlocksPerToken);

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE)   \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(    \
      &arConfig,                              \
      &twoshotAllreduceKernel<WORLD_SIZE, T>, \
      output,                                 \
      input,                                  \
      ucPtrs,                                 \
      mcastPtr,                               \
      numTokens,                              \
      tokenDim,                               \
      params.rank,                            \
      params.bufferFlags,                     \
      (!params.rmsNormFusion)));
  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcastPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T const* input = reinterpret_cast<T const*>(params.input);
  switch (params.nRanks) {
    case 2:
      LAUNCH_ALLREDUCE_KERNEL(2);
      break;
    case 4:
      LAUNCH_ALLREDUCE_KERNEL(4);
      break;
    case 8:
      LAUNCH_ALLREDUCE_KERNEL(8);
      break;
    case 16:
      LAUNCH_ALLREDUCE_KERNEL(16);
      break;
    case 32:
      LAUNCH_ALLREDUCE_KERNEL(32);
      break;
    case 64:
      LAUNCH_ALLREDUCE_KERNEL(64);
      break;
    default:
      FLASHINFER_ERROR(
          "[MNNVL AllReduceTwoShot] Unsupported world_size" + std::to_string(params.nRanks) +
          ". Supported sizes: {2, 4, 8, 16, 32, 64}");
      return cudaErrorInvalidValue;
  }
#undef LAUNCH_ALLREDUCE_KERNEL

  // Launch the rmsnorm lamport kernel if fusion is enabled
  if (params.rmsNormFusion) {
    static const int kSMVersionMajor = GetCudaComputeCapability().first;
    auto gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread, kSMVersionMajor);
    int rnBlockSize = std::get<0>(gridConfig);
    int rnClusterSize = std::get<1>(gridConfig);
    int rnLoadsPerThread = std::get<2>(gridConfig);

    int rnNumThreads = rnClusterSize * rnBlockSize;
    dim3 rnGrid(numTokens, rnClusterSize, 1);
    cudaLaunchConfig_t rnConfig;
    cudaLaunchAttribute rnAttrs[2];
    rnConfig.stream = params.stream;
    rnConfig.gridDim = rnGrid;
    rnConfig.blockDim = rnBlockSize;
    rnConfig.attrs = rnAttrs;
    rnAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    rnAttrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
    rnAttrs[1].id = cudaLaunchAttributeClusterDimension;
    rnAttrs[1].val.clusterDim.x = 1;
    rnAttrs[1].val.clusterDim.y = rnClusterSize;
    rnAttrs[1].val.clusterDim.z = 1;
    rnConfig.numAttrs = kSMVersionMajor >= 9 ? 2 : 1;

    bool const rnUseCGA = kSMVersionMajor >= 9 && rnClusterSize > 1;
    int const dimPadded = round_up(tokenDim, numEltsPerThread * rnNumThreads);
    int const iters = dimPadded / rnNumThreads;

    size_t const smemSize = 3 * rnBlockSize * iters * sizeof(T);

    FLASHINFER_LOG_DEBUG(
        "[MNNVL AllReduceTwoShotRMSNorm] Dispatch: grid size: (%d, %d, 1), block_size: %d, "
        "cluster_size: %d, "
        "loads_per_thread: %d, "
        "threads_needed: %d",
        numTokens,
        rnClusterSize,
        rnBlockSize,
        rnClusterSize,
        rnLoadsPerThread,
        ceil_div(tokenDim, numEltsPerThread));

#define RUN_RMSNORM_KERNEL(LOADS_PER_THREAD)                                                            \
  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                                                            \
      &rmsNormLamport<T, T, LOADS_PER_THREAD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize)); \
  rnConfig.dynamicSmemBytes = smemSize;                                                                 \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                                              \
      &rnConfig,                                                                                        \
      &rmsNormLamport<T, T, LOADS_PER_THREAD>,                                                          \
      residualOut,                                                                                      \
      output,                                                                                           \
      bufferInput,                                                                                      \
      gamma,                                                                                            \
      static_cast<float>(params.epsilon),                                                               \
      params.weightBias,                                                                                \
      residualIn,                                                                                       \
      numTokens,                                                                                        \
      tokenDim,                                                                                         \
      params.nRanks,                                                                                    \
      params.bufferFlags));

    T* residualOut = reinterpret_cast<T*>(params.residualOut);
    T* output = reinterpret_cast<T*>(params.output);
    T* bufferInput = reinterpret_cast<T*>(params.bufferPtrLocal);
    T const* gamma = reinterpret_cast<T const*>(params.gamma);
    T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
    if (rnUseCGA) {
      RUN_RMSNORM_KERNEL(1);
    } else {
      switch (rnLoadsPerThread) {
        case 1:
          RUN_RMSNORM_KERNEL(1);
          break;
        case 2:
          RUN_RMSNORM_KERNEL(2);
          break;
        case 3:
          RUN_RMSNORM_KERNEL(3);
          break;
        case 4:
          RUN_RMSNORM_KERNEL(4);
          break;
        case 5:
          RUN_RMSNORM_KERNEL(5);
          break;
        case 6:
          RUN_RMSNORM_KERNEL(6);
          break;
        case 7:
          RUN_RMSNORM_KERNEL(7);
          break;
        case 8:
          RUN_RMSNORM_KERNEL(8);
          break;
        default:
          FLASHINFER_ERROR(
              "[MNNVL AllReduceTwoShotRMSNorm] Unsupported loads_per_thread" + std::to_string(rnLoadsPerThread) +
              ". Supported sizes: {1, 2, 3, 4, 5, 6, 7, 8}");
          return cudaErrorInvalidValue;
      }  // switch (rnLoadsPerThread)
    }  // if (rnUseCGA)
#undef RUN_RMSNORM_KERNEL

  }  // if (params.rmsNormFusion)
  return cudaSuccess;
}
}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
