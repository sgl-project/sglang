// bs=1 specialization of the verbatim oneshot fused AR kernel
// (mnnvl_ar_fused.cuh): NumTokens and TokenDim become template constants so
// index arithmetic folds and loops unroll fully, while every VALUE-BEARING
// operation keeps the exact arithmetic, order, and launch geometry of the
// generic kernel — required because the e2e promote gate demands greedy
// output byte-identical to the stock path:
//   - the per-thread world-sum and the norm reduction tree are untouched;
//   - the launch geometry comes from the same adjustGridConfig, so the
//     cross-warp/cluster fp32 sum order is identical;
//   - the "fullSum / tokenDim" division uses the RUNTIME tokenDim kernel
//     argument (not the constant) so the compiler cannot fold the division
//     into a reciprocal multiply and change the quotient bits.
// Uncovered shape/parameter combinations fall back to the generic verbatim
// dispatch, which itself matches the deployed flashinfer binary.
#pragma once

#include "mnnvl_ar_fused.cuh"

namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

template <uint8_t WorldSize, typename T, int NumTokens, int TokenDim, typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshotArFusedNormConstKernel(
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
  int threadOffset = token * TokenDim + packedIdx * kELTS_PER_THREAD;

  cudaGridDependencySynchronize();
#else
  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  int threadOffset = token * TokenDim + packedIdx * kELTS_PER_THREAD;
#endif

  LamportFlags<PackedType> flag(bufferFlags, 1);
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], 0));

  if (packedIdx * kELTS_PER_THREAD >= TokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
    return;
  }

  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) val.elements[i] = fromFloat<T>(0.f);
  }

  reinterpret_cast<PackedType*>(&stagePtrMcast[token * TokenDim * WorldSize + rank * TokenDim])[packedIdx] = val.packed;

  flag.ctaArrive();
  flag.clearDirtyLamportBuf(inputPtrs[rank], -1);

  // Prefetch the norm operands BEFORE the Lamport wait so their global-load
  // latency hides behind the peer-arrival spin. Same addresses, same values,
  // same later use order — value-identical to the generic kernel; only the
  // load ISSUE time moves.
  PackedVec<PackedType, T> residualIn;
  residualIn.packed = *reinterpret_cast<PackedType const*>(&residualInPtr[threadOffset]);
  PackedVec<PackedType, T> gamma;
  gamma.packed = *reinterpret_cast<PackedType const*>(&gammaPtr[packedIdx * kELTS_PER_THREAD]);

  PackedVec<PackedType, float> valuesLamport[WorldSize];
  while (1) {
    bool valid = true;
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
      valuesLamport[r].packed = loadPackedVolatile<PackedType>(
          &stagePtrLocal[token * TokenDim * WorldSize + r * TokenDim + packedIdx * kELTS_PER_THREAD]);

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
  {
    packedAccum += residualIn;
    *reinterpret_cast<PackedType*>(&prenormedPtr[threadOffset]) = packedAccum.packed;

    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      threadSum += toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i]);
    }
    float blockSum = blockReduceSum<float, true>(threadSum);

    __shared__ float sharedVal[8];
    float fullSum = blockSum;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1) {
      fullSum = 0.F;
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
    // RUNTIME tokenDim on purpose: keeps the division emission (and thus the
    // quotient bits) identical to the generic kernel / deployed baseline.
    float rcpRms = rsqrtf(fullSum / tokenDim + epsilon);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] =
          fromFloat<T>(toFloat<T>(packedAccum.elements[i]) * rcpRms * (weightBias + toFloat<T>(gamma.elements[i])));
    }
  }
  reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = packedAccum.packed;
  flag.waitAndUpdate({static_cast<uint32_t>(NumTokens * TokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

// ---------------------------------------------------------------------------
// EXPERIMENTAL fence/flag-granularity variant (P1 idea-pool route closure):
// arrival at BLOCK granularity instead of CLUSTER granularity. The upstream
// protocol's per-call flag traffic is already one arrival counter round plus
// one rotation; the only removable synchronization on the critical path is
// the cluster.sync() inside ctaArrive (a 4-block barrier issued right after
// the multicast store). This variant replaces it with __syncthreads() + one
// release-increment per BLOCK, and the rotation waits for gridDim.x*gridDim.y
// block arrivals instead of cluster count. Ordering argument mirrors
// upstream at block scope: the execution barrier orders the block's
// multicast stores before the elected thread's release-increment; the
// rotation still happens only after every store in the grid has arrived.
// VALUE-NEUTRAL: no arithmetic, order, or geometry changes. Requires
// exact-fit geometry (no out-of-bounds threads), asserted by the dispatch —
// divergent-branch barriers would otherwise be undefined behavior.
// ---------------------------------------------------------------------------

template <typename PackedType>
inline __device__ void baBlockArrive(uint32_t* bufferFlags) {
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t* counter = &bufferFlags[8];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(counter), "r"(1) : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
    asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(counter), "r"(1) : "memory");
#else
    atomicAdd(counter, 1);
#endif
  }
}

template <typename PackedType>
inline __device__ void baWaitAndUpdate(
    uint32_t* bufferFlags,
    uint32_t bytesToClearStage0,
    LamportFlags<PackedType> const& flag,
    uint32_t currentIndex,
    uint32_t bytesPerBuffer,
    uint32_t numStages) {
  bool const isGridT0 = (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0);
  if (isGridT0) {
    uint32_t const targetCount = gridDim.x * gridDim.y;  // BLOCK arrivals
    uint32_t* counter = &bufferFlags[8];
    uint4* flagPtr = reinterpret_cast<uint4*>(bufferFlags);
    while (*reinterpret_cast<uint32_t volatile*>(counter) < targetCount) {
    }
    flagPtr[0] = {(currentIndex + 1) % 3, currentIndex, bytesPerBuffer, numStages};
    flagPtr[1] = {bytesToClearStage0, 0, 0, 0};
    *counter = 0;
  }
}

template <uint8_t WorldSize, typename T, int NumTokens, int TokenDim, typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshotArFusedNormConstKernelBA(
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
  int threadOffset = token * TokenDim + packedIdx * kELTS_PER_THREAD;

  cudaGridDependencySynchronize();
#else
  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  int threadOffset = token * TokenDim + packedIdx * kELTS_PER_THREAD;
#endif

  LamportFlags<PackedType> flag(bufferFlags, 1);
  // Snapshot rotation inputs before any state change (needed by the
  // block-granular updater, which cannot reuse the private members).
  uint32_t const curIdx = reinterpret_cast<uint4*>(bufferFlags)[0].x;
  uint32_t const bytesPerBuffer = reinterpret_cast<uint4*>(bufferFlags)[0].z;
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], 0));

  // Exact-fit geometry is asserted host-side; no out-of-bounds branch here.

  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) val.elements[i] = fromFloat<T>(0.f);
  }

  reinterpret_cast<PackedType*>(&stagePtrMcast[token * TokenDim * WorldSize + rank * TokenDim])[packedIdx] = val.packed;

  baBlockArrive<PackedType>(bufferFlags);
  flag.clearDirtyLamportBuf(inputPtrs[rank], -1);

  PackedVec<PackedType, T> residualIn;
  residualIn.packed = *reinterpret_cast<PackedType const*>(&residualInPtr[threadOffset]);
  PackedVec<PackedType, T> gamma;
  gamma.packed = *reinterpret_cast<PackedType const*>(&gammaPtr[packedIdx * kELTS_PER_THREAD]);

  PackedVec<PackedType, float> valuesLamport[WorldSize];
  while (1) {
    bool valid = true;
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
      valuesLamport[r].packed = loadPackedVolatile<PackedType>(
          &stagePtrLocal[token * TokenDim * WorldSize + r * TokenDim + packedIdx * kELTS_PER_THREAD]);

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
  {
    packedAccum += residualIn;
    *reinterpret_cast<PackedType*>(&prenormedPtr[threadOffset]) = packedAccum.packed;

    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      threadSum += toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i]);
    }
    float blockSum = blockReduceSum<float, true>(threadSum);

    __shared__ float sharedVal[8];
    float fullSum = blockSum;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1) {
      fullSum = 0.F;
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
  baWaitAndUpdate<PackedType>(
      bufferFlags,
      static_cast<uint32_t>(NumTokens * TokenDim * WorldSize * kELT_SIZE),
      flag,
      curIdx,
      bytesPerBuffer,
      1u);
}

// True when the chosen launch geometry covers the hidden dimension with no
// out-of-bounds threads (required by the block-arrival kernel's full-block
// barriers).
inline bool adjustGridConfigExactFit(int numTokens, int tokenDim, int eltsPerThread, int smVersionMajor) {
  auto [blockSize, clusterSize, loadsPerThread] = adjustGridConfig(numTokens, tokenDim, eltsPerThread, smVersionMajor);
  return loadsPerThread == 1 && blockSize * clusterSize * eltsPerThread == tokenDim;
}

// Specialized dispatch: constant instantiations for the frozen bs=1 shapes,
// identical launch geometry via the same adjustGridConfig; anything not
// covered falls back to the generic verbatim dispatch (correctness is never
// lost; dispatch cost is a handful of scalar compares on the host).
template <typename T>
cudaError_t oneshotArFusedConstDispatch(AllReduceFusionParams const& params) {
  bool const specialShape = params.rmsNormFusion && params.nRanks == 8 && params.tokenDim == 6144 &&
                            (params.numTokens == 6 || params.numTokens == 1) && std::is_same_v<T, __nv_bfloat16>;
  if (!specialShape) {
    return oneshotAllreduceFusionDispatch<T>(params);
  }

  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const eltsPerThread = sizeof(float4) / sizeof(T);
  static const int kSMVersionMajor = GetCudaComputeCapability().first;

  auto [blockSize, clusterSize, loadsPerThread] = adjustGridConfig(numTokens, tokenDim, eltsPerThread, kSMVersionMajor);
  dim3 grid(numTokens, clusterSize, 1);

  FLASHINFER_CHECK(
      blockSize <= 1024 && loadsPerThread == 1, "Hidden Dimension exceeds the maximum supported hidden dimension");

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = clusterSize;
  attrs[1].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{
      .gridDim = grid,
      .blockDim = static_cast<unsigned int>(blockSize),
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = attrs,
      .numAttrs = kSMVersionMajor >= 9 ? 2 : 1,
  };

  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T* residualOut = reinterpret_cast<T*>(params.residualOut);
  T const* input = reinterpret_cast<T const*>(params.input);
  T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
  T const* gamma = reinterpret_cast<T const*>(params.gamma);

  // ADOPTED (round 1): the block-arrival variant is bit-exact vs the
  // deployed baseline (randn + value zoo), stability-clean, and faster than
  // the cluster-arrival form (T=1 +6.9% by dropping the single-cluster
  // whole-grid sync; T=6 +0.8%). The frozen shapes satisfy its exact-fit
  // geometry requirement (asserted below); the cluster-arrival kernel is
  // kept above for A/B history and non-exact-fit fallback.
  bool const exactFit = adjustGridConfigExactFit(numTokens, tokenDim, eltsPerThread, kSMVersionMajor);

#define LAUNCH_CONST_KERNEL(NTOK, KERNEL)  \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx( \
      &config,                             \
      &KERNEL<8, T, NTOK, 6144>,           \
      output,                              \
      residualOut,                         \
      input,                               \
      residualIn,                          \
      gamma,                               \
      ucPtrs,                              \
      mcPtr,                               \
      numTokens,                           \
      tokenDim,                            \
      static_cast<float>(params.epsilon),  \
      params.weightBias,                   \
      params.rank,                         \
      params.bufferFlags));
  if (numTokens == 6) {
    if (exactFit) {
      LAUNCH_CONST_KERNEL(6, oneshotArFusedNormConstKernelBA);
    } else {
      LAUNCH_CONST_KERNEL(6, oneshotArFusedNormConstKernel);
    }
  } else {
    if (exactFit) {
      LAUNCH_CONST_KERNEL(1, oneshotArFusedNormConstKernelBA);
    } else {
      LAUNCH_CONST_KERNEL(1, oneshotArFusedNormConstKernel);
    }
  }
#undef LAUNCH_CONST_KERNEL
  return cudaSuccess;
}

// Experimental block-arrival variant dispatch (see kernel comment). Same
// specialization predicate PLUS the exact-fit geometry requirement.
template <typename T>
cudaError_t oneshotArFusedConstBaDispatch(AllReduceFusionParams const& params) {
  bool const specialShape = params.rmsNormFusion && params.nRanks == 8 && params.tokenDim == 6144 &&
                            (params.numTokens == 6 || params.numTokens == 1) && std::is_same_v<T, __nv_bfloat16>;
  if (!specialShape) {
    return oneshotAllreduceFusionDispatch<T>(params);
  }

  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const eltsPerThread = sizeof(float4) / sizeof(T);
  static const int kSMVersionMajor = GetCudaComputeCapability().first;

  auto [blockSize, clusterSize, loadsPerThread] = adjustGridConfig(numTokens, tokenDim, eltsPerThread, kSMVersionMajor);
  dim3 grid(numTokens, clusterSize, 1);

  FLASHINFER_CHECK(
      blockSize <= 1024 && loadsPerThread == 1, "Hidden Dimension exceeds the maximum supported hidden dimension");
  // Exact fit: no out-of-bounds thread may exist (the BA kernel uses
  // full-block barriers with no divergent early-exit path).
  FLASHINFER_CHECK(
      blockSize * clusterSize * eltsPerThread == tokenDim, "block-arrival variant requires exact-fit geometry");

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = clusterSize;
  attrs[1].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{
      .gridDim = grid,
      .blockDim = static_cast<unsigned int>(blockSize),
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = attrs,
      .numAttrs = kSMVersionMajor >= 9 ? 2 : 1,
  };

  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T* residualOut = reinterpret_cast<T*>(params.residualOut);
  T const* input = reinterpret_cast<T const*>(params.input);
  T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
  T const* gamma = reinterpret_cast<T const*>(params.gamma);

#define LAUNCH_BA_KERNEL(NTOK)                            \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                \
      &config,                                            \
      &oneshotArFusedNormConstKernelBA<8, T, NTOK, 6144>, \
      output,                                             \
      residualOut,                                        \
      input,                                              \
      residualIn,                                         \
      gamma,                                              \
      ucPtrs,                                             \
      mcPtr,                                              \
      numTokens,                                          \
      tokenDim,                                           \
      static_cast<float>(params.epsilon),                 \
      params.weightBias,                                  \
      params.rank,                                        \
      params.bufferFlags));
  if (numTokens == 6) {
    LAUNCH_BA_KERNEL(6);
  } else {
    LAUNCH_BA_KERNEL(1);
  }
#undef LAUNCH_BA_KERNEL
  return cudaSuccess;
}

}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
