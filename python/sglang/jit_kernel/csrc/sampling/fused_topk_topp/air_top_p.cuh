// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Standalone deterministic AIR Top-P radix algorithm.
//
// Derived from flashinfer/data/include/flashinfer/air_top_p.cuh (Apache 2.0,
// FlashInfer team, 2026), which in turn ports TensorRT-LLM's AIR Top-P kernel.
//
// Adaptations for the fused top-k + top-p kernel here:
//   * Per-row skip — when counter->len is initialized to 0 (set by our fused
//     init kernel for rows whose top_k <= MAX_K), every radix pass exits
//     immediately. Output threshold is then unused by the apply kernel.
//   * Threshold is left in counter->kthValueBits (no built-in apply kernel).
//   * Counter holds per-row total_sum so the apply kernel can renormalize.

#ifndef AIR_TOP_P_CUH_
#define AIR_TOP_P_CUH_

#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/atomic>

#include "nv_util.h"
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace air_top_p {

using IdxT = int;
using AccT = float;

static constexpr int BITS_PER_PASS = 11;
static constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
static constexpr int BLOCK_SIZE = 512;
using WideT = float4;

template <typename T>
static constexpr int NUM_PASSES = (sizeof(T) * 8 + BITS_PER_PASS - 1) / BITS_PER_PASS;

// Deterministic: 64-bit mantissa-only accumulator (radix exponent class is
// fixed per bucket because all values in a bucket share their top BITS_PER_PASS
// MSBs — so summing the mantissas plus tracking the count is enough to
// reconstruct an exact float per bucket regardless of summation order).
template <typename T>
using HisT = uint64_t;

template <typename T>
struct alignas(128) Counter {
  T const* in;
  IdxT oriLen;
  AccT sum;       // remaining sum budget at current refinement level
  IdxT len;       // candidates count in current refinement level
  float p;        // top_p threshold; 0 means "skip this row"
  AccT totalSum;  // total sum of the row (after pass 0)
  IdxT previousLen;
  typename cub::Traits<T>::UnsignedBits kthValueBits;
  alignas(128) IdxT filterCnt;
  alignas(128) uint32_t finishedBlockCnt;
};

template <typename IntType>
constexpr __host__ __device__ IntType ceilDiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}

template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b) {
  return ceilDiv(a, b) * b;
}

template <typename T>
__device__ int constexpr calcStartBit(int pass) {
  int startBit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BITS_PER_PASS;
  return startBit < 0 ? 0 : startBit;
}

template <typename T>
__device__ uint32_t constexpr calcMask(int pass) {
  int numBits = calcStartBit<T>(pass - 1) - calcStartBit<T>(pass);
  return (1 << numBits) - 1;
}

template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddleIn(T key, bool selectMin) {
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
  bits = cub::Traits<T>::TwiddleIn(bits);
  if (!selectMin) bits = ~bits;
  return bits;
}

template <typename T>
__device__ T twiddleOut(typename cub::Traits<T>::UnsignedBits bits, bool selectMin) {
  if (!selectMin) bits = ~bits;
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

template <typename T>
__device__ int calcBucket(T x, int startBit, uint32_t mask) {
  return (twiddleIn(x, false) >> startBit) & mask;
}

template <typename T>
__host__ __device__ IdxT calcBufLen(IdxT len) {
  IdxT constexpr ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
  IdxT bufLen = len / (ratio * 8);
  bufLen = alignTo(bufLen, 256);
  return bufLen;
}

template <typename T>
__host__ __device__ void setBufPointers(T const* in, T* buf1, T* buf2, int pass, T const*& inBuf, T*& outBuf) {
  if (pass == 0) {
    inBuf = in;
    outBuf = nullptr;
  } else if (pass == 1) {
    inBuf = in;
    outBuf = buf1;
  } else if (pass % 2 == 0) {
    inBuf = buf1;
    outBuf = buf2;
  } else {
    inBuf = buf2;
    outBuf = buf1;
  }
}

__device__ inline uint32_t calcMantissa(float value) {
  union {
    uint32_t bits;
    float value;
  } u;
  u.value = value;
  constexpr uint32_t numMantissa = 23;
  return u.bits & ((1u << numMantissa) - 1);
}

__device__ inline uint32_t calcExponent(float value) {
  union {
    uint32_t bits;
    float value;
  } u;
  u.value = value;
  constexpr uint32_t numMantissa = 23;
  return u.bits & ~((1u << numMantissa) - 1);
}

// Reconstruct sum-of-floats from {count, common exponent, accumulated mantissa
// bits}. The common exponent comes from the bucket's bit pattern; count adds
// in the implicit leading 1 bit each value has; bitSum stacks all mantissas.
__device__ inline float calcFloatValue(uint32_t count, uint32_t exponent, uint64_t bitSum) {
  constexpr uint32_t numTotalBits = 64;
  constexpr uint32_t numMantissa = 23;
  uint64_t extraInMantissa = (bitSum >> numMantissa);
  extraInMantissa = (exponent == 0) ? extraInMantissa : extraInMantissa + count;
  uint32_t numExtra = numTotalBits - __clzll(extraInMantissa);
  int numNorm = (exponent == 0) ? 0 : -1;
  exponent = exponent + ((numExtra + numNorm) << numMantissa);
  uint32_t mantissa;
  if (extraInMantissa != 0) {
    int numMove = numMantissa - (numExtra - 1);
    uint32_t mask = (1u << (numExtra - 1)) - 1;
    extraInMantissa = extraInMantissa & mask;
    if (numMove > 0) {
      extraInMantissa = extraInMantissa << numMove;
      mask = (1u << numMantissa) - 1;
      mantissa = ((bitSum & mask) >> (numExtra - 1)) | extraInMantissa;
    } else {
      mantissa = extraInMantissa >> (-1 * numMove);
    }
  } else {
    mantissa = bitSum;
  }
  uint32_t bitFloat = exponent | mantissa;
  return reinterpret_cast<float&>(bitFloat);
}

template <typename T, typename HisT_>
__device__ constexpr void histAtomicAdd(HisT_* dst, T value) {
  uint32_t m = calcMantissa(value);
  atomicAdd(reinterpret_cast<unsigned long long*>(dst), static_cast<uint64_t>(m));
}

template <typename T, typename Func>
__device__ void vectorizedProcess(size_t threadRank, size_t numThreads, T const* in, IdxT len, Func f) {
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = threadRank; i < len; i += numThreads)
      f(in[i], i);
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int itemsPerScalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[itemsPerScalar];
    } wide;
    int skipCnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                      ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                      : 0;
    if (skipCnt > len) skipCnt = len;
    WideT const* inCast = reinterpret_cast<WideT const*>(in + skipCnt);
    IdxT const lenCast = (len - skipCnt) / itemsPerScalar;
    for (IdxT i = threadRank; i < lenCast; i += numThreads) {
      wide.scalar = inCast[i];
      IdxT const real_i = skipCnt + i * itemsPerScalar;
#pragma unroll
      for (int j = 0; j < itemsPerScalar; ++j)
        f(wide.array[j], real_i + j);
    }
    if (static_cast<IdxT>(threadRank) < skipCnt) {
      f(in[threadRank], static_cast<IdxT>(threadRank));
    }
    IdxT const remain_i = skipCnt + lenCast * itemsPerScalar + threadRank;
    if (remain_i < len) f(in[remain_i], remain_i);
  }
}

template <typename T, typename HisT_>
__device__ __forceinline__ void filterAndHistogram(
    T const* inBuf,
    T* outBuf,
    int previousLen,
    Counter<T>* counter,
    HisT_* histogram,
    IdxT* countHistogram,
    HisT_* histogramSmem,
    IdxT* countHistogramSmem,
    int pass) {
  for (IdxT i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    histogramSmem[i] = 0;
    countHistogramSmem[i] = 0;
  }
  __syncthreads();
  int const startBit = calcStartBit<T>(pass);
  uint32_t const mask = calcMask<T>(pass);
  if (pass == 0) {
    auto f = [startBit, mask, histogramSmem, countHistogramSmem](T value, IdxT) {
      int bucket = calcBucket<T>(value, startBit, mask);
      histAtomicAdd<T>(histogramSmem + bucket, value);
      atomicAdd(countHistogramSmem + bucket, static_cast<IdxT>(1));
    };
    vectorizedProcess<T>(
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
        static_cast<size_t>(blockDim.x) * gridDim.x,
        inBuf,
        previousLen,
        f);
  } else {
    IdxT* pFilterCnt = &counter->filterCnt;
    auto const kthValueBits = counter->kthValueBits;
    int const previousStartBit = calcStartBit<T>(pass - 1);
    auto f = [outBuf, startBit, mask, previousStartBit, kthValueBits, pFilterCnt, histogramSmem, countHistogramSmem](
                 T value, IdxT) {
      auto const previousBits = (twiddleIn(value, false) >> previousStartBit) << previousStartBit;
      if (previousBits == kthValueBits) {
        if (outBuf) {
          IdxT pos = atomicAdd(pFilterCnt, static_cast<IdxT>(1));
          outBuf[pos] = value;
        }
        int bucket = calcBucket<T>(value, startBit, mask);
        histAtomicAdd<T>(histogramSmem + bucket, value);
        atomicAdd(countHistogramSmem + bucket, static_cast<IdxT>(1));
      }
    };
    vectorizedProcess<T>(
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
        static_cast<size_t>(blockDim.x) * gridDim.x,
        inBuf,
        previousLen,
        f);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    if (histogramSmem[i] != 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(histogram + i), static_cast<unsigned long long>(histogramSmem[i]));
    }
    if (countHistogramSmem[i] != 0) atomicAdd(countHistogram + i, countHistogramSmem[i]);
  }
}

template <typename T>
__global__ void
AirTopPRadixKernel(Counter<T>* counters, HisT<T>* histograms, IdxT* countHistograms, int const pass, T* buf1, T* buf2) {
  using HisT_ = HisT<T>;
  int const batchId = blockIdx.y;
  auto counter = counters + batchId;
  AccT currentSum;
  IdxT previousLen, currentLen;
  if (pass == 0) {
    currentSum = 0;
    previousLen = counter->len;
    currentLen = counter->len;
  } else {
    currentSum = counter->sum;
    currentLen = counter->len;
    previousLen = counter->previousLen;
  }
  if (currentLen == 0) return;
  IdxT const bufLen = calcBufLen<T>(counter->oriLen);
  T const* inBuf = nullptr;
  T* outBuf = nullptr;
  setBufPointers(counter->in, buf1 + bufLen * batchId, buf2 + bufLen * batchId, pass, inBuf, outBuf);
  if (pass == 0 || pass == 1 || previousLen > bufLen) {
    inBuf = counter->in;
    previousLen = counter->oriLen;
  }
  if (pass == 0 || currentLen > bufLen) {
    outBuf = nullptr;
  }
  auto histogram = histograms + batchId * NUM_BUCKETS;
  auto countHistogram = countHistograms + batchId * NUM_BUCKETS;
  __shared__ HisT_ histogramSmem[NUM_BUCKETS];
  __shared__ IdxT countHistogramSmem[NUM_BUCKETS];
  filterAndHistogram<T, HisT_>(
      inBuf, outBuf, previousLen, counter, histogram, countHistogram, histogramSmem, countHistogramSmem, pass);
  __syncthreads();
  __threadfence();
  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    uint32_t finished = atomicInc(&counter->finishedBlockCnt, gridDim.x - 1);
    isLastBlock = (finished == (gridDim.x - 1));
  }
  if (__syncthreads_or(isLastBlock)) {
    AccT* histValueSmem = reinterpret_cast<AccT*>(histogramSmem);
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
      uint64_t value = static_cast<uint64_t>(histogram[i]);
      IdxT count = countHistogram[i];
      if (count != 0) {
        uint32_t sb = calcStartBit<T>(pass);
        uint32_t bv = counter->kthValueBits;
        if (pass == 0) bv = i << sb;
        histValueSmem[i] = calcFloatValue(static_cast<uint32_t>(count), calcExponent(twiddleOut<T>(bv, false)), value);
      } else {
        histValueSmem[i] = 0.0f;
      }
    }
    __syncthreads();
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_COUNT = NUM_BUCKETS / WARP_SIZE;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    AccT* histPtr = histValueSmem;
    __shared__ AccT warpSum[WARP_COUNT];
    __shared__ cuda::atomic<AccT, cuda::thread_scope_block> blockSum;
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += BLOCK_SIZE) {
      reduce_store_async(warp, warpSum + i / WARP_SIZE, histPtr[i], cg::plus<float>{});
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
      reduce_store_async(warp, blockSum, warpSum[threadIdx.x], cg::plus<float>{});
      reduce_update_async(warp, blockSum, warpSum[threadIdx.x + WARP_SIZE], cg::plus<float>{});
    }
    __syncthreads();
    if (pass == 0 && threadIdx.x == 0) {
      counter->totalSum = blockSum.load();
    }
    if (pass == 0) currentSum = blockSum.load() * counter->p;
    if (threadIdx.x == 0) {
      AccT prev = 0;
      int targetStep = 0;
      for (int i = 0; i < WARP_COUNT; i++) {
        if (warpSum[i]) {
          targetStep = i;
          if ((prev + warpSum[i]) >= currentSum) break;
          prev += warpSum[i];
        }
      }
      int targetIdx = 0;
      for (int i = targetStep * WARP_SIZE; i < NUM_BUCKETS; i++) {
        if (countHistogram[i]) {
          targetIdx = i;
          if ((prev + histPtr[i]) >= currentSum) break;
          prev += histPtr[i];
        }
      }
      counter->sum = currentSum - prev;
      counter->len = countHistogram[targetIdx];
      typename cub::Traits<T>::UnsignedBits bucket = targetIdx;
      counter->kthValueBits |= bucket << calcStartBit<T>(pass);
    }
    __syncthreads();
    if (pass != NUM_PASSES<T> - 1) {
      for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        histogram[i] = 0;
        countHistogram[i] = 0;
      }
    }
    if (threadIdx.x == 0) {
      counter->previousLen = currentLen;
      counter->filterCnt = 0;
    }
  }
}

// Per-row init. Sets counter->len to oriLen (vocab size) only when this row is
// in top-p mode (top_k > maxTopK); otherwise len stays 0 so every radix pass
// short-circuits via the `if (currentLen == 0) return;` check.
template <typename T>
__global__ void AirTopPInitKernel(
    Counter<T>* counters,
    int len,
    T const* in,
    float const* top_p_arr,
    int32_t const* top_k_arr,
    int32_t maxTopK,
    HisT<T>* histograms,
    IdxT* countHistograms) {
  int const batchIdx = blockIdx.x;
  Counter<T>* counter = counters + batchIdx;
  int32_t k = top_k_arr[batchIdx];
  float p = top_p_arr[batchIdx];
  bool active = (k > maxTopK);
  if (threadIdx.x == 0) {
    counter->in = in + batchIdx * len;
    counter->oriLen = len;
    counter->len = active ? len : 0;
    counter->previousLen = len;
    counter->p = active ? p : 0.0f;
    counter->totalSum = 0.0f;
    counter->sum = 0;
    counter->kthValueBits = 0;
    counter->finishedBlockCnt = 0;
    counter->filterCnt = 0;
  }
  HisT<T>* hist = histograms + batchIdx * NUM_BUCKETS;
  IdxT* cntHist = countHistograms + batchIdx * NUM_BUCKETS;
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    hist[i] = 0;
    cntHist[i] = 0;
  }
}

template <typename T>
uint32_t CalcAirTopPBlockNum(int batchSize, int len, int smCnt) {
  constexpr int VECTORIZED_READ_SIZE = 16;
  int activeBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, AirTopPRadixKernel<T>, BLOCK_SIZE, 0);
  activeBlocks *= smCnt;
  IdxT bestNumBlocks = 0;
  float bestTailWavePenalty = 1.0f;
  IdxT const maxNumBlocks = ceilDiv<IdxT>(len, VECTORIZED_READ_SIZE / (int)sizeof(T) * BLOCK_SIZE);
  for (int numWaves = 1;; ++numWaves) {
    IdxT numBlocks = std::min(maxNumBlocks, static_cast<IdxT>(std::max(numWaves * activeBlocks / batchSize, 1)));
    IdxT itemsPerThread = ceilDiv<IdxT>(len, numBlocks * BLOCK_SIZE);
    itemsPerThread = alignTo<IdxT>(itemsPerThread, VECTORIZED_READ_SIZE / (int)sizeof(T));
    numBlocks = ceilDiv<IdxT>(len, itemsPerThread * BLOCK_SIZE);
    float actualNumWaves = static_cast<float>(numBlocks) * batchSize / activeBlocks;
    float tailWavePenalty = (ceilf(actualNumWaves) - actualNumWaves) / ceilf(actualNumWaves);
    if (tailWavePenalty < 0.15f) {
      bestNumBlocks = numBlocks;
      break;
    } else if (tailWavePenalty < bestTailWavePenalty) {
      bestNumBlocks = numBlocks;
      bestTailWavePenalty = tailWavePenalty;
    }
    if (numBlocks == maxNumBlocks) break;
  }
  return bestNumBlocks;
}

template <typename T>
size_t getWorkspaceBytes(int batchSize, int vocabSize) {
  auto align256 = [](size_t x) { return ((x + 255) / 256) * 256; };
  IdxT const bufLen = calcBufLen<T>(vocabSize);
  size_t countersSize = align256(sizeof(Counter<T>) * batchSize);
  size_t histSize = align256(sizeof(HisT<T>) * NUM_BUCKETS * batchSize);
  size_t countHistSize = align256(sizeof(IdxT) * NUM_BUCKETS * batchSize);
  size_t bufSize = align256(sizeof(T) * bufLen * batchSize);
  return countersSize + histSize + countHistSize + 2 * bufSize + 256;
}

// Resolve the workspace pointers and return them via the out-args. The caller
// is responsible for zeroing the counters/histograms/countHistograms (the
// unified init kernel does this so we can skip the separate AirTopPInitKernel).
template <typename T>
void resolveWorkspace(
    int batchSize,
    int vocabSize,
    void* workspace,
    Counter<T>*& outCounters,
    HisT<T>*& outHistograms,
    IdxT*& outCountHistograms,
    T*& outBuf1,
    T*& outBuf2) {
  auto align256 = [](size_t x) { return ((x + 255) / 256) * 256; };
  IdxT const bufLen = calcBufLen<T>(vocabSize);
  size_t countersSize = align256(sizeof(Counter<T>) * batchSize);
  size_t histSize = align256(sizeof(HisT<T>) * NUM_BUCKETS * batchSize);
  size_t countHistSize = align256(sizeof(IdxT) * NUM_BUCKETS * batchSize);
  size_t bufSize = align256(sizeof(T) * bufLen * batchSize);

  uint8_t* ws = static_cast<uint8_t*>(workspace);
  outCounters = reinterpret_cast<Counter<T>*>(ws);
  outHistograms = reinterpret_cast<HisT<T>*>(ws + countersSize);
  outCountHistograms = reinterpret_cast<IdxT*>(ws + countersSize + histSize);
  outBuf1 = reinterpret_cast<T*>(ws + countersSize + histSize + countHistSize);
  outBuf2 = reinterpret_cast<T*>(ws + countersSize + histSize + countHistSize + bufSize);
}

// Run just the 3 radix passes. Counters/histograms must already be initialized
// (e.g. by the fused kernel's unifiedInitKernel).
template <typename T>
void launchRadixOnly(
    Counter<T>* counters,
    HisT<T>* histograms,
    IdxT* countHistograms,
    T* buf1,
    T* buf2,
    int batchSize,
    int vocabSize,
    cudaStream_t stream) {
  auto launchPDL = [&](auto kernel, dim3 grid, dim3 block, size_t smem, auto... args) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = attr;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel, args...);
  };

  int dev = 0, smCnt = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smCnt, cudaDevAttrMultiProcessorCount, dev);
  uint32_t blockNum = CalcAirTopPBlockNum<T>(batchSize, vocabSize, smCnt);
  dim3 grid(blockNum, batchSize);
  constexpr int numPasses = NUM_PASSES<T>;
  for (int pass = 0; pass < numPasses; ++pass) {
    launchPDL(
        AirTopPRadixKernel<T>, grid, dim3(BLOCK_SIZE), 0, counters, histograms, countHistograms, pass, buf1, buf2);
  }
}

template <typename T>
void launch(
    T const* probs,
    float const* topPArr,
    int32_t const* topKArr,
    int32_t maxTopK,
    int batchSize,
    int vocabSize,
    void* workspace,
    Counter<T>*& outCounters,
    cudaStream_t stream) {
  auto align256 = [](size_t x) { return ((x + 255) / 256) * 256; };
  IdxT const bufLen = calcBufLen<T>(vocabSize);
  size_t countersSize = align256(sizeof(Counter<T>) * batchSize);
  size_t histSize = align256(sizeof(HisT<T>) * NUM_BUCKETS * batchSize);
  size_t countHistSize = align256(sizeof(IdxT) * NUM_BUCKETS * batchSize);
  size_t bufSize = align256(sizeof(T) * bufLen * batchSize);

  uint8_t* ws = static_cast<uint8_t*>(workspace);
  Counter<T>* counters = reinterpret_cast<Counter<T>*>(ws);
  HisT<T>* histograms = reinterpret_cast<HisT<T>*>(ws + countersSize);
  IdxT* countHistograms = reinterpret_cast<IdxT*>(ws + countersSize + histSize);
  T* buf1 = reinterpret_cast<T*>(ws + countersSize + histSize + countHistSize);
  T* buf2 = reinterpret_cast<T*>(ws + countersSize + histSize + countHistSize + bufSize);

  outCounters = counters;

  auto launchPDL = [&](auto kernel, dim3 grid, dim3 block, size_t smem, auto... args) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = attr;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel, args...);
  };

  launchPDL(
      AirTopPInitKernel<T>,
      dim3(batchSize),
      dim3(256),
      0,
      counters,
      vocabSize,
      probs,
      topPArr,
      topKArr,
      maxTopK,
      histograms,
      countHistograms);

  int dev = 0, smCnt = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smCnt, cudaDevAttrMultiProcessorCount, dev);
  uint32_t blockNum = CalcAirTopPBlockNum<T>(batchSize, vocabSize, smCnt);
  dim3 grid(blockNum, batchSize);
  constexpr int numPasses = NUM_PASSES<T>;
  for (int pass = 0; pass < numPasses; ++pass) {
    launchPDL(
        AirTopPRadixKernel<T>, grid, dim3(BLOCK_SIZE), 0, counters, histograms, countHistograms, pass, buf1, buf2);
  }
}

}  // namespace air_top_p

#endif  // AIR_TOP_P_CUH_
