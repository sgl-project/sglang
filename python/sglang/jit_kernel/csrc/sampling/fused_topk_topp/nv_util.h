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

#ifndef NV_UTIL_H_
#define NV_UTIL_H_
#include "cuda_runtime_api.h"
#include <cassert>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define TOPK_CUDA_CHECK(val)                    \
  {                                             \
    nv::cuda_check_((val), __FILE__, __LINE__); \
  }
#define TOPK_CUDA_CHECK_LAST_ERROR()                \
  {                                                 \
    nv::cuda_check_last_error_(__FILE__, __LINE__); \
  }

namespace nv {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
constexpr int WARP_SIZE = 32;

class CudaException : public std::runtime_error {
 public:
  explicit CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(
        std::string(file) + ":" + std::to_string(line) + ": CUDA error " + std::to_string(val) + ": " +
        cudaGetErrorString(val));
  }
}

inline void cuda_check_last_error_(const char* file, int line) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaPeekAtLastError();
  cuda_check_(err, file, line);
}

class Timer {
 public:
  Timer() {
    reset();
  }
  void reset() {
    start_time_ = std::chrono::steady_clock::now();
  }
  float elapsed_ms() {
    auto end_time = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end_time - start_time_);
    return dur.count();
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

inline size_t calc_aligned_size(const std::vector<size_t>& sizes) {
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
  size_t total = 0;
  for (auto sz : sizes) {
    total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }
  return total + ALIGN_BYTES - 1;
}

inline std::vector<void*> calc_aligned_pointers(const void* p, const std::vector<size_t>& sizes) {
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);

  char* ptr = reinterpret_cast<char*>((reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

  std::vector<void*> aligned_pointers;
  aligned_pointers.reserve(sizes.size());
  for (auto sz : sizes) {
    aligned_pointers.push_back(ptr);
    ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }

  return aligned_pointers;
}

}  // namespace nv
#endif
