#pragma once
#include <torch/extension.h>

#include <sstream>

struct cuda_error : public std::runtime_error {
  /**
   * @brief Constructs a `cuda_error` object with the given `message`.
   *
   * @param message The error char array used to construct `cuda_error`
   */
  cuda_error(const char* message) : std::runtime_error(message) {}
  /**
   * @brief Constructs a `cuda_error` object with the given `message` string.
   *
   * @param message The `std::string` used to construct `cuda_error`
   */
  cuda_error(std::string const& message) : cuda_error{message.c_str()} {}
};

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw cuda_error(_message.str());                                 \
    }                                                                   \
  } while (0)

#define CHECK_IS_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_IS_CUDA(x);         \
  CHECK_IS_CONTIGUOUS(x)

inline int getSMVersion() {
  int device{-1};
  CHECK_CUDA_SUCCESS(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}
