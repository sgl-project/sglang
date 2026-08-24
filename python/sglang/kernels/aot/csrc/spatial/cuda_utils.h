#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define CUDA_RT(call)                                                                                        \
  do {                                                                                                       \
    cudaError_t _status = (call);                                                                            \
    if (_status != cudaSuccess) {                                                                            \
      std::cerr << "ERROR: CUDA RT call \"" << #call << "\" in line " << __LINE__ << " of file " << __FILE__ \
                << " failed with " << cudaGetErrorString(_status) << std::endl;                              \
      TORCH_CHECK(                                                                                           \
          false,                                                                                             \
          c10::str(                                                                                          \
              "ERROR: CUDA RT call \"",                                                                      \
              #call,                                                                                         \
              "\" in line ",                                                                                 \
              __LINE__,                                                                                      \
              " of file ",                                                                                   \
              __FILE__,                                                                                      \
              " failed with ",                                                                               \
              cudaGetErrorString(_status)));                                                                 \
    }                                                                                                        \
  } while (0)

#define CUDA_DRV(call)                                                                                        \
  do {                                                                                                        \
    CUresult _status = (call);                                                                                \
    if (_status != CUDA_SUCCESS) {                                                                            \
      const char* err_str;                                                                                    \
      cuGetErrorString(_status, &err_str);                                                                    \
      std::cerr << "ERROR: CUDA DRV call \"" << #call << "\" in line " << __LINE__ << " of file " << __FILE__ \
                << " failed with " << err_str << std::endl;                                                   \
      TORCH_CHECK(                                                                                            \
          false,                                                                                              \
          c10::str(                                                                                           \
              "ERROR: CUDA DRV call \"",                                                                      \
              #call,                                                                                          \
              "\" in line ",                                                                                  \
              __LINE__,                                                                                       \
              " of file ",                                                                                    \
              __FILE__,                                                                                       \
              " failed with ",                                                                                \
              err_str));                                                                                      \
    }                                                                                                         \
  } while (0)
