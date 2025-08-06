#pragma once

namespace flashmla {

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)


#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm("trap;");                                                                                 \
        }                                                                                                 \
    } while(0)


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()


#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...) \
  [&] {                                              \
    if (NUM_SPLITS <= 32) {                          \
      constexpr static int NAME = 32;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 64) {                   \
      constexpr static int NAME = 64;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 96) {                   \
      constexpr static int NAME = 96;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 128) {                  \
      constexpr static int NAME = 128;               \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 160) {                  \
      constexpr static int NAME = 160;               \
      return __VA_ARGS__();                          \
    } else {                                         \
      FLASH_ASSERT(false);                           \
    }                                                \
  }()

} // flashmla