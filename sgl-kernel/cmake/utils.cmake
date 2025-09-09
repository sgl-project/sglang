# Adapt from: https://github.com/neuralmagic/vllm-flash-attention/blob/main/cmake/utils.cmake
#
# Clear all `-gencode` flags from `CMAKE_CUDA_FLAGS` and store them in
# `CUDA_ARCH_FLAGS`.
#
# Example:
#   CMAKE_CUDA_FLAGS="-Wall -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"
#   clear_cuda_arches(CUDA_ARCH_FLAGS)
#   CUDA_ARCH_FLAGS="-gencode arch=compute_70,code=sm_70;-gencode arch=compute_75,code=sm_75"
#   CMAKE_CUDA_FLAGS="-Wall"
#
macro(clear_cuda_arches CUDA_ARCH_FLAGS)
    # Extract all `-gencode` flags from `CMAKE_CUDA_FLAGS`
    string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS
      ${CMAKE_CUDA_FLAGS})

    # Remove all `-gencode` flags from `CMAKE_CUDA_FLAGS` since they will be modified
    # and passed back via the `CUDA_ARCHITECTURES` property.
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
      ${CMAKE_CUDA_FLAGS})
endmacro()
