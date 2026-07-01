# sage3 (SageAttention-3 Blackwell FP4 attention)
#
# SageAttention is NOT a pip package and ships no C++ header, so we FetchContent
# its source and #include its .cu directly (see csrc/attention/sage3.cu, which
# uses the PYBIND11_MODULE neutralization trick to merge the symbols in).
#
# sage3 uses arch-conditional FP4 MMA instructions that must be compiled for
# sm_120a, so the sage3_ops library is built ONLY when CUDA >= 13.0 (SM120a
# available). It is a separate .so — NOT added to the shared common_ops SOURCES,
# which would break the sm90/sm100 builds on the arch-conditional kernels.
# Allow a local vendored checkout to override the network fetch (useful on hosts
# where codeload tarball transfers are unreliable). Set SAGEATTENTION_SOURCE_DIR
# to a path containing sageattention3_blackwell/ to use it; otherwise fetch from
# the pinned upstream commit.
if(DEFINED SAGEATTENTION_SOURCE_DIR)
    FetchContent_Declare(repo-sageattention SOURCE_DIR ${SAGEATTENTION_SOURCE_DIR})
else()
    FetchContent_Declare(
        repo-sageattention
        URL      https://${GITHUB_ARTIFACTORY}/thu-ml/SageAttention/archive/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5.tar.gz
        URL_HASH SHA256=74b6667164f9e368e3799bc2ab59b9b08c4591630f1c6029560208b6fcf354c4
    )
endif()
FetchContent_Populate(repo-sageattention)

set(SAGE3_ENABLE_SM120 OFF)

# SageAttention-3 Blackwell kernels require sm_120a (FP4 MMA).
# Only build the sage3_ops library when the toolchain can target sm_120a.
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "13.0")
    set(SAGE3_CUDA_FLAGS
        "--expt-relaxed-constexpr"
        "--expt-extended-lambda"
        "--use_fast_math"
        "-Xcudafe=--diag_suppress=177"
        "-Xcudafe=--diag_suppress=2361"
        "-gencode=arch=compute_120a,code=sm_120a"
    )
    set(SAGE3_ENABLE_SM120 ON)
endif()

set(SAGE3_SOURCES
    "csrc/attention/sage3.cu"
    "csrc/sage3_extension.cc"
)

if(SAGE3_ENABLE_SM120)
    Python_add_library(sage3_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SAGE3_SOURCES})
    target_compile_options(sage3_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17 -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:${SAGE3_CUDA_FLAGS}>
    )
    target_include_directories(sage3_ops PRIVATE
        ${repo-sageattention_SOURCE_DIR}
        ${repo-sageattention_SOURCE_DIR}/sageattention3_blackwell/sageattn3
        ${repo-sageattention_SOURCE_DIR}/sageattention3_blackwell/sageattn3/blackwell
        ${repo-sageattention_SOURCE_DIR}/sageattention3_blackwell/sageattn3/quantization
        ${repo-cutlass_SOURCE_DIR}/include
        ${repo-cutlass_SOURCE_DIR}/tools/util/include
        ${repo-cutlass_SOURCE_DIR}/examples/common
        ${repo-cutlass_SOURCE_DIR}/examples/41_fused_multi_head_attention
    )
    target_link_libraries(sage3_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)
    # The #included upstream kernels pull in cudart_static/cudadevrt at link
    # time. CUDAToolkit resolves to the pip-installed CUDA's lib64, but pip CUDA
    # 13 lays out the runtime libs under lib/ — add it to the linker search path
    # so the torch-added -lcudart_static -lcudadevrt resolve.
    target_link_options(sage3_ops PRIVATE -L${CUDAToolkit_ROOT}/lib -L${CUDAToolkit_ROOT}/lib64)
    install(TARGETS sage3_ops LIBRARY DESTINATION "sgl_kernel")
else()
    message(STATUS "sage3_ops disabled (requires CUDA >= 13.0 / sm_120a)")
endif()
