"""Pinned repositories used only by the CUDA kernel toolchain."""

_CUDA_REDISTRIBUTABLES = [
    struct(
        name = "cuda_cccl",
        sha256 = "ed845eae8c1767706b6ee91e40c608a03f6f633551a849b63f7346d32d73ee60",
        strip_prefix = "cuda_cccl-linux-x86_64-13.0.85-archive",
        url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-13.0.85-archive.tar.xz",
    ),
    struct(
        name = "cuda_crt",
        sha256 = "5a3279a049ffc1cdb951c44cb95206acfdde9e9ae5e87825fc18d7e4a6878bb0",
        strip_prefix = "cuda_crt-linux-x86_64-13.0.88-archive",
        url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.0.88-archive.tar.xz",
    ),
    struct(
        name = "cuda_cudart",
        sha256 = "25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1",
        strip_prefix = "cuda_cudart-linux-x86_64-13.0.96-archive",
        url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.0.96-archive.tar.xz",
    ),
    struct(
        name = "cuda_nvcc",
        sha256 = "48e35be3cfbf4b4fbc16828eaec8a7048ee789403049dc409f7b643d6259cf7a",
        strip_prefix = "cuda_nvcc-linux-x86_64-13.0.88-archive",
        url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.0.88-archive.tar.xz",
    ),
    struct(
        name = "cuda_nvrtc",
        sha256 = "00038aac08e1dba6f1933237dbfb217ac6452ae24fab970edcac808f103ca64b",
        strip_prefix = "cuda_nvrtc-linux-x86_64-13.0.88-archive",
        url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-13.0.88-archive.tar.xz",
    ),
    struct(
        name = "libcublas",
        sha256 = "88bc951efd906032a371153ca61975e0d9c4761e4012169169a6b3a47931606e",
        strip_prefix = "libcublas-linux-x86_64-13.1.0.3-archive",
        url = "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-13.1.0.3-archive.tar.xz",
    ),
    struct(
        name = "libnvvm",
        sha256 = "17ef1665b63670887eeba7d908da5669fa8c66bb73b5b4c1367f49929c086353",
        strip_prefix = "libnvvm-linux-x86_64-13.0.88-archive",
        url = "https://developer.download.nvidia.com/compute/cuda/redist/libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.0.88-archive.tar.xz",
    ),
]

def _cuda_redistributable_repository_impl(rctx):
    for component in _CUDA_REDISTRIBUTABLES:
        rctx.download_and_extract(
            url = component.url,
            sha256 = component.sha256,
            stripPrefix = component.strip_prefix,
        )

    rctx.file(
        "BUILD.bazel",
        """
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "files",
    srcs = glob(["**"], exclude = ["BUILD.bazel"]),
)
""",
    )

cuda_redistributable_repository = repository_rule(
    implementation = _cuda_redistributable_repository_impl,
)
