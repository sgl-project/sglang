"""Declared contract for the transitional local CUDA kernel builder."""

KernelCudaToolchainInfo = provider(
    doc = "Configuration expected from the ambient CUDA and PyTorch toolchain.",
    fields = {
        "cuda_architectures": "CUDA code architectures emitted by the wheel.",
        "cuda_version": "Required CUDA toolkit major.minor version.",
        "torch_cxx11_abi": "Required torch._C._GLIBCXX_USE_CXX11_ABI value.",
    },
)

_SUPPORTED_ARCHITECTURES = {
    "13.0": ["80", "86", "89", "90", "90a", "100f", "120a"],
}

def _kernel_cuda_toolchain_impl(ctx):
    cuda_version = ctx.attr.cuda_version
    if cuda_version not in _SUPPORTED_ARCHITECTURES:
        fail(
            "unsupported CUDA toolkit version '{}'; supported versions are {}".format(
                cuda_version,
                sorted(_SUPPORTED_ARCHITECTURES.keys()),
            ),
        )

    cuda_architectures = ctx.attr.cuda_architectures
    if not cuda_architectures:
        fail("cuda_architectures must not be empty")
    if len(cuda_architectures) != len({arch: True for arch in cuda_architectures}):
        fail("cuda_architectures contains duplicates: {}".format(cuda_architectures))

    unsupported_architectures = [
        arch
        for arch in cuda_architectures
        if arch not in _SUPPORTED_ARCHITECTURES[cuda_version]
    ]
    if unsupported_architectures:
        fail(
            "CUDA {} does not support declared architectures {}; supported architectures are {}".format(
                cuda_version,
                unsupported_architectures,
                _SUPPORTED_ARCHITECTURES[cuda_version],
            ),
        )

    if ctx.attr.torch_cxx11_abi not in [0, 1]:
        fail("torch_cxx11_abi must be 0 or 1")

    return [
        platform_common.ToolchainInfo(
            kernel_cuda = KernelCudaToolchainInfo(
                cuda_architectures = cuda_architectures,
                cuda_version = cuda_version,
                torch_cxx11_abi = ctx.attr.torch_cxx11_abi,
            ),
        ),
    ]

kernel_cuda_toolchain = rule(
    implementation = _kernel_cuda_toolchain_impl,
    attrs = {
        "cuda_architectures": attr.string_list(mandatory = True),
        "cuda_version": attr.string(mandatory = True),
        "torch_cxx11_abi": attr.int(mandatory = True),
    },
)
