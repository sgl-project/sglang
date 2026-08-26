"""Declared CUDA kernel build inputs and configuration contract."""

KernelCudaToolchainInfo = provider(
    doc = "Pinned CUDA, PyTorch, and build-frontend inputs.",
    fields = {
        "build_python_paths": "Import roots for the pinned PEP 517 build frontend.",
        "cuda_architectures": "CUDA code architectures emitted by the wheel.",
        "cuda_root": "Exec-root-relative CUDA toolkit root.",
        "cuda_version": "Required CUDA toolkit major.minor version.",
        "inputs": "CUDA, PyTorch, and build-frontend files.",
        "python_abi": "Required CPython ABI tag.",
        "torch_cxx11_abi": "Required torch._C._GLIBCXX_USE_CXX11_ABI value.",
        "torch_root": "Exec-root-relative root of the unpacked PyTorch wheel.",
        "torch_version": "Required PyTorch wheel version.",
    },
)

_SUPPORTED_ARCHITECTURES = {
    "13.0": ["80", "86", "89", "90", "90a", "100f", "120a"],
}
_SUPPORTED_STACKS = [
    ("13.0", "2.13.0+cu130", "cp312", 1),
]

def _external_root(dep):
    files = dep[DefaultInfo].files.to_list()
    if not files:
        fail("{} does not provide any files".format(dep.label))

    root = dep.label.workspace_root
    if not root:
        fail("{} is not in an external repository".format(dep.label))
    for file in files:
        if not file.path.startswith(root + "/"):
            fail("{} contains a file outside {}: {}".format(dep.label, root, file.path))
    return root

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
    stack = (
        cuda_version,
        ctx.attr.torch_version,
        ctx.attr.python_abi,
        ctx.attr.torch_cxx11_abi,
    )
    if stack not in _SUPPORTED_STACKS:
        fail(
            "unsupported CUDA/PyTorch/Python/ABI stack {}; supported stacks are {}".format(
                stack,
                _SUPPORTED_STACKS,
            ),
        )

    cuda_root = _external_root(ctx.attr.cuda_files)
    torch_root = _external_root(ctx.attr.torch_files)
    build_python_paths = [
        _external_root(dep)
        for dep in ctx.attr.build_python_packages
    ]

    return [
        platform_common.ToolchainInfo(
            kernel_cuda = KernelCudaToolchainInfo(
                build_python_paths = build_python_paths,
                cuda_architectures = cuda_architectures,
                cuda_root = cuda_root,
                cuda_version = cuda_version,
                inputs = depset(
                    transitive = [
                        ctx.attr.cuda_files[DefaultInfo].files,
                        ctx.attr.torch_files[DefaultInfo].files,
                    ] + [
                        dep[DefaultInfo].files
                        for dep in ctx.attr.build_python_packages
                    ],
                ),
                python_abi = ctx.attr.python_abi,
                torch_cxx11_abi = ctx.attr.torch_cxx11_abi,
                torch_root = torch_root,
                torch_version = ctx.attr.torch_version,
            ),
        ),
    ]

kernel_cuda_toolchain = rule(
    implementation = _kernel_cuda_toolchain_impl,
    attrs = {
        "build_python_packages": attr.label_list(allow_files = True),
        "cuda_architectures": attr.string_list(mandatory = True),
        "cuda_files": attr.label(mandatory = True),
        "cuda_version": attr.string(mandatory = True),
        "python_abi": attr.string(mandatory = True),
        "torch_cxx11_abi": attr.int(mandatory = True),
        "torch_files": attr.label(mandatory = True),
        "torch_version": attr.string(mandatory = True),
    },
)
