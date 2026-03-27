import argparse
import logging
import os

from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

from sglang.jit_kernel.utils import (
    _REGISTERED_DEPENDENCIES,
    DEFAULT_INCLUDE,
    _get_default_target_flags,
    get_jit_cuda_arch,
    override_jit_cuda_arch,
)


def generate_clangd():
    logger = logging.getLogger()
    parser = argparse.ArgumentParser(
        description="Generate .clangd file for sglang jit kernel development."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .clangd file if it exists.",
    )
    parser.add_argument(
        "--dependencies",
        "--dep",
        nargs="*",
        default=[],
        choices=_REGISTERED_DEPENDENCIES.keys(),
        help="Extra dependency libraries to include.",
    )
    parser.add_argument(
        "--cuda-target",
        "--cuda",
        default=None,
        type=str,
        help="Target architecture to generate compile flags for.",
    )
    args = parser.parse_args()

    dep_include_paths = []
    for dep in args.dependencies:
        if dep not in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        dep_include_paths += _REGISTERED_DEPENDENCIES[dep]()

    include_paths = [
        *DEFAULT_INCLUDE,
        find_include_path(),
        find_dlpack_include_path(),
        *dep_include_paths,
    ]
    if args.cuda_target:
        assert args.cuda_target.count(".") == 1
        major, minor = args.cuda_target.split(".")
        major, minor = int(major), int(minor)
        context = override_jit_cuda_arch(major, minor)
        context.__enter__()
    else:
        arch = get_jit_cuda_arch()
        major, minor = arch.major, f"{arch.minor}{arch.suffix}"
        assert (
            major > 0
        ), "Cannot detect CUDA architecture, please specify --cuda-target explicitly."

    compile_flags = [
        "-xcuda",
        f"--cuda-gpu-arch=sm_{major}{minor}",
        "-Wall",
        "-Wextra",
        *_get_default_target_flags(),
        *[f"-isystem{path}" for path in include_paths],
    ]
    # NOTE: skip these flags because clangd don't recognize them
    UNSUPPORTED_FLAGS = {"--expt-relaxed-constexpr"}
    compile_flags = [flag for flag in compile_flags if flag not in UNSUPPORTED_FLAGS]
    compile_flags_str = ",\n    ".join(compile_flags)
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags_str}
  ]
"""
    if os.path.exists(".clangd") and not args.overwrite:
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning("Use --overwrite to force overwrite the existing .clangd file.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


assert __name__ == "__main__"

logging.basicConfig(level=logging.INFO)
generate_clangd()
