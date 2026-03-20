import argparse
import logging
import os
import subprocess

from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

from sglang.jit_kernel.utils import DEFAULT_INCLUDE, get_jit_cuda_arch


def generate_clangd():
    parser = argparse.ArgumentParser(
        description="Generate .clangd file for sglang jit kernel development."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .clangd file if it exists.",
    )

    logger = logging.getLogger()
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
    )
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    major, minor = compute_cap.split(".")
    compile_flags = ",\n    ".join(
        [
            "-xcuda",
            f"--cuda-gpu-arch=sm_{major}{minor}",
            "-Wall",
            "-Wextra",
            get_jit_cuda_arch().jit_flag,
            *[f"-isystem{path}" for path in include_paths],
        ]
    )
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    if os.path.exists(".clangd") and not parser.parse_args().overwrite:
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning("Use --overwrite to force overwrite the existing .clangd file.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


assert __name__ == "__main__"

generate_clangd()
