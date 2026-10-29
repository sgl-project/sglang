"""Build the M8/M12/M16 routed MoE bundle without a CUDA device."""

import argparse
from pathlib import Path

from runtime_build import compile_cuda, sha256, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    args = parser.parse_args()
    source = args.source / "python/sglang/kernels/jit/csrc/moe/q38_mkm_wide.cu"
    args.output.mkdir(parents=True, exist_ok=True)
    flags = [
        "--gpu-architecture=sm_107a",
        "-std=c++17",
        "-default-device",
        "--generate-line-info",
        "--ptxas-options=-v",
        "-DQ38_MKM_WIDE_SMEM_BYTES=215040",
        "-DSPEC_STAGES=4",
        "-DTAILV=3",
    ]
    cubin = args.output / "wide.cubin"
    compile_cuda(source, cubin, flags, args.cuda_root)
    write_manifest(
        args.output,
        {
            "arch": "sm_107a",
            "shape": [2560, 192, 512, 10, 16],
            "source_sha256": sha256(source),
            "cubin_sha256": sha256(cubin),
            "symbol": "q38_mkm_wide",
            "shared": 215040,
            "threads": 256,
            "grid": "M*10",
            "cluster": 1,
            "spec_stages": 4,
        },
    )


if __name__ == "__main__":
    main()
