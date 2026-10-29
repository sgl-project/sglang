"""Build the M8/M16 split-K router bundle without a CUDA device."""

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
    source = args.source / "python/sglang/kernels/jit/csrc/moe/q38_router_wide.cu"
    args.output.mkdir(parents=True, exist_ok=True)
    for capacity in (8, 16):
        flags = [
            "--gpu-architecture=sm_107a",
            "-std=c++17",
            "-default-device",
            f"-DQ35_ROUTER_MAX_M={capacity}",
            "-DQ35_ROUTER_KSPLIT=2",
            "-DFN_HIDDEN=2560",
            "-DFN_EXPERTS=512",
            "-DFN_TOP_K=10",
            "-DFN_STORAGE_I=192",
        ]
        compile_cuda(
            source, args.output / f"route-m{capacity}.cubin", flags, args.cuda_root
        )
    write_manifest(
        args.output,
        {
            "arch": "sm_107a",
            "shape": [2560, 512, 2],
            "capacities": [8, 16],
            "source_sha256": sha256(source),
            "sha256": {
                f"route-m{m}.cubin": sha256(args.output / f"route-m{m}.cubin")
                for m in (8, 16)
            },
        },
    )


if __name__ == "__main__":
    main()
