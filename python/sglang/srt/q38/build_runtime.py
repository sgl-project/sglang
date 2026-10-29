"""Build SM107a MoE bundles and optional fixed LM-head runtime files."""

import argparse
import ctypes
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from runtime_build import sha256, write_manifest


def build_lm_head(source: Path, inputs: Path, output: Path, cuda_root: Path) -> None:
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(inputs / name) != digest:
            raise ValueError(f"LM input hash mismatch: {name}")
    distribution = importlib.metadata.distribution("flashinfer-python")
    warp = Path(
        distribution.locate_file(
            "flashinfer/gemm/kernels/dense_bf16_gemm_warp_splitk.py"
        )
    )
    if sha256(warp) != manifest["FI_warp_source_SHA256"]:
        raise ValueError(
            "The fixed LM-head plan requires its matching FlashInfer source"
        )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs / "plan.json", output / "plan.json")
    subprocess.run(
        [
            str(cuda_root / "bin/nvcc"),
            "-arch=sm_107a",
            "-std=c++17",
            "-O3",
            "-shared",
            "-Xcompiler=-fPIC",
            str(source / "python/sglang/kernels/jit/csrc/gemm/q38_lm_head_lt.cc"),
            "-o",
            str(output / "lt_probe.so"),
            f"-L{cuda_root / 'lib64'}",
            "-lcublasLt",
            "-lcudart",
        ],
        check=True,
    )
    library = ctypes.CDLL(str(output / "lt_probe.so"))
    library.lm_lt_version.restype = ctypes.c_size_t
    if library.lm_lt_version() != manifest["cublasLt_version"]:
        raise ValueError(
            "The fixed LM-head plan requires its matching cuBLASLt version"
        )
    manifest["files"] = {
        name: sha256(output / name) for name in ("plan.json", "lt_probe.so")
    }
    write_manifest(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-lm", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    tools = Path(__file__).resolve().parent
    cuda_root = Path("/usr/local/cuda")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source / "python")
    environment["TRITON_PTXAS_PATH"] = str(cuda_root / "bin/ptxas")

    def run(script: Path, *arguments: str) -> None:
        subprocess.run(
            [sys.executable, str(script), *arguments], check=True, env=environment
        )

    aot = output / "aot"
    builder = source / "python/sglang/kernels/ops/moe/build_q38_aot.py"
    run(builder, "--source", str(source), "--output", str(aot), "--arch", "sm_107a")
    with tempfile.TemporaryDirectory(prefix="q38-k4-") as temporary:
        run(
            builder,
            "--source",
            str(source),
            "--output",
            temporary,
            "--arch",
            "sm_107a",
            "--router-ksplit",
            "4",
            "--router-only",
        )
        shutil.copytree(
            Path(temporary) / "router", aot / "router-k4", dirs_exist_ok=True
        )
    run(
        tools / "build_mkm_wide.py",
        "--source",
        str(source),
        "--output",
        str(aot / "mkm-wide"),
    )
    run(
        tools / "build_router_wide.py",
        "--source",
        str(source),
        "--output",
        str(aot / "router-wide"),
    )
    lm_inputs = tools / "lm_head"
    lm_built = not args.skip_lm and (lm_inputs / "plan.json").is_file()
    if lm_built:
        build_lm_head(source, lm_inputs, output / "lm-head", cuda_root)
    print(
        json.dumps(
            {
                "output": str(output),
                "arch": "sm_107a",
                "lm_built": lm_built,
            }
        )
    )


if __name__ == "__main__":
    main()
