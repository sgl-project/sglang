"""Build the two grouped-QSA bindings against the active SGLang PyTorch ABI.

Pass the flash-attention-v100 directory bundled in 1Cat-vLLM. Sources are
copied to a separate build directory; the supplied checkout is not modified.
The BSD-3-Clause license and source hashes are retained with the build.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "_sm70_grouped_qsa_v100.so"


def default_output() -> Path:
    """SGLang's SM70 prebuilt artifact directory, the single source of truth."""
    try:
        from sglang.kernels.sm70_paths import sm70_prebuilt_dir
    except Exception:
        return ROOT / "python/sglang/kernels/prebuilt" / OUTPUT_NAME
    return sm70_prebuilt_dir() / OUTPUT_NAME


BINDINGS = r"""
#include <torch/extension.h>
#include <optional>
at::Tensor flash_attention_grouped_sparse_page4_plan(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int, int, int);
at::Tensor flash_attention_grouped_sparse_page4(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::Tensor&, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_sparse_page4_plan_fwd", &flash_attention_grouped_sparse_page4_plan);
  m.def("grouped_sparse_page4_fwd", &flash_attention_grouped_sparse_page4);
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--build-dir", type=Path, default=Path.home() / ".cache/sglang-v100/grouped-qsa"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output(),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    build = args.build_dir.resolve()
    snapshot = build / "source"
    cuda_file = source / "kernel/flash_decode_paged.cu"
    contents = cuda_file.read_text()
    if "flash_attention_grouped_sparse_page4_plan" not in contents:
        raise RuntimeError("The source checkout lacks grouped Page4 prefill")
    snapshot.mkdir(parents=True, exist_ok=True)
    for directory in ["include", "kernel"]:
        for path in (source / directory).rglob("*"):
            if path.is_file() and (path.suffix in (".h", ".cuh") or path == cuda_file):
                target = snapshot / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    shutil.copy2(source / "LICENSE", snapshot / "LICENSE")
    bindings = build / "bindings.cpp"
    bindings.write_text(BINDINGS)
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    module = load(
        name="flash_attn_v100_cuda",
        sources=[str(bindings), str(snapshot / "kernel/flash_decode_paged.cu")],
        extra_include_paths=[str(snapshot / "include"), str(snapshot / "kernel")],
        build_directory=str(build),
        verbose=True,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
    )
    assert hasattr(module, "grouped_sparse_page4_plan_fwd") and hasattr(
        module, "grouped_sparse_page4_fwd"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(module.__file__, args.output)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    metadata = dict(
        source=str(source),
        source_revision=revision.stdout.strip() if revision.returncode == 0 else None,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        output=str(args.output),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        source_hashes={
            str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in snapshot.rglob("*")
            if p.is_file()
        },
    )
    (build / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
