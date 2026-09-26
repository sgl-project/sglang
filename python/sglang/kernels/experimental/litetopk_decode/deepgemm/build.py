#!/usr/bin/env python3
"""Build the pinned, patched DeepGEMM into a private directory; never install it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

HERE = Path(__file__).resolve().parent
MAIN_COMMIT = "1853080cf74589228fcd5dc333197c2ca6df49cf"
SUBMODULES = {
    "cutlass": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
    "fmt": "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28",
}
PATCH_SHA256 = "4397806ac79b5d03e69761832cc7b55d1ae1035f078460309e47091e57c17548"
# Four patched files plus the unchanged package __init__.py, which the fused loader cross-checks.
PATCHED_SOURCES = {
    "csrc/apis/attention.hpp": "fe27f86527e76165be99440488d389ac1307ba86f8961830030eded113278e29",
    "csrc/jit_kernels/impls/sm100_mqa_logits.hpp": "f41527f5717d3c1f7bb1dd289cf6e9b57e26b4eb2a33eb0dc3b5ed786e0272de",
    "deep_gemm/__init__.py": "05075513e4ddff743bd5d13745cc2c2e4600ae890b0114f5df2094292c4460f9",
    "deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh": "251ec364232113b0c66b945cd082673f277b704bd2c6b6326faf4abae742fc1b",
    "deep_gemm/include/deep_gemm/epilogue/coarse_histogram.cuh": "23a0c34fe2485b87891c13dcc8ad24ec6fc7d4b2bb1c8c708696849c0029f486",
}
# Unchanged baseline and dependency headers, checked even in containers without Git.
BASELINE_FILES = {
    "csrc/python_api.cpp": "3f67dbf5fcf9b5973f574307f8ef487db5864974f11849ca6905879656951576",
    "csrc/jit/compiler.hpp": "aec70f54bffc693a935ef9775ee9b40bc8857dcb8b464e6732f39a86e3292d62",
    "csrc/jit/device_runtime.hpp": "b567dbc0cc4882a4030bbb2935edbde454d3820d121388a00b1179dd23c3b9c7",
    "csrc/jit/kernel_runtime.hpp": "94453293fe1b360b51c3327d8e1587ac52b58c7a958907ed5cdce213f927b366",
    "deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh": "13d1485d31e976320b3e0f1bec10e6bbb9ef6f9173b3c5cb6bc84d572c808fbd",
    "deep_gemm/include/deep_gemm/layout/mqa_logits.cuh": "20af73fba45a7ab401d5f5834653725e6e777ac469a8b8f544cdda3351849294",
    "third-party/cutlass/include/cutlass/cutlass.h": "b0983633d014d2d7955d6e4963fe3338e8f444ad3ae4ff7096041f4d87fcf82e",
    "third-party/cutlass/include/cute/tensor.hpp": "165ce9484ddf256881662136a0d25004299d87e36986c1de33b27030a0955b5c",
    "third-party/cutlass/include/cute/arch/mma_sm100_umma.hpp": "7ba0cef9f7fe47148851c8c259df4943f6b1ffa1ba5ef268878596a375a54b1e",
    "third-party/cutlass/include/cutlass/arch/barrier.h": "90716cfd8e3300cac27e6d0bfeca63fd013406f47da0985835982c17c20da2cf",
    "third-party/fmt/include/fmt/format.h": "3bf317fcad21f1120a4df3cfa2d35064e22787683e4761a9e5fa9850098c1905",
    "third-party/fmt/include/fmt/core.h": "b2b6692145b11c7c2774db106078c97575a47d18e80c7783c68566475325ebb4",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_source(source):
    if sha(HERE / "opt_in_histogram.patch") != PATCH_SHA256:
        raise ValueError("companion patch SHA256 does not match the qualified patch")
    for name, expected in {**PATCHED_SOURCES, **BASELINE_FILES}.items():
        if sha(source / name) != expected:
            raise ValueError(f"source/dependency SHA256 mismatch: {name}")
    verification = dict(known_source_hashes_verified=True, git_verified=None,
                        git_heads={}, note="Git unavailable or checkout metadata absent; "
                        "commit/pins are declared provenance, not verified Git HEADs.")
    if not shutil.which("git"):
        return verification
    locations = {"main": (source, MAIN_COMMIT), **{
        name: (source / "third-party" / name, commit)
        for name, commit in SUBMODULES.items()}}
    for name, (directory, expected) in locations.items():
        top = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True)
        if top.returncode or Path(top.stdout.strip()).resolve() != directory:
            return verification
        head = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if head != expected:
            raise ValueError(f"unexpected Git HEAD for {name}: {head}")
        verification["git_heads"][name] = head
    verification.update(git_verified=True, note="Main and both submodule HEADs verified.")
    return verification


def source_inputs(source):
    suffixes = {".hpp", ".cuh", ".cpp", ".py", ".h", ".inl"}
    directories = [source / name for name in
                   ("csrc", "deep_gemm", "third-party/cutlass/include", "third-party/fmt/include")]
    return {str(path.relative_to(source)): sha(path)
            for directory in directories for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in suffixes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="pinned main checkout with submodules and companion patch applied")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE.parent / "build/fused/deepgemm",
                        help="fresh private output directory (no installation)")
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    verification = check_source(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    import torch

    before = source_inputs(source)
    target, work = output / "deep_gemm", output / "host-build"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "deep_gemm", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    dependencies = source / "third-party"
    for name in ("cute", "cutlass"):
        shutil.copytree(dependencies / "cutlass/include" / name, target / "include" / name,
                        dirs_exist_ok=True)
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    torchdir = Path(torch.__file__).parent
    nvidia = torchdir.parent / "nvidia/cu13"
    includes = [cuda / "include", cuda / "include/cccl", sysconfig.get_path("include"),
                torchdir / "include", torchdir / "include/torch/csrc/api/include",
                source / "deep_gemm/include", dependencies / "cutlass/include",
                dependencies / "fmt/include", nvidia / "include"]
    cxx = os.environ.get("CXX", "/usr/bin/g++-13")
    flags = ["-std=c++20", "-O3", "-shared", "-fPIC", "-Wno-psabi",
             "-Wno-deprecated-declarations", "-DTORCH_EXTENSION_NAME=_C",
             f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]
    library = work / "_C.so"
    command = [cxx, *flags, *["-I" + str(path) for path in includes],
               str(source / "csrc/python_api.cpp"), "-o", str(library),
               f"-L{cuda}/lib64", f"-L{torchdir}/lib", f"-L{nvidia}/lib",
               "-lcudart", "-l:libnvrtc.so.13", "-l:libcublasLt.so.13",
               "-l:libcublas.so.13", "-ltorch", "-ltorch_cpu", "-ltorch_python",
               "-lc10", "-lc10_cuda", "-ltorch_cuda"]
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (work / "compile.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"host build failed; see {work / 'compile.log'}")
    if source_inputs(source) != before:
        raise RuntimeError("source/dependencies changed during build")
    shutil.copy2(library, target / "_C.so")
    staged = {str(path.relative_to(output)): sha(path)
              for path in sorted(target.rglob("*")) if path.is_file()}
    report = dict(
        schema="litetopk-deepgemm-main-fusion-v1", status="ok",
        package="deep_gemm", library="deep_gemm/_C.so",
        library_sha256=sha(target / "_C.so"), staged_files=staged,
        main_commit=MAIN_COMMIT, submodules=SUBMODULES,
        patch_sha256=PATCH_SHA256, patched_source_sha256=PATCHED_SOURCES,
        baseline_files_sha256=BASELINE_FILES, base_verification=verification,
        source_sha256=before, source_directory=str(source), command=command,
        torch=torch.__version__, torch_cuda=torch.version.cuda,
        note="Private staged package; no installation, environment change or GPU execution.")
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "schema", "package", "library_sha256")}), flush=True)


if __name__ == "__main__":
    main()
