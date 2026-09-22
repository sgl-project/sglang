"""AOT-build the opt-in 1024-bin selector pair for the SGL main Q1 scorer.

This separate entry preserves the legacy builder and its B1/B2/B4/B8/B16 defaults.
Generate and verify sources without CUDA: ``python build_fused.py --check-sources``.
Build with CUDA_VISIBLE_DEVICES='', MAX_JOBS=1, CUTE_DSL_ARCH=sm_100a.
The artifact manifest records exact qualified source hashes; rebuilding a binary
does not itself qualify it numerically on a new toolchain or GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REFERENCE_LIBRARIES = {
    "small": "015c4f33e3f1fc0fb1aa8f1ae16d7fe79aef9b58bc9cf1bd5d34317c2f515018",
    "large": "a38033a368911ef2beaa00bfdb297b027430fea865aaa6afdd080f4e0374ecb2",
}
WORKSPACE_TAIL_OFFSET = 20_973_568
WORKSPACE_BYTES = WORKSPACE_TAIL_OFFSET + 128 * 8


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_sources():
    legacy = load("_litetopk_fused_legacy_build", HERE / "build.py")
    transform = load("_litetopk_fused_source", HERE / "selectors/fused_source.py")
    base = legacy._selector_source(1024)
    vendor = (HERE / "vendor/gvr2_topk_decode.py").read_text()
    texts = {route: transform.source(base, vendor, route) for route in ("small", "large")}
    return legacy, texts


def prepare_output_directory(directory):
    # The independent DeepGEMM build shares this root under deepgemm/.
    # Reserve only paths owned by this selector builder, including symlinks.
    for name in ("manifest.json", "small", "large"):
        path = directory / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite selector artifacts: {path}")
    directory.mkdir(parents=True, exist_ok=True)


def build_one(directory, route, text, legacy):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute import runtime

    threads, unroll = (512, 1) if route == "small" else (1024, 4)
    folder = directory / route
    folder.mkdir()
    generated = folder / "generated.py"
    generated.write_text(text)
    # Preserve the qualified derivative's import ABI and compilation module name.
    load("_gvr_hist_prologue_acqrel_hsplitq_original_20260918", HERE / "vendor/gvr2_topk_decode.py")
    module = load(f"_dynamic128v6_{route}", generated)

    def fake(dtype, rank, align=16):
        return legacy._selector_tensor(runtime, cute, dtype, rank, align)

    kernel = module.SplitQAcqRelHistogramPrologueGvrKernel(
        threads, unroll, 1, 256, 2, True,
        varlen=True, next_n=1, cr_shift=0, r_const=1,
    )
    compiled = cute.compile(
        kernel,
        fake(cutlass.Float32, 2),
        fake(cutlass.Int32, 2),
        fake(cutlass.Int32, 2),
        fake(cutlass.Int32, 1),
        *([cutlass.Int32(0)] * 11),
        fake(cutlass.Int32, 1, 4),
        *([cutlass.Int32(0)] * 5),
        fake(cutlass.Int32, 1),
        fake(cutlass.Int32, 1, 4),
        cutlass.Int32(0),
        fake(cutlass.Int32, 2),
        fake(cutlass.Int32, 1, 4),
        cutlass.Int32(0),
        cutlass.Int32(0),
        fake(cutlass.Int32, 1, 4),
        stream=runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --gpu-arch sm_100a",
    )
    obj, library = folder / "select.o", folder / "select.so"
    compiled.export_to_c(str(obj), function_name="select")
    runtime_libraries = runtime.find_runtime_libraries(enable_tvm_ffi=True)
    command = ["gcc", "-shared", "-o", str(library), str(obj), *map(str, runtime_libraries)]
    subprocess.run(command, check=True)
    return dict(
        name=f"dynamic128v6_{route}", batch=128, max_rows=128, bins=1024,
        threads=threads, u=unroll, R=1, diagnostic_records_per_row=1,
        flattened_grid_ctas=128, device_route=route, cutoff=8,
        runtime_row_parts="active<=2:64, <=8:16, <=16:8, <=32:4, <=64:2, else:1",
        geometry=f"flat128_guarded_{route}", dynamic_active_batch=True,
        candidate_capacity=16384, workspace_tail_offset=WORKSPACE_TAIL_OFFSET,
        workspace_bytes=WORKSPACE_BYTES, adaptive=False,
        certificate_miss="exact_fallback", sampling=True,
        source=f"{route}/generated.py", source_sha256=sha(generated),
        library=f"{route}/select.so", library_sha256=sha(library),
        reference_library_sha256=REFERENCE_LIBRARIES[route],
        matches_reference_binary=sha(library) == REFERENCE_LIBRARIES[route],
        link_command=command,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "build/fused")
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args()
    sources = (HERE / "build.py", HERE / "build_fused.py", HERE / "selectors/selector.py",
               HERE / "selectors/fused_source.py", HERE / "vendor/gvr2_topk_decode.py")
    hashes = {str(path.relative_to(HERE)): sha(path) for path in sources}
    legacy, texts = generate_sources()
    source_hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    if args.check_sources:
        print(json.dumps(dict(status="ok", qualified_source_hashes=source_hashes,
                              source_dependencies=hashes), indent=2))
        return
    legacy._require_build_environment()
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("build must start before CUDA initialization")
    directory = args.output_dir.resolve()
    prepare_output_directory(directory)
    records = {route: build_one(directory, route, text, legacy) for route, text in texts.items()}
    if any(sha(HERE / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("source changed during the build")
    manifest = dict(
        schema="sglang-litetopk-fused-selectors-v1", status="built",
        name="dynamic128v6_guarded_pair", geometry="device_guarded_pair", cutoff=8,
        target="sm_100a", bins=1024, max_rows=128, envelope=1048576, topk=2048,
        page_size=64, producer_diagnostics_words=148,
        workspace_tail_offset=WORKSPACE_TAIL_OFFSET, workspace_bytes=WORKSPACE_BYTES,
        source_dependencies=hashes, torch=torch.__version__, torch_cuda=torch.version.cuda,
        versions={name: importlib.metadata.version(name) for name in
                  ("nvidia-cutlass-dsl", "apache-tvm-ffi", "cuda-python")},
        qualified_source_provenance="Byte-identical V6 pair used by 80 SGL main 1853080c matrix observations; exact fallback and physical page mapping retained.",
        binary_validation="Compare source/library hashes and qualify rebuilt artifacts before claiming new-toolchain performance.",
        **records,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
