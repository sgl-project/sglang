"""AOT-build the LiteTopK decode selector pair (512- and 1024-thread routes) for sm_100a.

Build with CUDA_VISIBLE_DEVICES='', MAX_JOBS=1, CUTE_DSL_ARCH=sm_100a.
``--check-sources`` parses the selector source and prints its hash without CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "selectors/selector.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_one(directory, route, source, capacity):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute import runtime

    threads = 512 if route == "small" else 1024
    folder = directory / route
    folder.mkdir()
    copy = folder / "selector.py"
    copy.write_text(source)
    spec = importlib.util.spec_from_file_location(f"litetopk_{route}", copy)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def fake(dtype, rank, align=16):
        return runtime.make_fake_compact_tensor(
            dtype, tuple(cute.sym_int() for _ in range(rank)),
            stride_order=tuple(reversed(range(rank))), assumed_align=align)

    compiled = cute.compile(
        module.HistogramSelector(threads, capacity),
        fake(cutlass.Float32, 2),  # scores
        fake(cutlass.Int32, 2),  # output
        fake(cutlass.Int32, 1),  # workspace
        cutlass.Int32(0),  # logical length envelope
        fake(cutlass.Int32, 1, 4),  # per-row lengths
        fake(cutlass.Int32, 1),  # histogram
        fake(cutlass.Int32, 2),  # selector diagnostics
        fake(cutlass.Int32, 1, 4),  # page table
        cutlass.Int32(0),  # page-table row stride
        fake(cutlass.Int32, 1, 4),  # active rows
        stream=runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --gpu-arch sm_100a --enable-assertions")
    obj, library = folder / "select.o", folder / "select.so"
    compiled.export_to_c(str(obj), function_name="select")
    subprocess.run(["gcc", "-shared", "-o", str(library), str(obj),
                    *map(str, runtime.find_runtime_libraries(enable_tvm_ffi=True))], check=True)
    return dict(threads=threads, source=f"{route}/selector.py", source_sha256=sha(copy),
                library=f"{route}/select.so", library_sha256=sha(library))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "build/fused")
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--candidate-capacity", type=int, default=32768,
                        choices=(16384, 32768, 65536, 131072),
                        help="per-row candidate limit; overflow raises a CUDA device error")
    args = parser.parse_args()
    source = SOURCE.read_text()
    compile(source, str(SOURCE), "exec")
    if args.check_sources:
        print(json.dumps(dict(status="ok", source_sha256=sha(SOURCE)), indent=2))
        return
    expected = dict(CUDA_VISIBLE_DEVICES="", MAX_JOBS="1", CUTE_DSL_ARCH="sm_100a")
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"required build environment: {expected}")
    directory = args.output_dir.resolve()
    # The DeepGEMM build shares this root under deepgemm/; only the selector paths are reserved.
    for name in ("manifest.json", "small", "large"):
        if (directory / name).exists() or (directory / name).is_symlink():
            raise FileExistsError(f"refusing to overwrite selector artifacts: {directory / name}")
    directory.mkdir(parents=True, exist_ok=True)
    records = {route: build_one(directory, route, source, args.candidate_capacity) for route in ("small", "large")}
    if SOURCE.read_text() != source:
        raise RuntimeError("source changed during the build")
    manifest = dict(
        schema="sglang-litetopk-fused-selectors-v4", target="sm_100a", cutoff=8,
        histogram_mapping="hybrid1024-fp16rn16-unit-overflow-v1", candidate_capacity=args.candidate_capacity,
        workspace_layout="fixed-header-pair-v1", workspace_candidate_offset=4096, workspace_tail_offset=2048,
        workspace_bytes=4096 + 160 * args.candidate_capacity * 8,
        certificate_miss="error", overflow_policy="error", device_assertions=True,
        versions={name: importlib.metadata.version(name) for name in
                  ("nvidia-cutlass-dsl", "apache-tvm-ffi", "cuda-python")},
        **records,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
