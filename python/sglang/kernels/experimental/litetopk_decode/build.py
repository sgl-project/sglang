"""AOT-build the qualified B200 LiteTopK decode producer and selectors.

Run with the GPU hidden so CuTe compilation cannot accidentally execute kernels:

  CUDA_VISIBLE_DEVICES='' MAX_JOBS=1 CUTE_DSL_ARCH=sm_100a \
    python -m sglang.kernels.experimental.litetopk_decode.build --all
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
BUILD = HERE / "build"
ENVELOPE = 1_048_576
TOPK = 2_048
CTAS = 148
WORKSPACE_BYTES = 20_973_568
SELECTOR_CONFIGS = {
    1: ("selector_b1.py", 1),
    2: ("selector_coarse1024.py", 1),
    4: ("selector_coarse1024.py", 4),
    8: ("selector_coarse1024.py", 4),
    16: ("selector_coarse1024.py", 4),
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_build_environment() -> None:
    expected = {
        "CUDA_VISIBLE_DEVICES": "",
        "MAX_JOBS": "1",
        "CUTE_DSL_ARCH": "sm_100a",
    }
    actual = {name: os.environ.get(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"required build environment: {expected}; got {actual}")


def _deepgemm_include() -> Path:
    import torch

    site = Path(torch.__file__).resolve().parent.parent
    root = Path(
        os.environ.get(
            "LITETOPK_NATIVE_DEEPGEMM_INCLUDE",
            site / "vllm/third_party/deep_gemm/include",
        )
    ).resolve()
    for name in ("deep_gemm", "cute", "cutlass"):
        if not (root / name).is_dir():
            raise FileNotFoundError(root / name)
    return root


def _build_producer(
    *,
    name: str,
    source: Path,
    bins: int,
    namespace: str,
) -> dict:
    import torch
    from torch.utils.cpp_extension import load

    directory = BUILD / name
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / f"{namespace}.so"
    if library.exists():
        raise FileExistsError(library)

    include = _deepgemm_include()
    host_includes = sorted(
        (Path(torch.__file__).resolve().parent.parent / "nvidia").glob("*/include")
    )
    flags = [
        "-O3",
        "-std=c++20",
        "-DHPC_TARGET_ARCH=100",
        "-Dhpc=gred_probe_upstream",
    ]
    if bins == 1024:
        flags.extend(
            (
                "-DLITETOPK_COARSE_BINS=1024",
                f"-DLITETOPK_PRODUCER_NAMESPACE={namespace}",
            )
        )
    result = Path(
        load(
            name=namespace,
            sources=[str(source)],
            build_directory=str(directory),
            extra_include_paths=[
                str(include),
                str(HERE),
            ],
            extra_cflags=flags + ["-I" + str(path) for path in host_includes],
            extra_cuda_cflags=flags
            + [
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--ptxas-options=-v",
                "-gencode=arch=compute_100a,code=sm_100a",
            ],
            extra_ldflags=["-lcuda"],
            is_python_module=False,
            verbose=True,
        )
    )
    return {
        "name": name,
        "bins": bins,
        "namespace": namespace,
        "source_sha256": _sha256(source),
        "library": str(result),
        "library_sha256": _sha256(result),
    }


def build_producers() -> list[dict]:
    return [
        _build_producer(
            name="producer2048",
            source=HERE
            / "temporal_decode/decode_topk_batched/producer_v2/producer_batch.cu",
            bins=2048,
            namespace="litetopk_batched_producer_20260920",
        ),
        _build_producer(
            name="producer1024",
            source=HERE
            / "batchdecode-coarse-opt-20260921/producer/producer_batch.cu",
            bins=1024,
            namespace="litetopk_batched_producer_coarse1024_20260921",
        ),
    ]


def _selector_tensor(runtime, cute, dtype, rank: int, align: int = 16):
    return runtime.make_fake_compact_tensor(
        dtype,
        tuple(cute.sym_int() for _ in range(rank)),
        stride_order=tuple(reversed(range(rank))),
        assumed_align=align,
    )


def _build_selector(batch: int, source_name: str, unroll: int) -> dict:
    import torch
    from cutlass.cute import runtime

    module_name = "_gvr_hist_prologue_acqrel_hsplitq_original_20260918"
    original = _load(module_name, HERE / "vendor/gvr2_topk_decode.py")
    host = _load("_litetopk_gvr2_host", HERE / "vendor/gvr2_topk_host.py")
    derivative = _load(
        f"_litetopk_selector_b{batch}",
        HERE / "selectors" / source_name,
    )

    route = host.route_streaming(batch, ENVELOPE, ENVELOPE, TOPK, force_main=True)
    grid, template = tuple(route["grid"]), list(route["tpl"])
    r_const = CTAS // batch
    if grid != (r_const, batch) or route["block"] != 1024:
        raise RuntimeError(f"unexpected selector route for B={batch}: {route}")
    template[1] = unroll

    cute, cutlass = original.cute, original.cutlass
    scores = _selector_tensor(runtime, cute, cutlass.Float32, 2)
    hints = _selector_tensor(runtime, cute, cutlass.Int32, 2)
    output = _selector_tensor(runtime, cute, cutlass.Int32, 2)
    workspace = _selector_tensor(runtime, cute, cutlass.Int32, 1)
    lengths = _selector_tensor(runtime, cute, cutlass.Int32, 1, 4)
    histogram = _selector_tensor(runtime, cute, cutlass.Int32, 1)
    producer_diag = _selector_tensor(runtime, cute, cutlass.Int32, 1, 4)
    diagnostics = _selector_tensor(runtime, cute, cutlass.Int32, 2)
    page_table = _selector_tensor(runtime, cute, cutlass.Int32, 1, 4)
    stream = runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = derivative.SplitQAcqRelHistogramPrologueGvrKernel(
        1024,
        unroll,
        template[2],
        template[3],
        template[4],
        True,
        tshg=False,
        varlen=True,
        next_n=1,
        cr_shift=0,
        r_const=r_const,
    )
    compiled = cute.compile(
        kernel,
        scores,
        hints,
        output,
        workspace,
        *([cutlass.Int32(0)] * 11),
        lengths,
        *([cutlass.Int32(0)] * 5),
        histogram,
        producer_diag,
        cutlass.Int32(0),
        diagnostics,
        page_table,
        cutlass.Int32(0),
        cutlass.Int32(0),
        stream=stream,
        options="--enable-tvm-ffi --gpu-arch sm_100a",
    )

    directory = BUILD / "selectors" / f"b{batch}"
    directory.mkdir(parents=True, exist_ok=True)
    obj, library = directory / "select.o", directory / "select.so"
    if obj.exists() or library.exists():
        raise FileExistsError(directory)
    compiled.export_to_c(str(obj), function_name="select")
    runtime_libraries = runtime.find_runtime_libraries(enable_tvm_ffi=True)
    subprocess.run(
        ["gcc", "-shared", "-o", str(library), str(obj), *map(str, runtime_libraries)],
        check=True,
    )
    return {
        "batch": batch,
        "histogram_bins": 2048 if batch == 1 else 1024,
        "active_ctas_at_128k": 128 if batch == 1 else None,
        "unroll": unroll,
        "grid": [r_const, batch, 1],
        "workspace_bytes": WORKSPACE_BYTES,
        "source": source_name,
        "source_sha256": _sha256(HERE / "selectors" / source_name),
        "library": str(library),
        "library_sha256": _sha256(library),
    }


def build_selectors(batches: list[int]) -> list[dict]:
    unknown = sorted(set(batches) - set(SELECTOR_CONFIGS))
    if unknown:
        raise ValueError(f"unsupported batch sizes: {unknown}")
    return [
        _build_selector(batch, *SELECTOR_CONFIGS[batch]) for batch in batches
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--producers", action="store_true")
    parser.add_argument("--selectors", action="store_true")
    parser.add_argument("--batches", default="1,2,4,8,16")
    args = parser.parse_args()
    if not (args.all or args.producers or args.selectors):
        parser.error("choose --all, --producers, or --selectors")

    _require_build_environment()
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("build must start before CUDA initialization")
    report = {
        "schema": "sglang-litetopk-decode-aot-v1",
        "target": "sm_100a",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "cuda-python")
        },
    }
    if args.all or args.producers:
        report["producers"] = build_producers()
    if args.all or args.selectors:
        batches = [int(item) for item in args.batches.split(",")]
        report["selectors"] = build_selectors(batches)
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
