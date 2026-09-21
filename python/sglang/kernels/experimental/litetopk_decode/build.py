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
    1: (2048, 1),
    2: (1024, 1),
    4: (1024, 4),
    8: (1024, 4),
    16: (1024, 4),
}
SELECTOR_SHA256 = {
    2048: "2c9785d08b1a6e812c12fe417ea4fb551587dff629364c398c3958ce8d7f1326",
    1024: "faa13c98f6d59eb74961e3649b1175b89c9976c4d22a2239a3d560112600b2c4",
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


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"selector source anchor count != 1: {old[:80]!r}")
    return text.replace(old, new, 1)


def _selector_source(bins: int) -> str:
    text = (HERE / "selectors/selector.py").read_text()
    if bins not in SELECTOR_SHA256:
        raise ValueError(f"unsupported histogram size: {bins}")
    if bins == 1024:
        replacements = (
            (
                "    # Frozen coarse_boundary; exceptional endpoints are explicit, never half centers.\n",
                "    # Lower FP32 boundary of one grouped ordered-FP16 bin.\n",
            ),
            ("    if tau >= cutlass.Int32(2016):\n", "    if tau >= cutlass.Int32(1008):\n"),
            ("    elif tau >= cutlass.Int32(31):\n", "    elif tau >= cutlass.Int32(15):\n"),
            (
                "        maximum = cutlass.Uint32((tau << cutlass.Int32(5)) | cutlass.Int32(31))\n",
                "        maximum = cutlass.Uint32((tau << cutlass.Int32(6)) | cutlass.Int32(63))\n",
            ),
            (
                "            if cutlass.const_expr(self.r_const == 148):\n"
                "                if nv <= cutlass.Int32(131072):\n"
                "                    R = cutlass.Int32(128)\n",
                "",
            ),
            (
                "                    if cutlass.const_expr(self.r_const == 148):\n"
                "                        if R == cutlass.Int32(128):\n"
                "                            Q = (n4v + cutlass.Int32(127)) // cutlass.Int32(128)\n",
                "",
            ),
            (
                "            hbase = row * cutlass.Int32(2048)\n",
                "            hbase = row * cutlass.Int32(1024)\n",
            ),
            (
                "                h0 = histogram[hbase + tidx * cutlass.Int32(2)]\n"
                "                h1 = histogram[hbase + tidx * cutlass.Int32(2) + cutlass.Int32(1)]\n"
                "            hs = h0 + h1\n",
                "                h0 = histogram[hbase + tidx]\n"
                "            hs = h0\n",
            ),
            (
                "                    s_cbuf[NW + 0] = tidx * cutlass.Int32(2)\n",
                "                    s_cbuf[NW + 0] = tidx\n",
            ),
            (
                "                    if cert_bin >= 0 and cert_bin <= 2047 and cert_strict >= 0 and cert_strict <= k:\n",
                "                    if cert_bin >= 0 and cert_bin <= 1023 and cert_strict >= 0 and cert_strict <= k:\n",
            ),
            (
                "                            while hz < cutlass.Int32(2048):\n",
                "                            while hz < cutlass.Int32(1024):\n",
            ),
        )
        for old, new in replacements:
            text = _replace_once(text, old, new)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != SELECTOR_SHA256[bins]:
        raise RuntimeError(
            f"selector source drift for {bins} bins: {digest}"
        )
    return text


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
    flags.extend(
        (
            f"-DLITETOPK_COARSE_BINS={bins}",
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
    source = HERE / "csrc/producer_batch.cu"
    return [
        _build_producer(
            name="producer2048",
            source=source,
            bins=2048,
            namespace="litetopk_batched_producer_20260920",
        ),
        _build_producer(
            name="producer1024",
            source=source,
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


def _build_selector(batch: int, bins: int, unroll: int) -> dict:
    import torch
    from cutlass.cute import runtime

    module_name = "_gvr_hist_prologue_acqrel_hsplitq_original_20260918"
    original = _load(module_name, HERE / "vendor/gvr2_topk_decode.py")
    directory = BUILD / "selectors" / f"b{batch}"
    directory.mkdir(parents=True, exist_ok=True)
    source = _selector_source(bins)
    generated = directory / "generated.py"
    if generated.exists():
        raise FileExistsError(generated)
    generated.write_text(source)
    derivative = _load(
        f"_litetopk_selector_b{batch}",
        generated,
    )

    r_const = CTAS // batch
    template = (1024, unroll, 1, 256, 2, True, False)

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
        "histogram_bins": bins,
        "active_ctas_at_128k": 128 if batch == 1 else None,
        "unroll": unroll,
        "grid": [r_const, batch, 1],
        "workspace_bytes": WORKSPACE_BYTES,
        "source": str(generated),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
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
