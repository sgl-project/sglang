#!/usr/bin/env python3
"""Serially compile the fmha_sm100 variants used by the TP4 MSA A/B gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_variants(dtype_code: int) -> list[tuple]:
    """Return the BF16 sparse-paged variants reachable by this TP4 gate."""

    variants = []
    for single_wg in (True, False):
        for split_kv in (False, True):
            for pack_factor in (1, 16):
                variants.append(
                    (dtype_code, 128, single_wg, 0, 128, split_kv, pack_factor)
                )
    variants.append((dtype_code, 256, False, 0, 128, False, 1))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite precompile receipt: {output}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MINFER_FMHA_CACHE_DIR"] = str(cache_dir)

    import fmha_sm100
    import fmha_sm100.jit as jit
    import tvm_ffi

    if Path(jit.CACHE_BASE).resolve() != cache_dir:
        raise RuntimeError(
            f"fmha_sm100 ignored MINFER_FMHA_CACHE_DIR: {jit.CACHE_BASE}"
        )

    artifacts = []
    for runtime_args in runtime_variants(jit._BFLOAT16_CODE):
        variant_name, params = jit._variant_key_from_runtime(*runtime_args)
        jit._variant_manager._compile_only(variant_name, params)
        so_path = cache_dir / variant_name / f"{variant_name}.so"
        module = tvm_ffi.load_module(str(so_path))
        function = f"run_{variant_name}"
        getattr(module, function)
        artifacts.append(
            {
                "kind": "variant",
                "name": variant_name,
                "function": function,
                "runtime_args": list(runtime_args),
                "path": str(so_path),
                "sha256": sha256(so_path),
            }
        )

    auxiliary = (
        ("plan", jit._compile_plan_only, "plan/fmha_sm100_plan.so", "plan"),
        (
            "reduction",
            jit._do_compile_reduction,
            "reduction/fmha_sm100_reduction.so",
            "reduction",
        ),
        (
            "sparse_topk",
            jit._do_compile_sparse_topk,
            "sparse_topk/sparse_topk_select.so",
            "sparse_topk_select",
        ),
    )
    for name, compile_fn, relative_path, function in auxiliary:
        compile_fn()
        so_path = cache_dir / relative_path
        module = tvm_ffi.load_module(str(so_path))
        getattr(module, function)
        artifacts.append(
            {
                "kind": "auxiliary",
                "name": name,
                "function": function,
                "path": str(so_path),
                "sha256": sha256(so_path),
            }
        )

    payload = {
        "schema_version": 1,
        "status": "passed",
        "cache_dir": str(cache_dir),
        "fmha_sm100_path": str(Path(fmha_sm100.__file__).resolve()),
        "tvm_ffi_version": getattr(tvm_ffi, "__version__", "unknown"),
        "artifacts": artifacts,
    }
    with output.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
