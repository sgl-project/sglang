"""Build an isolated FlashInfer package from the ABI-repaired Step5 baseline."""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_sources(package, entries, stage):
    for relative, entry in entries.items():
        path = package / relative
        require(
            path.is_file() and sha256(path) == entry[stage],
            f"FlashInfer {stage} source mismatch: {path}",
        )


def copy_or_link(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def overlaps(first, second):
    return first == second or first in second.parents or second in first.parents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-package", type=Path, required=True)
    parser.add_argument("--parent-aot-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    patch_dir = Path(__file__).resolve().parent
    manifest = json.loads((patch_dir / "manifest.json").read_text())
    entries = manifest["files"]
    parent = args.parent_package.resolve()
    parent_aot = args.parent_aot_package.resolve()
    output = args.output_root.resolve()
    cache = args.cache_root.resolve()
    require(parent.is_dir() and parent_aot.is_dir(), "Missing parent packages")
    require(
        not output.exists() and not cache.exists(),
        "Use fresh output and cache directories",
    )
    for destination in (output, cache):
        for source in (parent, parent_aot, patch_dir):
            require(
                not overlaps(destination, source),
                f"Output/cache must be disjoint from source: {destination}, {source}",
            )
    require(not overlaps(output, cache), "Output and cache must be disjoint")
    verify_sources(parent, entries, "before")
    for entry in entries.values():
        require(
            sha256(patch_dir / entry["patch"]) == entry["patch_sha256"],
            f"Patch checksum mismatch: {entry['patch']}",
        )

    output.mkdir(parents=True)
    cache.mkdir(parents=True)
    private = output / "flashinfer"
    shutil.copytree(
        parent,
        private,
        copy_function=copy_or_link,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in entries:
        # Break hard links before patching, preserving the baseline package.
        temporary = (private / relative).with_suffix(".step5-new-inode")
        shutil.copyfile(parent / relative, temporary)
        temporary.replace(private / relative)
    for patch_name in dict.fromkeys(entry["patch"] for entry in entries.values()):
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "--fuzz=0",
                "-p1",
                "-i",
                str(patch_dir / patch_name),
            ],
            cwd=private,
            check=True,
        )
    verify_sources(parent, entries, "before")
    verify_sources(private, entries, "after")

    # An installed AOT module otherwise overrides the newly patched JIT source.
    modules = {"fused_moe_trtllm_sm100", "trtllm_mnnvl_comm"}
    aot = output / "flashinfer_jit_cache"
    aot.mkdir()
    for name in ("__init__.py", "_build_meta.py"):
        shutil.copy2(parent_aot / name, aot / name)
    (aot / "jit_cache").mkdir()
    for path in (parent_aot / "jit_cache").iterdir():
        if path.name not in modules:
            (aot / "jit_cache" / path.name).symlink_to(
                path.resolve(), target_is_directory=True
            )
    record = {
        "parent_package": str(parent),
        "package": str(private),
        "cache": str(cache),
        "sources": entries,
        "modules": [],
    }

    def save_record():
        (output / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    save_record()
    if args.prepare_only:
        print("Source hashes verified; CUDA build and runtime checks were not run.")
        return

    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(cache)
    os.environ["FLASHINFER_JIT_DEBUG"] = "0"
    sys.path.insert(0, str(output))
    importlib.invalidate_caches()
    flashinfer = importlib.import_module("flashinfer")
    require(
        Path(flashinfer.__file__).resolve().parent == private,
        "Imported the wrong FlashInfer package",
    )
    factories = [
        importlib.import_module(
            "flashinfer.jit.fused_moe"
        ).gen_trtllm_gen_fused_moe_sm100_module,
        importlib.import_module("flashinfer.jit.comm").gen_trtllm_mnnvl_comm_module,
    ]
    for factory in factories:
        spec = factory()
        require(
            not spec.is_aot and cache in spec.jit_library_path.parents,
            f"Expected a fresh JIT module: {spec.name}",
        )
        require(
            "-O3" in spec.extra_cuda_cflags
            and not {"-G", "-O0", "--device-debug"}.intersection(
                spec.extra_cuda_cflags
            ),
            "Expected release CUDA compilation",
        )
        module = spec.build_and_load()
        if spec.name == "fused_moe_trtllm_sm100":
            try:
                module.trtllm_fp8_block_scale_moe()
            except TypeError as error:
                require("Expected 46 but got 0 arguments" in str(error), str(error))
            else:
                raise RuntimeError("Expected the 46-argument MoE ABI")
        elif spec.name == "trtllm_mnnvl_comm":
            require(
                module.step5_tp4_rmsnorm_order_version() == 2,
                "Expected the Step5 TP4 RMSNorm multiplication-order fix",
            )
        library = spec.jit_library_path
        library_hash = sha256(library)
        (aot / "jit_cache" / spec.name).symlink_to(
            library.parent, target_is_directory=True
        )
        require(
            spec.is_aot and spec.aot_path.resolve() == library.resolve(),
            "The private AOT overlay does not point to the rebuilt module",
        )
        spec.build_and_load()
        require(sha256(library) == library_hash, "Library changed during reload")
        record["modules"].append(
            {
                "name": spec.name,
                "library": str(library),
                "sha256": library_hash,
                "cuda_flags": spec.extra_cuda_cflags,
            }
        )
        save_record()
    verify_sources(parent, entries, "before")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
