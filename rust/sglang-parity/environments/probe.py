#!/usr/bin/env python3
"""Verify a parity Python environment without installing packages or loading models."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import sys
import sysconfig
from pathlib import Path

RUST_MODULE = "sglang.srt.rust_extensions._server"


def library_paths() -> list[str]:
    paths = set()
    for root in {sysconfig.get_path("platlib"), sysconfig.get_path("purelib")}:
        if root:
            packages = Path(root)
            candidates = [packages / "torch/lib", *(packages / "nvidia").rglob("lib")]
            paths.update(str(path.resolve()) for path in candidates if path.is_dir())
    return sorted(paths)


def verify_python(expected: Path, version: str) -> None:
    if tuple(sys.version_info[:3]) != tuple(map(int, version.split("."))):
        raise RuntimeError(
            f"Python {version} is required; received {sys.version.split()[0]}"
        )
    # Resolving a virtualenv's python symlink would erase its environment identity.
    if os.path.abspath(sys.executable) != os.path.abspath(expected):
        raise RuntimeError(
            f"Python executable is {sys.executable}; expected {expected}"
        )


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_packages(lock: Path) -> dict[str, str]:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    environment = default_environment()
    environment["extra"] = ""
    pinned = {}
    # uv's requirements export uses backslash continuations for wheel hashes.
    content = lock.read_text().replace("\\\n", " ")
    for line in content.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        requirement = Requirement(re.split(r"\s+--hash(?:=|\s+)", line, maxsplit=1)[0])
        name = normalized_name(requirement.name)
        if name == "sglang":
            continue  # The local source installation is checked by import origin.
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        versions = list(requirement.specifier)
        if (
            requirement.url
            or len(versions) != 1
            or versions[0].operator != "=="
            or "*" in versions[0].version
        ):
            raise RuntimeError(f"Lock requirement must pin one exact version: {line}")
        version = versions[0].version
        if name in pinned and pinned[name] != version:
            raise RuntimeError(f"Conflicting lock versions for {name}")
        pinned[name] = version
    if not pinned:
        raise RuntimeError("Lock contains no applicable third-party package versions")
    return pinned


def verify_packages(lock: Path) -> dict[str, str]:
    expected = locked_packages(lock)
    installed = {}
    for distribution in importlib.metadata.distributions():
        name = normalized_name(distribution.metadata["Name"])
        version = distribution.version
        if name in installed and installed[name] != version:
            raise RuntimeError(f"Multiple installed versions of {name}")
        installed[name] = version
    problems = []
    for name, version in expected.items():
        actual = installed.get(name)
        if actual is None:
            problems.append(f"{name}=={version} is missing")
        elif actual != version:
            problems.append(f"{name}=={version} required; installed {actual}")
    if problems:
        raise RuntimeError("Lock verification failed: " + "; ".join(problems))
    # Extra packages are permitted, but remain visible in the full inventory.
    return dict(sorted(installed.items()))


def source_origin(module: object, source: Path) -> str:
    filename = getattr(module, "__file__", None)
    if not filename:
        raise RuntimeError(f"Module {module!r} has no source origin")
    origin = Path(filename).resolve()
    if not origin.is_file() or not origin.is_relative_to((source / "python").resolve()):
        raise RuntimeError(f"Import origin {origin} is outside {source / 'python'}")
    return str(origin)


def verify_backend(backend: str) -> dict:
    torch = importlib.import_module("torch")
    # Import the real companion extensions before sglang can install platform stubs.
    importlib.import_module("torchvision")
    importlib.import_module("torchaudio")
    if backend == "mlx":
        mx = importlib.import_module("mlx.core")
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal device is unavailable")
        result = mx.add(mx.array([1.0]), 2.0, stream=mx.gpu)
        mx.eval(result)
        if result.item() != 3.0:
            raise RuntimeError("MLX Metal arithmetic check failed")
        return {"name": backend, "device": "metal", "info": mx.metal.device_info()}
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA device is unavailable")
    result = torch.tensor([1.0, 2.0], device="cuda").sum().item()
    if result != 3.0:
        raise RuntimeError("CUDA arithmetic check failed")
    importlib.import_module("sgl_kernel")
    return {
        "name": backend,
        "device": "cuda",
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cuda_version": torch.version.cuda,
    }


def verify_rust(source: Path, loader: object) -> tuple[dict, dict]:
    workspace = source / "rust"
    crate = loader._discover_crate(workspace, RUST_MODULE)
    context = loader._build_context(crate)
    expected = loader._cached_extension_path(
        loader._cache_root(None), crate, context.fingerprint
    ).resolve()
    module = loader.load_rust_extension(RUST_MODULE, mode="auto", workspace=workspace)
    actual = Path(module.__file__).resolve()
    if actual != expected:
        raise RuntimeError(
            f"Rust extension loaded from {actual}; expected current source artifact {expected}"
        )
    with actual.open("rb") as artifact:
        digest = hashlib.file_digest(artifact, "sha256").hexdigest()
    extension = {
        "path": str(actual),
        "fingerprint": context.fingerprint,
        "source_digest": context.source_digest,
        "sha256": digest,
    }
    toolchain = {
        "rustc": loader._command_version("rustc", "-vV", cwd=crate.workspace),
        "cargo": loader._command_version("cargo", "--version", cwd=crate.workspace),
    }
    return extension, toolchain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-paths", type=Path, metavar="OUTPUT")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--backend", choices=("mlx", "cuda"))
    parser.add_argument("--python", type=Path)
    parser.add_argument("--python-version")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dependencies-only", action="store_true")
    args = parser.parse_args(argv)
    required = ("source", "lock", "backend", "python", "python_version", "output")
    if args.library_paths is not None:
        if args.dependencies_only or any(
            getattr(args, name) is not None for name in required
        ):
            parser.error("--library-paths must be used alone")
        args.library_paths.parent.mkdir(parents=True, exist_ok=True)
        args.library_paths.write_text(json.dumps(library_paths(), indent=2) + "\n")
        return 0
    missing = [
        "--" + name.replace("_", "-")
        for name in required
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    source = args.source.expanduser().resolve()
    lock = args.lock.expanduser().resolve()
    result = {
        "status": "failed",
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_executable": sys.executable,
        "source": str(source),
        "lock": str(lock),
        "dependencies_only": args.dependencies_only,
        "backend": {"name": args.backend},
    }
    try:
        verify_python(args.python.expanduser(), args.python_version)
        result["packages"] = verify_packages(lock)
        if not args.dependencies_only:
            result["backend"] = verify_backend(args.backend)
            sglang = importlib.import_module("sglang")
            result["source_origins"] = {"sglang": source_origin(sglang, source)}
            loader = importlib.import_module("sglang.srt.rust_extensions.loader")
            result["source_origins"]["rust_loader"] = source_origin(loader, source)
            result["rust_extension"], result["toolchain"] = verify_rust(source, loader)
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
