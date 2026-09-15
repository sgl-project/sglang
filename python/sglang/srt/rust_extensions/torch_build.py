"""Build settings for Rust extensions that link against the active PyTorch."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping

_MIN_SUPPORTED_TORCH = (2, 11)
_MAX_SUPPORTED_TORCH = (2, 13)


@dataclass(frozen=True)
class TorchBuildConfiguration:
    """Environment overrides plus stable inputs for the artifact fingerprint."""

    environment: dict[str, str]
    fingerprint: dict[str, object]


def torch_build_configuration(
    *,
    compat_header: Path,
    python_module: str,
    torch_module: ModuleType | None = None,
    base_environment: Mapping[str, str] | None = None,
    include_absolute_rpath: bool = True,
) -> TorchBuildConfiguration:
    """Describe a build against the torch package loaded by this interpreter."""
    if sys.platform != "linux":
        raise RuntimeError("the Rust TreeCore extension currently supports Linux only")

    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch must be installed before building the Rust TreeCore extension"
            ) from exc

    version = str(torch_module.__version__)
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"could not parse PyTorch version {version!r}")
    major_minor = (int(match.group(1)), int(match.group(2)))
    if not _MIN_SUPPORTED_TORCH <= major_minor <= _MAX_SUPPORTED_TORCH:
        minimum = ".".join(map(str, _MIN_SUPPORTED_TORCH))
        maximum = ".".join(map(str, _MAX_SUPPORTED_TORCH))
        raise RuntimeError(
            f"the Rust TreeCore supports PyTorch {minimum} through {maximum}; "
            f"found {version}"
        )

    torch_file = getattr(torch_module, "__file__", None)
    if torch_file is None:
        raise RuntimeError("the active PyTorch package has no filesystem location")
    torch_root = Path(torch_file).resolve().parent
    torch_lib = torch_root / "lib"
    if not torch_lib.is_dir():
        raise RuntimeError(
            f"the active PyTorch package has no library dir at {torch_lib}"
        )

    cxx11_abi_fn = getattr(torch_module, "compiled_with_cxx11_abi", None)
    if cxx11_abi_fn is not None:
        cxx11_abi = bool(cxx11_abi_fn())
    else:
        cxx11_abi = bool(torch_module._C._GLIBCXX_USE_CXX11_ABI)

    environment = dict(os.environ if base_environment is None else base_environment)
    environment.pop("LIBTORCH_USE_PYTORCH", None)
    environment["LIBTORCH"] = os.fspath(torch_root)
    environment["LIBTORCH_INCLUDE"] = os.fspath(torch_root)
    environment["LIBTORCH_LIB"] = os.fspath(torch_root)
    environment["LIBTORCH_CXX11_ABI"] = "1" if cxx11_abi else "0"
    # tch 0.24 targets Torch 2.11. The compatibility header below covers the
    # API removals in the supported 2.12/2.13 builds, after this explicit gate.
    environment["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
    environment["PYO3_PYTHON"] = sys.executable
    environment["PATH"] = os.pathsep.join(
        filter(None, (os.fspath(Path(sys.executable).parent), environment.get("PATH")))
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (os.fspath(torch_lib), environment.get("LD_LIBRARY_PATH")))
    )

    cxxflags = environment.get("CXXFLAGS", "")
    environment["CXXFLAGS"] = (
        f"{cxxflags} -include {shlex.quote(os.fspath(compat_header.resolve()))}"
    ).strip()

    package_depth = len(python_module.split(".")) - 1
    bundled_torch_lib = "$ORIGIN/" + "../" * package_depth + "torch/lib"
    rustflags = environment.get("RUSTFLAGS", "")
    rpath_flags = [f"-C link-arg=-Wl,-rpath,{bundled_torch_lib}"]
    if include_absolute_rpath:
        rpath_flags.append(f"-C link-arg=-Wl,-rpath,{torch_lib}")
    environment["RUSTFLAGS"] = " ".join(filter(None, (rustflags, *rpath_flags)))

    fingerprint = {
        "torch_version": version,
        "torch_root": os.fspath(torch_root),
        "torch_cxx11_abi": cxx11_abi,
        "torch_cuda": getattr(torch_module.version, "cuda", None),
        "torch_hip": getattr(torch_module.version, "hip", None),
        "include_absolute_rpath": include_absolute_rpath,
        "compat_header_sha256": (
            hashlib.sha256(compat_header.read_bytes()).hexdigest()
            if compat_header.is_file()
            else None
        ),
    }
    return TorchBuildConfiguration(environment=environment, fingerprint=fingerprint)
