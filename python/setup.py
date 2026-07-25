"""sglang build hooks.

Rust extensions are auto-discovered from the cargo workspace in ../rust: every
crate whose Cargo.toml declares

    [package.metadata.sglang]
    python-module = "sglang.srt.<pkg>._core"   # import path inside the wheel
    debug = false                              # optional RustExtension knob

is built as a PyO3 extension module at that import path. Adding a new extension
crate therefore needs no pyproject changes — declare the metadata in the crate.

Two filters can narrow the discovered set:

- [tool.sglang] rust-extensions in the active pyproject.toml: a list of
  case-insensitive substrings of the target module. Platform pyprojects use
  this to build a subset (e.g. pyproject_other.toml builds only "multimodal";
  grpc needs proto/tonic and is intentionally CUDA-only).
- SGLANG_BUILD_RUST_EXTS env var, applied at build time on top of the above:
  unset or "all" builds everything, "none" builds nothing, and a
  comma-separated list matches substrings, e.g. "grpc" matches
  "sglang.srt.grpc._core". It is read directly from os.environ instead of
  sglang.srt.environ, which is not importable until the package is built.
"""

import json
import os
import re
import subprocess
from pathlib import Path

from setuptools import setup

try:
    from setuptools_rust import Binding, RustExtension, build_rust
except ModuleNotFoundError as exc:
    if exc.name != "setuptools_rust":
        raise
    # Alternate platform pyprojects that build no Rust extensions do not
    # install setuptools-rust.
    build_rust = None

_BUILD_RUST_EXTS_ENV = "SGLANG_BUILD_RUST_EXTS"
_PYTHON_DIR = Path(__file__).resolve().parent
_RUST_WORKSPACE_DIR = _PYTHON_DIR.parent / "rust"


def _cargo_workspace_metadata():
    """The rust/ cargo workspace as JSON, straight from cargo's own parser."""
    manifest_path = _RUST_WORKSPACE_DIR / "Cargo.toml"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"no cargo workspace at {manifest_path} (building outside a repo "
            f"checkout?); set {_BUILD_RUST_EXTS_ENV}=none to build without "
            "Rust extensions"
        )
    try:
        out = subprocess.run(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--no-deps",
                "--manifest-path",
                str(manifest_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "cargo is required to discover the Rust extension modules in "
            f"{_RUST_WORKSPACE_DIR} (and to build them); install a Rust "
            f"toolchain, or set {_BUILD_RUST_EXTS_ENV}=none to build without "
            "Rust extensions"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"cargo metadata failed:\n{exc.stderr}") from exc
    return json.loads(out.stdout)


def _match_by_substring(declared, tokens, source):
    """Match tokens as case-insensitive substrings of extension names."""
    matched = set()
    unmatched = []
    for token in tokens:
        hits = {ext.name for ext in declared if token in ext.name.lower()}
        if hits:
            matched |= hits
        else:
            unmatched.append(token)
    if unmatched:
        declared_names = sorted(ext.name for ext in declared)
        raise ValueError(
            f"{source} matched no discovered Rust extension for: {unmatched}; "
            f"discovered extensions are {declared_names}"
        )
    return [ext for ext in declared if ext.name in matched]


def _discovered_rust_extensions():
    """One RustExtension per workspace crate declaring a python-module."""
    extensions = []
    for package in sorted(
        _cargo_workspace_metadata()["packages"], key=lambda p: p["name"]
    ):
        sglang_meta = (package["metadata"] or {}).get("sglang", {})
        if "python-module" not in sglang_meta:
            continue
        extensions.append(
            RustExtension(
                target=sglang_meta["python-module"],
                path=package["manifest_path"],
                binding=Binding.PyO3,
                debug=sglang_meta.get("debug"),
            )
        )
    if not extensions:
        raise RuntimeError(
            f"no crate under {_RUST_WORKSPACE_DIR} declares "
            "[package.metadata.sglang] python-module; set "
            f"{_BUILD_RUST_EXTS_ENV}=none to build without Rust extensions"
        )
    return extensions


# Deliberately not a TOML parser (keeps setup.py stdlib-only): the allowlist
# must be written as a single line, e.g. rust-extensions = ["multimodal"].
_ALLOWLIST_RE = re.compile(r"^rust-extensions\s*=\s*\[([^\]]*)\]", re.MULTILINE)


def _pyproject_rust_extensions(declared):
    """Apply the active pyproject's [tool.sglang] rust-extensions allowlist."""
    pyproject_text = (_PYTHON_DIR / "pyproject.toml").read_text(encoding="utf-8")
    match = _ALLOWLIST_RE.search(pyproject_text)
    if match is None:
        return declared
    tokens = re.findall(r'"([^"]*)"', match.group(1))
    return _match_by_substring(
        declared=declared,
        tokens=[token.lower() for token in tokens],
        source="[tool.sglang] rust-extensions",
    )


def _selected_rust_extensions(declared):
    """Apply the SGLANG_BUILD_RUST_EXTS build-time filter."""
    declared = list(declared)
    raw = os.environ.get(_BUILD_RUST_EXTS_ENV)
    if raw is None:
        return declared

    spec = raw.strip().lower()
    # An empty or whitespace-only value is treated as unset (build everything).
    if not spec or spec == "all":
        return declared
    if spec == "none":
        return []

    tokens = [token.strip() for token in spec.split(",")]
    if not all(tokens):
        raise ValueError(
            f"{_BUILD_RUST_EXTS_ENV}={raw!r} has an empty item; unset it or use "
            "'all', 'none', or a comma-separated list of extension names"
        )
    return _match_by_substring(
        declared=declared, tokens=tokens, source=_BUILD_RUST_EXTS_ENV
    )


def _declared_rust_extensions():
    # "none" short-circuits discovery so builds without a ../rust checkout
    # (e.g. from an sdist) still work.
    if (os.environ.get(_BUILD_RUST_EXTS_ENV) or "").strip().lower() == "none":
        return []
    return _pyproject_rust_extensions(_discovered_rust_extensions())


if build_rust is not None:

    class BuildRust(build_rust):
        """Build only the Rust extensions selected by SGLANG_BUILD_RUST_EXTS."""

        def run(self) -> None:
            rust_extensions = _selected_rust_extensions(self.extensions or [])
            self.extensions = rust_extensions
            self.distribution.rust_extensions = rust_extensions
            if not rust_extensions:
                return
            super().run()

    setup(
        cmdclass={"build_rust": BuildRust},
        rust_extensions=_declared_rust_extensions(),
    )
else:
    setup()
