"""Discover the Rust extension modules declared under rust/ and filter them.

setup.py and the CI build in scripts/ci/utils/build_rust_ext_modules.py load
this file by path, since the sglang package is not importable at build time, so
it depends on the standard library only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

BUILD_RUST_EXTS_ENV = "SGLANG_BUILD_RUST_EXTS"


@dataclass(frozen=True)
class DeclaredRustExtension:
    """A crate whose [package.metadata.sglang] declares a python-module."""

    # The python-module import path.
    name: str
    package_name: str
    manifest_path: str
    workspace_root: str
    # The crate's [package.metadata.sglang] table.
    metadata: dict


def _cargo_metadata(manifest_path, rust_dir):
    """One Cargo workspace/package manifest as Cargo's own JSON metadata."""
    if not manifest_path.is_file():
        raise RuntimeError(
            f"no Cargo manifest at {manifest_path} (building outside a repo "
            f"checkout?); set {BUILD_RUST_EXTS_ENV}=none to build without "
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
            f"{rust_dir} (and to build them); install a Rust "
            f"toolchain, or set {BUILD_RUST_EXTS_ENV}=none to build without "
            "Rust extensions"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"cargo metadata failed:\n{exc.stderr}") from exc
    return json.loads(out.stdout)


def _cargo_workspace_documents(rust_dir):
    """Root workspace metadata plus explicitly declared extension workspaces."""
    document = _cargo_metadata(rust_dir / "Cargo.toml", rust_dir)
    external_manifests = (
        (document.get("metadata") or {})
        .get("sglang", {})
        .get("extension-manifests", [])
    )
    documents = [document]
    for relative_manifest in external_manifests:
        external = (rust_dir / relative_manifest).resolve()
        if rust_dir not in external.parents:
            raise RuntimeError(
                f"external Rust extension manifest escapes rust/: {relative_manifest}"
            )
        documents.append(_cargo_metadata(external, rust_dir))
    return documents


def _discovered_rust_extensions(rust_dir):
    """One DeclaredRustExtension per workspace crate declaring a python-module."""
    packages = [
        (package, document["workspace_root"])
        for document in _cargo_workspace_documents(rust_dir)
        for package in document["packages"]
    ]
    extensions = []
    for package, workspace_root in sorted(packages, key=lambda p: p[0]["name"]):
        sglang_meta = (package["metadata"] or {}).get("sglang", {})
        if "python-module" not in sglang_meta:
            continue
        extensions.append(
            DeclaredRustExtension(
                name=sglang_meta["python-module"],
                package_name=package["name"],
                manifest_path=package["manifest_path"],
                workspace_root=workspace_root,
                metadata=sglang_meta,
            )
        )
    if not extensions:
        raise RuntimeError(
            f"no crate under {rust_dir} declares "
            "[package.metadata.sglang] python-module; set "
            f"{BUILD_RUST_EXTS_ENV}=none to build without Rust extensions"
        )
    return extensions


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


# Deliberately not a TOML parser (keeps setup.py stdlib-only): the allowlist
# must be written as a single line, e.g. rust-extensions = ["multimodal"].
_ALLOWLIST_RE = re.compile(r"^rust-extensions\s*=\s*\[([^\]]*)\]", re.MULTILINE)


def _pyproject_rust_extensions(declared, pyproject):
    """Apply the active pyproject's [tool.sglang] rust-extensions allowlist."""
    pyproject_text = pyproject.read_text(encoding="utf-8")
    match = _ALLOWLIST_RE.search(pyproject_text)
    if match is None:
        return declared
    tokens = re.findall(r'"([^"]*)"', match.group(1))
    return _match_by_substring(
        declared=declared,
        tokens=[token.lower() for token in tokens],
        source="[tool.sglang] rust-extensions",
    )


def selected_rust_extensions(declared):
    """Apply the SGLANG_BUILD_RUST_EXTS build-time filter.

    Matches on .name, so it takes setuptools-rust's RustExtension objects as
    well as DeclaredRustExtension.
    """
    declared = list(declared)
    raw = os.environ.get(BUILD_RUST_EXTS_ENV)
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
            f"{BUILD_RUST_EXTS_ENV}={raw!r} has an empty item; unset it or use "
            "'all', 'none', or a comma-separated list of extension names"
        )
    return _match_by_substring(
        declared=declared, tokens=tokens, source=BUILD_RUST_EXTS_ENV
    )


def declared_rust_extensions(*, rust_dir: Path, pyproject: Path):
    """The extensions under rust_dir that pass the pyproject's allowlist."""
    # "none" short-circuits discovery so builds without a ../rust checkout
    # (e.g. from an sdist) still work.
    if (os.environ.get(BUILD_RUST_EXTS_ENV) or "").strip().lower() == "none":
        return []
    return _pyproject_rust_extensions(_discovered_rust_extensions(rust_dir), pyproject)
