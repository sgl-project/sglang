#!/usr/bin/env python3
"""Create and verify metadata for CI-built SGLang wheel artifacts."""

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import types
import zipfile
from pathlib import Path

EXPECTED_EXTENSIONS = ("server", "grpc", "multimodal")
MANIFEST_NAME = "sglang-wheel-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _rust_channel(repo_root: Path) -> str:
    toolchain = (repo_root / "rust/rust-toolchain.toml").read_text()
    matches = re.findall(r'^channel\s*=\s*"([^"]+)"\s*$', toolchain, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one Rust channel in rust/rust-toolchain.toml"
        )
    return matches[0]


def _extension_members(wheel: Path) -> dict[str, str]:
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(extension_suffix, str) or not extension_suffix.endswith(".so"):
        raise RuntimeError(f"unsupported Python extension suffix: {extension_suffix!r}")
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
    selected = {}
    for package in EXPECTED_EXTENSIONS:
        package_dir = f"sglang/srt/{package}/"
        expected = f"{package_dir}_core{extension_suffix}"
        matches = [
            name
            for name in members
            if name.startswith(f"{package_dir}_core") and name.endswith(".so")
        ]
        if matches != [expected]:
            raise RuntimeError(
                f"expected only {expected} in {wheel.name}, found {matches}"
            )
        selected[package] = expected
    return selected


def _validate_wheel_identity(wheel: Path, members: dict[str, str]) -> None:
    if sys.implementation.name != "cpython":
        raise RuntimeError("wheel fan-out currently requires CPython")

    stem_parts = wheel.name.removesuffix(".whl").rsplit("-", 3)
    if len(stem_parts) != 4:
        raise RuntimeError(f"wheel has an invalid filename: {wheel.name}")
    python_tag, abi_tag, platform_tag = stem_parts[1:]
    interpreter_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine().lower().replace("-", "_").replace(".", "_")
    expected_tags = (interpreter_tag, interpreter_tag, f"linux_{machine}")
    actual_tags = (python_tag, abi_tag, platform_tag)
    if actual_tags != expected_tags:
        raise RuntimeError(
            f"wheel tags do not match this runner: artifact={actual_tags}, "
            f"runner={expected_tags}"
        )


def _compiler_identity(repo_root: Path) -> dict[str, str]:
    channel = _rust_channel(repo_root)
    if not shutil.which("rustup"):
        raise RuntimeError("rustup is required to verify the wheel compiler")

    selected_rustc = subprocess.check_output(
        ["rustup", "which", "rustc"], text=True
    ).strip()
    selected_cargo = subprocess.check_output(
        ["rustup", "which", "cargo"], text=True
    ).strip()
    expected_rustc = subprocess.check_output(
        ["rustup", "which", "--toolchain", channel, "rustc"], text=True
    ).strip()
    expected_cargo = subprocess.check_output(
        ["rustup", "which", "--toolchain", channel, "cargo"], text=True
    ).strip()
    if selected_rustc != expected_rustc or selected_cargo != expected_cargo:
        raise RuntimeError(f"active Rust tools do not resolve to toolchain {channel}")

    rustc_verbose = subprocess.check_output(
        ["rustc", "--version", "--verbose"], text=True
    ).strip()
    cargo_verbose = subprocess.check_output(
        ["cargo", "--version", "--verbose"], text=True
    ).strip()
    expected_rustc_verbose = subprocess.check_output(
        [expected_rustc, "--version", "--verbose"], text=True
    ).strip()
    expected_cargo_verbose = subprocess.check_output(
        [expected_cargo, "--version", "--verbose"], text=True
    ).strip()
    if (
        rustc_verbose != expected_rustc_verbose
        or cargo_verbose != expected_cargo_verbose
    ):
        raise RuntimeError(f"Rust commands on PATH do not match toolchain {channel}")
    return {
        "toolchain": channel,
        "rustc": rustc_verbose,
        "cargo": cargo_verbose,
    }


def _glibc_version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)+", value):
        raise ValueError(f"invalid glibc version: {value!r}")
    return tuple(int(part) for part in value.split("."))


def _host_glibc() -> str:
    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc" or not libc_version:
        raise RuntimeError(
            f"wheel fan-out requires glibc, found {libc_name or 'unknown'} "
            f"{libc_version or 'unknown'}"
        )
    _glibc_version(libc_version)
    return libc_version


def _verify_native_members(
    wheel: Path, members: dict[str, str], max_glibc: str
) -> str | None:
    allowed = _glibc_version(max_glibc)
    wheel_requirements: set[tuple[int, ...]] = set()
    with tempfile.TemporaryDirectory(prefix="sglang-wheel-check-") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(wheel) as archive:
            for member in members.values():
                archive.extract(member, root)

        for package, member in members.items():
            shared_object = root / member
            ldd = subprocess.run(
                ["ldd", str(shared_object)], capture_output=True, text=True, check=True
            )
            if "not found" in ldd.stdout or "not found" in ldd.stderr:
                raise RuntimeError(
                    f"unresolved library for {package}:\n{ldd.stdout}{ldd.stderr}"
                )

            symbols = subprocess.check_output(
                ["objdump", "-T", str(shared_object)], text=True
            )
            required = {
                _glibc_version(match)
                for match in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", symbols)
            }
            wheel_requirements.update(required)
            if required and max(required) > allowed:
                found = ".".join(str(part) for part in max(required))
                raise RuntimeError(
                    f"{package} extension requires GLIBC_{found}, above {max_glibc}"
                )

            module_name = f"sglang.srt.{package}._core"
            for parent_name in ("sglang", "sglang.srt", f"sglang.srt.{package}"):
                if parent_name not in sys.modules:
                    parent = types.ModuleType(parent_name)
                    parent.__path__ = []  # type: ignore[attr-defined]
                    sys.modules[parent_name] = parent
            spec = importlib.util.spec_from_file_location(module_name, shared_object)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not create import spec for {shared_object}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    if not wheel_requirements:
        return None
    return ".".join(str(part) for part in max(wheel_requirements))


def _runtime_tuple(repo_root: Path) -> dict[str, str]:
    return {
        "architecture": platform.machine(),
        "python_cache_tag": sys.implementation.cache_tag or "unknown",
        "python_soabi": sysconfig.get_config_var("SOABI") or "unknown",
        "rust_channel": _rust_channel(repo_root),
    }


def create_manifest(
    wheel: Path, repo_root: Path, output: Path, max_glibc: str
) -> dict[str, object]:
    members = _extension_members(wheel)
    _validate_wheel_identity(wheel, members)
    required_glibc = _verify_native_members(wheel, members, max_glibc)
    manifest: dict[str, object] = {
        "schema": 1,
        "commit": _git_revision(repo_root),
        **_runtime_tuple(repo_root),
        "compiler": _compiler_identity(repo_root),
        "max_glibc": max_glibc,
        "required_glibc": required_glibc,
        "wheel": wheel.name,
        "wheel_sha256": _sha256(wheel),
        "extensions": members,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_artifact(artifact_dir: Path, repo_root: Path) -> Path:
    manifest_path = artifact_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise RuntimeError(
            f"unsupported wheel manifest schema: {manifest.get('schema')!r}"
        )

    expected = {"commit": _git_revision(repo_root), **_runtime_tuple(repo_root)}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"wheel compatibility mismatch for {field}: "
                f"artifact={manifest.get(field)!r}, runner={value!r}"
            )

    compiler = manifest.get("compiler")
    if not isinstance(compiler, dict) or compiler.get("toolchain") != _rust_channel(
        repo_root
    ):
        raise RuntimeError(
            "wheel manifest does not contain the verified compiler identity"
        )
    for command in ("rustc", "cargo"):
        if not isinstance(compiler.get(command), str) or not compiler[command]:
            raise RuntimeError(
                f"wheel manifest does not contain the verified {command} identity"
            )

    required_glibc = manifest.get("required_glibc")
    if not isinstance(required_glibc, str):
        raise RuntimeError("wheel manifest does not declare required_glibc")
    host_glibc = _host_glibc()
    if _glibc_version(host_glibc) < _glibc_version(required_glibc):
        raise RuntimeError(
            f"wheel requires glibc {required_glibc}, runner provides {host_glibc}"
        )

    wheel_name = str(manifest["wheel"])
    if Path(wheel_name).name != wheel_name:
        raise RuntimeError(
            f"wheel manifest contains an unsafe filename: {wheel_name!r}"
        )
    wheel = artifact_dir / wheel_name
    if not wheel.is_file() or wheel.suffix != ".whl" or ".editable" in wheel.name:
        raise RuntimeError(f"artifact does not contain a regular wheel: {wheel}")
    if _sha256(wheel) != manifest.get("wheel_sha256"):
        raise RuntimeError("wheel checksum does not match its manifest")
    members = _extension_members(wheel)
    _validate_wheel_identity(wheel, members)
    if members != manifest.get("extensions"):
        raise RuntimeError("wheel extension members do not match its manifest")
    return wheel.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--wheel", type=Path, required=True)
    create.add_argument("--repo-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--max-glibc", default="2.35")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact-dir", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "create":
            manifest = create_manifest(
                args.wheel.resolve(),
                args.repo_root.resolve(),
                args.output.resolve(),
                args.max_glibc,
            )
            print(json.dumps(manifest, sort_keys=True))
        else:
            print(
                verify_artifact(args.artifact_dir.resolve(), args.repo_root.resolve())
            )
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
