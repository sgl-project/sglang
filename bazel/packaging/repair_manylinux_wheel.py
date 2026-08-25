#!/usr/bin/env python3
"""Repair and audit a direct Bazel wheel without rebuilding its native modules."""

from __future__ import annotations

import argparse
import base64
import csv
import email.policy
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from auditwheel.main import main as auditwheel_main
from elftools.elf.elffile import ELFFile
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

_DETERMINISTIC_ZIP_EPOCH = "315532800"  # 1980-01-01T00:00:00Z
_GLIBC_RE = re.compile(r"^GLIBC_(\d+(?:\.\d+)*)$")


def _sha256(contents: bytes) -> str:
    digest = hashlib.sha256(contents).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_path(paths: list[str], description: str) -> str:
    if len(paths) != 1:
        raise ValueError(f"expected one {description}, found {paths}")
    return paths[0]


def _safe_archive_path(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe wheel archive path: {name!r}")


def _read_wheel(path: Path, expected_tag: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"{path.name} contains duplicate archive paths")
        for name in names:
            _safe_archive_path(name)

        manifest = sorted(info.filename for info in infos if not info.is_dir())
        payload = {name: archive.read(name) for name in manifest}
        metadata_path = _single_path(
            [name for name in manifest if name.endswith(".dist-info/METADATA")],
            "METADATA",
        )
        wheel_path = _single_path(
            [name for name in manifest if name.endswith(".dist-info/WHEEL")],
            "WHEEL",
        )
        record_path = _single_path(
            [name for name in manifest if name.endswith(".dist-info/RECORD")],
            "RECORD",
        )

    package_metadata = BytesParser(policy=email.policy.default).parsebytes(
        payload[metadata_path]
    )
    wheel_metadata = BytesParser(policy=email.policy.default).parsebytes(
        payload[wheel_path]
    )
    wheel_tags = sorted(wheel_metadata.get_all("Tag", []))
    if wheel_tags != [expected_tag]:
        raise ValueError(f"{path.name} WHEEL tags {wheel_tags!r} != {[expected_tag]!r}")
    if wheel_metadata["Root-Is-Purelib"].lower() != "false":
        raise ValueError(f"{path.name} must set Root-Is-Purelib: false")

    filename_name, filename_version, _, filename_tags = parse_wheel_filename(path.name)
    parsed_filename_tags = sorted(str(tag) for tag in filename_tags)
    if parsed_filename_tags != [expected_tag]:
        raise ValueError(
            f"{path.name} filename tags {parsed_filename_tags!r} "
            f"!= {[expected_tag]!r}"
        )
    if canonicalize_name(package_metadata["Name"]) != filename_name:
        raise ValueError(f"{path.name} filename and METADATA names differ")
    if Version(package_metadata["Version"]) != filename_version:
        raise ValueError(f"{path.name} filename and METADATA versions differ")

    rows = list(
        csv.reader(io.StringIO(payload[record_path].decode("utf-8"), newline=""))
    )
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{path.name} RECORD contains a malformed row")
    record_names = [row[0] for row in rows]
    if len(record_names) != len(set(record_names)):
        raise ValueError(f"{path.name} RECORD contains duplicate paths")
    if set(record_names) != set(manifest):
        missing = sorted(set(manifest) - set(record_names))
        extra = sorted(set(record_names) - set(manifest))
        raise ValueError(
            f"{path.name} RECORD/manifest mismatch: missing={missing}, extra={extra}"
        )
    for record_name, digest, size in rows:
        if record_name == record_path:
            if digest or size:
                raise ValueError(f"{path.name} RECORD must not hash itself")
            continue
        contents = payload[record_name]
        expected_digest = f"sha256={_sha256(contents)}"
        if digest != expected_digest or size != str(len(contents)):
            raise ValueError(f"{path.name} has invalid RECORD entry for {record_name}")

    return {
        "manifest": manifest,
        "metadata_path": metadata_path,
        "native_paths": sorted(name for name in manifest if name.endswith(".so")),
        "payload": payload,
        "record_entries": len(rows),
        "record_path": record_path,
        "wheel_path": wheel_path,
    }


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split("."))


def _elf_audit(path: Path, max_glibc: tuple[int, ...]) -> dict[str, Any]:
    with path.open("rb") as native_file:
        elf = ELFFile(native_file)
        if (
            elf.elfclass != 64
            or not elf.little_endian
            or elf.header["e_machine"] != "EM_X86_64"
            or elf.header["e_type"] != "ET_DYN"
        ):
            raise ValueError(f"{path.name} is not an ELF64 x86_64 shared object")

        glibc_versions: set[tuple[int, ...]] = set()
        version_section = elf.get_section_by_name(".gnu.version_r")
        if version_section is not None:
            for _, auxiliaries in version_section.iter_versions():
                for auxiliary in auxiliaries:
                    match = _GLIBC_RE.fullmatch(auxiliary.name)
                    if match:
                        glibc_versions.add(_version_tuple(match.group(1)))
                    elif auxiliary.name.startswith("GLIBC_"):
                        raise ValueError(
                            f"{path.name} uses unsupported symbol version "
                            f"{auxiliary.name}"
                        )
        if not glibc_versions:
            raise ValueError(f"{path.name} has no GLIBC symbol requirements")
        observed_glibc = max(glibc_versions)
        if observed_glibc > max_glibc:
            raise ValueError(
                f"{path.name} requires GLIBC {'.'.join(map(str, observed_glibc))}, "
                f"above {'.'.join(map(str, max_glibc))}"
            )

        dynamic = elf.get_section_by_name(".dynamic")
        if dynamic is None:
            raise ValueError(f"{path.name} has no dynamic section")
        needed = sorted(tag.needed for tag in dynamic.iter_tags("DT_NEEDED"))
        search_paths = [
            getattr(tag, "rpath", getattr(tag, "runpath", ""))
            for tag in dynamic.iter_tags()
            if tag.entry.d_tag in {"DT_RPATH", "DT_RUNPATH"}
        ]
        if search_paths:
            raise ValueError(
                f"{path.name} has forbidden RPATH/RUNPATH entries: {search_paths}"
            )

    return {
        "max_glibc": ".".join(map(str, observed_glibc)),
        "needed": needed,
        "sha256": _file_sha256(path),
    }


def _import_native_modules(
    root: Path, modules: list[str], expected_paths: dict[str, str]
) -> list[str]:
    package_names = sorted(
        {
            ".".join(module.split(".")[:index])
            for module in modules
            for index in range(1, len(module.split(".")))
        },
        key=lambda name: (name.count("."), name),
    )
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(root / package_name.replace(".", "/"))]
        sys.modules[package_name] = package

    imported = []
    for module_name in sorted(modules):
        native_path = root / expected_paths[module_name]
        spec = importlib.util.spec_from_file_location(module_name, native_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        imported.append(module_name)
    return imported


def _tool_path(distribution_name: str, relative_path: str) -> Path:
    distribution = metadata.distribution(distribution_name)
    package_root = Path(distribution.locate_file("."))
    candidates = [
        package_root / relative_path,
        package_root.parent / relative_path,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{distribution_name} does not provide expected file {relative_path}; "
        f"checked {candidates}"
    )


def _run_auditwheel(input_wheel: Path, output: Path, policy: str) -> None:
    patchelf = _tool_path("patchelf", "bin/patchelf")
    version_output = subprocess.check_output(
        [patchelf, "--version"],
        text=True,
    ).strip()
    if not version_output.startswith("patchelf "):
        raise RuntimeError(f"unexpected patchelf output: {version_output!r}")

    old_argv = sys.argv
    old_path = os.environ.get("PATH")
    old_source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    try:
        os.environ["PATH"] = str(patchelf.parent)
        os.environ["SOURCE_DATE_EPOCH"] = _DETERMINISTIC_ZIP_EPOCH
        with tempfile.TemporaryDirectory(prefix="sglang-auditwheel-") as temporary:
            wheel_dir = Path(temporary)
            sys.argv = [
                "auditwheel",
                "repair",
                "--plat",
                policy,
                "--only-plat",
                "--patcher",
                "patchelf",
                "--zip-compression-level",
                "9",
                "--wheel-dir",
                str(wheel_dir),
                str(input_wheel),
            ]
            status = auditwheel_main()
            if status not in (None, 0):
                raise RuntimeError(f"auditwheel failed with status {status}")
            repaired = sorted(wheel_dir.glob("*.whl"))
            if len(repaired) != 1 or repaired[0].name != output.name:
                raise RuntimeError(
                    f"auditwheel produced {[path.name for path in repaired]!r}, "
                    f"expected {output.name!r}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(repaired[0], output)
    finally:
        sys.argv = old_argv
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path
        if old_source_date_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = old_source_date_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--input-tag", required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--extension-suffix", required=True)
    parser.add_argument("--max-glibc", required=True)
    parser.add_argument("--expect-auditwheel", required=True)
    parser.add_argument("--expect-patchelf", required=True)
    parser.add_argument("--native-module", action="append", default=[])
    args = parser.parse_args()

    actual_versions = {
        "auditwheel": metadata.version("auditwheel"),
        "patchelf": metadata.version("patchelf"),
    }
    expected_versions = {
        "auditwheel": args.expect_auditwheel,
        "patchelf": args.expect_patchelf,
    }
    if actual_versions != expected_versions:
        raise ValueError(
            f"repair tool versions {actual_versions!r} != {expected_versions!r}"
        )

    _run_auditwheel(args.input, args.output, args.policy)
    input_info = _read_wheel(args.input, args.input_tag)
    output_info = _read_wheel(args.output, args.output_tag)
    if input_info["manifest"] != output_info["manifest"]:
        raise ValueError("auditwheel unexpectedly changed the wheel manifest")

    mutable_paths = {input_info["record_path"], input_info["wheel_path"]}
    changed_payloads = [
        name
        for name in input_info["manifest"]
        if name not in mutable_paths
        and input_info["payload"][name] != output_info["payload"][name]
    ]
    if changed_payloads:
        raise ValueError(
            f"auditwheel unexpectedly changed wheel payloads: {changed_payloads}"
        )

    expected_native_paths = {
        module: module.replace(".", "/") + args.extension_suffix
        for module in args.native_module
    }
    expected_native = sorted(expected_native_paths.values())
    if input_info["native_paths"] != expected_native:
        raise ValueError(
            f"input native payloads {input_info['native_paths']!r} "
            f"!= {expected_native!r}"
        )
    if output_info["native_paths"] != expected_native:
        raise ValueError(
            f"output native payloads {output_info['native_paths']!r} "
            f"!= {expected_native!r}"
        )

    max_glibc = _version_tuple(args.max_glibc)
    with tempfile.TemporaryDirectory(prefix="sglang-wheel-import-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(args.output) as archive:
            archive.extractall(root)
        native_audit = {
            module: _elf_audit(root / expected_native_paths[module], max_glibc)
            for module in sorted(args.native_module)
        }
        imported = _import_native_modules(
            root,
            args.native_module,
            expected_native_paths,
        )

    report = {
        "input": {
            "filename": args.input.name,
            "record_entries": input_info["record_entries"],
            "sha256": _file_sha256(args.input),
            "tag": args.input_tag,
        },
        "native_modules": native_audit,
        "native_payloads_unchanged": True,
        "output": {
            "filename": args.output.name,
            "record_entries": output_info["record_entries"],
            "sha256": _file_sha256(args.output),
            "tag": args.output_tag,
        },
        "policy": args.policy,
        "smoke_imports": imported,
        "tool_versions": actual_versions,
    }
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    args.audit_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
