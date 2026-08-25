#!/usr/bin/env python3
"""Compare the install-visible structure of two wheels."""

from __future__ import annotations

import argparse
import csv
import email.policy
import io
import json
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

_NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")
_EXTENSION_TAG_RE = re.compile(r"\.(?:cpython-[^.]+|abi3)\.(?:so|pyd)$")


def _resolve_wheel(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_file() and path.suffix == ".whl":
        return path
    if path.is_dir():
        wheels = sorted(path.glob("*.whl"))
        if len(wheels) == 1:
            return wheels[0]
        raise ValueError(f"{path} contains {len(wheels)} wheels; expected one")
    raise ValueError(f"{path} is neither a wheel nor a wheel directory")


def _installed_path(archive_path: str) -> str | None:
    parts = archive_path.split("/")
    if len(parts) >= 3 and parts[0].endswith(".data"):
        if parts[1] in {"purelib", "platlib"}:
            return "/".join(parts[2:])
        return None
    return archive_path


def _import_path(installed_path: str) -> str | None:
    if ".dist-info/" in installed_path or ".data/" in installed_path:
        return None
    if installed_path.endswith(".py"):
        module = installed_path[:-3]
        if module.endswith("/__init__"):
            module = module[: -len("/__init__")]
        return module.replace("/", ".")
    if installed_path.endswith(_NATIVE_SUFFIXES):
        parent, _, leaf = installed_path.rpartition("/")
        module_leaf = leaf.split(".", 1)[0]
        return ".".join(filter(None, (parent.replace("/", "."), module_leaf)))
    return None


def _wheel_info(wheel_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(wheel_path) as archive:
        manifest = sorted(
            name for name in archive.namelist() if name and not name.endswith("/")
        )
        metadata_paths = [
            name for name in manifest if name.endswith(".dist-info/METADATA")
        ]
        wheel_paths = [name for name in manifest if name.endswith(".dist-info/WHEEL")]
        record_paths = [name for name in manifest if name.endswith(".dist-info/RECORD")]
        if not (len(metadata_paths) == len(wheel_paths) == len(record_paths) == 1):
            raise ValueError(
                f"{wheel_path} must contain exactly one METADATA, WHEEL, and RECORD"
            )

        metadata = BytesParser(policy=email.policy.default).parsebytes(
            archive.read(metadata_paths[0])
        )
        wheel_metadata = BytesParser(policy=email.policy.default).parsebytes(
            archive.read(wheel_paths[0])
        )
        record_entries = {
            row[0]
            for row in csv.reader(
                io.StringIO(archive.read(record_paths[0]).decode("utf-8"))
            )
            if row
        }

    installed_paths = sorted(
        installed
        for archive_path in manifest
        if (installed := _installed_path(archive_path)) is not None
    )
    import_paths = sorted(
        import_path
        for installed_path in installed_paths
        if (import_path := _import_path(installed_path)) is not None
    )
    native_files = sorted(
        installed
        for installed in installed_paths
        if installed.endswith(_NATIVE_SUFFIXES)
    )
    native_modules = sorted(
        import_path
        for import_path in import_paths
        if any(
            installed_path.startswith(import_path.replace(".", "/") + ".")
            and _EXTENSION_TAG_RE.search(installed_path)
            for installed_path in native_files
        )
    )

    return {
        "path": str(wheel_path),
        "distribution": metadata["Name"],
        "version": metadata["Version"],
        "wheel_tags": sorted(wheel_metadata.get_all("Tag", [])),
        "manifest": manifest,
        "record_matches_manifest": record_entries == set(manifest),
        "installed_paths": installed_paths,
        "import_paths": import_paths,
        "native_files": native_files,
        "native_modules": native_modules,
    }


def _difference(left: list[str], right: list[str]) -> dict[str, list[str]]:
    left_set = set(left)
    right_set = set(right)
    return {
        "only_authoritative": sorted(left_set - right_set),
        "only_candidate": sorted(right_set - left_set),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("authoritative")
    parser.add_argument("candidate")
    parser.add_argument("--expect-version")
    parser.add_argument("--expect-native", action="append", default=[])
    args = parser.parse_args()

    try:
        authoritative = _wheel_info(_resolve_wheel(args.authoritative))
        candidate = _wheel_info(_resolve_wheel(args.candidate))
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))

    scalar_fields = ("distribution", "version", "wheel_tags")
    list_fields = (
        "manifest",
        "installed_paths",
        "import_paths",
        "native_files",
        "native_modules",
    )
    differences = {
        field: (
            _difference(authoritative[field], candidate[field])
            if field in list_fields
            else {
                "authoritative": authoritative[field],
                "candidate": candidate[field],
            }
        )
        for field in (*scalar_fields, *list_fields)
        if authoritative[field] != candidate[field]
    }

    expectation_failures: list[str] = []
    if args.expect_version:
        for side, info in (
            ("authoritative", authoritative),
            ("candidate", candidate),
        ):
            if info["version"] != args.expect_version:
                expectation_failures.append(
                    f"{side} version {info['version']!r} != {args.expect_version!r}"
                )
    if args.expect_native:
        expected_native = sorted(args.expect_native)
        for side, info in (
            ("authoritative", authoritative),
            ("candidate", candidate),
        ):
            if info["native_modules"] != expected_native:
                expectation_failures.append(
                    f"{side} native modules {info['native_modules']!r} "
                    f"!= {expected_native!r}"
                )
    for side, info in (
        ("authoritative", authoritative),
        ("candidate", candidate),
    ):
        if not info["record_matches_manifest"]:
            expectation_failures.append(f"{side} RECORD does not match its manifest")

    report = {
        "authoritative": {
            key: value for key, value in authoritative.items() if key not in list_fields
        },
        "candidate": {
            key: value for key, value in candidate.items() if key not in list_fields
        },
        "counts": {
            field: {
                "authoritative": len(authoritative[field]),
                "candidate": len(candidate[field]),
            }
            for field in list_fields
        },
        "differences": differences,
        "expectation_failures": expectation_failures,
        "parity": not differences and not expectation_failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["parity"] else 1


if __name__ == "__main__":
    sys.exit(main())
