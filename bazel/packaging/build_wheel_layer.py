#!/usr/bin/env python3
"""Install a wheel into a deterministic OCI filesystem layer."""

from __future__ import annotations

import argparse
import configparser
import io
import re
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

_ENTRY_POINT = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):"
    r"(?P<function>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$"
)
_PYTHON = "/usr/local/bin/python3"
_SCRIPT_DIR = PurePosixPath("usr/local/bin")


def _safe_relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ValueError(f"unsafe wheel path: {raw!r}")
    return path


def _wheel_destination(
    path: PurePosixPath, site_packages: PurePosixPath
) -> PurePosixPath:
    if path.parts[0].endswith(".data"):
        if len(path.parts) < 3:
            raise ValueError(f"malformed wheel data path: {path}")
        scheme = path.parts[1]
        relative = PurePosixPath(*path.parts[2:])
        roots = {
            "data": PurePosixPath("usr/local"),
            "headers": PurePosixPath("usr/local/include"),
            "platlib": site_packages,
            "purelib": site_packages,
            "scripts": _SCRIPT_DIR,
        }
        if scheme not in roots:
            raise ValueError(f"unsupported wheel install scheme {scheme!r}")
        return roots[scheme] / relative
    return site_packages / path


def _tar_info(name: PurePosixPath, mode: int, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(str(name))
    info.mode = mode
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def _add_file(
    archive: tarfile.TarFile,
    destinations: set[PurePosixPath],
    destination: PurePosixPath,
    contents: bytes,
    mode: int,
) -> None:
    if destination in destinations:
        raise ValueError(f"duplicate wheel layer destination: {destination}")
    destinations.add(destination)
    archive.addfile(_tar_info(destination, mode, len(contents)), io.BytesIO(contents))


def _console_script(target: str) -> bytes:
    target = target.partition("[")[0].strip()
    match = _ENTRY_POINT.fullmatch(target)
    if not match:
        raise ValueError(f"unsupported console entry point: {target!r}")
    module = match.group("module")
    function = match.group("function")
    return f"""#!{_PYTHON}
import sys
from {module} import {function}

if __name__ == "__main__":
    sys.exit({function}())
""".encode()


def _version_smoke(version: str) -> bytes:
    return f"""#!{_PYTHON}
from importlib.metadata import version

expected = {version!r}
actual = version("sglang")
if actual != expected:
    raise SystemExit(f"expected sglang {{expected}}, got {{actual}}")
print(f"sglang {{actual}}")
""".encode()


def _metadata(wheel: zipfile.ZipFile) -> tuple[str, str, str]:
    names = wheel.namelist()
    metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
    entry_point_paths = [
        name for name in names if name.endswith(".dist-info/entry_points.txt")
    ]
    if len(metadata_paths) != 1 or len(entry_point_paths) > 1:
        raise ValueError(
            "wheel must contain one METADATA and at most one entry_points.txt"
        )
    metadata = BytesParser(policy=default).parsebytes(wheel.read(metadata_paths[0]))
    distribution = metadata["Name"]
    version = metadata["Version"]
    if not distribution or not version:
        raise ValueError("wheel METADATA must contain Name and Version")
    entry_points = (
        wheel.read(entry_point_paths[0]).decode() if entry_point_paths else ""
    )
    return distribution, version, entry_points


def build_layer(wheel_path: Path, output: Path, site_packages_raw: str) -> None:
    site_packages = _safe_relative_path(site_packages_raw.strip("/"))
    if not site_packages.parts:
        raise ValueError("site-packages must not be empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel_path) as wheel:
        distribution, version, entry_points_raw = _metadata(wheel)
        if distribution.lower().replace("_", "-") != "sglang":
            raise ValueError(f"expected the sglang wheel, got {distribution!r}")

        destinations: set[PurePosixPath] = set()
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            for member in sorted(wheel.infolist(), key=lambda item: item.filename):
                if member.is_dir():
                    continue
                source = _safe_relative_path(member.filename)
                destination = _wheel_destination(source, site_packages)
                raw_mode = member.external_attr >> 16
                mode = stat.S_IMODE(raw_mode) if raw_mode else 0o644
                _add_file(
                    archive,
                    destinations,
                    destination,
                    wheel.read(member),
                    mode,
                )

            if entry_points_raw:
                entry_points = configparser.ConfigParser(interpolation=None)
                entry_points.read_string(entry_points_raw)
                for name, target in sorted(entry_points["console_scripts"].items()):
                    script_name = _safe_relative_path(name)
                    if len(script_name.parts) != 1:
                        raise ValueError(f"unsafe console script name: {name!r}")
                    _add_file(
                        archive,
                        destinations,
                        _SCRIPT_DIR / script_name,
                        _console_script(target),
                        0o755,
                    )

            _add_file(
                archive,
                destinations,
                _SCRIPT_DIR / "sglang-wheel-version-smoke",
                _version_smoke(version),
                0o755,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-packages", required=True)
    args = parser.parse_args()
    build_layer(args.wheel, args.output, args.site_packages)


if __name__ == "__main__":
    main()
