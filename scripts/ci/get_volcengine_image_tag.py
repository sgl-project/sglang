#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TAG_SUFFIX_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z_.-]*")


def validate_suffix(flag: str, value: str) -> None:
    if value and not TAG_SUFFIX_RE.fullmatch(value):
        raise SystemExit(f"--{flag} must be a Docker tag-safe suffix")


def build_tag(
    mode: str,
    version: str,
    timestamp: str,
    tag_value: str = "",
    variant_suffix: str = "",
    cuda_suffix: str = "",
    format_suffix: str = "",
) -> str:
    """Compose the final image tag.

    Suffix order is fixed as ``variant`` -> ``cuda`` -> ``format`` so that the
    image format marker (e.g. ``zstd`` / ``nydus``) always trails the CUDA
    marker: ``v<ver>.byted.<val>.<ts>[-<variant>][-cu130][-zstd]``.
    """
    if mode == "manual":
        tag = f"v{version}.iaas.dev.{timestamp}"
    elif mode == "nightly":
        tag = f"v{version}.iaas.nightly.{timestamp}"
    else:
        if not tag_value:
            raise SystemExit("--tag-value is required when --mode=version")
        tag = f"v{version}.byted.{tag_value}.{timestamp}"

    if variant_suffix:
        tag = f"{tag}-{variant_suffix}"

    if cuda_suffix:
        tag = f"{tag}-{cuda_suffix}"

    if format_suffix:
        tag = f"{tag}-{format_suffix}"

    return tag


def get_sglang_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    version_file = repo_root / "python/sglang/_version.py"
    if version_file.exists():
        content = version_file.read_text()
        match = re.search(r"__version__\s*=\s*version\s*=\s*'([^']+)'", content)
        if match:
            return match.group(1)

    result = subprocess.run(
        ["python3", "python/tools/get_version_tag.py", "--tag-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().lstrip("v")

    raise SystemExit(
        "failed to extract sglang version from python/sglang/_version.py or git tags"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Volcengine CR image tags for fork workflows."
    )
    parser.add_argument(
        "--mode", choices=["manual", "nightly", "version"], required=True
    )
    parser.add_argument(
        "--tag-value",
        default="",
        help="Required for version mode; inserted after .byted.",
    )
    parser.add_argument("--cuda-suffix", choices=["", "cu129", "cu130"], default="")
    parser.add_argument(
        "--variant-suffix",
        default="",
        help="Optional build variant suffix appended before the CUDA suffix.",
    )
    parser.add_argument(
        "--format-suffix",
        default="",
        help="Optional image format suffix (e.g. zstd, nydus) appended after "
        "the CUDA suffix.",
    )
    args = parser.parse_args()

    validate_suffix("variant-suffix", args.variant_suffix)
    validate_suffix("format-suffix", args.format_suffix)

    version = get_sglang_version()
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M")

    tag = build_tag(
        mode=args.mode,
        version=version,
        timestamp=timestamp,
        tag_value=args.tag_value,
        variant_suffix=args.variant_suffix,
        cuda_suffix=args.cuda_suffix,
        format_suffix=args.format_suffix,
    )

    print(tag)


if __name__ == "__main__":
    main()
