#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TAG_SUFFIX_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z_.-]*")
VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9A-Za-z]+)+(?:[._+-][0-9A-Za-z]+)*")
DOCKER_TAG_RE = re.compile(r"[0-9A-Za-z_][0-9A-Za-z_.-]{0,127}")


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

    Manual tags use ``tag_value`` verbatim when it is provided, otherwise they
    fall back to the generated ``v<ver>.iaas.dev.<ts>`` name.
    Suffix order is fixed as ``variant`` -> ``cuda`` -> ``format`` so that the
    image format marker (e.g. ``zstd`` / ``nydus``) always trails the CUDA
    marker: ``<manual-tag>[-<variant>][-cu130][-zstd]``.
    """
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"invalid SGLang version: {version}")
    for name, value in (
        ("tag-value", tag_value),
        ("variant-suffix", variant_suffix),
        ("cuda-suffix", cuda_suffix),
        ("format-suffix", format_suffix),
    ):
        validate_suffix(name, value)

    if mode == "manual":
        tag = tag_value or f"v{version}.iaas.dev.{timestamp}"
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

    if not DOCKER_TAG_RE.fullmatch(tag):
        raise SystemExit("generated Docker tag is invalid or exceeds 128 characters")
    return tag


def get_sglang_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["python3", "scripts/release/get_version_tag.py", "--tag-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().lstrip("v")

    raise SystemExit(
        "failed to extract sglang version from scripts/release/get_version_tag.py"
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
        help="Used verbatim as the base tag in manual mode; required for "
        "version mode and inserted after .byted.",
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

    validate_suffix("tag-value", args.tag_value)
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
