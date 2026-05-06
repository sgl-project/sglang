#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


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
    parser.add_argument("--cuda-suffix", choices=["", "cu130"], default="")
    args = parser.parse_args()

    version = get_sglang_version()
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M")

    if args.mode == "manual":
        tag = f"v{version}.iaas.dev.{timestamp}"
    elif args.mode == "nightly":
        tag = f"v{version}.iaas.nightly.{timestamp}"
    else:
        if not args.tag_value:
            raise SystemExit("--tag-value is required when --mode=version")
        tag = f"v{version}.byted.{args.tag_value}.{timestamp}"

    if args.cuda_suffix:
        tag = f"{tag}-{args.cuda_suffix}"

    print(tag)


if __name__ == "__main__":
    main()
