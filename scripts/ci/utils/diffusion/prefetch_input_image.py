#!/usr/bin/env python3
"""Cache the image-edit inputs before NPU CI starts loading models."""

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

# (env var exported to the test step, URL, SHA-256)
INPUTS = (
    (
        "SGLANG_TEST_TI2I_INPUT_IMAGE",
        "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        "0d56a63fca6ef8fe2a0ca27a0774c2bada9b2a198aeedbfdfb23bec09d0d3901",
    ),
    (
        "SGLANG_TEST_MULTI_IMAGE_INPUT_IMAGE",
        "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/rabbit.jpg",
        "7e138910e522855f4dfd49d3c80351e1d69e0956fd041cba56fab329300c991f",
    ),
)


def prefetch(cache_root: Path, url: str, sha256: str) -> Path:
    cache_dir = cache_root.expanduser().resolve() / sha256
    cache_dir.mkdir(parents=True, exist_ok=True)
    image = cache_dir / url.rsplit("/", 1)[-1]
    if image.is_file() and hashlib.sha256(image.read_bytes()).hexdigest() == sha256:
        return image

    # Unique staging files and atomic replacement allow concurrent partitions.
    with tempfile.TemporaryDirectory(prefix="download-", dir=cache_dir) as staging:
        downloaded = Path(staging) / image.name
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                "120",
                "--output",
                str(downloaded),
                os.environ.get("GITHUB_PROXY_URL", "") + url,
            ],
            check=True,
            timeout=400,
        )
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"SHA-256 mismatch for {url}")
        downloaded.replace(image)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("~/.cache/sglang/inputs")
    )
    args = parser.parse_args()

    lines = []
    for env_name, url, sha256 in INPUTS:
        image = prefetch(args.cache_root, url, sha256)
        lines.append(f"{env_name}={image}")

    if github_env := os.environ.get("GITHUB_ENV"):
        with open(github_env, "a", encoding="utf-8") as output:
            output.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
