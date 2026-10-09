#!/usr/bin/env python3
"""Cache the image-edit input before NPU CI starts loading models."""

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

IMAGE_URL = "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg"
IMAGE_SHA256 = "0d56a63fca6ef8fe2a0ca27a0774c2bada9b2a198aeedbfdfb23bec09d0d3901"


def prefetch(cache_root: Path) -> Path:
    cache_dir = cache_root.expanduser().resolve() / IMAGE_SHA256
    cache_dir.mkdir(parents=True, exist_ok=True)
    image = cache_dir / "TI2I_Qwen_Image_Edit_Input.jpg"
    if (
        image.is_file()
        and hashlib.sha256(image.read_bytes()).hexdigest() == IMAGE_SHA256
    ):
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
                os.environ.get("GITHUB_PROXY_URL", "") + IMAGE_URL,
            ],
            check=True,
            timeout=400,
        )
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != IMAGE_SHA256:
            raise ValueError("Image-edit input SHA-256 mismatch")
        downloaded.replace(image)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("~/.cache/sglang/inputs")
    )
    args = parser.parse_args()
    image = prefetch(args.cache_root)
    line = f"SGLANG_TEST_TI2I_INPUT_IMAGE={image}"
    if github_env := os.environ.get("GITHUB_ENV"):
        with open(github_env, "a", encoding="utf-8") as output:
            output.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
