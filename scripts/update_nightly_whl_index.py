#!/usr/bin/env python3
"""
Update the wheel index for nightly SGLang releases.

This script generates a single PyPI-compatible index.html file at nightly/index.html
containing all historical nightly builds, ordered by commit count (newest first).

Reference: https://github.com/flashinfer-ai/flashinfer/blob/v0.2.0/scripts/update_whl_index.py
"""

import argparse
import hashlib
import pathlib
import re


def compute_sha256(file_path: pathlib.Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def update_wheel_index(commit_hash: str, nightly_version: str, build_date: str = None):
    """Update the wheel index for nightly releases.

    Creates a single index at nightly/index.html containing all historical nightlies.

    Args:
        commit_hash: Short git commit hash (e.g., 'c5f1e86')
        nightly_version: Full nightly version string (e.g., '0.5.6.post1.dev7716+gc5f1e86')
        build_date: Build date in YYYY-MM-DD format (e.g., '2025-12-13')
    """
    dist_dir = pathlib.Path("dist")
    whl_repo_dir = pathlib.Path("sgl-whl")

    if not dist_dir.exists():
        print(f"Warning: {dist_dir} does not exist, skipping index update")
        return

    # Base URL for wheels stored in GitHub Releases
    base_url = "https://github.com/sgl-project/whl/releases/download"
    # Use date-based tag if build_date is provided, otherwise fall back to commit-only
    if build_date:
        release_tag = f"nightly-{build_date}-{commit_hash}"
    else:
        release_tag = f"nightly-{commit_hash}"

    # Create nightly directory structure following PEP 503
    # /nightly/index.html -> links to sglang/
    # /nightly/sglang/index.html -> contains wheel links
    nightly_dir = whl_repo_dir / "nightly"
    nightly_dir.mkdir(parents=True, exist_ok=True)

    sglang_dir = nightly_dir / "sglang"
    sglang_dir.mkdir(parents=True, exist_ok=True)

    root_index = nightly_dir / "index.html"
    package_index = sglang_dir / "index.html"

    print(f"\nUpdating nightly wheel index")
    print(f"  Root index: {root_index}")
    print(f"  Package index: {package_index}")

    # Read existing package index if it exists
    existing_links = []
    if package_index.exists():
        with open(package_index, "r") as f:
            content = f.read()
            # Extract existing links (skip header and HTML boilerplate)
            existing_links = [
                line for line in content.split("\n") if line.startswith("<a href=")
            ]

    # Generate new links for current wheels
    new_links = []
    for wheel_path in sorted(dist_dir.glob("*.whl")):
        try:
            filename = wheel_path.name
            sha256 = compute_sha256(wheel_path)

            # URL format: {base_url}/{release_tag}/{filename}#sha256={hash}
            wheel_url = f"{base_url}/{release_tag}/{filename}#sha256={sha256}"
            link = f'<a href="{wheel_url}">{filename}</a><br>'

            new_links.append(link)
            print(f"  Added: {filename}")
        except Exception as e:
            print(f"  Error processing {wheel_path.name}: {e}")
            continue

    if not new_links:
        print("  No new wheels to add")
        return

    # Combine existing and new links (new links first for latest)
    all_links = new_links + existing_links

    # Remove duplicates while preserving order (newer first)
    seen = set()
    unique_links = []
    for link in all_links:
        # Extract filename from link to check for duplicates
        filename_match = re.search(r">([^<]+\.whl)</a>", link)
        if filename_match:
            filename = filename_match.group(1)
            if filename not in seen:
                seen.add(filename)
                unique_links.append(link)

    # Write root index (links to sglang package directory)
    with open(root_index, "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write('<a href="sglang/">sglang</a>\n')

    print(f"  Written root index: {root_index}")

    # Write package index in minimal format (matching production sgl-kernel index)
    with open(package_index, "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<h1>SGLang Nightly Wheels</h1>\n")
        # Write links only
        f.write("\n".join(unique_links))
        f.write("\n")

    print(f"  Written {len(unique_links)} total wheels to {package_index}")
    print(f"\nDone! Users can install with:")
    print(
        f"  pip install sglang --pre --extra-index-url https://sgl-project.github.io/whl/nightly/"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Update wheel index for nightly SGLang releases"
    )
    parser.add_argument(
        "--commit-hash",
        type=str,
        required=True,
        help="Short git commit hash (e.g., 'c5f1e86')",
    )
    parser.add_argument(
        "--nightly-version",
        type=str,
        required=True,
        help="Full nightly version string (e.g., '0.5.6.post1.dev7716+gc5f1e86')",
    )
    parser.add_argument(
        "--build-date",
        type=str,
        required=False,
        help="Build date in YYYY-MM-DD format (e.g., '2025-12-13')",
    )

    args = parser.parse_args()

    print(f"Updating nightly wheel index")
    print(f"  Commit: {args.commit_hash}")
    print(f"  Version: {args.nightly_version}")
    if args.build_date:
        print(f"  Build date: {args.build_date}")

    update_wheel_index(args.commit_hash, args.nightly_version, args.build_date)


if __name__ == "__main__":
    main()
