#!/usr/bin/env python3

import argparse
import html
import pathlib
import re

ROOT_LINK = '<a href="others/">others</a><br>'
OTHERS_HEADER = "<!DOCTYPE html>\n<h1>SGLang Other Files</h1>\n"
ASSET_URL_PREFIX = "https://github.com/sgl-project/whl/releases/download/"
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
TAG_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the sgl-project/whl indexes for a special file."
    )
    parser.add_argument("--repo-dir", type=pathlib.Path, required=True)
    parser.add_argument("--asset-url", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()

    if not args.repo_dir.is_dir():
        parser.error(f"repository directory does not exist: {args.repo_dir}")
    if not (args.repo_dir / "index.html").is_file():
        parser.error(f"root index does not exist: {args.repo_dir / 'index.html'}")
    if SHA256_PATTERN.fullmatch(args.sha256) is None:
        parser.error("--sha256 must contain exactly 64 hexadecimal characters")
    if TAG_PATTERN.fullmatch(args.tag) is None:
        parser.error("--tag may contain only ASCII letters, digits, '.', '_', and '-'")
    if not args.asset_url.startswith(ASSET_URL_PREFIX):
        parser.error(f"--asset-url must start with {ASSET_URL_PREFIX}")

    return args


def update_root_index(repo_dir: pathlib.Path) -> bool:
    index_path = repo_dir / "index.html"
    content = index_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    root_link_count = sum(line.rstrip("\r\n") == ROOT_LINK for line in lines)
    if root_link_count == 1:
        return False

    if root_link_count > 1:
        content = "".join(line for line in lines if line.rstrip("\r\n") != ROOT_LINK)
    if content and not content.endswith("\n"):
        content += "\n"
    index_path.write_text(f"{content}{ROOT_LINK}\n", encoding="utf-8")
    return True


def update_others_index(
    repo_dir: pathlib.Path,
    asset_url: str,
    filename: str,
    tag: str,
    sha256: str,
) -> bool:
    index_dir = repo_dir / "others"
    index_path = index_dir / "index.html"
    escaped_url = html.escape(asset_url, quote=True)
    escaped_filename = html.escape(filename, quote=True)
    escaped_tag = html.escape(tag, quote=True)
    identity = f'href="{escaped_url}#sha256='
    entry = (
        f'<a href="{escaped_url}#sha256={sha256.lower()}">'
        f"{escaped_filename}</a> ({escaped_tag})<br>\n"
    )

    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        if not content.startswith(OTHERS_HEADER):
            raise ValueError(
                f"{index_path} does not start with the expected SGLang header"
            )
    else:
        content = OTHERS_HEADER

    if identity in content:
        return False

    index_dir.mkdir(parents=True, exist_ok=True)
    updated = f"{OTHERS_HEADER}{entry}{content[len(OTHERS_HEADER) :]}"
    index_path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    root_changed = update_root_index(args.repo_dir)
    others_changed = update_others_index(
        repo_dir=args.repo_dir,
        asset_url=args.asset_url,
        filename=args.filename,
        tag=args.tag,
        sha256=args.sha256,
    )
    print("updated" if root_changed or others_changed else "unchanged")


if __name__ == "__main__":
    main()
