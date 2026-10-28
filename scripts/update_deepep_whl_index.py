"""Update the PEP 503 indexes for sgl-deep-ep and sgl-deep-ep-v25 release wheels."""

import argparse
import hashlib
import pathlib
import re

SUPPORTED_CUDA_VERSIONS = ("130", "134")
SUPPORTED_PACKAGES = ("sgl-deep-ep", "sgl-deep-ep-v25")
ANCHOR_PATTERN = re.compile(r'^<a href="[^"]+">(?P<filename>[^<]+)</a><br>$')


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_pattern(package: str) -> re.Pattern:
    distribution = re.escape(package.replace("-", "_"))
    return re.compile(
        rf"^{distribution}-(?P<version>[0-9][^-]*)-[^-]+-[^-]+-[^-]+\.whl$"
    )


def _ensure_root_link(cuda_root: pathlib.Path, package: str) -> None:
    root_index = cuda_root / "index.html"
    lines = root_index.read_text().splitlines() if root_index.exists() else []
    doctype = [line for line in lines if line.startswith("<!DOCTYPE")]
    anchors = [line for line in lines if line.startswith("<a ")]
    link = f'<a href="{package}/">{package}</a>'
    if link not in anchors:
        anchors.append(link)
    content = [*(doctype or ["<!DOCTYPE html>"]), *sorted(set(anchors))]
    root_index.write_text("\n".join(content) + "\n")


def update_wheel_index(
    cuda_version: str,
    wheel_dir: pathlib.Path,
    repository: pathlib.Path,
    package: str = "sgl-deep-ep",
    release_tag_prefix: str = "v",
) -> None:
    if cuda_version not in SUPPORTED_CUDA_VERSIONS:
        raise ValueError(f"Unsupported CUDA version: {cuda_version}")
    if package not in SUPPORTED_PACKAGES:
        raise ValueError(f"Unsupported package: {package}")

    cuda_root = repository / f"cu{cuda_version}"
    index_dir = cuda_root / package
    index_dir.mkdir(exist_ok=True, parents=True)
    _ensure_root_link(cuda_root, package)
    wheel_pattern = _wheel_pattern(package)

    index_path = index_dir / "index.html"
    entries = {}
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            match = ANCHOR_PATTERN.match(line)
            if match:
                entries[match.group("filename")] = line

    suffix = f"+cu{cuda_version}"
    release_base = "https://github.com/sgl-project/whl/releases/download"
    for path in sorted(wheel_dir.glob("*.whl")):
        match = wheel_pattern.match(path.name)
        if not match or suffix not in match.group("version"):
            continue
        public_version = match.group("version").split("+", 1)[0]
        tag = f"{release_tag_prefix}{public_version}"
        url = f"{release_base}/{tag}/{path.name}#sha256={_sha256(path)}"
        entries[path.name] = f'<a href="{url}">{path.name}</a><br>'

    content = ["<!DOCTYPE html>", *(entries[name] for name in sorted(entries))]
    index_path.write_text("\n".join(content) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", required=True, choices=SUPPORTED_CUDA_VERSIONS)
    parser.add_argument("--package", default="sgl-deep-ep", choices=SUPPORTED_PACKAGES)
    parser.add_argument(
        "--release-tag-prefix",
        default="v",
        help="Prefix of the sgl-project/whl release tag that holds the wheels",
    )
    parser.add_argument("--wheel-dir", type=pathlib.Path, default=pathlib.Path("dist"))
    parser.add_argument(
        "--repository", type=pathlib.Path, default=pathlib.Path("sgl-whl")
    )
    args = parser.parse_args()
    update_wheel_index(
        args.cuda,
        args.wheel_dir,
        args.repository,
        args.package,
        args.release_tag_prefix,
    )


if __name__ == "__main__":
    main()
