"""Fail when sgl-router's cache key misses a crate outside sgl-router/ it builds.

sgl-router is its own cargo workspace, but path dependencies pull some rust/
crates into its build, and the key lists those rather than all of rust/. Any
change that adds such a dependency edits a manifest the key already hashes, so
the key misses and the compile job, which runs this, sees the stale list.
"""

import json
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
# The two places that compute the key, which must hash the same files.
KEY_FILES = [
    ".github/workflows/_pr-test-sgl-router-build.yml",
    ".github/actions/download-sgl-router/action.yml",
]
# Seeds the key from main, so it has to trigger on everything the key hashes.
SEED_FILE = ".github/workflows/seed-sgl-router-cache.yml"


def required_patterns():
    """`<dir>/**` per path dependency outside sgl-router/, plus the manifest of the
    workspace each inherits `workspace = true` fields from."""
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked"]
            + ["--manifest-path", str(REPO_ROOT / "sgl-router/Cargo.toml")],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    router = REPO_ROOT / "sgl-router"
    patterns = set()
    for package in metadata["packages"]:
        if package["source"] is not None:
            continue
        crate_dir = pathlib.Path(package["manifest_path"]).parent
        if crate_dir.is_relative_to(router):
            continue
        patterns.add(f"{crate_dir.relative_to(REPO_ROOT)}/**")
        for parent in crate_dir.parents:
            if not parent.is_relative_to(REPO_ROOT):
                break
            manifest = parent / "Cargo.toml"
            if manifest.is_file() and "[workspace]" in manifest.read_text():
                patterns.add(str(manifest.relative_to(REPO_ROOT)))
                break
    return patterns


def key_patterns(path):
    text = (REPO_ROOT / path).read_text()
    match = re.search(r"key: sgl-router-.*hashFiles\(([^)]*)\)", text)
    assert match, f"no sgl-router key in {path}"
    return match.group(1)


def main():
    errors = []
    keys = {path: key_patterns(path) for path in KEY_FILES}
    if len(set(keys.values())) != 1:
        errors.append(f"the key hashes different files in {', '.join(KEY_FILES)}")
    seed = (REPO_ROOT / SEED_FILE).read_text()
    for pattern in sorted(required_patterns()):
        quoted = f"'{pattern}'"
        for path, key in keys.items():
            if quoted not in key:
                errors.append(f"{path}: the key does not hash {quoted}")
        if quoted not in seed:
            errors.append(f"{SEED_FILE}: paths does not list {quoted}")
    for error in errors:
        print(f"::error::{error}")
    if errors:
        sys.exit(1)
    print("sgl-router cache key covers every crate the router builds")


if __name__ == "__main__":
    main()
