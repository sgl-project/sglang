"""Shrink the Rust Ext build's cargo target dir before it is cached.

Keeps the third-party crates' deps/, build/ and .fingerprint/ entries, which are
what the cache saves compiling. Drops what every build compiles again anyway:
the sglang crates (a fresh checkout gives their sources new mtimes), the final
cdylibs and their dep-info, and torch-sys with tch on top of it (the build step
cleans torch-sys, since it links against that run's venv).

Usage: prune_rust_ext_target.py <target_dir>
"""

import json
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REBUILT_EVERY_RUN = {"torch-sys", "tch"}
KEPT_PROFILE_ENTRIES = {"deps", "build", ".fingerprint"}


def cargo_metadata(manifest):
    output = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"]
        + ["--manifest-path", str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def rebuilt_unit_names():
    """Package and target names of the rebuilt crates, with `-` as `_`."""
    root = REPO_ROOT / "rust/Cargo.toml"
    workspace = cargo_metadata(root)
    # The extension crates rust/Cargo.toml excludes, as setup.py finds them.
    extensions = workspace["metadata"]["sglang"]["extension-manifests"]
    packages = list(workspace["packages"])
    for manifest in extensions:
        packages.extend(cargo_metadata(root.parent / manifest)["packages"])
    names = set(REBUILT_EVERY_RUN)
    for package in packages:
        names.add(package["name"])
        names.update(target["name"] for target in package["targets"])
    return {name.replace("-", "_") for name in names}


def unit_name(entry):
    """`libfoo_bar-<hash>.rlib` -> `libfoo_bar`; `foo-bar-<hash>` -> `foo_bar`."""
    stem = entry.name.split(".", 1)[0]
    return stem.rsplit("-", 1)[0].replace("-", "_")


def prune(target_dir):
    names = rebuilt_unit_names()
    removed = 0
    # <target>/<profile>, or <target>/<triple>/<profile> under CARGO_BUILD_TARGET.
    fingerprints = [
        *target_dir.glob("*/.fingerprint"),
        *target_dir.glob("*/*/.fingerprint"),
    ]
    for fingerprint in fingerprints:
        profile_dir = fingerprint.parent
        for entry in list(profile_dir.iterdir()):
            if entry.name not in KEPT_PROFILE_ENTRIES:
                remove(entry)
                removed += 1
        for kept in KEPT_PROFILE_ENTRIES:
            kept_dir = profile_dir / kept
            if not kept_dir.is_dir():
                continue
            for entry in list(kept_dir.iterdir()):
                name = unit_name(entry)
                if name in names or name.removeprefix("lib") in names:
                    remove(entry)
                    removed += 1
    print(f"pruned {removed} entries from {target_dir}")


def remove(entry):
    if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(entry)
    else:
        entry.unlink()


if __name__ == "__main__":
    prune(pathlib.Path(sys.argv[1]))
