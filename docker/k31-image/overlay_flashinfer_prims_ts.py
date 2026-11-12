# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Overlay FlashInfer main's Prims-TS sparse MLA onto the image's FlashInfer 0.7.2.

The opt-in `prims_ts` DSA backend needs FlashInfer's `flashinfer/attention/prims_ts/` from main
26bb6dc9 (flashinfer-ai/flashinfer #7410 and #7984); 0.7.2 ships an older copy that fails at
startup. This replaces only that directory's Python files (CuTe DSL; no csrc, AOT or jit-cache
module changes), the same tree as the earlier GB300 runs used. Everything else in
FlashInfer, including the DynBlock-rebuilt modules, stays 0.7.2.

Then applies flashinfer-prims-ts-k31ht.patch (K31HT_MLA_TAILROW, default off: unset, the patched files run
26bb6dc9's code). Idempotent: a marker records the overlaid tree and the patch, so a rerun is a no-op unless a
reinstall replaced the files. Needs network access only when it overlays.
"""

import hashlib
import io
import json
import subprocess
import sysconfig
import tarfile
import urllib.request
from importlib.metadata import version
from pathlib import Path

FLASHINFER_COMMIT = "26bb6dc93831c0192e524357dd25d7092917ae9b"
FLASHINFER_BASE_VERSION = "0.7.2"
SUBDIR = "flashinfer/attention/prims_ts"

SITE = Path(sysconfig.get_paths()["purelib"])
TARGET = SITE / SUBDIR
MARKER = TARGET / "k31-prims-ts.json"
PATCH = Path(__file__).resolve().parent / "flashinfer-prims-ts-k31ht.patch"


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    installed = version("flashinfer-python")
    if installed != FLASHINFER_BASE_VERSION:
        raise SystemExit(
            f"flashinfer-python {installed} is installed; the Prims-TS overlay is validated on "
            f"{FLASHINFER_BASE_VERSION} only"
        )
    if MARKER.exists():
        marker = json.loads(MARKER.read_text())
        if (
            marker.get("commit") == FLASHINFER_COMMIT
            and marker.get("patch_sha256") == hashlib.sha256(PATCH.read_bytes()).hexdigest()
            and marker.get("tree_sha256") == tree_sha256(TARGET)
        ):
            print(f"Prims-TS already overlaid from flashinfer {FLASHINFER_COMMIT[:8]}")
            return

    url = f"https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/{FLASHINFER_COMMIT}"
    with urllib.request.urlopen(url, timeout=600) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    prefix = f"flashinfer-{FLASHINFER_COMMIT}/{SUBDIR}/"
    members = [
        m
        for m in archive.getmembers()
        if m.isfile() and m.name.startswith(prefix) and m.name.endswith(".py")
    ]
    if not members:
        raise SystemExit(f"{SUBDIR} not found in flashinfer {FLASHINFER_COMMIT}")

    for path in TARGET.rglob("*.py"):
        path.unlink()
    for member in members:
        dest = TARGET / member.name[len(prefix) :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.extractfile(member).read())
    subprocess.run(["git", "apply", "-p1", str(PATCH)], cwd=TARGET, check=True)

    MARKER.write_text(
        json.dumps(
            {
                "commit": FLASHINFER_COMMIT,
                "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
                "tree_sha256": tree_sha256(TARGET),
            }
        )
    )
    print(
        f"overlaid {len(members)} Prims-TS files from flashinfer {FLASHINFER_COMMIT[:8]} + {PATCH.name}"
    )


if __name__ == "__main__":
    main()
