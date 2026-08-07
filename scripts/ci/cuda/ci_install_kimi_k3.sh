#!/bin/bash
# Install the standard CUDA CI dependencies plus Kimi-K3's Transformers fix.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source (not bash) so the generic install's Python/venv selection remains
# active while applying the Transformers compatibility fix.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ci_install_dependency.sh" "$@"

apply_transformers_symlink_patch() {
    # transformers 5.12.1 resolves custom-code symlinks out of the HF snapshot
    # and into blobs/, then looks for relative imports by filename in blobs/.
    # Mirror the hot patch in docker/rocm.Dockerfile until upstream PR #46618
    # reaches a transformers release.
    python3 - <<'PY'
import importlib
import pathlib
import tempfile

import transformers.dynamic_module_utils as dynamic_module_utils

marks = ["Path(resolved_module_file).resolve()", "Path(source_file).resolve()"]
path = pathlib.Path(dynamic_module_utils.__file__)
src = path.read_text()
if not any(mark in src for mark in marks):
    print("transformers dynamic_module_utils already fixed; no patch needed")
else:
    patched = (
        src.replace(
            "Path(resolved_module_file).resolve()", "Path(resolved_module_file)"
        ).replace("Path(source_file).resolve()", "Path(source_file)")
    )
    assert patched != src, "FATAL: transformers symlink patch matched nothing"
    path.write_text(patched)
    print("patched transformers dynamic_module_utils.py (symlink hash fix)")

# Exercise the exact HF-cache layout that failed in the B300 Kimi-K3 run.
dynamic_module_utils = importlib.reload(dynamic_module_utils)
with tempfile.TemporaryDirectory() as tmp:
    cache = pathlib.Path(tmp) / "models--org--model"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "modeling-blob").write_text("from .media_utils import VALUE\n")
    (blobs / "media-blob").write_text("VALUE = 1\n")
    (snapshot / "modeling.py").symlink_to("../../blobs/modeling-blob")
    (snapshot / "media_utils.py").symlink_to("../../blobs/media-blob")
    source_hash = dynamic_module_utils._compute_local_source_files_hash(
        snapshot, snapshot / "modeling.py"
    )
    assert len(source_hash) == 16, source_hash
PY
}

apply_transformers_symlink_patch
