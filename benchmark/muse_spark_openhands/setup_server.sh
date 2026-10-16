#!/usr/bin/env bash
# Use the runtime built by the public muse-spark Dockerfile; do not rebuild stale dependency recipes here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PY=${SERVER_PYTHON:-python3}
"$PY" - "$REPO_ROOT" <<'PY'
import importlib.metadata as m
import sys
from pathlib import Path
import torch

assert torch.version.cuda and torch.version.cuda.startswith("13."), "Use the Muse Spark CUDA 13 image"
for package, version in {
    "torch": "2.14.1", "transformers": "5.19.0", "sglang-kernel": "0.4.9",
    "flashinfer-python": "0.7.0.post1", "flashinfer-cubin": "0.7.0.post1",
    "flashinfer-jit-cache": "0.7.0.post1",
}.items():
    actual = m.version(package).split("+")[0]
    assert actual == version, f"{package}: {actual} != {version}; rebuild docker/muse_spark/muse-spark.Dockerfile"
assert m.version("sgl-deep-gemm") == "0.2.1.post1+muse.cbe1727", "Use the Dockerfile's patched DeepGEMM"
assert (Path(sys.argv[1]) / "docker/muse_spark/apply_deep_gemm_patches.sh").is_file()
PY
"$PY" -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT/python"
"$PY" - "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
import sglang
from sglang.srt.layers.moe.muse_spark_megamoe import require_muse_megamoe_patch
assert Path(sglang.__file__).resolve() == Path(sys.argv[1]) / "python/sglang/__init__.py"
require_muse_megamoe_patch()
print("Serving source:", sglang.__file__)
PY
