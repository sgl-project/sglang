"""Write muse_image_manifest.json for a build context made by prepare_context.sh.

Usage: make_manifest.py <context dir> <tree commit> <changed-files listing (git diff --name-status vs the base)>
The context must also hold .aiter_patched/aiter/... (the aiter files as patches 05 and 06 leave them).
"""

import hashlib
import json
import sys
from pathlib import Path

ctx, commit, changes = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
PKG = "/sgl-workspace/sglang/python/sglang"
AITER = "/sgl-workspace/aiter"
PATCHED = (
    "aiter/ops/triton/attention/unified_attention.py",
    "aiter/ops/triton/_triton_kernels/attention/unified_attention.py",
    "aiter/configs/model_configs/muse_spark_bf16_tuned_gemm.csv",
    "aiter/configs/model_configs/qwen3_5_397b_bf16_tuned_gemm.csv",
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


files = {
    f"{PKG}/{p.relative_to(ctx / 'python/sglang')}": sha(p)
    for p in sorted((ctx / "python/sglang").rglob("*"))
    if p.is_file()
}
aiter_files = {f"{AITER}/{f}": sha(ctx / ".aiter_patched" / f) for f in PATCHED}
manifest = {
    "schema": 2,
    "tree_commit": commit,
    "tree_base": "upstream PR #44667 37197b72708c (muse-spark ROCm, 4/5) on main d33dd0566a",
    "base_image": "lmsysorg/sglang-rocm:muse-spark-open@sha256:"
    "72065dfc0b87818ab415f96c0fc08848233d25f22b002ac8f28c33fad6ec82a4",
    "install": f"editable mapping 'sglang' -> {PKG} (package directory replaced); "
    "aiter patches 05 + 06 applied to /sgl-workspace/aiter (Python/Triton files and GEMM CSVs only)",
    "aiter_patch_sha256": {
        p.name: sha(p) for p in sorted(ctx.glob("0*-aiter-*.patch"))
    },
    "runtime_env_required": [],
    "changes_vs_base": [
        line.split("\t") for line in changes.read_text().splitlines() if line
    ],
    "tree_sha256": hashlib.sha256(
        "".join(f"{k} {v}\n" for k, v in files.items()).encode()
    ).hexdigest(),
    "files": files,
    "aiter_files": aiter_files,
}
json.dump(manifest, sys.stdout, indent=1)
print()
