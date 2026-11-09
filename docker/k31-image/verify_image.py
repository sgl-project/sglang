"""Fail the image build when a compiled K3.1 component would silently fall back.

Runs inside the image without a GPU, so it checks artifacts rather than loading CUDA code:
- the flashinfer DynBlock header patch is in place (patch_flashinfer.py), and every installed
  copy of the modules that include it was rebuilt from this header for 1024 experts
  (build_flashinfer_dynblock.py marker), so a flashinfer upgrade cannot silently undo it;
- the FlashInfer Prims-TS overlay (overlay_flashinfer_prims_ts.py) and the patched Dynamo runtime
  (build_dynamo.py) are installed and unchanged since they were made;
- the Rust TreeCore extension loads.
"""

import hashlib
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

SITE = Path(sysconfig.get_paths()["purelib"])

errors = []

header = (
    SITE / "flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
)
match = re.search(r"DynBlockKernelMaxNumExperts = (\d+);", header.read_text())
print(f"flashinfer DynBlockKernelMaxNumExperts: {match and match.group(1)}")
if not match or int(match.group(1)) < 896:
    errors.append("flashinfer DynBlock routing does not cover 896 experts")
header_sha256 = hashlib.sha256(header.read_bytes()).hexdigest()
modules = [
    path
    for name in ("fused_moe_trtllm_sm100", "trtllm_gen_routing")
    for path in SITE.glob(f"flashinfer*/**/{name}.so")
]
if not modules:
    errors.append(
        "no installed fused_moe_trtllm_sm100 / trtllm_gen_routing module found"
    )
for module_path in modules:
    marker_path = module_path.parent / "k31-dynblock.json"
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
    stale = (
        marker.get("header_sha256") != header_sha256
        or marker.get("DynBlockKernelMaxNumExperts") != 1024
    )
    print(
        f"{module_path.relative_to(SITE)}: {'stale' if stale else 'rebuilt for 1024 experts'}"
    )
    if stale:
        errors.append(
            f"{module_path} was not rebuilt from the patched header (build_flashinfer_dynblock.py)"
        )

prims_ts = SITE / "flashinfer/attention/prims_ts"
prims_marker_path = prims_ts / "k31-prims-ts.json"
prims_marker = (
    json.loads(prims_marker_path.read_text()) if prims_marker_path.exists() else {}
)
prims_digest = hashlib.sha256()
for path in sorted(prims_ts.rglob("*.py")):
    prims_digest.update(
        str(path.relative_to(prims_ts)).encode() + b"\0" + path.read_bytes()
    )
prims_ok = prims_marker.get("tree_sha256") == prims_digest.hexdigest()
print(
    f"flashinfer Prims-TS: {'overlaid from ' + prims_marker['commit'][:8] if prims_ok else 'not overlaid'}"
)
if not prims_ok:
    errors.append(
        "flashinfer prims_ts is not the overlaid tree (overlay_flashinfer_prims_ts.py)"
    )

dynamo_core = SITE / "dynamo/_core.abi3.so"
dynamo_marker_path = SITE / "dynamo/k31-dynamo.json"
dynamo_marker = (
    json.loads(dynamo_marker_path.read_text()) if dynamo_marker_path.exists() else {}
)
dynamo_ok = (
    dynamo_core.exists()
    and dynamo_marker.get("core_sha256")
    == hashlib.sha256(dynamo_core.read_bytes()).hexdigest()
)
print(
    f"Dynamo runtime: {'patched build ' + dynamo_marker['key'] if dynamo_ok else 'missing or stock'}"
)
if not dynamo_ok:
    errors.append("dynamo/_core is not the patched runtime (build_dynamo.py)")
dynamo_sglang = SITE / "dynamo/sglang"
for name, marker in (
    ("_compat.py", 'KimiK31ForConditionalGeneration"} & set('),
    ("args.py", "# K3.1 DSA DCP decode edit"),
    ("register.py", "# K3.1 model card edit"),
    ("register.py", "# K3.1 image: stage the K3.1 model card on first use"),
):
    path = dynamo_sglang / name
    if not path.exists() or marker not in path.read_text():
        errors.append(
            f"dynamo/sglang/{name} lacks the K3.1 edit {marker!r} (build_dynamo.py)"
        )

tree_core = subprocess.run(
    [sys.executable, "-c", "from sglang.srt.mem_cache.rust_tree_core import extension"],
    cwd="/tmp",
    capture_output=True,
    text=True,
    timeout=600,
)
print(f"Rust TreeCore import: exit {tree_core.returncode}")
if tree_core.returncode != 0:
    errors.append(f"Rust TreeCore does not load:\n{tree_core.stderr[-2000:]}")

if errors:
    sys.exit("image verification failed:\n- " + "\n- ".join(errors))
print("image verification passed")
