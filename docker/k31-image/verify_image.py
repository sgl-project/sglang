"""Fail the image build when a compiled K3.1 component would silently fall back.

Runs inside the image without a GPU, so it checks artifacts rather than loading CUDA code:
- every op that our FlashMLA extension source registers is in the installed flashmla_ops;
- the flashinfer DynBlock header patch is in place (patch_flashinfer.py), and every installed
  copy of the modules that include it was rebuilt from this header for 1024 experts
  (build_flashinfer_dynblock.py marker), so a flashinfer upgrade cannot silently undo it;
- the Rust TreeCore extension loads.
"""

import hashlib
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

SGLANG = Path("/sgl-workspace/sglang")
SITE = Path(sysconfig.get_paths()["purelib"])

errors = []

source = (SGLANG / "python/sglang/kernels/aot/csrc/flashmla_extension.cc").read_text()
ops = re.findall(r'm\.def\(\s*"(\w+)\(', source)
module = (SITE / "sgl_kernel/flashmla_ops.abi3.so").read_bytes()
missing = [op for op in ops if op.encode() not in module]
print(f"flashmla_ops: {len(ops) - len(missing)}/{len(ops)} registered ops present")
if not ops or missing:
    errors.append(f"flashmla_ops lacks {missing or 'all ops'}: the PyPI build is installed")

header = SITE / "flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
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
    errors.append("no installed fused_moe_trtllm_sm100 / trtllm_gen_routing module found")
for module_path in modules:
    marker_path = module_path.parent / "k31-dynblock.json"
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
    stale = (
        marker.get("header_sha256") != header_sha256
        or marker.get("DynBlockKernelMaxNumExperts") != 1024
    )
    print(f"{module_path.relative_to(SITE)}: {'stale' if stale else 'rebuilt for 1024 experts'}")
    if stale:
        errors.append(
            f"{module_path} was not rebuilt from the patched header (build_flashinfer_dynblock.py)"
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
