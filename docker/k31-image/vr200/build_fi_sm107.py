# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Build FlashInfer's SM107 TRT-LLM MoE module with the K3.1 DynBlock header.

docker/k31-image/patch_flashinfer.py raises DynBlockKernelMaxNumExperts from 512 to 1024 so the
DynBlock routing kernel covers K3.1's 896 experts, and build_flashinfer_dynblock.py rebuilds the
jit-cache modules `fused_moe_trtllm_sm100` and `trtllm_gen_routing`. On SM107 FlashInfer loads a
third module, `fused_moe_trtllm_sm107` (gen_trtllm_gen_fused_moe_sm100_module(enable_rubin=True)),
and the routing kernels are compiled into it. Its prebuilt copy in the flashinfer-jit-cache sm107a
provider was compiled with 512, so this script rebuilds it from the patched header.

The build needs nvcc (in the image) but no GPU. Rebuilt in the served base image, the module's device code
(.nv_fatbin) is byte-identical to the module the K3.1 VR200 points ran; the host code differs only in the
embedded build workspace path and nvcc's temporary symbol names.

Usage, inside the serving image:
  python3 build_fi_sm107.py OUT_DIR
Writes OUT_DIR/fused_moe_trtllm_sm107.so, OUT_DIR/RoutingCustomPolicy.cuh (the patched header)
and OUT_DIR/k31-dynblock.json (marker). install_k31_parts.sh OUT_DIR then installs them.
Patches the installed header in place first (a no-op when already patched).
Environment: K31_BUILD_TMP = parent of the temporary FlashInfer workspace (default: the system
temporary directory); MAX_JOBS = parallel nvcc jobs (default: all CPUs).
"""

import hashlib
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

OLD = "static constexpr int DynBlockKernelMaxNumExperts = 512;"
NEW = "static constexpr int DynBlockKernelMaxNumExperts = 1024;  // K3.1 (896 experts)"
MODULE = "fused_moe_trtllm_sm107"
ARCH = "10.7a"

_BUILD = r"""
import shutil, sys
from flashinfer.jit import fused_moe
spec = fused_moe.gen_trtllm_gen_fused_moe_sm100_module(enable_rubin=True)
print("module", spec.name, flush=True)
if spec.name != sys.argv[2]:
    sys.exit("unexpected module name " + spec.name)
spec.build(verbose=False)
shutil.copyfile(spec.jit_library_path, sys.argv[1])
"""


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    site = Path(sysconfig.get_paths()["purelib"])
    header = (
        site / "flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
    )
    text = header.read_text()
    if NEW not in text:
        if text.count(OLD) != 1:
            sys.exit(f"{header}: expected exactly one {OLD!r}")
        header.write_text(text.replace(OLD, NEW))
    shutil.copyfile(header, out / "RoutingCustomPolicy.cuh")

    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=os.environ.get("K31_BUILD_TMP")) as ws:
        env = dict(
            os.environ,
            FLASHINFER_CUDA_ARCH_LIST=ARCH,
            FLASHINFER_WORKSPACE_BASE=ws,
            MAX_JOBS=os.environ.get("MAX_JOBS", str(os.cpu_count() or 1)),
        )
        subprocess.run(
            [sys.executable, "-c", _BUILD, str(out / f"{MODULE}.so"), MODULE],
            env=env,
            check=True,
        )
    marker = {
        "DynBlockKernelMaxNumExperts": 1024,
        "module": MODULE,
        "arch": ARCH,
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "seconds": round(time.time() - t0),
        "flashinfer": md.version("flashinfer-python"),
    }
    (out / "k31-dynblock.json").write_text(json.dumps(marker, indent=1))
    print("BUILD_FI_SM107_OK", marker)


if __name__ == "__main__":
    main()
