"""Let FlashInfer's TRT-LLM MoE routing use the DynBlock kernel for K3.1's 896 experts.

Raises DynBlockKernelMaxNumExperts from 512 to 1024 in the installed flashinfer headers;
a no-op when already applied. Only builds read this header: build_flashinfer_dynblock.py
then recompiles the prebuilt jit-cache modules that include it.
"""

import sys
import sysconfig
from pathlib import Path

HEADER = (
    Path(sysconfig.get_paths()["purelib"])
    / "flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
)
OLD = "static constexpr int DynBlockKernelMaxNumExperts = 512;"
NEW = "static constexpr int DynBlockKernelMaxNumExperts = 1024;  // K3.1 (896 experts)"

text = HEADER.read_text()
if NEW in text:
    print(f"{HEADER}: already patched")
elif text.count(OLD) == 1:
    HEADER.write_text(text.replace(OLD, NEW))
    print(f"{HEADER}: patched")
else:
    sys.exit(f"{HEADER}: expected exactly one {OLD!r}")
