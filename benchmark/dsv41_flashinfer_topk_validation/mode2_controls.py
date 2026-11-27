"""Negative controls for test_sgl_kernel_mode2_matches_flashinfer_small_tie_break.

    PYTHONPATH=<tree>/python python3 mode2_controls.py <tree>

Each control breaks one premise of the mode-2 parity check and runs it in its own
process (with SGLANG_TOPK_V2_MODE=2, as the test does); the check must fail:
  - sgl-kernel's top-k v2 module is built in mode 1 although the environment says 2;
  - FlashInfer runs the large tie-break instead of the small one.
The unmodified check runs last and must pass.
"""

from __future__ import annotations

import os
import subprocess
import sys

TREE = os.path.abspath(sys.argv[1])
TEST = f"{TREE}/test/registered/kernels/ops/attention/test_dsv41_topk_selector.py"

LOAD = f"""
import runpy
g = runpy.run_path({TEST!r}, run_name="mode2_control")
"""

CONTROLS = {
    "sgl-kernel top-k v2 built in mode 1": LOAD
    + """
from sglang.kernels.ops.attention.dsv4 import topk
from sglang.srt.environ import envs
build = topk._jit_topk_v2_module
def mode1(*args, **kwargs):
    with envs.SGLANG_TOPK_V2_MODE.override(1):
        return build(*args, **kwargs)
topk._jit_topk_v2_module = mode1
g["mode2_parity"]()
""",
    "FlashInfer large tie-break": LOAD
    + """
g["FLASHINFER_SMALL"][:] = [(b, "large", det) for b, _, det in g["FLASHINFER_SMALL"]]
g["mode2_parity"]()
""",
}


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SGLANG_TOPK_V2_MODE": "2"},
        capture_output=True,
        text=True,
        timeout=1800,
    )


def main() -> int:
    missed = 0
    for name, code in CONTROLS.items():
        proc = run(code)
        tail = [l for l in (proc.stdout + proc.stderr).splitlines() if "AssertionError" in l]
        if proc.returncode != 0 and tail:
            print(f"CAUGHT  {name}: {tail[-1][:240]}", flush=True)
        else:
            missed += 1
            print(f"MISSED  {name}: exit {proc.returncode}", flush=True)
            print((proc.stdout + proc.stderr)[-3000:], flush=True)
    proc = run(LOAD + '\ng["mode2_parity"]()\n')
    clean = proc.returncode == 0
    print(("PASSED" if clean else "FAILED") + "  unmodified check", flush=True)
    if not clean:
        print((proc.stdout + proc.stderr)[-3000:], flush=True)
    print("all controls caught" if missed == 0 and clean else f"controls missed: {missed}", flush=True)
    return 0 if missed == 0 and clean else 1


if __name__ == "__main__":
    sys.exit(main())
