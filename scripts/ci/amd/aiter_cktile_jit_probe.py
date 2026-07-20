"""Cold-build one AITER CKTile module and report whether it is importable."""

import importlib
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODULE = "module_gemm_a8w8_blockscale_bpreshuffle_cktile"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from aiter.jit.core import get_user_jit_dir
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale_bpreshuffle_cktile,
    )

    commit = subprocess.check_output(
        ["git", "-C", "/sgl-workspace/aiter", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    jit_dir = Path(get_user_jit_dir())
    shared_object = jit_dir / f"{MODULE}.so"
    case_name = os.environ.get("PROBE_CASE_NAME", "unknown")

    print(
        f"PROBE_START case={case_name} module={MODULE} commit={commit} "
        f"jit_dir={jit_dir} utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        flush=True,
    )
    if shared_object.exists():
        print(f"PRECONDITION_FAILED preexisting_so={shared_object}", flush=True)
        return 3

    started = time.perf_counter()
    try:
        # compile_ops loads/builds the extension before pybind validates the
        # deliberately omitted runtime tensors. This isolates compilation from
        # kernel execution and model/runtime behavior.
        gemm_a8w8_blockscale_bpreshuffle_cktile()
    except TypeError as exc:
        if not shared_object.exists():
            print("BUILD_FAILED TypeError occurred before .so was produced", flush=True)
            traceback.print_exc()
            return 2
        print(f"EXPECTED_POST_BUILD_CALL_ERROR {exc}", flush=True)
    except BaseException:
        print(f"BUILD_FAILED so_exists={shared_object.exists()}", flush=True)
        traceback.print_exc()
        return 2

    elapsed = time.perf_counter() - started
    if not shared_object.exists():
        print(f"BUILD_FAILED missing_so={shared_object}", flush=True)
        return 2

    imported = importlib.import_module(f"aiter.jit.{MODULE}")
    size = shared_object.stat().st_size
    print(
        f"BUILD_SUCCESS case={case_name} elapsed={elapsed:.3f}s "
        f"so={shared_object} size={size} imported={imported.__name__}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
