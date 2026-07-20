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

    from aiter.jit.core import build_module, get_args_of_build, get_user_jit_dir

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
        # This is the same get_args_of_build -> build_module path taken by
        # compile_ops after its initial import raises ModuleNotFoundError. Calling
        # it directly isolates JIT compilation from torch dispatcher schema
        # validation and from launching the compiled GPU kernel.
        args = get_args_of_build(MODULE)
        build_module(
            MODULE,
            args["srcs"],
            args["flags_extra_cc"],
            args["flags_extra_hip"],
            args["blob_gen_cmd"],
            args["extra_include"],
            args["extra_ldflags"],
            args["verbose"],
            args["is_python_module"],
            args["is_standalone"],
            args["torch_exclude"],
            args.get("third_party", []),
            args.get("hipify", False),
            flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
        )
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
