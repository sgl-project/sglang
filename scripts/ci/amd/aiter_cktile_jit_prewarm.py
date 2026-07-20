"""Prebuild the AITER CKTile module used by DeepSeek-R1 on ROCm 7.0."""

import importlib
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODULE = "module_gemm_a8w8_blockscale_cktile"


def import_and_report(shared_object: Path, status: str, elapsed: float) -> int:
    imported = importlib.import_module(f"aiter.jit.{MODULE}")
    print(
        f"PREWARM_{status} module={MODULE} elapsed={elapsed:.3f}s "
        f"so={shared_object} size={shared_object.stat().st_size} "
        f"imported={imported.__name__}",
        flush=True,
    )
    return 0


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

    print(
        f"PREWARM_START module={MODULE} commit={commit} jit_dir={jit_dir} "
        f"utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        if shared_object.exists():
            return import_and_report(shared_object, "CACHE_HIT", 0.0)

        # This is the same get_args_of_build -> build_module path taken by
        # compile_ops after its initial import raises ModuleNotFoundError.
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
        print(f"PREWARM_FAILED so_exists={shared_object.exists()}", flush=True)
        traceback.print_exc()
        return 2

    elapsed = time.perf_counter() - started
    if not shared_object.exists():
        print(f"PREWARM_FAILED missing_so={shared_object}", flush=True)
        return 2

    return import_and_report(shared_object, "SUCCESS", elapsed)


if __name__ == "__main__":
    sys.exit(main())
