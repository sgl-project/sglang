"""Single entry to generate all configs for the max-concurrency sweep.

Runs:
  1. gen_heter_configs.py  → per-mc base heter assignments
  2. gen_dyna_variants.py  → 11 runtime-dispatch variants per mc

Flags forwarded to both:
  --task <name>         write to configs/<task>/ instead of flat configs/
  --calib_json <path>   amortized KV sizing from calib_kv.py output
                        (only applies to gen_heter_configs)

After this, run `bash run_sweep.sh <task>`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> int:
    path = THIS_DIR / script
    print(f"\n>>> {script} {' '.join(extra)}")
    return subprocess.run(
        [sys.executable, str(path), *extra], check=False
    ).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task")
    ap.add_argument("--calib_json")
    args = ap.parse_args()

    heter_args: list[str] = []
    variants_args: list[str] = []
    if args.task:
        heter_args += ["--task", args.task]
        variants_args += ["--task", args.task]
    if args.calib_json:
        heter_args += ["--calib_json", args.calib_json]

    steps = [
        ("gen_heter_configs.py", heter_args),
        ("gen_dyna_variants.py", variants_args),
    ]
    for script, extra in steps:
        rc = _run(script, extra)
        if rc != 0:
            print(f"FAILED: {script} (exit {rc})", file=sys.stderr)
            return rc
    dest = f"configs/{args.task}/mc*/" if args.task else "configs/mc*/"
    print(f"\nAll configs generated under {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
