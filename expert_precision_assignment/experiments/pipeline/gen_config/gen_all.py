"""Single entry to generate all configs for the max-concurrency sweep.

Runs:
  1. gen_heter_configs.py  → per-mc base heter assignments
  2. gen_dyna_variants.py  → 11 runtime-dispatch variants per mc

Flags forwarded to both:
  --task <name>         write to data/configs/<task>/ instead of flat
                        data/configs/
  --calib_json <path>   amortized KV sizing from calib_kv.py output
                        (only applies to gen_heter_configs)
  --attention_num_bits {16,4}
                        baked into heter_config.json so the runtime swaps
                        attention qkv_proj+o_proj to INT4 GPTQ-Marlin
                        (only applies to gen_heter_configs; variants
                        deep-copy the base config so they inherit it).

After this, run `bash pipeline/run_sweep.sh <task>`.
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
    ap.add_argument(
        "--attention_num_bits",
        type=int,
        choices=(16, 4),
        default=16,
        help="Forwarded to gen_heter_configs.py; baked into heter_config.json.",
    )
    args = ap.parse_args()

    heter_args: list[str] = []
    variants_args: list[str] = []
    if args.task:
        heter_args += ["--task", args.task]
        variants_args += ["--task", args.task]
    if args.calib_json:
        heter_args += ["--calib_json", args.calib_json]
    heter_args += ["--attention_num_bits", str(args.attention_num_bits)]

    steps = [
        ("gen_heter_configs.py", heter_args),
        ("gen_dyna_variants.py", variants_args),
    ]
    for script, extra in steps:
        rc = _run(script, extra)
        if rc != 0:
            print(f"FAILED: {script} (exit {rc})", file=sys.stderr)
            return rc
    dest = (
        f"data/configs/{args.task}/mc*/" if args.task else "data/configs/mc*/"
    )
    print(f"\nAll configs generated under {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
