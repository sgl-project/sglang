#!/usr/bin/env python3
"""Report which K3 tuned-config tables this GPU actually resolves.

Two K3 kernels select behavior from JSON tables keyed on
torch.cuda.get_device_name(). A miss is silent: sp_collective falls back to
NCCL, moe_front falls back to a default heuristic. Run this on the target host
before assuming either fused path is live.

  python3 benchmark/kernels/kimi_k3/check_tuned_configs.py [--world 8]
"""

from __future__ import annotations

import argparse
import os

import torch

from sglang.kernels.ops.kimi_k3 import sp_collective
from sglang.kernels.ops.moe import moe_front


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=torch.cuda.device_count())
    ap.add_argument("--hidden-size", type=int, default=7168)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return 1

    raw = torch.cuda.get_device_name(torch.device("cuda:0"))
    # Same normalization the lookups apply; the filename must match it exactly.
    norm = raw.replace(" ", "_").replace("/", "_")
    cap = torch.cuda.get_device_capability(0)
    print(f"device            : {raw!r}")
    print(f"normalized        : {norm}")
    print(f"capability        : sm_{cap[0]}{cap[1]}")
    print(f"visible GPUs      : {torch.cuda.device_count()}  (assuming world={args.world})")
    print()

    missing = []

    sp_dir = os.path.join(os.path.dirname(sp_collective.__file__), "configs", "sp_collective")
    sp_name = f"world={args.world},H={args.hidden_size},device_name={norm}.json"
    sp_hit = os.path.exists(os.path.join(sp_dir, sp_name))
    print(f"sp_collective     : {'TUNED' if sp_hit else 'MISS -> falls back to NCCL'}")
    print(f"  expects         : {sp_name}")
    if not sp_hit:
        missing.append(os.path.join(sp_dir, sp_name))
        have = sorted(f for f in os.listdir(sp_dir) if f.endswith(".json"))
        print(f"  present         : {have or '(none)'}")

    mf_dir = os.path.join(os.path.dirname(moe_front.__file__), "configs", "moe_front")
    for kind in ("strategy", "epilogue"):
        name = f"{kind},E={moe_front.NUM_EXPERTS},topk={moe_front.TOPK},device_name={norm}.json"
        hit = os.path.exists(os.path.join(mf_dir, name))
        print(f"moe_front/{kind:9s}: {'TUNED' if hit else 'MISS -> default heuristic'}")
        print(f"  expects         : {name}")
        if not hit:
            missing.append(os.path.join(mf_dir, name))

    print()
    if missing:
        print("Untuned on this device. Generate with bench_sp_collective.py --tune")
        print("and write each table to the exact path above.")
    else:
        print("All K3 tuned tables resolve on this device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
