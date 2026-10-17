"""Standalone repro: npu_hc_pre tail-OOB x segment-boundary placement.

Hypothesis under test (see docs/a5_pd_mf.md): torch.ops.custom.npu_hc_pre
has a small out-of-tensor tail access in its T=2048 tiling path that only
faults when the op's 16MB y output lands at the end of an allocator segment
with unmapped VA right after it.

Random padding cannot reach that layout (the allocator recycles a handful of
slots and the serving address band sits ~100GB up). Instead this script
hunts boundaries deterministically: it lays down contiguous 16MB blocks and
watches data_ptr for a discontinuity -- the block right before the jump ends
at a segment boundary. It frees exactly that block (leaving a boundary-sized
hole), then calls the op so its internally-allocated y lands there.

Run on the device (no server needed; source the CANN env first):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 test/manual/npu_hc_pre_boundary_repro.py \
    --so /usr/local/python3.12.13/lib/python3.12/site-packages/custom_ops/\
custom_ops_lib.cpython-312-x86_64-linux-gnu.so

Fault -> tail-OOB confirmed in isolation; hand PC+0xe80 to the kernel owner.
All boundaries survived -> the fault needs more than a bare boundary; we move
the probe into the serving process instead.
"""

import argparse
import os
import random

import torch

# The custom ops (.so) that registers torch.ops.custom.* is loaded by the
# serving environment, not by this script's imports.
import torch_npu  # noqa: F401  (also makes device="npu" usable)

T_CRASH = 2048  # args[12]=0x800 in every recorded crash
HIDDEN = 16384
HC_MULT = 4
# Kernel signature from the IDEDD dump json: mixes[T,24] fp32 and
# hc_base[24] fp32 arrive inside the op, so the wrapper's hc_fn weight is
# [24, H] (F.linear(x_flat, hc_fn) -> mixes[T,24]) and hc_base is [24].
HC_CHANNELS = 24

BLOCK = 16 * 1024 * 1024  # y is [T, H/m] bf16 = exactly 16MB


def call_op(x, hc_fn, hc_scale, hc_base):
    return torch.ops.custom.npu_hc_pre(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )


def ensure_custom_ops(so_arg: str) -> None:
    if hasattr(torch.ops.custom, "npu_hc_pre"):
        return
    candidates = [so_arg, os.environ.get("CUSTOM_OPS_SO_PATH", "")]
    for path in candidates:
        if path and os.path.isfile(path):
            torch.ops.load_library(path)
            if hasattr(torch.ops.custom, "npu_hc_pre"):
                print(f"loaded custom ops from {path}", flush=True)
                return
    raise SystemExit(
        "torch.ops.custom.npu_hc_pre is not registered. Locate the op "
        "library and rerun with --so (or export CUSTOM_OPS_SO_PATH):\n"
        "  grep -rl npu_hc_pre /usr/local/python3.12.13/lib/python3.12/"
        "site-packages --include='*.so' 2>/dev/null"
    )


def hunt_boundaries(weights, max_blocks: int) -> int:
    """Lay contiguous 16MB blocks; at every data_ptr jump, the previous
    block ends at a boundary. Free it and run the op with y landing in the
    boundary hole. Returns the number of boundaries exercised."""
    hc_fn, hc_scale, hc_base = weights
    blocks = []
    ptrs = []
    exercised = 0
    for _ in range(max_blocks):
        # Allocate x first so it cannot occupy the boundary hole we make.
        x = torch.randn(
            T_CRASH, HC_MULT, HIDDEN // HC_MULT, dtype=torch.bfloat16, device="npu"
        )
        block = torch.empty(BLOCK, dtype=torch.uint8, device="npu")
        ptr = block.data_ptr()
        if ptrs and ptr != ptrs[-1] + BLOCK:
            boundary_end = ptrs[-1] + BLOCK
            victim = blocks.pop()
            ptrs.pop()
            del victim  # 16MB hole ending exactly at the segment boundary
            y, post, comb = call_op(x, hc_fn, hc_scale, hc_base)
            print(
                f"boundary #{exercised}: hole ends 0x{boundary_end:x} "
                f"-> y=0x{y.data_ptr():x} "
                f"y_end=0x{y.data_ptr() + y.numel() * y.element_size():x}",
                flush=True,
            )
            del y, post, comb
            exercised += 1
            blocks.append(block)
            ptrs.append(ptr)
            del x
            continue
        blocks.append(block)
        ptrs.append(ptr)
        del x
    return exercised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=512,
                        help="16MB blocks to lay down (512 = 8GB of probing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--so", default="", help="path to the custom ops .so registering torch.ops.custom"
    )
    args = parser.parse_args()
    ensure_custom_ops(args.so)
    random.seed(args.seed)
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)

    print("op schema:", torch.ops.custom.npu_hc_pre.default._schema, flush=True)
    weights = (
        torch.randn(HC_CHANNELS, HIDDEN, dtype=torch.float32, device="npu"),
        torch.ones(3, dtype=torch.float32, device="npu"),
        torch.zeros(HC_CHANNELS, dtype=torch.float32, device="npu"),
    )

    exercised = hunt_boundaries(weights, args.blocks)
    print(f"survived: exercised {exercised} segment boundaries without fault")


if __name__ == "__main__":
    main()
