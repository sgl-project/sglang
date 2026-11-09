# SPDX-License-Identifier: Apache-2.0

"""Locate the first split/unsplit Mamba2 state difference on fixed inputs.

Run in the SGLang environment under investigation. The three state-producing
stages run directly, so this diagnosis is separate from the original full-scan
regression. Optional block-size overrides affect only this process.
"""

import argparse
import inspect
import json
import math
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.mamba.triton_ops import ssd_chunk_state, ssd_state_passing


class _FixedStatePassing:
    def __init__(self, kernel, block_size, num_stages):
        self.kernel = kernel.fn
        self.block_size = block_size
        self.num_stages = num_stages

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            return self.kernel[grid](
                *args,
                **kwargs,
                BLOCK_SIZE=self.block_size,
                num_warps=4,
                num_stages=self.num_stages,
            )

        return run


def _json_scalar(tensor):
    value = tensor.item()
    return value if math.isfinite(value) else str(value)


def first_difference(expected, actual):
    expected = expected.to(actual.dtype)
    mismatch = expected != actual
    count = mismatch.count_nonzero().item()
    if count == 0:
        return None
    index = tuple(mismatch.nonzero()[0].tolist())
    return {
        "count": count,
        "index": index,
        "expected": _json_scalar(expected[index]),
        "actual": _json_scalar(actual[index]),
        "max_abs": _json_scalar((expected.float() - actual.float()).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--length", type=int, default=1025)
    parser.add_argument("--split", type=int, default=512)
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43413)
    parser.add_argument("--decay-scale", type=float, default=1.0)
    parser.add_argument("--dt-shift", type=float, default=-4.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--state-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--with-initial-state", action="store_true")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dimensions = (
        args.chunk_size,
        args.length,
        args.heads,
        args.head_dim,
        args.state_dim,
        args.groups,
        args.repeats,
    )
    if min(dimensions) <= 0 or args.chunk_size & (args.chunk_size - 1):
        parser.error(
            "dimensions and repeats must be positive, and chunk-size must be a power of two"
        )
    if not (0 < args.split < args.length and args.split % args.chunk_size == 0):
        parser.error("split must be an interior chunk boundary")
    if args.heads % args.groups:
        parser.error("heads must be divisible by groups")
    if not math.isfinite(args.decay_scale) or args.decay_scale <= 0:
        parser.error("decay-scale must be finite and positive")
    if not math.isfinite(args.dt_shift):
        parser.error("dt-shift must be finite")
    if args.num_stages is not None and args.block_size is None:
        parser.error("num-stages requires block-size to select a fixed configuration")
    if args.block_size is not None:
        args.num_stages = 1 if args.num_stages is None else args.num_stages
        if (
            args.block_size <= 0
            or args.block_size & (args.block_size - 1)
            or args.num_stages < 1
        ):
            parser.error(
                "block-size must be a positive power of two, and num-stages must be positive"
            )
        ssd_state_passing._state_passing_fwd_kernel = _FixedStatePassing(
            ssd_state_passing._state_passing_fwd_kernel,
            args.block_size,
            args.num_stages,
        )
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(
        1, args.length, args.heads, args.head_dim, device="cuda", dtype=dtype
    )
    dt = torch.randn(1, args.length, args.heads, device="cuda") + args.dt_shift
    B = torch.randn(
        1, args.length, args.groups, args.state_dim, device="cuda", dtype=dtype
    )
    A = -torch.exp(torch.rand(args.heads, device="cuda")) * args.decay_scale
    bias = torch.randn(args.heads, device="cuda")
    state_dtype = getattr(torch, args.state_dtype)
    initial = (
        torch.randn(
            1,
            args.heads,
            args.head_dim * args.state_dim,
            device="cuda",
            dtype=state_dtype,
        )
        if args.with_initial_state
        else None
    )
    cumsum_options = (
        {"preserve_zero_dt": True}
        if "preserve_zero_dt"
        in inspect.signature(ssd_chunk_state._chunk_cumsum_fwd).parameters
        else {}
    )

    def stages(start, end, initial):
        cumsum, effective_dt = ssd_chunk_state._chunk_cumsum_fwd(
            dt[:, start:end],
            A,
            args.chunk_size,
            dt_bias=bias,
            dt_softplus=True,
            **cumsum_options,
        )
        seq_idx = torch.zeros(1, end - start, device="cuda", dtype=torch.int32)
        local = ssd_chunk_state._chunk_state_fwd(
            B[:, start:end],
            x[:, start:end],
            effective_dt,
            cumsum,
            seq_idx=seq_idx,
            states_in_fp32=True,
        )
        passed, final = ssd_state_passing._state_passing_fwd(
            local.flatten(-2),
            cumsum,
            initial_states=initial,
            seq_idx=seq_idx,
            chunk_size=args.chunk_size,
            out_dtype=state_dtype,
            is_cont_batched=True,
            chunk_offsets=torch.zeros(
                cumsum.shape[2], device="cuda", dtype=torch.int32
            ),
            round_boundary_states=True,
        )
        config = {
            "cumsum": str(ssd_chunk_state._chunk_cumsum_fwd_kernel.best_config),
            "chunk_state": str(ssd_chunk_state._chunk_state_fwd_kernel.best_config),
            "state_passing": (
                f"BLOCK_SIZE={args.block_size}, num_warps=4, num_stages={args.num_stages}"
                if args.block_size is not None
                else str(ssd_state_passing._state_passing_fwd_kernel.best_config)
            ),
        }
        return effective_dt, cumsum, local, passed, final, config

    failures = []
    reference = stages(0, args.length, initial)
    first_chunk = args.split // args.chunk_size
    for iteration in range(args.repeats):
        full = stages(0, args.length, initial)
        prefix = stages(0, args.split, initial)
        suffix = stages(args.split, args.length, prefix[4].to(state_dtype))
        checks = (
            ("full_replay_local", reference[2], full[2]),
            ("full_replay_final", reference[4], full[4]),
            ("prefix_cumsum", full[1][:, :, :first_chunk], prefix[1]),
            ("suffix_cumsum", full[1][:, :, first_chunk:], suffix[1]),
            ("prefix_local", full[2][:, :first_chunk], prefix[2]),
            ("suffix_local", full[2][:, first_chunk:], suffix[2]),
            ("prefix_final", full[3][:, first_chunk], prefix[4]),
            ("suffix_passed", full[3][:, first_chunk:], suffix[3]),
            ("suffix_final", full[4], suffix[4]),
        )
        for name, expected, actual in checks:
            difference = first_difference(expected, actual)
            if difference is not None:
                failures.append({"iteration": iteration, "stage": name, **difference})
                if len(failures) == 1:
                    torch.save(
                        {"expected": expected.cpu(), "actual": actual.cpu()},
                        args.output.with_suffix(".pt"),
                    )
        if failures:
            break
    properties = torch.cuda.get_device_properties(0)
    record = {
        "configuration": vars(args) | {"output": str(args.output)},
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "preserve_zero_dt": bool(cumsum_options),
        "device": properties.name,
        "gcn_arch": getattr(properties, "gcnArchName", None)
        if torch.version.hip
        else None,
        "tuning": {
            "full": full[5],
            "prefix": prefix[5],
            "suffix": suffix[5],
        },
        "completed_repeats": iteration + 1,
        "failures": failures,
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
