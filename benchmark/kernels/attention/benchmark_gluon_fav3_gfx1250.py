"""Correctness-gated AITER BF16 vs Gluon FAv3 benchmark on gfx1250."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

import aiter
import torch
import triton
from aiter.jit.core import ENABLE_CK

from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
    FAv3LaunchConfig,
    gluon_fav3_attention,
    select_fav3_launch_config,
)

PRESETS = {
    "reference": (1, 8192, 8192, 16),
    # One Wan2.2-T2V-A14B request at 720x1280x193: 49*45*80 tokens.
    # Conditional and unconditional CFG branches run in separate B1 calls.
    "wan-cross": (1, 176_400, 512, 40),
    "wan-self": (1, 176_400, 176_400, 40),
}


def aiter_attention(query, key, value):
    """Match the default BF16 AITerImpl call, including RTZ conversion."""
    return aiter.flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=query.shape[-1] ** -0.5,
        causal=False,
        return_attn_probs=False,
        return_lse=False,
        how_v3_bf16_cvt=2,
    )


def check_sampled_fp64(query, key, value, outputs):
    """Check selected batch/head/query positions against all keys in FP64."""
    batch, seqlen_q, num_heads, head_size = query.shape
    batches = torch.tensor(sorted({0, batch - 1}), device=query.device)
    rows = torch.tensor(
        sorted(
            {
                0,
                min(127, seqlen_q - 1),
                min(128, seqlen_q - 1),
                min(255, seqlen_q - 1),
                min(256, seqlen_q - 1),
                seqlen_q // 2,
                seqlen_q - 1,
            }
        ),
        device=query.device,
    )
    heads = torch.tensor(
        sorted({0, num_heads // 2, num_heads - 1}), device=query.device
    )

    def sample_query(tensor):
        return (
            tensor.index_select(1, rows).index_select(2, heads).index_select(0, batches)
        )

    query_fp64 = sample_query(query).transpose(1, 2).double()
    key_fp64 = (
        key.index_select(2, heads).index_select(0, batches).transpose(1, 2).double()
    )
    value_fp64 = (
        value.index_select(2, heads).index_select(0, batches).transpose(1, 2).double()
    )
    probabilities = torch.softmax(
        query_fp64 @ key_fp64.transpose(-1, -2) * head_size**-0.5, dim=-1
    )
    reference = (probabilities @ value_fp64).transpose(1, 2)
    reference_rms = reference.square().mean().sqrt().item()
    metrics = {}
    for name, output in outputs.items():
        actual = sample_query(output).double()
        difference = actual - reference
        rms_error = difference.square().mean().sqrt().item()
        relative_rms = rms_error / max(reference_rms, 1e-30)
        if not torch.isfinite(actual).all() or relative_rms > 0.005:
            raise AssertionError(
                f"{name} failed independent FP64 check: relative RMS={relative_rms}"
            )
        torch.testing.assert_close(
            actual, reference, rtol=0.02, atol=0.01 * reference_rms
        )
        metrics[name] = {
            "relative_rms": relative_rms,
            "max_abs": difference.abs().max().item(),
        }
    print(
        "independent_fp64="
        + json.dumps(
            {
                "batches": batches.tolist(),
                "query_rows": rows.tolist(),
                "heads": heads.tolist(),
                "all_keys": key.shape[1],
                "metrics": metrics,
            },
            sort_keys=True,
        )
    )


def package_provenance(module):
    source = Path(module.__file__).resolve()
    result = {
        "version": getattr(module, "__version__", "unknown"),
        "source": str(source),
    }
    for parent in source.parents:
        if not (parent / ".git").exists():
            continue
        try:
            revision = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            result["revision"] = revision.stdout.strip()
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(parent),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            result["tracked_source_modified"] = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            result["revision"] = "unavailable"
        break
    return result


def print_provenance(device):
    properties = torch.cuda.get_device_properties(device)
    provenance = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "hip": torch.version.hip,
        "device": properties.name,
        "architecture": getattr(properties, "gcnArchName", "unknown"),
        "triton": package_provenance(triton),
        "aiter": package_provenance(aiter),
        "ENABLE_CK": bool(ENABLE_CK),
        "ENABLE_CK_env": os.environ.get("ENABLE_CK", "<unset>"),
        "aiter_baseline": "AITerImpl-equivalent BF16, default scale, how_v3_bf16_cvt=2",
    }
    print("provenance=" + json.dumps(provenance, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="reference")
    parser.add_argument(
        "--batch",
        type=int,
        help="Override preset batch; Wan defaults to one CFG branch.",
    )
    parser.add_argument("--sq", type=int)
    parser.add_argument("--sk", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--provider", choices=("both", "aiter", "gluon"), default="both"
    )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--trace-provider", choices=("aiter", "gluon"))
    parser.add_argument(
        "--schedule",
        choices=("auto", "pipeline", "pingpong", "persistent"),
    )
    parser.add_argument("--scheduler")
    return parser.parse_args()


def main():
    args = parse_args()
    batch, seqlen_q, seqlen_k, num_heads = PRESETS[args.preset]
    batch = args.batch or batch
    seqlen_q = args.sq or seqlen_q
    seqlen_k = args.sk or seqlen_k
    num_heads = args.heads or num_heads

    torch.manual_seed(0)
    query = torch.randn(
        batch,
        seqlen_q,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        seqlen_k,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    print_provenance(query.device)

    config = select_fav3_launch_config(
        batch,
        num_heads,
        seqlen_q,
        seqlen_k=seqlen_k,
    )
    if args.schedule not in (None, "auto"):
        schedule_configs = {
            "pipeline": FAv3LaunchConfig("pipeline", 128, 64, 4, False, ""),
            "pingpong": FAv3LaunchConfig("pingpong", 256, 64, 8, True, ""),
            "persistent": FAv3LaunchConfig(
                "persistent", 128, 256, 4, True, "amdgpu-sched-strategy=iterative-ilp"
            ),
        }
        config = schedule_configs[args.schedule]
    if args.scheduler is not None:
        if config.schedule == "persistent":
            raise ValueError(
                "--scheduler cannot override the persistent kernel scheduler"
            )
        scheduler = (
            ""
            if args.scheduler in ("", "default")
            else f"amdgpu-sched-strategy={args.scheduler}"
        )
        config = config._replace(llvm_fn_attrs=scheduler)

    def gluon_attention():
        return gluon_fav3_attention(
            query,
            key,
            value,
            launch_config=config,
        )

    if args.trace_provider is not None:
        function = (
            (lambda: aiter_attention(query, key, value))
            if args.trace_provider == "aiter"
            else gluon_attention
        )
        function()
        torch.cuda.synchronize()
        function()
        torch.cuda.synchronize()
        print(f"traced {args.trace_provider}")
        return

    if not args.skip_check:
        expected = aiter_attention(query, key, value)
        actual = gluon_attention()
        torch.cuda.synchronize()
        for name, output in (("aiter", expected), ("gluon", actual)):
            if not torch.isfinite(output).all():
                raise AssertionError(f"{name} produced non-finite output")
        check_sampled_fp64(query, key, value, {"aiter": expected, "gluon": actual})
        difference = (actual.float() - expected.float()).abs()
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=0.04,
            atol=0.04,
        )
        print(
            "correctness "
            f"max={difference.max().item():.6f} "
            f"mean={difference.mean().item():.6f}"
        )

    print(f"shape=B{batch},Sq{seqlen_q},Sk{seqlen_k},H{num_heads},D128 config={config}")

    all_functions = {
        "aiter": lambda: aiter_attention(query, key, value),
        "gluon": gluon_attention,
    }
    names = ("aiter", "gluon") if args.provider == "both" else (args.provider,)
    functions = {name: all_functions[name] for name in names}
    timings = {name: [] for name in functions}
    for round_index in range(args.rounds):
        order = (
            names if round_index % 2 == 0 or len(names) == 1 else tuple(reversed(names))
        )
        for name in order:
            milliseconds = triton.testing.do_bench(
                functions[name],
                warmup=args.warmup,
                rep=args.rep,
            )
            timings[name].append(float(milliseconds))

    flops = 4.0 * batch * seqlen_q * seqlen_k * num_heads * 128
    means = {}
    for name in names:
        mean_ms = statistics.fmean(timings[name])
        means[name] = mean_ms
        tflops = flops / (mean_ms * 1e-3) / 1e12
        print(
            f"{name:6s} mean={mean_ms:9.4f} ms "
            f"tflops={tflops:9.1f} runs={timings[name]}"
        )
    if args.provider == "both":
        print(f"speedup={means['aiter'] / means['gluon']:.4f}x")


if __name__ == "__main__":
    main()
