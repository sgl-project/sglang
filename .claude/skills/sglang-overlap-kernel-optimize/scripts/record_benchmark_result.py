#!/usr/bin/env python3
"""
Record benchmark result for an overlap kernel version.

Writes a unified JSON result file (<kernel_name>_result_v<version>.json) that
contains both performance data (if correctness passed) and error info (if any).
If the file already exists for that version, the new data overwrites it.

Usage:
    # Record a successful benchmark (correctness PASSED, with performance data)
    python3 record_benchmark_result.py \
        --kernel_name allgather_gemm_overlap \
        --version v1 \
        --pod sglang-test-pod-abc123 \
        --iteration 1 \
        --status PASSED \
        --max_diff 0.0001 --mean_diff 0.00005 \
        --compute_only_us 1234.5 \
        --comm_only_us 5678.9 \
        --non_overlap_us 6913.4 \
        --overlap_us 3456.7 \
        --speedup 2.0 \
        --overlap_efficiency 0.85 \
        --config M=1024,N=7168,K=2048,dtype=bf16,world_size=2,num_comm_sms=8 \
        --tuning_changes "Initial generation"

    # Record a failed benchmark (correctness FAILED, with error details)
    python3 record_benchmark_result.py \
        --kernel_name allgather_gemm_overlap \
        --version v2 \
        --pod sglang-test-pod-abc123 \
        --iteration 2 \
        --status FAILED \
        --max_diff 0.5 --mean_diff 0.2 \
        --error_type runtime_crash \
        --error_details "CUDA error: an illegal memory access was encountered at kernel line 89" \
        --config M=1024,N=7168,K=2048,dtype=bf16,world_size=2,num_comm_sms=8 \
        --tuning_changes "Reduced BLOCK_M from 64 to 32, added address clamping"

    # Record an environment failure (no benchmark ran at all)
    python3 record_benchmark_result.py \
        --kernel_name allgather_gemm_overlap \
        --version v1 \
        --pod sglang-test-pod-abc123 \
        --iteration 1 \
        --status FAILED \
        --error_type environment_failure \
        --error_details "torch.cuda.is_available() == False; pod has no GPU" \
        --config M=1024,N=7168,K=2048,dtype=bf16,world_size=2,num_comm_sms=8
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def parse_config_string(config_str: str) -> dict:
    """Parse comma-separated key=value pairs into a dict with typed values."""
    if not config_str:
        return {}
    result = {}
    for pair in config_str.split(","):
        key, value = pair.strip().split("=")
        # Try to cast to int, then float, then keep as string
        try:
            result[key] = int(value)
        except ValueError:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def record_result(args: argparse.Namespace) -> None:
    """Write the unified result JSON file."""
    result = {
        "kernel_name": args.kernel_name,
        "version": args.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pod": args.pod,
        "iteration": args.iteration,
        "config": parse_config_string(args.config),
        "correctness": {
            "status": args.status,
            "max_diff": args.max_diff,
            "mean_diff": args.mean_diff,
        },
        "performance": None,
        "errors": None,
        "tuning_changes": args.tuning_changes,
    }

    # Fill performance data if correctness passed
    if args.status == "PASSED":
        result["performance"] = {
            "compute_only_us": args.compute_only_us,
            "comm_only_us": args.comm_only_us,
            "non_overlap_us": args.non_overlap_us,
            "overlap_us": args.overlap_us,
            "speedup": args.speedup,
            "overlap_efficiency": args.overlap_efficiency,
        }

    # Fill error info if any (can coexist with PASSED for minor warnings,
    # but typically only present when FAILED)
    if args.error_type or args.error_details:
        result["errors"] = {
            "error_type": args.error_type or "unknown",
            "error_details": args.error_details or "",
        }
        if args.error_traceback:
            result["errors"]["traceback"] = args.error_traceback
        if args.error_env_info:
            result["errors"]["env_info"] = args.error_env_info

    # Write to output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{args.kernel_name}_result_{args.version}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result written to: {filepath}")
    print(f"  Status: {args.status}")
    if args.status == "PASSED":
        print(f"  Speedup: {args.speedup}x, Overlap: {args.overlap_us} us, Efficiency: {args.overlap_efficiency}")
    else:
        print(f"  Error type: {args.error_type}, Error: {args.error_details}")


def main():
    parser = argparse.ArgumentParser(
        description="Record benchmark result for an overlap kernel version."
    )

    # Core fields
    parser.add_argument("--kernel_name", required=True, help="Kernel name (e.g., allgather_gemm_overlap)")
    parser.add_argument("--version", required=True, help="Version string (e.g., v1, v2)")
    parser.add_argument("--pod", required=True, help="Pod name where benchmark ran")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number in the optimization loop")
    parser.add_argument("--status", required=True, choices=["PASSED", "FAILED"],
                        help="Correctness test status")

    # Correctness metrics
    parser.add_argument("--max_diff", type=float, default=None, help="Max diff from correctness test")
    parser.add_argument("--mean_diff", type=float, default=None, help="Mean diff from correctness test")

    # Performance metrics (only used when status=PASSED)
    parser.add_argument("--compute_only_us", type=float, default=None,
                        help="Compute-only latency in microseconds")
    parser.add_argument("--comm_only_us", type=float, default=None,
                        help="Comm-only latency in microseconds")
    parser.add_argument("--non_overlap_us", type=float, default=None,
                        help="Non-overlap (sequential) latency in microseconds")
    parser.add_argument("--overlap_us", type=float, default=None,
                        help="Overlap kernel latency in microseconds")
    parser.add_argument("--speedup", type=float, default=None,
                        help="Speedup = (compute + comm) / overlap")
    parser.add_argument("--overlap_efficiency", type=float, default=None,
                        help="Overlap efficiency = max(compute, comm) / overlap")

    # Error fields (only used when status=FAILED or when errors occurred)
    parser.add_argument("--error_type", default=None,
                        choices=["compile_error", "cuda_illegal_access", "cuda_misaligned",
                                 "hang_deadlock", "correctness_mismatch", "environment_failure", "unknown"],
                        help="Error category from taxonomy")
    parser.add_argument("--error_details", default=None, help="Error description string")
    parser.add_argument("--error_traceback", default=None, help="Full Python/CUDA traceback")
    parser.add_argument("--error_env_info", default=None, help="Environment info (nvidia-smi, versions, etc.)")

    # Config and metadata
    parser.add_argument("--config", default=None,
                        help="Comma-separated key=value config pairs (e.g., M=1024,N=7168,K=2048)")
    parser.add_argument("--tuning_changes", default=None,
                        help="Description of tuning changes for this version")
    parser.add_argument("--output_dir", default="./results",
                        help="Directory to write result files (default: ./results)")

    args = parser.parse_args()
    record_result(args)


if __name__ == "__main__":
    main()
