"""
Run live profiling.

Usage:
python3 -m sglang.profiler
"""

import argparse
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import requests

PROFILER_DIR = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _run_profile(
    url: Optional[str],
    num_steps: int,
    activities: List[str],
    output_dir: Optional[str] = None,
    profile_name: Optional[str] = None,
    profile_by_stage: bool = False,
    merge_profiles: bool = False,
) -> str:
    if output_dir is None:
        output_dir = PROFILER_DIR

    output_dir = os.path.normpath(output_dir)
    output_dir = os.path.abspath(output_dir)
    output_dir = Path(output_dir)

    # Add "profile_name/timestamp" to the path.
    if profile_name:
        output_dir = output_dir / profile_name
    output_dir = output_dir / str(time.time())
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Dump profiling traces to {output_dir}")
    print(
        f"Waiting for {num_steps} steps and the trace to be flushed.... ({profile_by_stage=})"
    )

    # Dump server args.
    file_path = Path(output_dir) / "server_args.json"
    if not file_path.exists():
        response = requests.get(url + "/get_server_info")
        response.raise_for_status()
        server_args_data = response.json()
        with open(file_path, "w") as file:
            file.write(json.dumps(server_args_data))

    # Start profiler. The API replies when all steps are processed
    # and files are generated.
    json_data = {
        "output_dir": str(output_dir),
        "num_steps": str(num_steps),
        "activities": activities,
        "profile_by_stage": profile_by_stage,
        "merge_profiles": merge_profiles,
    }

    response = requests.post(url=url + "/start_profile", json=json_data)
    response.raise_for_status()

    trace_link = str(output_dir)
    return trace_link


def run_profile(
    url: Optional[str],
    num_steps: int,
    activities: List[str],
    output_dir: Optional[str] = None,
    profile_name: Optional[str] = None,
    profile_by_stage: bool = False,
    merge_profiles: bool = False,
):
    # step based profile will self terminate on num_steps constraints
    link = _run_profile(
        url,
        num_steps,
        activities,
        output_dir,
        profile_name,
        profile_by_stage,
        merge_profiles,
    )
    return link


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:30000",
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Profile directory to dump profile traces.",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="The name of this profile run.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="The number of forward steps to profile.",
    )
    parser.add_argument(
        "--profile-by-stage",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Whether to profile prefill and decode separately",
    )
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="Whether to profile CPU activity",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="Whether to profile GPU activity",
    )
    parser.add_argument(
        "--mem",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Whether to memory usage (https://pytorch.org/memory_viz)",
    )
    parser.add_argument(
        "--rpd",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Whether to use rpd profiler (https://github.com/ROCm/rocmProfileData)",
    )
    parser.add_argument(
        "--merge-profiles",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Whether to merge profiles from all ranks into a single trace file",
    )

    args = parser.parse_args()
    activities = []
    if args.cpu:
        activities.append("CPU")
    if args.gpu:
        activities.append("GPU")
    if args.mem:
        activities.append("MEM")
    if args.rpd:
        activities.append("RPD")
    run_profile(
        args.url,
        args.num_steps,
        activities,
        args.output_dir,
        args.profile_name,
        args.profile_by_stage,
        args.merge_profiles,
    )
