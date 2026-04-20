#!/usr/bin/env python3
import argparse

try:
    import mooncake

    BENCH_TOOL_PATH = f"{mooncake.__path__[0]}/transfer_engine_bench"
    print(f"Mooncake is installed. Bench tool path:\n{BENCH_TOOL_PATH}")
except ImportError:
    BENCH_TOOL_PATH = None
    print("Mooncake is not installed.")
    exit(0)


def run_cmd(args):
    cmd = [BENCH_TOOL_PATH]
    if args.initiator:
        cmd += ["--mode=initiator"]
    elif args.target:
        cmd += ["--mode=target"]

    if args.metadata_server:
        cmd += [f"--metadata_server={args.metadata_server}"]
    if args.mc_segment_id:
        cmd += [f"--segment_id={args.mc_segment_id}"]
    if args.device:
        cmd += [f"--device_name={args.device}"]

    if args.bench_h2h:
        cmd += ["--use_vram=false"]

    cmd += ["--auto_discovery"]
    print(f"Executing command: {' '.join(cmd)}")
    import subprocess

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        exit(1)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--initiator", action="store_true", help="Run as initiator")
    group.add_argument("--target", action="store_true", help="Run as target")
    parser.add_argument("--metadata-server", type=str, default="P2PHANDSHAKE")
    parser.add_argument("--mc-segment-id", type=str, default=None)
    parser.add_argument("--bench-h2h", action="store_true")
    parser.add_argument("--device", type=str, default="mlx5_0")
    args = parser.parse_args()

    print("Running Mooncake transfer engine benchmark...")
    if not args.initiator and not args.target:
        parser.error("Please specify --initiator or --target")
    if args.initiator and args.mc_segment_id is None:
        parser.error("Please specify --mc-segment-id for initiator")

    run_cmd(args)


if __name__ == "__main__":
    main()
