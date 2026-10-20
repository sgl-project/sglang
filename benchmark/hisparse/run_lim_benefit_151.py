"""Run the LIM A/B matrix on 151 while auditing target-device process ownership.

Run on the 151 host. Uses only stdlib and the existing container runner. A
foreign process on the target device invalidates the run; only this task's
benchmark process is terminated. Other services are never changed.
"""

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import threading
from pathlib import Path


def device_processes(raw, device):
    # 151 A3 maps physical devices 0..15 to eight two-chip boards.
    board, chip = divmod(device, 2)
    process_table = raw.split("Process id", 1)
    if len(process_table) != 2:
        raise RuntimeError("npu-smi process table missing; cannot audit ownership")
    processes = []
    for line in process_table[1].splitlines():
        match = re.match(r"\|\s+(\d+)\s+(\d+)\s+\|\s+(\d+)\s+\|", line)
        if match and tuple(map(int, match.group(1, 2))) == (board, chip):
            processes.append(int(match.group(3)))
    return processes


def stamp():
    # The 151 host uses Python 3.9, before datetime.UTC was introduced.
    return datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.task_root / "src/benchmark/hisparse/bench_lim_framework.py"
    stop = threading.Event()
    invalid = threading.Event()
    monitor_failures = []

    def ownership():
        raw = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10)
        own, foreign = [], []
        for pid in device_processes(raw, args.device):
            try:
                command = (
                    Path(f"/proc/{pid}/cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode()
                )
                if not command:
                    # npu-smi can retain a just-exited PID whose cmdline is
                    # already empty. Zombies cannot issue more device work.
                    stat = Path(f"/proc/{pid}/stat").read_text()
                    if stat.rsplit(")", 1)[1].split()[0] == "Z":
                        continue
            except FileNotFoundError:
                continue
            (own if str(source) in command else foreign).append(
                {"pid": pid, "command": command}
            )
        return raw, own, foreign

    def monitor():
        while not stop.is_set():
            try:
                raw, own, foreign = ownership()
                with (args.output_dir / "occupancy-raw.log").open("a") as f:
                    f.write(stamp() + "\n" + raw + "\n")
                with (args.output_dir / "occupancy.jsonl").open("a") as f:
                    f.write(
                        json.dumps({"utc": stamp(), "owned": own, "foreign": foreign})
                        + "\n"
                    )
                if foreign:
                    invalid.set()
                    # Verify ownership again immediately before terminating.
                    for item in own:
                        try:
                            command = Path(f"/proc/{item['pid']}/cmdline").read_bytes()
                            if str(source).encode() in command:
                                os.kill(item["pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        except FileNotFoundError:
                            pass
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                monitor_failures.append(str(error))
                invalid.set()
            stop.wait(args.monitor_interval)

    raw, own, foreign = ownership()
    (args.output_dir / "occupancy-before.txt").write_text(raw)
    if own or foreign:
        raise RuntimeError(f"Target device is occupied: own={own}, foreign={foreign}")
    subprocess.run(
        [
            "python3",
            str(args.task_root / "lim-manifest.py"),
            str(args.task_root / "src"),
            str(args.task_root / "lim-files.json"),
            str(args.output_dir / "source-sha256.json"),
        ],
        check=True,
    )
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    # B, prefix, C, TopK set difference, tail, shared layers, hotset layout.
    cases = [
        (1, 32768, 2048, 128, 1, 1, "scattered"),
        (1, 32768, 2048, 1024, 1, 1, "scattered"),
        (4, 32768, 2048, 128, 1, 1, "scattered"),
        (4, 8192, 2048, 1024, 128, 1, "scattered"),
        (4, 32768, 4096, 1024, 1, 1, "scattered"),
        (1, 32768, 4096, 1024, 1, 4, "scattered"),
        (4, 32768, 4096, 1024, 1, 1, "prefix"),
        (1, 32768, 4096, 1024, 1, 4, "prefix"),
        (4, 131072, 4096, 1024, 1, 1, "prefix"),
    ]
    records = []
    started = stamp()
    try:
        for batch, length, capacity, misses, tail, layers, layout in cases:
            if invalid.is_set():
                break
            name = (
                f"b{batch}-l{length}-c{capacity}-m{misses}-t{tail}-x{layers}-{layout}"
            )
            command = [
                "bash",
                str(args.task_root / "run_lim_151.sh"),
                "--modes",
                "li_reference,lim",
                "--batch",
                str(batch),
                "--context",
                str(length),
                "--capacity",
                str(capacity),
                "--misses",
                str(misses),
                "--tail",
                str(tail),
                "--layers",
                str(layers),
                "--hotset-layout",
                layout,
                "--include-new-token-d2h",
                "--warmup",
                "10",
                "--iterations",
                str(args.iterations),
                "--rounds",
                str(args.rounds),
                "--pairs-per-graph",
                "8",
                "--output",
                str(args.output_dir / f"{name}.json"),
            ]
            print(stamp(), "RUN", name, flush=True)
            case_started = stamp()
            with (args.output_dir / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    check=False,
                    env=os.environ | {"NPU_DEVICE": str(args.device)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            record = {
                "name": name,
                "started_utc": case_started,
                "ended_utc": stamp(),
                "returncode": result.returncode,
                "device_contention_observed": invalid.is_set(),
                "command": command,
            }
            records.append(record)
            if result.returncode:
                print((args.output_dir / f"{name}.log").read_text()[-6000:], flush=True)
                break
            data = json.loads((args.output_dir / f"{name}.json").read_text())
            print(
                json.dumps(
                    {
                        "case": name,
                        "us": data["median_us_per_decode"],
                        "change_percent": data[
                            "lim_vs_reference_latency_change_percent"
                        ],
                        "misses": data["actual_steady_topk_misses"],
                    }
                ),
                flush=True,
            )
    finally:
        stop.set()
        thread.join(timeout=15)
        raw, own, foreign = ownership()
        (args.output_dir / "occupancy-after.txt").write_text(raw)
        manifest = {
            "started_utc": started,
            "ended_utc": stamp(),
            "device": args.device,
            "monitor_interval_seconds": args.monitor_interval,
            "monitor_failures": monitor_failures,
            "foreign_process_observed": invalid.is_set() or bool(foreign),
            "cases": records,
            "completed_cases": len(records),
            "expected_cases": len(cases),
            "valid": len(records) == len(cases)
            and all(r["returncode"] == 0 for r in records)
            and not invalid.is_set()
            and not foreign,
        }
        (args.output_dir / "run-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    if not manifest["valid"]:
        raise RuntimeError("Incomplete or contended run; inspect run-manifest.json")
    print("VALID MATRIX", str(args.output_dir), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path("/home/y00970600/personal/hisparse-lim-framework-20261020"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=range(16), default=7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--monitor-interval", type=float, default=2)
    run(parser.parse_args())
