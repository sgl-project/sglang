"""Compare two native source trees on the CPU detokenizer output pipeline.

No HTTP server, model weights or GPU. Both arms use native IPC, a real tiny
tokenizer and the selected tree's actual multi-worker output loop.
"""

import argparse
import hashlib
import json
import os
import random
import select
import statistics
import subprocess
import sys
from pathlib import Path


def read_reply(process):
    if not select.select([process.stdout], [], [], 90)[0]:
        raise TimeoutError("Controller did not reply")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"Controller exited: {process.poll()}")
    return json.loads(line)


def stop(process):
    if process.poll() is not None:
        return
    try:
        process.stdin.write('{"stop":true}\n')
        process.stdin.flush()
        process.wait(timeout=15)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(args):
    """Randomize paired arm order, retain raw trials and verify all outputs."""
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    processes, logs, sources, raw = {}, [], {}, []
    rng = random.Random(20261002)
    try:
        for arm, source in (
            ("reference", args.reference),
            ("candidate", args.candidate),
        ):
            env = dict(
                os.environ,
                PYTHONPATH=str(source.resolve()),
                SGLANG_USE_PICKLE_IPC="1" if args.transport == "pickle" else "0",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1",
            )
            log = (args.output / f"{arm}-controller.log").open("w")
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "fanout_controller.py"),
                    "--log",
                    str(args.output / f"{arm}-worker.log"),
                ],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
            )
            processes[arm] = process
            sources[arm] = read_reply(process)
            print(f"Initialized {arm}", flush=True)
        with (args.output / "raw.jsonl").open("w") as saved:
            for count in (1, 32, 310):
                for rich in (False, True):
                    for round_index in range(-1, args.rounds):
                        order = list(processes)
                        rng.shuffle(order)
                        command = dict(
                            count=count,
                            rich=rich,
                            round=round_index,
                            repetitions=args.repetitions,
                            seed=10000
                            + count * 100
                            + int(rich) * 100000
                            + round_index * 10,
                        )
                        for arm in order:
                            process = processes[arm]
                            process.stdin.write(json.dumps(command) + "\n")
                            process.stdin.flush()
                            result = dict(read_reply(process), arm=arm)
                            raw.append(result)
                            saved.write(json.dumps(result) + "\n")
                            saved.flush()
                    print(f"Verified count={count}, rich={rich}", flush=True)
        summary = []
        for count in (1, 32, 310):
            for rich in (False, True):
                cells = [
                    row
                    for row in raw
                    if row["count"] == count
                    and row["rich"] == rich
                    and row["round"] >= 0
                ]
                for metric in ("complete_ms", "first_ms"):
                    paired = {
                        arm: {
                            row["round"]: statistics.mean(
                                trial[metric] for trial in row["trials"]
                            )
                            for row in cells
                            if row["arm"] == arm
                        }
                        for arm in processes
                    }
                    reductions = [
                        100 * (1 - paired["candidate"][i] / paired["reference"][i])
                        for i in range(args.rounds)
                    ]
                    summary.append(
                        dict(
                            count=count,
                            rich=rich,
                            metric=metric,
                            reference_ms=statistics.median(
                                paired["reference"].values()
                            ),
                            candidate_ms=statistics.median(
                                paired["candidate"].values()
                            ),
                            median_paired_reduction_pct=statistics.median(reductions),
                            faster_rounds=sum(value > 0 for value in reductions),
                        )
                    )
        result = dict(
            transport=args.transport,
            rounds=args.rounds,
            repetitions=args.repetitions,
            sources=sources,
            verified_rows=sum(row["verified_rows"] for row in raw),
            fence_rows=sum(row["fence_rows"] for row in raw),
            summary=summary,
            scripts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    root / "fanout_controller.py",
                    root / "fanout_fixtures.py",
                ]
            },
            scope="CPU native detokenizer and output fanout, real IPC and tiny tokenizer. First output includes consumer metadata reads. No HTTP request state, scheduler, model or GPU.",
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for process in processes.values():
            stop(process)
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Baseline checkout's python directory",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate checkout's python directory",
    )
    parser.add_argument("--transport", choices=("pickle", "msgpack"), default="msgpack")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 12 or not 1 <= args.repetitions <= 5:
        parser.error("Use 1..12 rounds and 1..5 repetitions")
    if any(
        not (source / "sglang").is_dir() for source in (args.reference, args.candidate)
    ):
        parser.error("Both source paths must contain the sglang package")
    run(args)
