# SPDX-License-Identifier: Apache-2.0
"""Calibrate DPCache schedules for one model and request configuration.

A DPCache schedule is bound to the exact request it was calibrated for
(checkpoint, resolution, step count, scheduler, guidance, attention backend,
dtype), so calibrate once per configuration you serve. Two steps:

  record: generate the calibration prompts natively. Each request scores the
  PACT errors of its final transformer-block features inside the worker and
  writes one capture file; no features leave the worker. Shard across GPUs
  with --shard, one process per GPU, into the same --capture-dir.

    python -m sglang.multimodal_gen.tools.dpcache_calibrate record \\
      --model-path Qwen/Qwen-Image-2.1 --prompts-file prompts.txt \\
      --height 1024 --width 1024 --num-inference-steps 40 \\
      --capture-dir dpcache-captures

  plan: average the captures and write one schedule per budget ``K``.

    python -m sglang.multimodal_gen.tools.dpcache_calibrate plan \\
      --capture-dir dpcache-captures --budgets 12 16 20 24 --out-dir schedules

Serve the schedules with ``--dpcache-schedule-dir`` and pick one per request
with ``dpcache_budget`` (or ``--dpcache-default-budget``). Choose ``K`` on
prompts you did not calibrate on: the planner minimises feature error, not the
image difference you care about.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# default gap bound: scoring cost grows with it, and wider gaps predict poorly
DEFAULT_MAX_GAP = 8


def load_prompts(path: str, default_seed: int) -> list[dict]:
    """One item per generation: ``{"prompt": str, "seed": int}``.

    The file is plain text (one prompt per line) or JSONL with a ``prompt`` and
    an optional ``seed`` per line.
    """
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line) if line.startswith("{") else {"prompt": line}
            items.append(
                {"prompt": item["prompt"], "seed": item.get("seed", default_seed)}
            )
    if not items:
        raise ValueError(f"{path} contains no prompts")
    return items


def shard_indices(count: int, shard: str | None) -> list[int]:
    if shard is None:
        return list(range(count))
    index, total = (int(x) for x in shard.split("/"))
    if not 0 <= index < total:
        raise ValueError(f"--shard must be i/n with 0 <= i < n, got {shard}")
    return list(range(index, count, total))


def capture_path(capture_dir: str, index: int) -> Path:
    # named by the unsharded index, so shards never collide
    return Path(capture_dir) / f"{index:05d}.pt"


def sampling_kwargs(args: argparse.Namespace, item: dict, output: Path) -> dict:
    kwargs = {
        "prompt": item["prompt"],
        "seed": item["seed"],
        "save_output": False,
        "dpcache_calibration": {
            "output": str(output.resolve()),
            "max_gap": args.max_gap or None,
        },
    }
    for name in ("height", "width", "num_inference_steps", "guidance_scale"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    return kwargs


def default_source_commit() -> str:
    import sglang

    package = os.path.dirname(sglang.__file__)
    try:
        return subprocess.check_output(
            ["git", "-C", package, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        from sglang.version import __version__

        return f"sglang-{__version__}"


def run_record(args: argparse.Namespace, unknown_args: list[str]) -> None:
    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    items = load_prompts(args.prompts_file, args.seed)
    indices = shard_indices(len(items), args.shard)
    todo = [i for i in indices if not capture_path(args.capture_dir, i).exists()]
    print(f"{len(indices) - len(todo)} of {len(indices)} captures already exist")
    if not todo:
        return
    os.makedirs(args.capture_dir, exist_ok=True)
    server_args = ServerArgs.from_cli_args(args, unknown_args)
    with DiffGenerator.from_server_args(server_args) as generator:
        for index in todo:
            output = capture_path(args.capture_dir, index)
            generator.generate(
                sampling_params_kwargs=sampling_kwargs(args, items[index], output)
            )
            if not output.exists():
                raise SystemExit(
                    f"calibration request {index} wrote no capture; see the log above"
                )
            print(f"captured {output}")


def run_plan(args: argparse.Namespace) -> None:
    from sglang.multimodal_gen.runtime.cache.dpcache import (
        load_calibration_capture,
        plan_schedules,
    )

    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    captures = [load_calibration_capture(str(p)) for p in paths]
    if not captures:
        raise SystemExit(f"no captures in {args.capture_dir}")
    max_gap = captures[0]["max_gap"] if args.max_gap is None else args.max_gap or None
    schedules = plan_schedules(
        captures,
        args.budgets,
        source_commit=args.source_commit or default_source_commit(),
        mandatory=tuple(range(args.mandatory_steps)),
        max_gap=max_gap,
        force_last_full=args.force_last_full,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    for budget, schedule in schedules.items():
        path = Path(args.out_dir) / f"K{budget}.json"
        path.write_text(json.dumps(schedule, indent=1) + "\n")
        print(
            f"K={budget}: cost {schedule['calibrated_cost']:.4f}, "
            f"full steps {schedule['full_steps']} -> {path}"
        )
    print(f"planned from {len(captures)} captures in {args.capture_dir}")


def build_parser() -> argparse.ArgumentParser:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser(
        "record", help="Generate calibration prompts and write PACT error captures."
    )
    record.add_argument("--prompts-file", required=True)
    record.add_argument("--capture-dir", required=True)
    record.add_argument(
        "--seed", type=int, default=42, help="Seed for prompts without one."
    )
    record.add_argument("--height", type=int)
    record.add_argument("--width", type=int)
    record.add_argument("--num-inference-steps", type=int)
    record.add_argument("--guidance-scale", type=float)
    record.add_argument(
        "--max-gap",
        type=int,
        default=DEFAULT_MAX_GAP,
        help="Longest gap between full steps to score; 0 scores every gap.",
    )
    record.add_argument("--shard", help="Record only shard i of n, as i/n.")
    ServerArgs.add_cli_args(record)

    plan = commands.add_parser(
        "plan", help="Average captures and write one schedule per budget."
    )
    plan.add_argument("--capture-dir", required=True)
    plan.add_argument("--out-dir", required=True)
    plan.add_argument("--budgets", type=int, nargs="+", required=True)
    plan.add_argument(
        "--max-gap",
        type=int,
        help="Gap bound for planning, at most the recorded one; 0 means none. "
        "Defaults to the recorded bound.",
    )
    plan.add_argument(
        "--mandatory-steps",
        type=int,
        default=3,
        help="Leading steps that always run full (at least 2).",
    )
    plan.add_argument("--force-last-full", action="store_true")
    plan.add_argument(
        "--source-commit", help="Provenance; defaults to the sglang git revision."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args, unknown_args = build_parser().parse_known_args(argv)
    if args.command == "record":
        run_record(args, unknown_args)
    else:
        if unknown_args:
            raise SystemExit(f"unrecognized arguments: {' '.join(unknown_args)}")
        run_plan(args)


if __name__ == "__main__":
    main(sys.argv[1:])
